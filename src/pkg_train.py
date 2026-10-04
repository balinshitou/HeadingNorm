"""可审计训练入口：A6、RoNIN-ResNet18 与数据规模曲线。

默认使用 ``subject_disjoint_v2``：RoNIN 的 88 条公开训练/验证缓存先按
受试者重新划分，验证受试者与训练受试者完全不重合。25/50/100% 实验使用
固定、嵌套的训练受试者顺序；所有架构共享同一份文件清单和训练步数。

历史 ``legacy`` 划分仅用于复算早期实验，不得据此声称验证集受试者独立。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models
import pkg_common as P


SPLIT_CONFIG = P.ROOT / "config" / "splits_subject_disjoint_v2.json"


def _subject(path: Path) -> str:
    return path.stem.split("__", 1)[1].split("_", 1)[0]


def _manifest_hashes():
    out = {}
    manifest = P.ROOT / "provenance" / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def select_files(split, sources, protocol, data_frac):
    """返回冻结缓存；数据比例只应用于训练集。"""
    dirs = {"ronin": P.CACHE_RONIN, "oxiod": P.CACHE_OXIOD}
    if protocol == "subject_disjoint_v2":
        if tuple(sources) != ("ronin",):
            raise ValueError("subject_disjoint_v2 当前只注册 RoNIN 单源实验")
        cfg = json.load(open(SPLIT_CONFIG, encoding="utf-8"))
        wanted = set(cfg["validation_subjects"] if split == "val"
                     else cfg["fraction_subjects"][f"{data_frac:.2f}"])
        files = sorted(P.CACHE_RONIN.glob("*.npz"))
        return [f for f in files if _subject(f) in wanted]
    files = []
    for source in sources:
        files.extend(sorted(dirs[source].glob(f"{split}__*.npz")))
    if split == "train" and data_frac < 1.0:
        raise ValueError("数据规模曲线必须使用 subject_disjoint_v2 的冻结嵌套子集")
    return sorted(files)


def load_split(split, sources, win_sec=1.0, rate=200.0,
               protocol="subject_disjoint_v2", data_frac=1.0):
    """按窗长/采样率重建样本，并保留逐文件证据标识。"""
    step = int(round(200.0 / rate))
    assert abs(200.0 / step - rate) < 1e-6, f"采样率 {rate} 必须是 200 的整数分频"
    window = int(round(win_sec * rate))
    files = select_files(split, sources, protocol, data_frac)
    seqs, hours = [], 0.0
    for path in files:
        with np.load(path) as z:
            feat = np.asarray(z["feat"], np.float32)[::step]
            gt = np.asarray(z["gt"], np.float32)[::step]
            ts = np.asarray(z["ts"], np.float64)[::step]
        n = min(len(feat), len(gt), len(ts)) - window
        if n <= 100:
            continue
        dt = (ts[window:window + n] - ts[:n])[:, None]
        if np.any(dt <= 0):
            continue
        targ = ((gt[window:window + n] - gt[:n]) / dt).astype(np.float32)
        seqs.append((feat, targ, n, path))
        hours += (ts[-1] - ts[0]) / 3600.0
    return seqs, window, hours


def sample_batch(seqs, bs, rng, window, sampling="uniform_window"):
    if sampling == "uniform_window":
        counts = np.fromiter((row[2] for row in seqs), dtype=np.float64)
        idx = rng.choice(len(seqs), size=bs, p=counts / counts.sum())
    else:
        idx = rng.integers(0, len(seqs), bs)
    x = np.empty((bs, window, 6), np.float32)
    y = np.empty((bs, 2), np.float32)
    for b, i in enumerate(idx):
        feat, targ, n, _ = seqs[i]
        j = int(rng.integers(0, n))
        x[b] = feat[j:j + window]
        y[b] = targ[j]
    return torch.from_numpy(x.transpose(0, 2, 1)), torch.from_numpy(y)


def make_loss(name, delta):
    if name == "mse":
        return torch.nn.MSELoss()

    def huber(pred, target):
        residual = pred[:, :2] - target
        absolute = residual.abs()
        return torch.where(absolute <= delta, 0.5 * residual ** 2,
                           delta * (absolute - 0.5 * delta)).mean()
    return huber


def evaluate(net, seqs, dev, loss_fn, window, sampling="uniform_window",
             seed=123, n_batch=40, bs=256):
    """每个检查点使用完全相同的验证窗口。"""
    rng = np.random.default_rng(seed)
    net.eval(); total = error = 0.0; count = 0
    with torch.no_grad():
        for _ in range(n_batch):
            x, y = sample_batch(seqs, bs, rng, window, sampling)
            x, y = x.to(dev), y.to(dev)
            pred = net(x)
            total += float(loss_fn(pred, y)) * len(y)
            error += float((pred[:, :2] - y).norm(dim=1).sum())
            count += len(y)
    net.train()
    return total / count, error / count


def _data_records(seqs):
    manifest = _manifest_hashes()
    records = []
    for *_, path in seqs:
        rel = path.relative_to(P.ROOT).as_posix()
        records.append({"path": rel, "sha256": manifest.get(rel),
                        "subject": _subject(path)})
    if any(row["sha256"] is None for row in records):
        raise RuntimeError("训练输入未进入 SHA-256 清单；请先冻结数据清单")
    return records


def global_norm_stats(seqs):
    """只用当前训练文件计算旋转对称的六通道归一化常数。"""
    count = 0
    total = np.zeros(6, np.float64)
    square = np.zeros(6, np.float64)
    for feat, _, _, _ in seqs:
        values = np.asarray(feat, np.float64)
        count += len(values)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
    raw_mu = total / count
    mu = np.array([0.0, 0.0, raw_mu[2], 0.0, 0.0, raw_mu[5]])
    # Horizontal pairs share one RMS scale around zero, preserving SO(2) symmetry.
    gyro_xy = np.sqrt((square[0] + square[1]) / (2.0 * count))
    acc_xy = np.sqrt((square[3] + square[4]) / (2.0 * count))
    variance = square / count - np.square(raw_mu)
    sd = np.array([gyro_xy, gyro_xy, np.sqrt(max(variance[2], 1e-12)),
                   acc_xy, acc_xy, np.sqrt(max(variance[5], 1e-12))])
    if np.any(sd <= 1e-8) or not np.isfinite(sd).all():
        raise RuntimeError(f"非法GlobalNorm尺度：{sd}")
    return mu.tolist(), sd.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", choices=("transformer", "ronin_resnet18", "imunet2024"),
                    default="transformer")
    ap.add_argument("--sources", default="ronin")
    ap.add_argument("--split-protocol", choices=("subject_disjoint_v2", "legacy"),
                    default="subject_disjoint_v2")
    ap.add_argument("--data-frac", type=float, choices=(0.25, 0.50, 1.00), default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--win-sec", type=float, default=1.0)
    ap.add_argument("--loss", choices=("huber", "mse"), default="huber")
    ap.add_argument("--delta", type=float, default=0.27)
    ap.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    ap.add_argument("--scheduler", choices=("onecycle", "none"), default="onecycle")
    ap.add_argument("--pre", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hnorm", type=int, choices=(0, 1), default=1)
    ap.add_argument("--gnorm", type=int, choices=(0, 1), default=1)
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--sampling", choices=("uniform_window", "uniform_sequence"),
                    default="uniform_window")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    if a.seed < 0:
        raise SystemExit("投稿补强实验必须使用非负显式种子")
    if a.arch in ("ronin_resnet18", "imunet2024") and a.pre:
        raise SystemExit(
            f"{a.arch} 保持六通道拓扑时必须显式传 --pre 0；"
            "HeadingNorm/GlobalNorm 可用于RQ2统一前端主干比较")
    if a.arch in ("ronin_resnet18", "imunet2024") and (
            a.rate != 200.0 or a.win_sec != 1.0):
        raise SystemExit(f"{a.arch} 的冻结拓扑只接受 1 s @ 200 Hz（200点）输入")

    out_dir = P.GENERATED_MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    out, done = out_dir / f"{a.tag}.pt", out_dir / f"{a.tag}.json"
    if done.exists():
        print(f"[跳过] {a.tag} 已训练完成")
        return
    if out.exists():
        print(f"[续跑] 发现不完整权重 {out.name}，从头执行并仅在完成后写元数据", flush=True)

    sources = tuple(s.strip() for s in a.sources.split(",") if s.strip())
    unknown = sorted(set(sources) - {"ronin", "oxiod"})
    if unknown:
        raise SystemExit(f"未知训练数据源：{unknown}")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = P.pick_device()
    train, window, train_hours = load_split(
        "train", sources, a.win_sec, a.rate, a.split_protocol, a.data_frac)
    val, _, val_hours = load_split(
        "val", sources, a.win_sec, a.rate, a.split_protocol, 1.0)
    assert train and val, "训练或验证集为空"
    train_subjects = sorted({_subject(row[3]) for row in train})
    val_subjects = sorted({_subject(row[3]) for row in val})
    overlap = sorted(set(train_subjects) & set(val_subjects))
    if a.split_protocol == "subject_disjoint_v2" and overlap:
        raise RuntimeError(f"受试者泄漏：{overlap}")
    print(f"[数据] 协议={a.split_protocol} 比例={a.data_frac:.2f} 源={sources} "
          f"训练 {len(train)}条/{len(train_subjects)}人/{train_hours:.3f}h；"
          f"验证 {len(val)}条/{len(val_subjects)}人/{val_hours:.3f}h", flush=True)

    gnorm_mu, gnorm_sd = (global_norm_stats(train) if a.gnorm else (None, None))
    if a.gnorm:
        print(f"[归一化] 仅训练受试者 mu={gnorm_mu} sd={gnorm_sd}", flush=True)
    net = models.build(a.arch, pre=bool(a.pre), width=a.width,
                       hnorm=bool(a.hnorm), nout=2, gnorm=bool(a.gnorm),
                       gnorm_mu=gnorm_mu, gnorm_sd=gnorm_sd, rate=a.rate).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    optimizer_cls = torch.optim.AdamW if a.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(net.parameters(), lr=a.lr,
                              **({"weight_decay": 1e-4} if a.optimizer == "adamw" else {}))
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.lr, total_steps=a.steps, pct_start=0.15)
        if a.scheduler == "onecycle" else None)
    loss_fn = make_loss(a.loss, a.delta)
    rng = np.random.default_rng(a.seed)
    print(f"[模型] {a.arch} 参数={n_params:,} loss={a.loss} optimizer={a.optimizer} "
          f"steps={a.steps} seed={a.seed}", flush=True)

    best, history, started = float("inf"), [], time.time()
    every = max(1, a.steps // 10)
    for step in range(1, a.steps + 1):
        x, y = sample_batch(train, a.bs, rng, window, a.sampling)
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(net(x), y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % every == 0 or step == a.steps:
            val_loss, val_verr = evaluate(net, val, dev, loss_fn, window, a.sampling)
            elapsed = time.time() - started
            history.append({"step": step, "val_loss": val_loss, "val_verr": val_verr})
            star = ""
            if val_loss < best:
                best = val_loss
                torch.save({
                    "model_state_dict": net.state_dict(), "arch": a.arch,
                    "pre": bool(a.pre), "width": a.width, "hnorm": bool(a.hnorm),
                    "gnorm": int(a.gnorm), "nout": 2, "win_sec": a.win_sec,
                    "gnorm_mu": gnorm_mu, "gnorm_sd": gnorm_sd,
                    "rate": a.rate, "loss": a.loss, "delta": a.delta,
                    "steps": a.steps, "bs": a.bs, "lr": a.lr,
                    "sources": list(sources), "train_hours": train_hours,
                    "seed": a.seed, "best_step": step, "val_loss": val_loss,
                    "val_verr": val_verr, "split_protocol": a.split_protocol,
                    "data_frac": a.data_frac, "sampling": a.sampling}, out)
                star = " *"
            remaining = elapsed / step * (a.steps - step)
            print(f"  step {step:6d}/{a.steps} val_loss={val_loss:.6f} "
                  f"val_verr={val_verr:.4f}m/s 用时={elapsed/60:.1f}min "
                  f"剩={remaining/60:.1f}min{star}", flush=True)

    split_digest = hashlib.sha256(SPLIT_CONFIG.read_bytes()).hexdigest()
    meta = {
        "schema_version": "2.0", "tag": a.tag, "arch": a.arch,
        "sources": list(sources), "split_protocol": a.split_protocol,
        "split_config_sha256": split_digest, "data_frac_requested": a.data_frac,
        "train_hours": train_hours, "val_hours": val_hours,
        "n_train_seq": len(train), "n_val_seq": len(val),
        "train_subjects": train_subjects, "val_subjects": val_subjects,
        "subject_overlap": overlap, "train_files": _data_records(train),
        "val_files": _data_records(val), "parameters": n_params,
        "steps": a.steps, "batch_size": a.bs, "learning_rate": a.lr,
        "rate_hz": a.rate, "window_seconds": a.win_sec, "loss": a.loss,
        "gnorm_mu": gnorm_mu, "gnorm_sd": gnorm_sd,
        "huber_delta": a.delta if a.loss == "huber" else None,
        "optimizer": a.optimizer, "scheduler": a.scheduler,
        "sampling": a.sampling, "seed": a.seed,
        "best_val_loss": best, "minutes": (time.time() - started) / 60,
        "history": history, "device": str(dev), "python": platform.python_version(),
        "torch": torch.__version__, "platform": platform.platform(), "pid": os.getpid()
    }
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[完成] {a.tag} -> {out.relative_to(P.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
