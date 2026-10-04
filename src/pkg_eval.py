"""可移植主评测：186 条（RoNIN seen 32 / unseen 32 / RIDI 94 / OxIOD-large 28）。

TLIO 的36条头戴设备官方test序列属于事后跨设备形态检验，由
``eval_tlio_posthoc.py``单独执行，避免与正文注册主矩阵混合。

默认口径：**registered**。RoNIN 只做起点平移；RIDI/OxIOD 分别只用
前 10 s/5 s 估计无尺度、无镜像的 SE(2) 刚体变换。该口径用于正文外部对比。
``start_yaw``（利用全轨迹估计旋转）只用于补充材料中的内部诊断。

每条序列输出：ATE / RTE / TLR / MCS / R₄ / 残差通道分解（速率 vs 方向）。
残差通道分解是本项目独有的仪器：它在**积分之前**直接看误差落在哪个通道，
用来回答"加 OxIOD 改善的是长度通道还是方向通道"。
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkg_common as P
import models


def load_net(tag, dev):
    candidates = (P.CKPT / f"{tag}.pt", P.GENERATED_MODELS / f"{tag}.pt",
                  P.SUBMISSION_MODELS / f"{tag}.pt")
    path = next((p for p in candidates if p.exists()), candidates[0])
    o = torch.load(path, map_location="cpu", weights_only=False)
    net = models.build(o["arch"], pre=o.get("pre", True), width=o.get("width", 1.0),
                       hnorm=o.get("hnorm", False), nout=o.get("nout", 2),
                       gnorm=bool(o.get("gnorm", 0)),
                       gnorm_mu=o.get("gnorm_mu"), gnorm_sd=o.get("gnorm_sd"),
                       rate=float(o.get("rate", 200.0)))
    net.load_state_dict(o["model_state_dict"])
    return net.eval().to(dev), float(o.get("win_sec", 1.0)), float(o.get("rate", 200.0)), o


def predict(net, feat, dev, win_sec, rate, chunk=2048):
    ds = int(round(200.0 / rate)); w = int(round(win_sec * rate))
    f = feat[::ds]
    ids = np.arange(0, len(f) - w, max(1, int(round(rate * 0.05))), dtype=int)
    if len(ids) < 2:
        return None, None
    out = np.empty((len(ids), 2), np.float32)
    with torch.no_grad():
        for i0 in range(0, len(ids), chunk):
            idx = ids[i0:i0 + chunk]
            x = np.stack([f[j:j + w].T for j in idx], 0).astype(np.float32)
            out[i0:i0 + len(idx)] = net(torch.from_numpy(x).to(dev)).cpu().numpy()[:, :2]
    return ids * ds, np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--datasets", default="ronin,ridi,oxiod",
                    help="逗号分隔的数据集；默认评测全部 186 条")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个数据组最多评测多少条（RoNIN seen/unseen分开）；仅用于冒烟测试")
    ap.add_argument("--alignment", choices=("registered", "start_yaw", "both"),
                    default="registered",
                    help="正文默认 registered；start_yaw 仅作补充诊断")
    a = ap.parse_args()
    dev = P.pick_device()
    wanted = {x.strip() for x in a.datasets.split(",") if x.strip()}
    files = []
    per_dataset = {}
    for fp in sorted(P.CACHE_EVAL.glob("*.npz")):
        with np.load(fp, allow_pickle=False) as z:
            dsn, split_name = str(z["dataset"]), str(z["split"])
        if dsn not in wanted:
            continue
        group = (dsn, split_name)
        n = per_dataset.get(group, 0)
        if a.limit is not None and n >= a.limit:
            continue
        files.append(fp)
        per_dataset[group] = n + 1
    print(f"[评测] {len(files)} 条序列 × {len(a.tags)} 个模型", flush=True)
    rows = []
    for tag in a.tags:
        try:
            net, wsec, mrate, meta = load_net(tag, dev)
        except Exception as e:
            print(f"  跳过 {tag}: {type(e).__name__}: {e}"); continue
        t0 = time.time()
        for k, fp in enumerate(files):
            with np.load(fp, allow_pickle=False) as z:
                feat = np.asarray(z["feat"], float); tm = np.asarray(z["tm"], float)
                te = np.asarray(z["te"], float); gt = np.asarray(z["gt"], float)
                rate = float(z["rate"]); dsn = str(z["dataset"])
                seq = str(z["seq"]); split = str(z["split"])
            ids, vel = predict(net, feat, dev, wsec, mrate)
            if ids is None:
                continue
            traj = P.trajectory_from_velocity(vel, ids, tm, te)
            # 残差通道：与训练目标同定义的真值速度
            t0w = tm[ids]
            gx = np.interp(t0w, te, gt[:, 0]); gy = np.interp(t0w, te, gt[:, 1])
            hx = np.interp(t0w + wsec, te, gt[:, 0]); hy = np.interp(t0w + wsec, te, gt[:, 1])
            vg = np.stack([(hx - gx) / wsec, (hy - gy) / wsec], 1)
            rc = P.residual_channels(vel, vg)
            alignments = ("registered", "start_yaw") if a.alignment == "both" else (a.alignment,)
            for alignment in alignments:
                if alignment == "registered":
                    al, n_prefix, th0 = P.align_registered(dsn, traj, gt, te)
                else:
                    al, th0 = P.align_start_yaw(traj, gt)
                    n_prefix = len(gt)  # 明示该诊断使用了完整真值轨迹
                ate, rte = P.ate_rte(al, gt, rate)
                tlr, mcs = P.shape_metrics(al, gt)
                rows.append(dict(model=tag, dataset=dsn, split=split, seq=seq,
                                 align=alignment, alignment_prefix_samples=n_prefix,
                                 ate=ate, rte=rte, tlr=tlr, mcs=mcs, theta0=th0,
                                 r4=P.rectilinearity(vel),
                                 **{f"rc_{k2}": v for k2, v in rc.items()}))
            if (k + 1) % 40 == 0:
                print(f"    {tag}  {k+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
        print(f"  {tag} 完成  {time.time()-t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit("没有生成任何评测记录；请检查模型名与评测缓存")
    out = P.RESULTS / a.out
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[写出] {out.relative_to(P.ROOT)}   {len(rows)} 条记录", flush=True)


if __name__ == "__main__":
    main()
