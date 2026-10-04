"""在 TLIO 官方测试集上评测冻结的 HN-Transformer、ResNet18 和 IMUNet。

这是事后的跨设备形态压力测试：输入是 TLIO 由 VIO 派生的世界系处理后 IMU，评价只用
开头 10 s 估计无尺度、无镜像的 SE(2) 变换。它与 186 条手机序列主评价严格分开。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import pkg_common as P
import pkg_eval as E


TAGS = [
    *(f"A6_v2_f100_s{seed}" for seed in range(4)),
    *(f"ResNet18_v2_matched_s{seed}" for seed in range(4)),
    *(f"IMUNet2024_v21_matched_s{seed}" for seed in range(4)),
]


def family(tag):
    if tag.startswith("A6_v2_f100_"):
        return "A6-100%"
    if tag.startswith("ResNet18_v2_matched_"):
        return "RoNIN-ResNet18-matched"
    if tag.startswith("IMUNet2024_v21_matched_"):
        return "IMUNet-2024-matched"
    raise ValueError(tag)


def seed_of(tag):
    return int(tag.rsplit("s", 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval_tlio_posthoc_v22.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    files = sorted(P.CACHE_TLIO.glob("tlio__*.npz"))
    if args.limit is not None:
        files = files[:args.limit]
    expected = 36 if args.limit is None else min(args.limit, 36)
    if len(files) != expected:
        raise RuntimeError(f"TLIO official-test cache count={len(files)}, expected={expected}")
    device = P.pick_device()
    rows = []
    for index, tag in enumerate(TAGS, 1):
        net, win_sec, rate, _ = E.load_net(tag, device)
        started = time.time()
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                feat = np.asarray(data["feat"], float)
                tm = np.asarray(data["tm"], float)
                te = np.asarray(data["te"], float)
                gt = np.asarray(data["gt"], float)
                seq = str(data["seq"])
                input_attitude_source = str(data["input_attitude_source"])
            ids, velocity = E.predict(net, feat, device, win_sec, rate)
            if ids is None:
                raise RuntimeError(f"empty TLIO inference: {tag}/{seq}")
            trajectory = P.trajectory_from_velocity(velocity, ids, tm, te)
            aligned, prefix_samples, theta0 = P.align_prefix_rigid_2d(
                trajectory, gt, te, 10.0)
            ate, rte = P.ate_rte(aligned, gt, 200.0)
            tlr, mcs = P.shape_metrics(aligned, gt)
            values = np.asarray([ate, rte, tlr, mcs, theta0], float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite TLIO metric: {tag}/{seq}")
            rows.append({
                "model": tag,
                "family": family(tag),
                "seed": seed_of(tag),
                "dataset": "tlio",
                "split": "official_test",
                "seq": seq,
                "analysis_status": "post_hoc_exploratory_not_confirmatory",
                "device_form_factor": "head_mounted_imu",
                "input_attitude_source": input_attitude_source,
                "align": "tlio_posthoc_prefix_se2_10s",
                "alignment_prefix_samples": prefix_samples,
                "allow_scale": False,
                "allow_reflection": False,
                "ate": ate,
                "rte": rte,
                "tlr": tlr,
                "mcs": mcs,
                "theta0": theta0,
            })
        print(f"[TLIO {index:02d}/{len(TAGS):02d}] {tag}: {len(files)} sequences, "
              f"{time.time() - started:.1f}s", flush=True)
    if len(rows) != len(TAGS) * expected:
        raise RuntimeError(f"TLIO result rows={len(rows)}")
    out = P.RESULTS / args.out
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[TLIO写出] {out.relative_to(P.ROOT)}: {len(rows)} rows")


if __name__ == "__main__":
    main()
