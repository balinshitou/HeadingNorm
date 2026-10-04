"""在 50 条自采路线数据上评测本项目模型（不依赖外部绝对路径）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import pkg_common as P
import pkg_eval as E


def load_gt(route: str):
    dense = np.loadtxt(P.SELF_GT / f"{route}_真值密点.csv", delimiter=",", skiprows=1)
    vertices, lengths = [], []
    with open(P.SELF_GT / f"{route}_真值折线.csv", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            fields = line.strip().split(",")
            vertices.append([float(fields[1]), float(fields[2])])
            lengths.append(float(fields[3]))
    return dense, np.asarray(vertices, float), float(sum(lengths))


def evaluate_tag(tag, dev):
    net, win_sec, model_rate, _ = E.load_net(tag, dev)
    rows = []
    for fp in sorted(P.SELF_DATA.glob("*__synced.npz")):
        stem = fp.name.removesuffix("__synced.npz")
        route = stem.split("_")[0]
        dense, vertices, total = load_gt(route)
        with np.load(fp, allow_pickle=False) as z:
            time = np.asarray(z["time"], float)
            feat = np.concatenate([z["gyro_world"], z["acc_world"]], axis=1)
        ids, vel = E.predict(net, feat, dev, win_sec, model_rate)
        if ids is None:
            continue
        tp = time[ids]
        dt = np.diff(tp, prepend=tp[0] - np.median(np.diff(tp)))
        xy = np.cumsum(vel * dt[:, None], axis=0)
        n = min(len(xy), 4000)
        idx = np.linspace(0, len(xy) - 1, n).astype(int)
        u = np.linspace(0, 1, n)
        ug = np.linspace(0, 1, len(dense))
        gt_pair = np.stack([np.interp(u, ug, dense[:, i]) for i in (0, 1)], axis=1)
        aligned, theta0 = P.align_start_yaw(xy[idx], gt_pair)
        path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        end_err = float(np.linalg.norm(aligned[-1] - gt_pair[-1]))
        directions = np.diff(vertices, axis=0)
        directions = directions[np.linalg.norm(directions, axis=1) > 1e-9]
        gt_theta = np.arctan2(directions[:, 1], directions[:, 0])
        rows.append({
            "model": tag,
            "seq": stem,
            "route": route,
            "align": "start_yaw",
            "tlr": path_len / total,
            "path_len": path_len,
            "gt_len": total,
            "end_err": end_err,
            "end_err_pct": 100.0 * end_err / total,
            "theta0": theta0,
            "r4": P.rectilinearity(vel),
            "gt_r4": float(np.abs(np.mean(np.exp(4j * gt_theta)))),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default="eval_self_collected_reproduced.json")
    args = ap.parse_args()
    dev = P.pick_device()
    rows = []
    for tag in args.tags:
        current = evaluate_tag(tag, dev)
        if len(current) != 50:
            raise RuntimeError(f"{tag} 只评测出 {len(current)} 条，自采数据应为 50 条")
        rows.extend(current)
        print(f"[自采] {tag}: 50 条", flush=True)
    out = P.RESULTS / args.out
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[写出] {out.relative_to(P.ROOT)}")


if __name__ == "__main__":
    main()
