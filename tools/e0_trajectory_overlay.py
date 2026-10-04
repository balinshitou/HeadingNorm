"""Dump aligned trajectories at eight initial reference directions.

This is an evaluation-only script. It reuses the exact protocol of
tools/e1_heading_repeatability.py, but instead of storing only the ATE and RTE
of each (sequence, angle) pair it stores the aligned trajectory itself, so that
the eight trajectories produced from one recording can be drawn on one axis.

No training, no checkpoint selection, no change to any frozen result file.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import pkg_common as P
import pkg_eval as E
from sensors_evaluate import load
from e1_heading_repeatability import rotate_horizontal, rotate_2d, sequences

OUT = ROOT / 'results/e0_trajectory_overlay'

ARMS = {
    'GN + yaw augmentation': 'Sensors_v4_resnet_yaw_s0',
    'GN + HeadingNorm': 'ResNet18_v3_hn_gn_s0',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=int, default=8)
    ap.add_argument('--seq', action='append', required=True,
                    help='sequence name to dump; repeatable')
    ap.add_argument('--stride', type=int, default=5,
                    help='keep every n-th trajectory sample to bound file size')
    args = ap.parse_args()

    device = P.pick_device()
    torch.set_num_threads(4)
    angles = [2 * np.pi * k / args.angles for k in range(args.angles)]
    OUT.mkdir(parents=True, exist_ok=True)

    files = sequences()
    rows = []
    gt_rows = []
    seen = set()

    for arm, tag in ARMS.items():
        net, win_sec, rate, checkpoint = load(tag, device)
        for path, dataset, split in files:
            with np.load(path, allow_pickle=False) as z:
                seq = str(z['seq'])
                if seq not in args.seq:
                    continue
                feat = np.asarray(z['feat'], np.float32)
                tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
                score_rate = float(z['rate']) if 'rate' in z else 200.
            if seq not in seen:
                seen.add(seq)
                for i in range(0, len(gt), args.stride):
                    gt_rows.append(dict(seq=seq, dataset=dataset, split=split,
                                        sample=i, x=float(gt[i, 0]), y=float(gt[i, 1])))
            for index, psi in enumerate(angles):
                rotated = feat if index == 0 else rotate_horizontal(feat, psi)
                ids, velocity = E.predict(net, rotated, device, win_sec, rate, chunk=512)
                restored = velocity if index == 0 else rotate_2d(velocity, -psi)
                trajectory = P.trajectory_from_velocity(restored, ids, tm, te)
                aligned, _, _ = P.align_registered(dataset, trajectory, gt, te)
                ate, rte = P.ate_rte(aligned, gt, score_rate)
                for i in range(0, len(aligned), args.stride):
                    rows.append(dict(arm=arm, model=tag, seq=seq, dataset=dataset,
                                     split=split, angle_index=index,
                                     angle_deg=round(np.degrees(psi), 1),
                                     ate=round(ate, 4), rte=round(rte, 4),
                                     sample=i, x=float(aligned[i, 0]),
                                     y=float(aligned[i, 1])))
                print(f'[{arm}] {seq} psi={np.degrees(psi):5.1f} '
                      f'ate={ate:.3f} rte={rte:.3f}', flush=True)

    with (OUT / 'trajectories.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / 'ground_truth.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(gt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gt_rows)
    print(f'wrote {len(rows)} trajectory rows and {len(gt_rows)} ground-truth rows to {OUT}')


if __name__ == '__main__':
    main()
