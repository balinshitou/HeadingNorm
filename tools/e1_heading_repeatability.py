"""E1: initial-heading repeatability.

A different initial heading of the upstream reference frame is a global yaw
rotation of the horizontal IMU channels. The physical motion is unchanged, so a
frontend that is exactly yaw-equivariant must return the identical trajectory
score at every angle. This script measures that spread.

Protocol per sequence and per angle psi:
  1. rotate horizontal gyroscope (channels 0,1) and horizontal accelerometer
     (channels 3,4) by psi; vertical channels untouched;
  2. predict window velocities with the frozen checkpoint;
  3. rotate the predicted velocity back by -psi, so the trajectory is expressed
     in the original world frame and the stored ground truth applies unchanged;
  4. integrate and score with the paper's registered protocol
     (RoNIN: translation to the ground-truth start; RIDI: 10 s SE(2) prefix).

psi = 0 reproduces the frozen sensors_v4 evaluation and is kept as a self-check.
This is an evaluation-only experiment: no training, no checkpoint selection and
no change to the frozen result files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import pkg_common as P
import pkg_eval as E
import sensors_train as S
from sensors_evaluate import load

OUT = ROOT / 'results/e1_heading_repeatability'

# Four frontends on the shared ResNet18 backbone are the controlled comparison.
# The three HN backbones test whether exact invariance is backbone independent.
FRONTENDS = {
    'GN only': 'Sensors_v4_resnet_gn_s{}',
    'GN + yaw augmentation': 'Sensors_v4_resnet_yaw_s{}',
    'GN + mixed-sensor PCA (adapted)': 'Sensors_v4_resnet_mixed_pca_s{}',
    'GN + HeadingNorm': 'ResNet18_v3_hn_gn_s{}',
}
BACKBONES = {
    'Transformer + HN + GN (no PreFilter)': 'A6_v3_nopre_s{}',
    'IMUNet + HN + GN': 'IMUNet2024_v3_hn_gn_s{}',
}


def rotate_horizontal(feat, psi):
    """Rotate the horizontal gyro and accelerometer vectors by psi."""
    c, s = float(np.cos(psi)), float(np.sin(psi))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    out = np.array(feat, dtype=np.float64, copy=True)
    out[:, 0:2] = out[:, 0:2] @ rotation.T
    out[:, 3:5] = out[:, 3:5] @ rotation.T
    return np.asarray(out, np.float32)


def rotate_2d(vectors, psi):
    c, s = float(np.cos(psi)), float(np.sin(psi))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(vectors, float) @ rotation.T


def sequences():
    files = []
    for path in sorted(P.CACHE_EVAL.glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            dataset, split = str(z['dataset']), str(z['split'])
        if dataset in {'ronin', 'ridi'}:
            files.append((path, dataset, split))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=int, default=8,
                    help='number of yaw angles uniformly covering [0, 2*pi)')
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--include-backbones', action='store_true')
    ap.add_argument('--limit-per-group', type=int, help='smoke test only')
    ap.add_argument('--tag')
    args = ap.parse_args()

    device = P.pick_device()
    torch.set_num_threads(4)
    angles = [2 * np.pi * k / args.angles for k in range(args.angles)]
    families = dict(FRONTENDS)
    if args.include_backbones:
        families.update(BACKBONES)

    output = OUT if args.limit_per_group is None else ROOT / 'verification/generated/e1_smoke'
    output.mkdir(parents=True, exist_ok=True)
    script_hash = S.digest(Path(__file__))
    files = sequences()

    for family, template in families.items():
        for seed in range(args.seeds):
            tag = template.format(seed)
            if args.tag and tag != args.tag:
                continue
            started = time.monotonic()
            net, win_sec, rate, checkpoint = load(tag, device)
            wanted, counts = [], {}
            for path, dataset, split in files:
                key = (dataset, split)
                if args.limit_per_group and counts.get(key, 0) >= args.limit_per_group:
                    continue
                wanted.append((path, dataset, split))
                counts[key] = counts.get(key, 0) + 1

            destination = output / f'{tag}.json'
            provenance = dict(model=tag, family=family, seed=seed,
                              checkpoint_sha256=S.digest(checkpoint),
                              script_sha256=script_hash, angles=list(map(float, angles)),
                              expected_rows=len(wanted) * len(angles), device=str(device))
            rows = []
            if destination.exists():
                previous = json.loads(destination.read_text())
                if all(previous.get(k) == v for k, v in provenance.items()):
                    if previous['complete']:
                        print(f'[verified existing] {tag}', flush=True)
                        continue
                    rows = previous['rows']
            finished = {(r['cache_path'], r['angle_index']) for r in rows}

            for path, dataset, split in wanted:
                relative = str(path.relative_to(ROOT))
                if all((relative, k) in finished for k in range(len(angles))):
                    continue
                with np.load(path, allow_pickle=False) as z:
                    feat = np.asarray(z['feat'], np.float32)
                    tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
                    seq = str(z['seq'])
                    score_rate = float(z['rate']) if 'rate' in z else 200.
                for index, psi in enumerate(angles):
                    if (relative, index) in finished:
                        continue
                    rotated = feat if index == 0 else rotate_horizontal(feat, psi)
                    ids, velocity = E.predict(net, rotated, device, win_sec, rate, chunk=512)
                    if ids is None or not np.isfinite(velocity).all():
                        raise RuntimeError(f'Invalid predictions {tag}/{seq}/{index}')
                    # express the prediction back in the original world frame
                    restored = velocity if index == 0 else rotate_2d(velocity, -psi)
                    trajectory = P.trajectory_from_velocity(restored, ids, tm, te)
                    aligned, n_prefix, angle0 = P.align_registered(dataset, trajectory, gt, te)
                    ate, rte = P.ate_rte(aligned, gt, score_rate)
                    if not np.isfinite([ate, rte]).all():
                        raise RuntimeError(f'Invalid metrics {tag}/{seq}/{index}')
                    rows.append(dict(model=tag, family=family, seed=seed, dataset=dataset,
                                     split=split, seq=seq, subject=seq.split('_')[0],
                                     cache_path=relative, angle_index=index,
                                     angle_rad=float(psi), ate=ate, rte=rte,
                                     alignment_prefix_samples=n_prefix, theta0=angle0))
                S.atomic_json(destination, {**provenance, 'complete': False, 'rows': rows})
            if len(rows) != len(wanted) * len(angles):
                raise RuntimeError(f'Row count mismatch for {tag}')
            S.atomic_json(destination, {**provenance, 'complete': True, 'rows': rows,
                                        'elapsed_seconds': time.monotonic() - started})
            print(f'[evaluated] {tag}: {len(rows)} rows; '
                  f'{time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
