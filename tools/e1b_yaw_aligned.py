"""E1b: initial-heading repeatability scored with an optimal yaw alignment.

The paper's registered protocol translates RoNIN predictions to the ground-truth
start without rotating them, following the official RoNIN metric. A reviewer may
object that absolute heading is unobservable from inertial data alone, so every
method should be granted a best-fit rotation about the start before scoring; on
RoNIN the registered protocol does not grant one, which could book a pure global
rotation as error.

This script repeats E1 and scores every (sequence, angle) pair twice from the
same forward pass: once with the registered protocol (a bit-exact self-check
against the frozen E1 files) and once with `align_start_yaw`, which places the
start on the ground-truth start and then applies the closed-form optimal
rotation about it, using the whole trajectory rather than a prefix. No scale, no
mirroring. Decision rules are pre-registered in config/e1b_yaw_aligned.md.

Evaluation only: no training, no checkpoint selection, no change to frozen files.
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
from e1_heading_repeatability import (FRONTENDS, BACKBONES, rotate_horizontal,
                                      rotate_2d, sequences)

OUT = ROOT / 'results/e1b_yaw_aligned'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--limit-per-group', type=int, help='smoke test only')
    ap.add_argument('--tag')
    args = ap.parse_args()

    device = P.pick_device()
    torch.set_num_threads(4)
    angles = [2 * np.pi * k / args.angles for k in range(args.angles)]
    families = {**FRONTENDS, **BACKBONES}

    output = OUT if args.limit_per_group is None else ROOT / 'verification/generated/e1b_smoke'
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
                    restored = velocity if index == 0 else rotate_2d(velocity, -psi)
                    trajectory = P.trajectory_from_velocity(restored, ids, tm, te)
                    # (a) the paper's registered protocol, for the bit-exact self-check
                    aligned, n_prefix, angle0 = P.align_registered(dataset, trajectory, gt, te)
                    ate, rte = P.ate_rte(aligned, gt, score_rate)
                    # (b) the most permissive protocol: optimal yaw about the start
                    yaw_aligned, theta_opt = P.align_start_yaw(trajectory, gt)
                    ate_y, rte_y = P.ate_rte(yaw_aligned, gt, score_rate)
                    if not np.isfinite([ate, rte, ate_y, rte_y]).all():
                        raise RuntimeError(f'Invalid metrics {tag}/{seq}/{index}')
                    rows.append(dict(model=tag, family=family, seed=seed, dataset=dataset,
                                     split=split, seq=seq, subject=seq.split('_')[0],
                                     cache_path=relative, angle_index=index,
                                     angle_rad=float(psi), ate=ate, rte=rte,
                                     ate_yaw=ate_y, rte_yaw=rte_y, theta_opt=theta_opt,
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
