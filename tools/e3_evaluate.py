#!/usr/bin/env python3
"""E3: evaluate the Transformer and IMUNet yaw-augmentation arms.

Runs exactly the protocol of tools/e1_heading_repeatability.py so the new arms are
directly comparable with the arms already measured. The rotation helpers and the
sequence list are imported from that script rather than copied, and that script is
not modified, so the provenance of the completed E1 evaluations is untouched.
The psi = 0 rows are the ordinary accuracy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import models
import pkg_common as P
import pkg_eval as E
from e1_heading_repeatability import rotate_horizontal, rotate_2d, sequences

MODELS = ROOT / 'models/e3_backbone_frontend'
OUT = ROOT / 'results/e3_backbone_frontend'
FAMILIES = {
    'Transformer + GN + yaw augmentation': 'E3_transformer_yaw_s{}',
    'IMUNet + GN + yaw augmentation': 'E3_imunet_yaw_s{}',
}
ANGLES = [2 * np.pi * k / 8 for k in range(8)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n')
    temp.replace(path)


def load(tag, device):
    path = MODELS / f'{tag}.pt'
    metadata = MODELS / f'{tag}.json'
    if not metadata.exists():
        raise RuntimeError(f'Incomplete training: {tag}')
    recorded = json.loads(metadata.read_text())['checkpoint_sha256']
    if digest(path) != recorded:
        raise RuntimeError(f'Checkpoint digest mismatch: {tag}')
    o = torch.load(path, map_location='cpu', weights_only=False)
    net = models.build(o['arch'], pre=o['pre'], width=o['width'], hnorm=o['hnorm'],
                       nout=o['nout'], gnorm=bool(o['gnorm']), gnorm_mu=o['gnorm_mu'],
                       gnorm_sd=o['gnorm_sd'], rate=o['rate'])
    net.load_state_dict(o['model_state_dict'], strict=True)
    return net.eval().to(device), float(o['win_sec']), float(o['rate']), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    args = ap.parse_args()
    device = P.pick_device()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    files = sequences()
    script_hash = digest(Path(__file__))

    for family, template in FAMILIES.items():
        for seed in range(args.seeds):
            tag = template.format(seed)
            if not (MODELS / f'{tag}.json').exists():
                print(f'[skip, not trained yet] {tag}', flush=True)
                continue
            destination = OUT / f'{tag}.json'
            started = time.monotonic()
            net, win_sec, rate, checkpoint = load(tag, device)
            provenance = dict(model=tag, family=family, seed=seed,
                              checkpoint_sha256=digest(checkpoint), script_sha256=script_hash,
                              angles=[float(a) for a in ANGLES],
                              expected_rows=len(files) * len(ANGLES), device=str(device))
            rows = []
            if destination.exists():
                previous = json.loads(destination.read_text())
                if all(previous.get(k) == v for k, v in provenance.items()):
                    if previous['complete']:
                        print(f'[verified existing] {tag}', flush=True)
                        continue
                    rows = previous['rows']
            finished = {(r['cache_path'], r['angle_index']) for r in rows}

            for path, dataset, split in files:
                relative = str(path.relative_to(ROOT))
                if all((relative, k) in finished for k in range(len(ANGLES))):
                    continue
                with np.load(path, allow_pickle=False) as z:
                    feat = np.asarray(z['feat'], np.float32)
                    tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
                    seq = str(z['seq'])
                    score_rate = float(z['rate']) if 'rate' in z else 200.
                for index, psi in enumerate(ANGLES):
                    if (relative, index) in finished:
                        continue
                    rotated = feat if index == 0 else rotate_horizontal(feat, psi)
                    ids, velocity = E.predict(net, rotated, device, win_sec, rate, chunk=512)
                    if ids is None or not np.isfinite(velocity).all():
                        raise RuntimeError(f'Invalid predictions {tag}/{seq}/{index}')
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
                atomic_json(destination, {**provenance, 'complete': False, 'rows': rows})
            if len(rows) != len(files) * len(ANGLES):
                raise RuntimeError(f'Row count mismatch for {tag}')
            atomic_json(destination, {**provenance, 'complete': True, 'rows': rows,
                                      'elapsed_seconds': time.monotonic() - started})
            print(f'[evaluated] {tag}: {len(rows)} rows; '
                  f'{time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
