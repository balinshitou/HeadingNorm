#!/usr/bin/env python3
"""E3: train the yaw-augmentation arms for the Transformer and IMUNet backbones.

The manuscript compares four horizontal-reference-direction frontends on a single
ResNet18 backbone, and separately reports the Transformer and IMUNet backbones with
HeadingNorm only. The cross-backbone statement therefore has no augmentation control.
This script supplies it, completing a 3 backbone x 2 frontend factorial design.

The reference HeadingNorm arms for these backbones (A6_v3_nopre_s*, IMUNet2024_v3_hn_gn_s*)
were produced by src/pkg_train.py under config/rq_experiment_matrix_v3.json. The recipe
below duplicates that module term by term -- same split loader, GlobalNorm statistics,
optimiser, OneCycle schedule, gradient clipping, Huber loss, window sampling stream,
validation cadence, fixed validation windows and checkpoint-selection rule -- and adds
only the per-window yaw augmentation, drawn from an independent generator constructed
exactly as in src/sensors_train.py. The recipe is duplicated rather than imported
because src/pkg_train.py has no augmentation hook and every frozen run records the
SHA-256 of the source files it used; editing them would invalidate that metadata.

Checkpoints are written in the src/pkg_train.py checkpoint format so that
src/pkg_eval.py and models.build reconstruct them without any special case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import models
import pkg_common as P
import pkg_train as T
from sensors_frontend import rotate_batch

PROTOCOL = ROOT / 'config/e3_backbone_frontend.json'
OUT = ROOT / 'models/e3_backbone_frontend'
SOURCE_FILES = ['tools/run_e3_train.py', 'src/models.py', 'src/pkg_train.py',
                'src/pkg_common.py', 'src/sensors_frontend.py',
                'config/splits_subject_disjoint_v2.json']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def atomic_save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def specs():
    protocol = json.loads(PROTOCOL.read_text())
    out = []
    for family in protocol['families']:
        for seed in family['seeds']:
            out.append({**protocol['common'], **family['args'], 'seed': seed,
                        'tag': family['tag_template'].format(seed=seed),
                        'family': family['id'], 'phase': family['phase'],
                        'reference_arm': family['reference_arm'].format(seed=seed)})
    return out


def train(spec, protocol_sha, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    tag, seed = spec['tag'], spec['seed']
    done, best_path = OUT / f'{tag}.json', OUT / f'{tag}.pt'
    if done.exists():
        old = json.loads(done.read_text())
        if old['spec'] != spec or old['source_sha256'] != sources:
            raise RuntimeError(f'Existing run metadata does not match: {tag}')
        if digest(best_path) != old['checkpoint_sha256']:
            raise RuntimeError(f'Existing checkpoint changed: {tag}')
        print(f'[complete, verified] {tag}', flush=True)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = P.pick_device()

    train_data, window, train_hours = T.load_split(
        'train', ('ronin',), spec['win_sec'], spec['rate'],
        spec['split_protocol'], spec['data_frac'])
    val_data, _, val_hours = T.load_split(
        'val', ('ronin',), spec['win_sec'], spec['rate'], spec['split_protocol'], 1.0)
    train_subjects = sorted({T._subject(row[3]) for row in train_data})
    val_subjects = sorted({T._subject(row[3]) for row in val_data})
    overlap = sorted(set(train_subjects) & set(val_subjects))
    if overlap:
        raise RuntimeError(f'Subject leakage: {overlap}')

    mu, sd = T.global_norm_stats(train_data)
    net = models.build(spec['arch'], pre=bool(spec['pre']), width=spec['width'],
                       hnorm=bool(spec['hnorm']), nout=2, gnorm=bool(spec['gnorm']),
                       gnorm_mu=mu, gnorm_sd=sd, rate=spec['rate']).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=spec['lr'], total_steps=spec['steps'], pct_start=0.15)
    loss_fn = T.make_loss(spec['loss'], spec['delta'])

    window_rng = np.random.default_rng(seed)
    aug_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260906]))

    n_params = sum(p.numel() for p in net.parameters())
    provenance = dict(
        spec=spec, protocol_sha256=protocol_sha, source_sha256=sources,
        split_config_sha256=digest(T.SPLIT_CONFIG),
        train_hours=train_hours, val_hours=val_hours,
        n_train_seq=len(train_data), n_val_seq=len(val_data),
        train_subjects=train_subjects, val_subjects=val_subjects, subject_overlap=overlap,
        train_files=T._data_records(train_data), val_files=T._data_records(val_data),
        gnorm_mu=mu, gnorm_sd=sd, parameters=n_params, device=str(device),
        python=platform.python_version(), torch=torch.__version__,
        platform=platform.platform(), pid=os.getpid())
    print(f'[training] {tag} arch={spec["arch"]} parameters={n_params:,} '
          f'reference={spec["reference_arm"]}', flush=True)

    # Checkpoint payload keys follow src/pkg_train.py so pkg_eval.load_net works unchanged.
    checkpoint_header = dict(
        arch=spec['arch'], pre=bool(spec['pre']), width=spec['width'],
        hnorm=bool(spec['hnorm']), gnorm=int(spec['gnorm']), nout=2,
        win_sec=spec['win_sec'], gnorm_mu=mu, gnorm_sd=sd, rate=spec['rate'],
        loss=spec['loss'], delta=spec['delta'], steps=spec['steps'], bs=spec['bs'],
        lr=spec['lr'], sources=['ronin'], train_hours=train_hours, seed=seed,
        split_protocol=spec['split_protocol'], data_frac=spec['data_frac'],
        sampling=spec['sampling'], yaw_augmentation=True)

    best, history, started = float('inf'), [], time.time()
    every = max(1, spec['steps'] // 10)
    net.train()
    for step in range(1, spec['steps'] + 1):
        x, y = T.sample_batch(train_data, spec['bs'], window_rng, window, spec['sampling'])
        angles = torch.from_numpy(aug_rng.uniform(-np.pi, np.pi, len(y)).astype(np.float32))
        x, y = rotate_batch(x, y, angles)
        x, y = x.to(device), y.to(device)
        loss = loss_fn(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % every == 0 or step == spec['steps']:
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at step {step}; retain this failed run')
            val_loss, val_verr = T.evaluate(net, val_data, device, loss_fn, window, spec['sampling'])
            if not np.isfinite([val_loss, val_verr]).all():
                raise RuntimeError('Nonfinite validation result')
            history.append(dict(step=step, val_loss=val_loss, val_verr=val_verr))
            star = ''
            if val_loss < best:
                best = val_loss
                atomic_save(best_path, {**checkpoint_header,
                                        'model_state_dict': net.state_dict(),
                                        'best_step': step, 'val_loss': val_loss,
                                        'val_verr': val_verr})
                star = ' *'
            elapsed = time.time() - started
            print(f'  {tag} step {step:6d}/{spec["steps"]} val_loss={val_loss:.6f} '
                  f'val_verr={val_verr:.4f} elapsed={elapsed/60:.1f}min'
                  f' remaining={elapsed/step*(spec["steps"]-step)/60:.1f}min{star}', flush=True)

    atomic_json(done, {**provenance, 'history': history, 'best_val_loss': best,
                       'best_step': min(history, key=lambda h: h['val_loss'])['step'],
                       'minutes': (time.time() - started) / 60,
                       'checkpoint_sha256': digest(best_path), 'status': 'complete'})
    print(f'[saved] {done}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--family', help='restrict to one family id')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    protocol_sha = digest(PROTOCOL)
    sources = {p: digest(ROOT / p) for p in SOURCE_FILES}
    for spec in specs():
        if spec['seed'] >= args.seeds:
            continue
        if args.family and spec['family'] != args.family:
            continue
        if args.dry_run:
            print(spec['tag'], spec['arch'], 'ref=', spec['reference_arm'])
            continue
        train(spec, protocol_sha, sources)


if __name__ == '__main__':
    main()
