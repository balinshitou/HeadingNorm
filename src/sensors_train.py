"""Auditable, resumable training for the bounded Sensors revision protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import pkg_common as P
import pkg_train as T
import sensors_frontend as F

ROOT = P.ROOT
PROTOCOL = ROOT / 'config/sensors_revision_v4.json'
OUT = ROOT / 'models/sensors_v4'
SOURCE_FILES = ['src/sensors_train.py', 'src/sensors_frontend.py', 'src/models.py',
                'src/imunet_baseline.py', 'src/pkg_train.py', 'src/pkg_common.py',
                'config/splits_subject_disjoint_v2.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def atomic_save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def expanded_runs(protocol):
    out = []
    for family in protocol['families']:
        for seed in family['seeds']:
            spec = {**protocol['common'], **family['args'], 'seed': seed,
                    'tag': family['tag_template'].format(seed=seed),
                    'family': family['id'], 'phase': family['phase']}
            out.append(spec)
    return out


def train(spec, protocol_sha, source_hashes):
    OUT.mkdir(parents=True, exist_ok=True)
    tag, seed = spec['tag'], spec['seed']
    done, best_path = OUT / f'{tag}.json', OUT / f'{tag}.pt'
    last_path, progress = OUT / f'{tag}.last.pt', OUT / f'{tag}.progress.json'
    if done.exists():
        old = json.loads(done.read_text())
        if old['spec'] != spec or old['protocol_sha256'] != protocol_sha or old['source_sha256'] != source_hashes:
            raise RuntimeError(f'Existing run metadata does not match: {tag}')
        if digest(best_path) != old['checkpoint_sha256']:
            raise RuntimeError(f'Existing checkpoint changed: {tag}')
        print(f'[complete, verified] {tag}', flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    dev = P.pick_device()
    train_data, window, train_hours = T.load_split('train', ('ronin',))
    val_data, _, val_hours = T.load_split('val', ('ronin',))
    mu, sd = T.global_norm_stats(train_data) if spec['gnorm'] else (None, None)
    net = F.build(spec['frontend'], mu, sd, spec['arch'], spec['pre'], spec['width']).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=spec['lr'],
                                                   total_steps=spec['steps'], pct_start=.15)
    loss_fn = T.make_loss('huber', .27)
    # Independent RNG streams retain identical window sampling across augmentation arms.
    window_rng = np.random.default_rng(seed)
    aug_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260906]))
    start_step, history, best_loss, elapsed_before = 0, [], float('inf'), 0.0
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if saved['spec'] != spec or saved['protocol_sha256'] != protocol_sha or saved['source_sha256'] != source_hashes:
            raise RuntimeError(f'Resume provenance mismatch: {tag}')
        net.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'])
        if dev.type == 'mps':
            torch.mps.set_rng_state(saved['accelerator_rng'])
        elif dev.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['accelerator_rng'])
        window_rng.bit_generator.state = saved['window_rng']
        aug_rng.bit_generator.state = saved['aug_rng']
        start_step, history = saved['step'], saved['history']
        best_loss, elapsed_before = saved['best_loss'], saved['elapsed_seconds']
        print(f'[resume] {tag} at step {start_step}', flush=True)
    provenance = dict(spec=spec, protocol_sha256=protocol_sha, source_sha256=source_hashes,
                      gnorm_mu=mu, gnorm_sd=sd, train_hours=train_hours, val_hours=val_hours,
                      train_files=T._data_records(train_data), val_files=T._data_records(val_data),
                      parameters=sum(p.numel() for p in net.parameters()), device=str(dev),
                      torch=torch.__version__, platform=platform.platform())
    print(f'[training] {tag} {spec} parameters={provenance["parameters"]}', flush=True)
    started = time.monotonic()
    net.train()
    for step in range(start_step + 1, spec['steps'] + 1):
        x, y = T.sample_batch(train_data, spec['bs'], window_rng, window)
        if spec['yaw_augmentation']:
            angles = torch.from_numpy(aug_rng.uniform(-np.pi, np.pi, len(y)).astype(np.float32))
            x, y = F.rotate_batch(x, y, angles)
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at step {step}; retain this failed run')
            elapsed = elapsed_before + time.monotonic() - started
            atomic_json(progress, dict(tag=tag, step=step, total_steps=spec['steps'],
                                      elapsed_seconds=elapsed, last_validation=history[-1] if history else None))
        if step % 2000 == 0 or step == spec['steps']:
            vl, ve = T.evaluate(net, val_data, dev, loss_fn, window)
            if not np.isfinite([vl, ve]).all():
                raise RuntimeError('Nonfinite validation result')
            elapsed = elapsed_before + time.monotonic() - started
            history.append(dict(step=step, val_loss=vl, val_verr=ve))
            if vl < best_loss:
                best_loss = vl
                atomic_save(best_path, {**provenance, 'model_state_dict': net.state_dict(),
                                        'best_step': step, 'val_loss': vl, 'val_verr': ve})
            accelerator_rng = (torch.mps.get_rng_state() if dev.type == 'mps' else
                               torch.cuda.get_rng_state_all() if dev.type == 'cuda' else None)
            atomic_save(last_path, {**provenance, 'model_state_dict': net.state_dict(),
                                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                   'torch_rng': torch.get_rng_state(), 'accelerator_rng': accelerator_rng,
                                   'window_rng': window_rng.bit_generator.state,
                                   'aug_rng': aug_rng.bit_generator.state, 'step': step, 'history': history,
                                   'best_loss': best_loss, 'elapsed_seconds': elapsed})
            print(f'[validation] {tag} {step}/{spec["steps"]} loss={vl:.6f} velocity={ve:.5f} '
                  f'elapsed={elapsed / 60:.1f} min', flush=True)
    atomic_json(done, {**provenance, 'history': history, 'best_val_loss': best_loss,
                       'best_step': min(history, key=lambda h:h['val_loss'])['step'],
                       'minutes': (elapsed_before + time.monotonic() - started) / 60,
                       'checkpoint_sha256': digest(best_path), 'status': 'complete'})
    print(f'[saved] {done}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--expected-protocol-sha256', required=True)
    args = ap.parse_args()
    if digest(PROTOCOL) != args.expected_protocol_sha256:
        raise SystemExit('Protocol changed after dispatch')
    protocol = json.loads(PROTOCOL.read_text())
    specs = {s['tag']: s for s in expanded_runs(protocol)}
    spec = specs[args.tag]
    sources = {p:digest(ROOT / p) for p in SOURCE_FILES}
    train(spec, args.expected_protocol_sha256, sources)


if __name__ == '__main__':
    main()
