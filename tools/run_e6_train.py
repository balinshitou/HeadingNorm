"""Train the E6 sign-rule ablation arm under the reference training recipe.

This duplicates the recipe of src/sensors_train.py rather than importing it,
because that module records the SHA-256 of src/models.py and src/sensors_frontend.py
in every run's metadata and re-verifies them. Adding the ablated frontend to
either file would invalidate the metadata of every existing frozen run. The
recipe below is kept identical term by term: same split loader, GlobalNorm
statistics, optimiser, schedule, gradient clipping, loss, window sampling stream,
validation cadence and checkpoint-selection rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import pkg_common as P
import pkg_train as T
import e6_frontend as F

PROTOCOL = ROOT / 'config/e6_sign_ablation.json'
OUT = ROOT / 'models/e6_sign_ablation'
SOURCE_FILES = ['tools/run_e6_train.py', 'src/e6_frontend.py', 'src/models.py',
                'src/pkg_train.py', 'src/pkg_common.py',
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


def specs():
    protocol = json.loads(PROTOCOL.read_text())
    out = []
    for family in protocol['families']:
        for seed in family['seeds']:
            out.append({**protocol['common'], **family['args'], 'seed': seed,
                        'tag': family['tag_template'].format(seed=seed),
                        'family': family['id'], 'phase': family['phase']})
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
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    device = P.pick_device()
    train_data, window, train_hours = T.load_split('train', ('ronin',))
    val_data, _, val_hours = T.load_split('val', ('ronin',))
    mu, sd = T.global_norm_stats(train_data)
    net = F.build(spec['frontend'], mu, sd, spec['arch'], spec['pre'], spec['width']).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=spec['lr'],
                                                    total_steps=spec['steps'], pct_start=.15)
    loss_fn = T.make_loss('huber', .27)
    window_rng = np.random.default_rng(seed)
    provenance = dict(spec=spec, protocol_sha256=protocol_sha, source_sha256=sources,
                      gnorm_mu=mu, gnorm_sd=sd, train_hours=train_hours, val_hours=val_hours,
                      parameters=sum(p.numel() for p in net.parameters()), device=str(device),
                      torch=torch.__version__, platform=platform.platform())
    print(f'[training] {tag} parameters={provenance["parameters"]}', flush=True)
    history, best_loss = [], float('inf')
    started = time.monotonic()
    net.train()
    for step in range(1, spec['steps'] + 1):
        x, y = T.sample_batch(train_data, spec['bs'], window_rng, window)
        x, y = x.to(device), y.to(device)
        loss = loss_fn(net(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if step % 2000 == 0 or step == spec['steps']:
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at step {step}')
            vl, ve = T.evaluate(net, val_data, device, loss_fn, window)
            if not np.isfinite([vl, ve]).all():
                raise RuntimeError('Nonfinite validation result')
            history.append(dict(step=step, val_loss=vl, val_verr=ve))
            if vl < best_loss:
                best_loss = vl
                atomic_save(best_path, {**provenance, 'model_state_dict': net.state_dict(),
                                        'best_step': step, 'val_loss': vl, 'val_verr': ve})
            print(f'[validation] {tag} {step}/{spec["steps"]} loss={vl:.6f} velocity={ve:.5f} '
                  f'elapsed={(time.monotonic() - started) / 60:.1f} min', flush=True)
    atomic_json(done, {**provenance, 'history': history, 'best_val_loss': best_loss,
                       'best_step': min(history, key=lambda h: h['val_loss'])['step'],
                       'minutes': (time.monotonic() - started) / 60,
                       'checkpoint_sha256': digest(best_path), 'status': 'complete'})
    print(f'[saved] {done}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    args = ap.parse_args()
    protocol_sha = digest(PROTOCOL)
    sources = {p: digest(ROOT / p) for p in SOURCE_FILES}
    for spec in specs():
        if spec['seed'] < args.seeds:
            train(spec, protocol_sha, sources)


if __name__ == '__main__':
    main()
