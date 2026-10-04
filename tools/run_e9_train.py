"""Train the E9 arms: EqNIO learned canonical frame, retrained under the reference recipe (docs/83).

Copied from tools/run_e6_train.py term by term (split loader, GlobalNorm statistics, AdamW, OneCycle,
gradient clipping, Huber 0.27, window sampling stream, validation cadence, checkpoint rule). Differences:
  - the model comes from src/e9_eqnio_frontend.py (GlobalNorm + EqNIO, verbatim model classes);
  - the learning rate is a spec field, chosen from config lr_candidates on seed 0 by validation loss only;
  - --smoke runs 50 steps into verification/generated/e9_smoke and is never used for results.
Usage (HN_DEVICE=mps, at most two processes):
  .venv/bin/python tools/run_e9_train.py --family eqnio_so2 --phase select
  .venv/bin/python tools/run_e9_train.py --family eqnio_so2 --phase main
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
import e9_eqnio_frontend as F

PROTOCOL = ROOT / 'config/e9_eqnio_retrain.json'
OUT = ROOT / 'models/e9_eqnio_retrain'
SELECT = ROOT / 'results/eqnio_retrain_20260919/lr_selection.json'
SOURCE_FILES = ['tools/run_e9_train.py', 'src/e9_eqnio_frontend.py', 'tools/eqnio_models_20260912.py', 'src/models.py',
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


def spec_for(family, lr, seed, steps=None):
    protocol = json.loads(PROTOCOL.read_text())
    fam = next(f for f in protocol['families'] if f['id'] == family)
    common = dict(protocol['common'])
    if steps is not None:
        common['steps'] = steps
    return {**common, 'frontend': fam['frontend'], 'lr': lr, 'seed': seed, 'family': family,
            'phase': fam['phase'], 'tag': protocol['tag_template'].format(family=family, lr=lr, seed=seed)}


def train(spec, protocol_sha, sources, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    tag, seed = spec['tag'], spec['seed']
    done, best_path = out / f'{tag}.json', out / f'{tag}.pt'
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
        if step % min(2000, spec['steps']) == 0 or step == spec['steps']:
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


def best_loss(spec):
    return json.loads((OUT / f"{spec['tag']}.json").read_text())['best_val_loss']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=('eqnio_so2', 'eqnio_o2'))
    ap.add_argument('--phase', required=True, choices=('smoke', 'select', 'main'))
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    protocol_sha = digest(PROTOCOL)
    sources = {p: digest(ROOT / p) for p in SOURCE_FILES}
    if args.phase == 'smoke':
        spec = spec_for(args.family, protocol['lr_candidates'][0], 0, steps=50)
        train(spec, protocol_sha, sources, out=ROOT / 'verification/generated/e9_smoke')
        return
    cands = [spec_for(args.family, lr, 0) for lr in protocol['lr_candidates']]
    if args.phase == 'select':
        for spec in cands:
            train(spec, protocol_sha, sources)
        sel = json.loads(SELECT.read_text()) if SELECT.exists() else {}
        losses = {f"{c['lr']:g}": best_loss(c) for c in cands}
        chosen = min(cands, key=best_loss)['lr']
        sel[args.family] = dict(val_loss_by_lr=losses, chosen_lr=chosen, rule=protocol['lr_selection'])
        SELECT.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(SELECT, sel)
        print(f'[lr selection] {args.family}: {losses} -> {chosen:g}', flush=True)
        return
    chosen = json.loads(SELECT.read_text())[args.family]['chosen_lr']
    for seed in protocol['seeds']:
        train(spec_for(args.family, chosen, seed), protocol_sha, sources)


if __name__ == '__main__':
    main()
