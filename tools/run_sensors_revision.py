#!/usr/bin/env python3
"""Run the locked Sensors experiment batch with durable logs and exclusive lock."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import sensors_train as S

LOG = ROOT / 'logs/sensors_v4'


def run(command, logfile, env):
    with logfile.open('a') as stream:
        stream.write('\nRUN ' + json.dumps(command) + '\n')
        stream.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        code = proc.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); inspect {logfile}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['mps','cuda','cpu'], default='mps')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    protocol = json.loads(S.PROTOCOL.read_text())
    runs = S.expanded_runs(protocol)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_hash = S.digest(S.PROTOCOL)
        source_hashes = {p:S.digest(ROOT / p) for p in S.SOURCE_FILES}
        snapshot = LOG / 'training_sources.json'
        evidence = dict(protocol_sha256=protocol_hash, source_sha256=source_hashes)
        if snapshot.exists() and json.loads(snapshot.read_text()) != evidence:
            raise RuntimeError('Training source or protocol changed; create a new revision rather than mix runs')
        S.atomic_json(snapshot, evidence)
        env = dict(os.environ, HN_DEVICE=args.device, PKG_THREADS='4', OMP_NUM_THREADS='4')
        status = dict(pid=os.getpid(), started_at=time.time(), total_runs=len(runs),
                      protocol_sha256=protocol_hash, device=args.device)
        try:
            for index, spec in enumerate(runs):
                if S.digest(S.PROTOCOL) != protocol_hash or any(S.digest(ROOT / p) != h for p,h in source_hashes.items()):
                    raise RuntimeError('Frozen training dependencies changed during batch')
                S.atomic_json(LOG / 'status.json', {**status, 'state':'training',
                              'current_run':spec['tag'], 'run_index':index + 1})
                print(f'[{index+1}/{len(runs)}] {spec["tag"]}', flush=True)
                run([sys.executable, str(ROOT / 'src/sensors_train.py'), '--tag', spec['tag'],
                     '--expected-protocol-sha256', protocol_hash], LOG / (spec['tag'] + '.log'), env)
            for script in ['sensors_evaluate.py', 'sensors_analyze.py', 'build_sensors_manuscript.py']:
                S.atomic_json(LOG / 'status.json', {**status, 'state':'postprocessing','current_step':script})
                run([sys.executable, str(ROOT / 'tools' / script)], LOG / (script + '.log'), env)
            analysis = json.loads((ROOT / 'results/sensors_v4/analysis.json').read_text())
            S.atomic_json(LOG / 'status.json', {**status,
                          'state':'needs_budget_extension' if analysis['budget_material_change'] else 'batch_complete',
                          'finished_at':time.time(), 'author_information_pending':True})
        except BaseException as exc:
            S.atomic_json(LOG / 'status.json', {**status, 'state':'failed', 'error':repr(exc),
                          'failed_at':time.time()})
            raise


if __name__ == '__main__':
    main()
