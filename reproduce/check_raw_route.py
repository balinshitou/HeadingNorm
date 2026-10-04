"""Check the raw-data route: re-run the pipeline from the public datasets on a few sequences and compare with the archive.

    python reproduce/check_raw_route.py

Needs the evaluation caches built from the public datasets (data/README.md) and, for the released RoNIN networks, the RoNIN
code and weights (external/README.md). Each step re-computes something the archive contains, writes only to
verification/generated/ or reproduce/output/raw_route/, and compares with the archived value:

  1. tools/archive_small_inputs_20261004.py --check   phone-data statistics (Table S4) and the inputs of Figures 2 and 7
  2. tools/data_stats_20260929.py                     sequence counts, subjects and durations of Table 1
  3. tools/unified_eval_20260913.py --group yawhn --limit 2
                                                      inference with our ResNet18-YawAug checkpoints (4 seeds) and plug-in HN,
                                                      2 sequences per test set; per-sequence ATE, RTE, shape error
  4. tools/hn_plugin_p0_20260912.py --weights ext_resnet --limit 1
                                                      the released RoNIN ResNet under 8 headings and 8 conventions (Table 4, S17)
  5. tools/confirm_anglefair_eval_20260927.py --selfcheck
                                                      three networks on three TLIO-confirm sequences (Tables 8, 9, S21);
                                                      this script runs only on Apple MPS (as the archived runs did) and is
                                                      skipped on other machines
Inference uses HN_DEVICE=auto (CUDA or MPS when available) unless HN_DEVICE is set; on a CPU steps 3-4 take about 30 min.
Agreement: identical files for steps 1-2; per-sequence values within 1e-4 m for steps 3-5 (GPU/CPU kernels differ in the
last digits). After a full re-run of an experiment (README, "Starting from the raw datasets"), run_all.py regenerates every
table and figure from the new records.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
OUT = ROOT / 'reproduce/output/raw_route'
TOL = 1e-4
CACHES = ['data/eval/benchmark', 'data/eval/tlio_posthoc', 'data/eval/imunet_owndata', 'data/eval/confirm_tlio', 'data/eval/confirm_imunet']


DEVICE = {'HN_DEVICE': os.environ.get('HN_DEVICE', 'auto')}


def run(args, env=None):
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **DEVICE, **(env or {})})
    return r, time.time() - t0


def rows(path):
    return {(r['dataset'], r['sequence'], r.get('seed')): r for r in json.load(open(path))['rows']}


def main():
    import hn_paths
    missing = [c for c in CACHES if not any((ROOT / c).glob('*.npz'))]
    if not (hn_paths.RONIN_SRC / 'model_resnet1d.py').exists():
        missing.append(f'RoNIN official code in {hn_paths.RONIN_SRC}')
    if missing:
        print('Not available (see data/README.md and external/README.md):\n  ' + '\n  '.join(missing))
        sys.exit(2)
    OUT.mkdir(parents=True, exist_ok=True)
    lines, bad = [], 0

    # 1. small archives
    r, t = run(['tools/archive_small_inputs_20261004.py', '--check'])
    bad += r.returncode != 0
    lines.append(f"1. small archives (Table S4, Figures 2 and 7): {'identical' if r.returncode == 0 else 'DIFFER'} ({t:.0f} s)")
    lines += ['     ' + l for l in r.stdout.splitlines()]

    # 2. Table 1 statistics: the script is run with its output redirected
    src = (ROOT / 'tools/data_stats_20260929.py').read_text()
    old = "(ROOT / 'results/data_stats_20260929/stats.json').write_text"
    assert src.count(old) == 1
    code = src.replace(old, f"Path({str(OUT / 'stats.json')!r}).write_text")
    r = subprocess.run([sys.executable, '-c', f"__file__ = {str(ROOT / 'tools/data_stats_20260929.py')!r}\n" + code], cwd=ROOT,
                       capture_output=True, text=True)
    same = r.returncode == 0 and json.load(open(OUT / 'stats.json')) == json.load(open(ROOT / 'results/data_stats_20260929/stats.json'))
    bad += not same
    lines.append(f"2. dataset statistics (Table 1): {'identical' if same else 'DIFFER'}")

    # 3. our checkpoints + plug-in HN
    smoke = ROOT / 'verification/generated/unified_eval_smoke'
    shutil.rmtree(smoke, ignore_errors=True)
    r, t = run(['tools/unified_eval_20260913.py', '--group', 'yawhn', '--limit', '2'])
    worst, n = 0.0, 0
    if r.returncode == 0:
        for s in range(4):
            new, arc = rows(smoke / f'ours_yaw_hn_s{s}.json'), rows(ROOT / f'results/unified_eval_20260913/ours_yaw_hn_s{s}.json')
            for k, v in new.items():
                for q in ('ate_u', 'rte_u', 'ate_lit', 'ate_shape'):
                    worst = max(worst, abs(v[q] - arc[k][q])); n += 1
    ok = r.returncode == 0 and n > 0 and worst <= TOL
    bad += not ok
    lines.append(f"3. ResNet18-YawAug + plug-in HN, 4 seeds x 5 test sets x 2 sequences: {n} values, max |difference| "
                 f"{worst:.1e} m {'(ok)' if ok else 'DIFFER' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip().splitlines()[-1]} ({t:.0f} s)")

    # 4. released RoNIN ResNet over 8 headings / 8 conventions
    smoke = ROOT / 'verification/generated/hn_plugin_p0_smoke'
    shutil.rmtree(smoke, ignore_errors=True)
    r, t = run(['tools/hn_plugin_p0_20260912.py', '--weights', 'ext_resnet', '--limit', '1'])
    worst, n = 0.0, 0
    if r.returncode == 0:
        new = {(x['dataset'], x['sequence']): x for x in json.load(open(smoke / 'ext_resnet.json'))['rows']}
        arc = {(x['dataset'], x['sequence']): x for x in json.load(open(ROOT / 'results/hn_plugin_p0_20260912/ext_resnet.json'))['rows']}
        for k, v in new.items():
            for q in ('orig_ate_by_psi', 'hn_ate_by_alpha'):
                for a, b in zip(v[q], arc[k][q]):
                    worst = max(worst, abs(a - b)); n += 1
    ok = r.returncode == 0 and n > 0 and worst <= TOL
    bad += not ok
    lines.append(f"4. RoNIN ResNet (released), 8 headings + 8 conventions, 1 sequence per test set: {n} values, max |difference| "
                 f"{worst:.1e} m {'(ok)' if ok else 'DIFFER' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip().splitlines()[-1]} ({t:.0f} s)")

    # 5. confirmation sets (the script accepts only Apple MPS)
    import torch
    if torch.backends.mps.is_available():
        r, t = run(['tools/confirm_anglefair_eval_20260927.py', '--selfcheck'], {'HN_DEVICE': 'mps'})
        bad += r.returncode != 0
        last = [l for l in r.stdout.splitlines() if l.strip()][-1:] or r.stderr.strip().splitlines()[-1:]
        lines.append(f"5. confirmation sets, 3 networks x 3 TLIO-confirm sequences: {'ok' if r.returncode == 0 else 'DIFFER'} — "
                     f"{last[0] if last else ''} ({t:.0f} s)")
    else:
        lines.append('5. confirmation sets: skipped (tools/confirm_anglefair_eval_20260927.py runs only on Apple MPS)')

    lines.append(f"RESULT: {'the raw-data route reproduces the archived values' if not bad else f'{bad} step(s) differ'}")
    text = '\n'.join(lines)
    print(text)
    (OUT / 'summary.txt').write_text(text + '\n')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
