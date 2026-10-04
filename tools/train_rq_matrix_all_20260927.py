"""Expand all 40 training runs of the historical full-pipeline matrix (Supplementary Table S19) into src/pkg_train.py commands.

tools/run_rq_experiment_matrix.py trains only the 28 `new_families` and expects the 12 `reused_frozen_references`
(A6_v2_f100, ResNet18_v2_matched, IMUNet2024_v21_matched) to exist already; their original driver scripts are no longer
available. This script builds the commands for both groups in the same way (config/rq_experiment_matrix_v3.json:
`common` updated by each family's `args`, plus tag and seed).

    python tools/train_rq_matrix_all_20260927.py --dry-run          # print the 40 commands
    python tools/train_rq_matrix_all_20260927.py --only ResNet18_v3_hn_gn_s0
    python tools/train_rq_matrix_all_20260927.py                    # run all (about 40 x 15 min on an Apple M-series GPU)

Checkpoints go to models/generated/ (as in the original runs). Training on MPS is not bitwise deterministic; compare the
validation losses with models/submission_frozen/*.json and the evaluation with results/submission_frozen/ rather than hashes.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def flag(key):
    return '--' + key.replace('_', '-')


def commands():
    m = json.load(open(ROOT / 'config/rq_experiment_matrix_v3.json'))
    out = []
    for group in ('reused_frozen_references', 'new_families'):
        for fam in m[group]:
            for seed in fam['seeds']:
                v = dict(m['common']); v.update(fam['args']); v.update(tag=fam['tag_template'].format(seed=seed), seed=seed)
                cmd = [sys.executable, str(ROOT / 'src/pkg_train.py')]
                for k, x in v.items():
                    cmd += [flag(k), str(x)]
                out.append((v['tag'], cmd))
    assert len(out) == 40 and len({t for t, _ in out}) == 40
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', help='comma-separated tags')
    a = ap.parse_args()
    runs = commands()
    if a.only:
        keep = set(a.only.split(','))
        runs = [r for r in runs if r[0] in keep]
    for tag, cmd in runs:
        print(' '.join(cmd[1:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
