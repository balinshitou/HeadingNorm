"""Aggregate the E1 initial-heading repeatability evaluation.

Primary metric follows the paper: average sequences within a subject, then give
subjects equal weight. Repeatability is reported at two levels:
  * subject-macro ATE computed separately at each yaw angle, then its spread
    across angles (what a reader would see as "the reported accuracy moves");
  * per-sequence spread across angles, aggregated over sequences (what a single
    recording would see).
Seeds are training repeats, so spreads are computed per seed and then averaged
across the four seeds; the seed-to-seed sample standard deviation is reported
alongside so training noise is not mistaken for heading sensitivity.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'results/e1_heading_repeatability'
ABLATION = ROOT / 'results/e6_sign_ablation'
OUT = ROOT / 'results/e1_heading_repeatability/analysis.json'

ORDER = ['GN only', 'GN + yaw augmentation', 'GN + mixed-sensor PCA (adapted)',
         'GN + HeadingNorm without sign rule', 'GN + HeadingNorm',
         'Transformer + HN + GN (no PreFilter)', 'IMUNet + HN + GN']
GROUPS = [('ronin', 'unseen'), ('ridi', ''), ('ronin', 'seen')]
LABEL = {('ronin', 'unseen'): 'independent-subject group',
         ('ridi', ''): 'RIDI', ('ronin', 'seen'): 'training-subject group'}


def subject_macro(rows):
    """Mean over subjects of the within-subject mean ATE."""
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r['subject']].append(r['ate'])
    return mean(mean(v) for v in by_subject.values())


def main():
    files = sorted(SRC.glob('*.json')) + sorted(ABLATION.glob('*.json'))
    if not files:
        raise SystemExit('no E1 evaluation files yet')
    rows = []
    incomplete = []
    for path in files:
        payload = json.loads(path.read_text())
        if 'rows' not in payload:      # e.g. the analytic frame check
            continue
        if not payload.get('complete'):
            incomplete.append(path.name)
            continue
        rows.extend(payload['rows'])
    if incomplete:
        print(f'[note] skipping {len(incomplete)} incomplete file(s): {", ".join(incomplete)}')
    if not rows:
        raise SystemExit('no complete E1 evaluations yet')

    angles = sorted({r['angle_index'] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        cells[(r['family'], r['seed'], r['dataset'], r['split'], r['angle_index'])].append(r)

    report = []
    for family in ORDER:
        seeds = sorted({r['seed'] for r in rows if r['family'] == family})
        if not seeds:
            continue
        for dataset, split in GROUPS:
            per_seed_range, per_seed_worst, per_seed_base = [], [], []
            per_seed_seq_range = []
            for seed in seeds:
                curve = []
                for index in angles:
                    bucket = cells.get((family, seed, dataset, split, index))
                    if not bucket:
                        break
                    curve.append(subject_macro(bucket))
                if len(curve) != len(angles):
                    break
                per_seed_range.append(max(curve) - min(curve))
                per_seed_worst.append(max(curve))
                per_seed_base.append(curve[0])
                # per-sequence spread across angles
                by_sequence = defaultdict(dict)
                for index in angles:
                    for r in cells[(family, seed, dataset, split, index)]:
                        by_sequence[r['seq']][index] = r['ate']
                spreads = [max(v.values()) - min(v.values())
                           for v in by_sequence.values() if len(v) == len(angles)]
                per_seed_seq_range.append(dict(
                    median=float(np.median(spreads)), p95=float(np.percentile(spreads, 95)),
                    maximum=float(max(spreads)), sequences=len(spreads)))
            if len(per_seed_range) != len(seeds):
                continue
            entry = dict(
                family=family, dataset=dataset, split=split, group=LABEL[(dataset, split)],
                seeds=len(seeds), angles=len(angles),
                ate_at_zero=mean(per_seed_base),
                subject_macro_range=mean(per_seed_range),
                subject_macro_range_seed_sd=stdev(per_seed_range) if len(seeds) > 1 else 0.,
                worst_angle_subject_macro=mean(per_seed_worst),
                worst_angle_relative=mean(per_seed_worst) / mean(per_seed_base) - 1.,
                range_relative=mean(per_seed_range) / mean(per_seed_base),
                sequence_range_median=mean(s['median'] for s in per_seed_seq_range),
                sequence_range_p95=mean(s['p95'] for s in per_seed_seq_range),
                sequence_range_max=mean(s['maximum'] for s in per_seed_seq_range))
            report.append(entry)

    OUT.write_text(json.dumps(dict(angles=len(angles), rows=report), indent=1))
    partial = sorted({e['family'] for e in report if e['seeds'] < 4})
    if partial:
        print('\n[WARNING] these families are still incomplete and MUST NOT be quoted:')
        for family in partial:
            n = next(e['seeds'] for e in report if e['family'] == family)
            print(f'    {family}  ({n}/4 seeds)')
        print()
    header = (f"{'frontend':34s} {'seeds':>5s} {'group':28s} {'ATE(psi=0)':>10s} {'range':>8s} "
              f"{'range%':>7s} {'worst':>8s} {'worst%':>7s} {'seq med':>8s} {'seq max':>8s}")
    print(header)
    print('-' * len(header))
    for e in report:
        flag = '' if e['seeds'] == 4 else '  <-- PARTIAL'
        print(f"{e['family']:34s} {e['seeds']:5d} {e['group']:28s} {e['ate_at_zero']:10.3f} "
              f"{e['subject_macro_range']:8.3f} {100*e['range_relative']:6.1f}% "
              f"{e['worst_angle_subject_macro']:8.3f} {100*e['worst_angle_relative']:6.1f}% "
              f"{e['sequence_range_median']:8.3f} {e['sequence_range_max']:8.3f}{flag}")
    print(f'\n[written] {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
