#!/usr/bin/env python3
"""E3: the 3 backbone x 2 frontend factorial, from the psi-sweep evaluations.

Every arm on both sides of every contrast is scored by the same script
(tools/e1_heading_repeatability.py for the four arms that already existed,
tools/e3_evaluate.py for the two new ones), on the same sequences, with the same
registered alignment. The psi = 0 rows are the ordinary accuracy evaluation, so
accuracy and repeatability come from one pass and cannot drift apart.

Primary quantity: subject-macro ATE, i.e. average sequences within a subject,
then give subjects equal weight, after averaging the four training seeds.
Uncertainty: paired subject-cluster bootstrap, conditional on the saved runs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / 'results/e1_heading_repeatability', ROOT / 'results/e3_backbone_frontend']
OUT = ROOT / 'results/e3_backbone_frontend/analysis.json'
BOOTSTRAP_SEED = 20260906   # the project's established bootstrap stream

BACKBONES = {
    'ResNet18': {'hn': 'ResNet18_v3_hn_gn_s{}', 'yaw': 'Sensors_v4_resnet_yaw_s{}'},
    'Transformer': {'hn': 'A6_v3_nopre_s{}', 'yaw': 'E3_transformer_yaw_s{}'},
    'IMUNet': {'hn': 'IMUNet2024_v3_hn_gn_s{}', 'yaw': 'E3_imunet_yaw_s{}'},
}
GROUPS = [('ronin', 'unseen', 'independent-subject group'),
          ('ridi', '', 'RIDI'),
          ('ronin', 'seen', 'training-subject group')]
SEEDS = (0, 1, 2, 3)


def load_rows():
    rows = []
    for directory in SOURCES:
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.json')):
            if path.name in {'analysis.json', 'analytic_frame_check.json'}:
                continue
            payload = json.loads(path.read_text())
            if 'rows' not in payload:
                continue
            if not payload.get('complete'):
                print(f'[note] skipping incomplete {path.name}')
                continue
            rows.extend(payload['rows'])
    return rows


def cells(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r['model'], r['dataset'], r['split'], r['angle_index'])].append(r)
    return out


def subject_table(bucket, metric):
    """subject -> mean over that subject's sequences."""
    by = defaultdict(list)
    for r in bucket:
        by[r['subject']].append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def arm_values(cell, template, dataset, split, metric, angle=0):
    """subject -> array over the four seeds; None when an arm is missing."""
    tables = []
    for seed in SEEDS:
        bucket = cell.get((template.format(seed), dataset, split, angle))
        if not bucket:
            return None
        tables.append(subject_table(bucket, metric))
    keys = set(tables[0])
    if any(set(t) != keys for t in tables):
        raise RuntimeError(f'Subject sets differ across seeds for {template}')
    return {s: np.array([t[s] for t in tables]) for s in sorted(keys)}


def sequence_values(cell, template, dataset, split, metric, angle=0):
    per_seed = []
    for seed in SEEDS:
        bucket = cell.get((template.format(seed), dataset, split, angle))
        if not bucket:
            return None
        per_seed.append(np.mean([float(r[metric]) for r in bucket]))
    return np.array(per_seed)


def effect(treatment, control):
    if treatment.keys() != control.keys():
        raise RuntimeError('Unpaired subject sets')
    keys = sorted(treatment)
    a = np.array([treatment[k] for k in keys])
    b = np.array([control[k] for k in keys])
    per_subject = (a - b).mean(axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = per_subject[rng.integers(0, len(keys), (10000, len(keys)))].mean(axis=1)
    return dict(n_subjects=len(keys),
                treatment_mean=float(a.mean()), control_mean=float(b.mean()),
                difference=float(per_subject.mean()),
                conditional_subject_bootstrap95=np.quantile(draws, [.025, .975]).tolist(),
                improved_subjects=int((per_subject < 0).sum()),
                paired_seed_differences=(a - b).mean(axis=0).tolist())


def spread(cell, template, dataset, split):
    """Per-seed per-sequence cross-angle ATE range, averaged over seeds."""
    medians, maxima, relatives = [], [], []
    for seed in SEEDS:
        per_seq = defaultdict(dict)
        for angle in range(8):
            bucket = cell.get((template.format(seed), dataset, split, angle))
            if not bucket:
                return None
            for r in bucket:
                per_seq[r['seq']][angle] = float(r['ate'])
        ranges, rel = [], []
        for seq, v in per_seq.items():
            if len(v) != 8:
                continue
            ranges.append(max(v.values()) - min(v.values()))
            rel.append((max(v.values()) - min(v.values())) / v[0] if v[0] > 0 else np.nan)
        medians.append(float(np.median(ranges)))
        maxima.append(float(np.max(ranges)))
        relatives.append(float(np.nanmedian(rel)))
    return dict(sequence_range_median=float(np.mean(medians)),
                sequence_range_max=float(np.mean(maxima)),
                sequence_range_median_relative=float(np.mean(relatives)))


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit('no evaluation files found')
    cell = cells(rows)

    summaries, contrasts, repeatability, missing = [], [], [], []
    for backbone, arms in BACKBONES.items():
        for dataset, split, label in GROUPS:
            for metric in ('ate', 'rte'):
                values = {}
                for arm, template in arms.items():
                    v = arm_values(cell, template, dataset, split, metric)
                    if v is None:
                        missing.append(f'{backbone}/{arm}/{label}/{metric}')
                        continue
                    seq = sequence_values(cell, template, dataset, split, metric)
                    subject_seed = np.array(list(v.values())).mean(axis=0)
                    summaries.append(dict(
                        backbone=backbone, arm=arm, dataset=dataset, split=split,
                        group=label, metric=metric, subjects=len(v),
                        subject_macro_mean=float(subject_seed.mean()),
                        subject_macro_seed_sd=float(subject_seed.std(ddof=1)),
                        sequence_macro_mean=float(seq.mean()),
                        sequence_macro_seed_sd=float(seq.std(ddof=1))))
                    values[arm] = v
                if {'hn', 'yaw'} <= set(values):
                    contrasts.append(dict(
                        backbone=backbone, treatment='GN + HeadingNorm',
                        control='GN + yaw augmentation', dataset=dataset, split=split,
                        group=label, metric=metric,
                        primary=(metric == 'ate' and (dataset == 'ridi' or split == 'unseen')),
                        **effect(values['hn'], values['yaw'])))
        for dataset, split, label in GROUPS[:2]:
            for arm, template in arms.items():
                s = spread(cell, template, dataset, split)
                if s:
                    repeatability.append(dict(backbone=backbone, arm=arm, dataset=dataset,
                                              split=split, group=label, **s))

    sign_consistent = None
    primary = [c for c in contrasts if c['primary']]
    covered = {c['backbone'] for c in primary}
    if covered == set(BACKBONES):
        sign_consistent = bool(len({np.sign(c['difference']) for c in primary}) == 1)

    payload = dict(
        protocol='config/e3_backbone_frontend.json',
        backbones=[b for b in BACKBONES],
        bootstrap_seed=BOOTSTRAP_SEED,
        summaries=summaries, contrasts=contrasts, repeatability=repeatability,
        missing_arms=missing,
        primary_sign_consistent_across_backbones=sign_consistent,
        note=('Subject percentile bootstrap conditions on four saved training runs per arm. '
              'Subject identity is the sequence-name prefix. No p-values. The psi = 0 rows of '
              'the heading sweep are the ordinary accuracy evaluation.'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')

    if missing:
        print('[WARNING] missing arms (not yet trained or evaluated):')
        for m in sorted(set(missing)):
            print('   ', m)
    header = f"{'backbone':12s} {'group':28s} {'metric':6s} {'HN':>8s} {'yaw':>8s} {'diff':>8s} {'95% interval':>20s}"
    print(header)
    print('-' * len(header))
    for c in contrasts:
        lo, hi = c['conditional_subject_bootstrap95']
        star = ' *' if c['primary'] else ''
        print(f"{c['backbone']:12s} {c['group']:28s} {c['metric']:6s} "
              f"{c['treatment_mean']:8.3f} {c['control_mean']:8.3f} {c['difference']:+8.3f} "
              f"[{lo:+7.3f}, {hi:+7.3f}]{star}")
    print()
    print(f"primary sign consistent across backbones: {sign_consistent}")
    print(f'[written] {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
