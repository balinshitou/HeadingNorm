#!/usr/bin/env python3
"""Recompute the initial-reference-direction repeatability table under one convention.

The manuscript and results/e1_heading_repeatability/analysis.json currently disagree on
how the per-sequence column is aggregated: the manuscript pools (seed, sequence) pairs,
the analysis artefact takes the per-seed median and then averages over seeds. Seeds are
training repeats everywhere else in the paper, so this script adopts the per-seed
convention throughout and prints both so the change is auditable.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / 'results/e1_heading_repeatability', ROOT / 'results/e6_sign_ablation',
       ROOT / 'results/e3_backbone_frontend']
ORDER = ['GN only', 'GN + yaw augmentation', 'GN + mixed-sensor PCA (adapted)',
         'GN + HeadingNorm without sign rule', 'GN + HeadingNorm',
         'Transformer + HN + GN (no PreFilter)', 'IMUNet + HN + GN',
         'Transformer + GN + yaw augmentation', 'IMUNet + GN + yaw augmentation']
GROUPS = [('ronin', 'unseen', 'independent-subject group'), ('ridi', '', 'RIDI'),
          ('ronin', 'seen', 'training-subject group')]


def rows_all():
    out = []
    for d in SRC:
        if not d.exists():
            continue
        for p in sorted(d.glob('*.json')):
            if p.name in {'analysis.json', 'analytic_frame_check.json'}:
                continue
            payload = json.loads(p.read_text())
            if payload.get('complete') and 'rows' in payload:
                out.extend(payload['rows'])
    return out


def subject_macro(bucket):
    by = defaultdict(list)
    for r in bucket:
        by[r['subject']].append(r['ate'])
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    rows = rows_all()
    cell = defaultdict(list)
    for r in rows:
        cell[(r['family'], r['seed'], r['dataset'], r['split'], r['angle_index'])].append(r)
    fams = [f for f in ORDER if any(k[0] == f for k in cell)]
    out = []
    hdr = (f"{'arm':38s} {'group':28s} {'ATE(0)':>8s} {'range':>7s} {'range%':>7s} "
           f"{'worst%':>7s} {'seqMed':>10s} {'seqMed%':>8s} {'seqMax':>10s} {'pooledMed':>10s}")
    print(hdr); print('-' * len(hdr))
    for fam in fams:
        seeds = sorted({k[1] for k in cell if k[0] == fam})
        for ds, sp, label in GROUPS:
            if not any(k[0] == fam and k[2] == ds and k[3] == sp for k in cell):
                continue
            curves, med, medrel, mx, pooled = [], [], [], [], []
            ok = True
            for s in seeds:
                curve = []
                per_seq = defaultdict(dict)
                for i in range(8):
                    b = cell.get((fam, s, ds, sp, i))
                    if not b:
                        ok = False
                        break
                    curve.append(subject_macro(b))
                    for r in b:
                        per_seq[r['seq']][i] = r['ate']
                if not ok:
                    break
                curves.append(curve)
                rng = [max(v.values()) - min(v.values()) for v in per_seq.values() if len(v) == 8]
                rel = [(max(v.values()) - min(v.values())) / v[0]
                       for v in per_seq.values() if len(v) == 8 and v[0] > 0]
                med.append(np.median(rng)); medrel.append(np.median(rel)); mx.append(max(rng))
                pooled.extend(rng)
            if not ok:
                continue
            arr = np.array(curves)
            base = arr[:, 0].mean()
            rangev = (arr.max(axis=1) - arr.min(axis=1)).mean()
            worst = arr.max(axis=1).mean()
            e = dict(arm=fam, group=label, seeds=len(seeds), ate_at_zero=base,
                     subject_macro_range=rangev, range_relative=rangev / base,
                     worst_angle_relative=worst / base - 1,
                     sequence_range_median=float(np.mean(med)),
                     sequence_range_median_relative=float(np.mean(medrel)),
                     sequence_range_max=float(np.mean(mx)),
                     pooled_sequence_range_median=float(np.median(pooled)))
            out.append(e)
            print(f"{fam:38s} {label:28s} {base:8.3f} {rangev:7.4f} {100*e['range_relative']:6.1f}% "
                  f"{100*e['worst_angle_relative']:+6.1f}% {e['sequence_range_median']:10.3e} "
                  f"{100*e['sequence_range_median_relative']:7.1f}% {e['sequence_range_max']:10.3e} "
                  f"{e['pooled_sequence_range_median']:10.3e}")
    dest = ROOT / 'results/e1_heading_repeatability/table_per_seed_convention.json'
    dest.write_text(json.dumps(out, indent=1) + '\n')
    print(f'\n[written] {dest.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
