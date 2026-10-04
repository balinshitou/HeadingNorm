"""E4: where the ATE and RTE orderings separate.

On the independent-subject group HeadingNorm has the lower ATE but the higher
60 s RTE, while its direct per-window velocity error is higher in every geometry
bin. Those three facts are consistent if HN trades a slightly noisier per-window
estimate for a smaller systematic component that accumulates over long horizons.

RTE is already a displacement error over a fixed horizon: the frozen
implementation computes (est[d:]-est[:-d]) - (gt[d:]-gt[:-d]) with d = 60 s. This
script evaluates that same quantity as a function of the horizon h. Reading the
curve is enough to test the explanation, with no new metric definition:
  * uncorrelated error  -> displacement error grows as sqrt(h);
  * systematic error    -> displacement error grows as h.
ATE corresponds to the long-horizon end (position error accumulated from the
start), so a crossing between 60 s and the sequence length would account for the
two metrics disagreeing.

Evaluation only: frozen checkpoints, the paper's registered alignment, no
training and no checkpoint selection.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import pkg_common as P
import pkg_eval as E
import sensors_train as S
from sensors_evaluate import load

OUT = ROOT / 'results/e4_error_timescale'
HORIZONS = [1., 2., 5., 10., 20., 30., 45., 60., 90., 120.]
# A horizon is only admissible when the record is at least MIN_RATIO times as
# long, so every point rests on many start times rather than on a handful of
# end-of-sequence pairs. Without this the curve turns over near h ~ duration,
# which is a sampling artefact and not a property of the error.
MIN_RATIO = 3.0
# ATE is not a point on the D(h) curve: it is the start-anchored error
# ||integral of e from 0 to t||, averaged over t, whereas D(h) averages over all
# start times. For a random-walk-like error the ATE magnitude corresponds to a
# horizon of order half the record, which is far beyond the admissible range of
# D(h). ANCHORED_TIMES therefore samples the ATE integrand directly.
ANCHORED_TIMES = [10., 20., 30., 60., 90., 120., 180., 240., 300., 420., 600.]
FRONTENDS = {
    'GN only': 'Sensors_v4_resnet_gn_s{}',
    'GN + yaw augmentation': 'Sensors_v4_resnet_yaw_s{}',
    'GN + mixed-sensor PCA (adapted)': 'Sensors_v4_resnet_mixed_pca_s{}',
    'GN + HeadingNorm': 'ResNet18_v3_hn_gn_s{}',
}


def displacement_error(est, gt, rate_hz, seconds):
    """RoNIN's RTE numerator at an arbitrary horizon; None if the sequence is short.

    No length extrapolation: a horizon a sequence cannot support is left out
    rather than rescaled, so every reported point is measured. A horizon is
    admissible only when the record is at least MIN_RATIO times as long.
    """
    delta = int(round(seconds * rate_hz))
    if delta < 1 or len(est) < delta * MIN_RATIO:
        return None
    relative = est[delta:] + gt[:-delta] - est[:-delta] - gt[delta:]
    return float(np.sqrt(np.mean(relative ** 2)))


def evaluate(args):
    device = P.pick_device()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(P.CACHE_EVAL.glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            dataset, split = str(z['dataset']), str(z['split'])
        if dataset == 'ridi' or (dataset == 'ronin' and split == 'unseen'):
            files.append((path, dataset, split))
    if args.limit:
        keep, counts = [], {}
        for entry in files:
            key = (entry[1], entry[2])
            if counts.get(key, 0) < args.limit:
                keep.append(entry); counts[key] = counts.get(key, 0) + 1
        files = keep
    script_hash = S.digest(Path(__file__))

    for family, template in FRONTENDS.items():
        for seed in range(args.seeds):
            tag = template.format(seed)
            destination = OUT / f'{tag}.json'
            started = time.monotonic()
            net, win_sec, rate, checkpoint = load(tag, device)
            provenance = dict(model=tag, family=family, seed=seed,
                              checkpoint_sha256=S.digest(checkpoint),
                              script_sha256=script_hash, horizons=HORIZONS,
                              expected_sequences=len(files), device=str(device))
            if destination.exists():
                previous = json.loads(destination.read_text())
                if all(previous.get(k) == v for k, v in provenance.items()) and previous['complete']:
                    print(f'[verified existing] {tag}', flush=True)
                    continue
            rows = []
            for path, dataset, split in files:
                with np.load(path, allow_pickle=False) as z:
                    feat = np.asarray(z['feat'], np.float32)
                    tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
                    seq = str(z['seq'])
                    score_rate = float(z['rate']) if 'rate' in z else 200.
                ids, velocity = E.predict(net, feat, device, win_sec, rate, chunk=512)
                trajectory = P.trajectory_from_velocity(velocity, ids, tm, te)
                aligned, _, _ = P.align_registered(dataset, trajectory, gt, te)
                n = min(len(aligned), len(gt))
                aligned, truth = aligned[:n], gt[:n]
                ate, rte60 = P.ate_rte(aligned, truth, score_rate)
                curve = {f'{h:g}': displacement_error(aligned, truth, score_rate, h)
                         for h in HORIZONS}
                # start-anchored error: exactly what ATE averages over
                elapsed = te[:n] - te[0]
                offset = aligned - truth
                anchored = {}
                for moment in ANCHORED_TIMES:
                    if elapsed[-1] < moment:
                        anchored[f'{moment:g}'] = None
                        continue
                    index = int(np.searchsorted(elapsed, moment))
                    index = min(index, n - 1)
                    anchored[f'{moment:g}'] = float(
                        np.sqrt(np.mean(offset[index] ** 2)))
                rows.append(dict(model=tag, family=family, seed=seed, dataset=dataset,
                                 split=split, seq=seq, subject=seq.split('_')[0],
                                 duration_seconds=float(te[n - 1] - te[0]),
                                 ate=ate, rte60=rte60, curve=curve, anchored=anchored))
            S.atomic_json(destination, {**provenance, 'complete': True, 'rows': rows,
                                        'elapsed_seconds': time.monotonic() - started})
            print(f'[evaluated] {tag}: {len(rows)} sequences; '
                  f'{time.monotonic() - started:.1f}s', flush=True)


def subject_macro(values):
    by_subject = defaultdict(list)
    for subject, value in values:
        by_subject[subject].append(value)
    return mean(mean(v) for v in by_subject.values())


def summarise():
    payloads = [json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))
                if p.name != 'analysis.json']
    payloads = [p for p in payloads if p.get('complete')]
    if not payloads:
        raise SystemExit('no complete E4 evaluations yet')
    rows = [r for p in payloads for r in p['rows']]

    report = []
    for dataset, split, label in [('ronin', 'unseen', 'independent-subject group'),
                                  ('ridi', '', 'RIDI')]:
        scoped = [r for r in rows if r['dataset'] == dataset and r['split'] == split]
        # one common sequence set per horizon so curves are comparable across models
        for horizon in HORIZONS:
            key = f'{horizon:g}'
            usable = {r['seq'] for r in scoped if r['curve'].get(key) is not None}
            if not usable:
                continue
            for family in FRONTENDS:
                seeds = sorted({r['seed'] for r in scoped if r['family'] == family})
                per_seed = []
                for seed in seeds:
                    values = [(r['subject'], r['curve'][key]) for r in scoped
                              if r['family'] == family and r['seed'] == seed
                              and r['seq'] in usable]
                    if values:
                        per_seed.append(subject_macro(values))
                if len(per_seed) == len(seeds) and per_seed:
                    report.append(dict(group=label, dataset=dataset, split=split,
                                       horizon_seconds=horizon, family=family,
                                       sequences=len(usable),
                                       subject_macro=mean(per_seed),
                                       seed_sd=stdev(per_seed) if len(per_seed) > 1 else 0.))
    anchored_report = []
    for dataset, split, label in [('ronin', 'unseen', 'independent-subject group')]:
        scoped = [r for r in rows if r['dataset'] == dataset and r['split'] == split]
        for moment in ANCHORED_TIMES:
            key = f'{moment:g}'
            usable = {r['seq'] for r in scoped if r.get('anchored', {}).get(key) is not None}
            if not usable:
                continue
            for family in FRONTENDS:
                seeds = sorted({r['seed'] for r in scoped if r['family'] == family})
                per_seed = []
                for seed in seeds:
                    values = [(r['subject'], r['anchored'][key]) for r in scoped
                              if r['family'] == family and r['seed'] == seed
                              and r['seq'] in usable]
                    if values:
                        per_seed.append(subject_macro(values))
                if len(per_seed) == len(seeds) and per_seed:
                    anchored_report.append(dict(group=label, elapsed_seconds=moment,
                                                family=family, sequences=len(usable),
                                                subject_macro=mean(per_seed)))
    (OUT / 'analysis.json').write_text(json.dumps(
        dict(rows=report, anchored=anchored_report), indent=1))

    print('\n=== independent-subject group: START-ANCHORED error (m) vs elapsed time ===')
    print('    (this is the quantity ATE averages over)')
    print(f"{'elapsed(s)':>10s} {'seqs':>5s} " +
          ' '.join(f'{f.replace("GN + ","").replace("GN only","GN")[:14]:>15s}' for f in FRONTENDS)
          + f"  {'HN-yaw':>9s}")
    for moment in ANCHORED_TIMES:
        cells, values, n = [], {}, ''
        for family in FRONTENDS:
            hit = [e for e in anchored_report if e['elapsed_seconds'] == moment
                   and e['family'] == family]
            if hit:
                values[family] = hit[0]['subject_macro']; n = hit[0]['sequences']
                cells.append(f"{hit[0]['subject_macro']:15.3f}")
            else:
                cells.append(f"{'--':>15s}")
        if not values:
            continue
        gap = ''
        if 'GN + HeadingNorm' in values and 'GN + yaw augmentation' in values:
            gap = f"{values['GN + HeadingNorm'] - values['GN + yaw augmentation']:+9.3f}"
        print(f'{moment:10g} {n:>5} ' + ' '.join(cells) + f'  {gap}')

    for label in ['independent-subject group', 'RIDI']:
        print(f'\n=== {label}: displacement error (m) vs horizon ===')
        print(f"{'horizon(s)':>10s} {'seqs':>5s} " +
              ' '.join(f'{f.replace("GN + ","").replace("GN only","GN")[:14]:>15s}'
                       for f in FRONTENDS) + f"  {'HN-yaw':>9s}")
        for horizon in HORIZONS:
            cells, values = [], {}
            n = ''
            for family in FRONTENDS:
                hit = [e for e in report if e['group'] == label
                       and e['horizon_seconds'] == horizon and e['family'] == family]
                if hit:
                    values[family] = hit[0]['subject_macro']
                    n = hit[0]['sequences']
                    cells.append(f"{hit[0]['subject_macro']:15.3f}")
                else:
                    cells.append(f"{'--':>15s}")
            if not values:
                continue
            gap = ''
            if 'GN + HeadingNorm' in values and 'GN + yaw augmentation' in values:
                gap = f"{values['GN + HeadingNorm'] - values['GN + yaw augmentation']:+9.3f}"
            print(f'{horizon:10g} {n:>5} ' + ' '.join(cells) + f'  {gap}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--summarise-only', action='store_true')
    ap.add_argument('--limit', type=int, help='sequences per group; smoke test only')
    args = ap.parse_args()
    if not args.summarise_only:
        evaluate(args)
    summarise()


if __name__ == '__main__':
    main()
