"""Shared helpers for reproduce/make_tables.py: loading the archived per-sequence results, the cluster bootstrap used
throughout the paper, and number formatting identical to the manuscript (Unicode minus, signed intervals).

Every table is computed from files under results/ (and a few small archived summaries); nothing here reads raw data."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
B, SEED = 10000, 20260906          # 10,000 percentile-bootstrap resamples, seed 20260906 (all intervals in the paper)
MINUS = '−'
Q95 = (0.025, 0.975)


# ------------------------------------------------------------------------------------------------------------- numbers
def boot(x, q=Q95):
    """Percentile bootstrap of the mean of x; a fresh generator with the paper's seed for every call."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1), q)


def level_q(level):
    a = (1 - level) / 2
    return (a, 1 - a)


def _m(s):
    return s.replace('-', MINUS)


def num(x, d=3):
    return _m(f'{x:.{d}f}')


def sgn(x, d=3):
    return _m(f'{x:+.{d}f}')


def pct(x, d=1, sign=False):
    return _m(f'{x:+.{d}f}%' if sign else f'{x:.{d}f}%')


def ci(lo, hi, d=3):
    return f'[{sgn(lo, d)}, {sgn(hi, d)}]'


def sci(x, d=2):
    return _m(f'{x:.{d}e}')


def pm(m, s, d=3):
    return f'{num(m, d)}±{num(s, d)}'


def bold(s, on=True):
    return f'**{s}**' if on else s


def thousands(n):
    return f'{n:,}'


# ------------------------------------------------------------------------------------------------------------- loading
def rows_of(path):
    j = json.load(open(path))
    return j['rows'] if isinstance(j, dict) else j


def subject(ds, seq):
    """Analysis unit: subjects for RoNIN/RIDI/phone data, sequences for TLIO (no subject information released)."""
    if ds.startswith('imunet'):
        return re.search(r'Subje[ct]+_(\d+)', seq).group(1)
    if ds.startswith('tlio'):
        return seq
    return seq.split('_')[0]


def load_cells(dirs):
    """Unified-evaluation style results: (model, dataset) -> sequence -> seed -> row."""
    cell = defaultdict(lambda: defaultdict(dict))
    for d in dirs:
        for p in sorted((R / d).glob('*.json')):
            j = json.load(open(p))
            if not isinstance(j, dict) or 'rows' not in j:
                continue
            assert j.get('complete', True), p
            for r in j['rows']:
                s = r.get('seed')
                cell[(r['model'], r['dataset'])][r['sequence']][0 if s is None else s] = r
    return cell


_CELLS = {}


def cells():
    """Unified evaluation (test sets), confirmation sets and the cross-architecture runs."""
    if not _CELLS:
        _CELLS.update(load_cells(['unified_eval_20260913', 'confirm_20260913', 'plugin_arch_20260923']))
    return _CELLS


def seqmeans(cell, m, ds, metric):
    """Per-sequence value, averaged over seeds."""
    A = cell[(m, ds)]
    assert A, (m, ds)
    return {s: float(np.mean([r[metric] for r in A[s].values()])) for s in A}


def compare(a, b, ds, q=Q95):
    """Paired comparison of two {sequence: value} maps: subject means (TLIO: sequences) weighted equally;
    subject-level and sequence-level percentile intervals."""
    assert set(a) == set(b), (ds, len(a), len(b))
    g = defaultdict(list)
    for s in a:
        g[subject(ds, s)].append(s)
    keys = sorted(g)
    ua = np.array([np.mean([a[s] for s in g[k]]) for k in keys])
    ub = np.array([np.mean([b[s] for s in g[k]]) for k in keys])
    dsub = ua - ub
    dseq = np.array([a[s] - b[s] for s in sorted(a)])
    return dict(t=ua.mean(), c=ub.mean(), d=dsub.mean(), rel=100 * dsub.mean() / ub.mean(), cs=boot(dsub, q), cq=boot(dseq, q),
                nsub=len(keys), nseq=len(a), imp_sub=int((dsub < 0).sum()), imp_seq=int((dseq < 0).sum()),
                seq_t=float(np.mean(list(a.values()))), seq_c=float(np.mean(list(b.values()))))


_PAIRED = {}


def paired(t, c, ds, metric='ate_u', q=Q95):
    k = (t, c, ds, metric, tuple(q))
    if k not in _PAIRED:
        C = cells()
        _PAIRED[k] = compare(seqmeans(C, t, ds, metric), seqmeans(C, c, ds, metric), ds, q)
    return _PAIRED[k]


def seq_weighted(m, ds, metric):
    """Sequence-weighted mean (seeds averaged within each sequence first)."""
    return float(np.mean(list(seqmeans(cells(), m, ds, metric).values())))


def csv_rows(path):
    return list(csv.DictReader(open(path)))


# ------------------------------------------------------------------------------------------------ controlled front ends
GMAP = {('ronin', 'seen'): 'seen', ('ronin', 'unseen'): 'unseen', ('ridi', ''): 'ridi'}
FROZEN = {'gn': 'Sensors_v4_resnet_gn', 'yaw': 'Sensors_v4_resnet_yaw', 'pca': 'Sensors_v4_resnet_mixed_pca',
          'hn': 'ResNet18_v3_hn_gn', 'raw': 'ResNet18_v2_matched'}
SWEEP = {'gn': ('e1_heading_repeatability', 'Sensors_v4_resnet_gn'), 'yaw': ('e1_heading_repeatability', 'Sensors_v4_resnet_yaw'),
         'pca': ('e1_heading_repeatability', 'Sensors_v4_resnet_mixed_pca'), 'hn': ('e1_heading_repeatability', 'ResNet18_v3_hn_gn'),
         'tf_hn': ('e1_heading_repeatability', 'A6_v3_nopre'), 'imu_hn': ('e1_heading_repeatability', 'IMUNet2024_v3_hn_gn'),
         'tf_yaw': ('e3_backbone_frontend', 'E3_transformer_yaw'), 'imu_yaw': ('e3_backbone_frontend', 'E3_imunet_yaw'),
         'nosign': ('e6_sign_ablation', 'E6_resnet_heading_nosign')}
_EV, _SW = {}, {}


def sweep(k, d=None):
    """Yaw sweep over 8 initial headings: seed -> group -> sequence -> {angle_index: row}."""
    d, name = (d, SWEEP[k][1]) if d else SWEEP[k]
    key = (d, name)
    if key not in _SW:
        out = {}
        for s in range(4):
            g = defaultdict(lambda: defaultdict(dict))
            for r in rows_of(R / f'{d}/{name}_s{s}.json'):
                if (r['dataset'], r['split']) in GMAP:
                    g[GMAP[(r['dataset'], r['split'])]][r['seq']][r['angle_index']] = r
            out[s] = g
        _SW[key] = out
    return _SW[key]


def frozen(k):
    """Evaluation at the dataset reference heading (psi = 0): seed -> group -> sequence -> row with ate, rte, subject."""
    if k in _EV:
        return _EV[k]
    if k in FROZEN and k != 'raw':
        out = {}
        for s in range(4):
            g = defaultdict(dict)
            for r in rows_of(R / f'sensors_v4/evaluation/{FROZEN[k]}_s{s}.json'):
                if (r['dataset'], r['split']) in GMAP:
                    g[GMAP[(r['dataset'], r['split'])]][r['seq']] = r
            out[s] = g
    elif k == 'raw':
        # no GN, no HN: the registered evaluation of the 40 training runs (RoNIN groups and RIDI)
        out = {s: defaultdict(dict) for s in range(4)}
        for r in rows_of(R / 'submission_frozen/eval_rq1_rq4_v3_registered.json'):
            if (r['dataset'], r['split']) in GMAP and r['model'].startswith('ResNet18_v2_matched_s'):
                s = int(r['model'][-1])
                out[s][GMAP[(r['dataset'], r['split'])]][r['seq']] = dict(r, subject=subject(r['dataset'], r['seq']))
    else:                                   # psi = 0 column of a sweep
        sw = sweep(k)
        out = {s: {grp: {q: {'ate': v[0]['ate'], 'rte': v[0]['rte'], 'subject': v[0]['subject']} for q, v in g.items()}
                   for grp, g in sw[s].items()} for s in sw}
    _EV[k] = out
    return out


def subj_macro(g, key='ate'):
    by = defaultdict(list)
    for r in g.values():
        by[r['subject']].append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()]))


def per_seed(k, grp, key='ate', how='subj'):
    ev = frozen(k)
    return np.array([subj_macro(ev[s][grp], key) if how == 'subj' else np.mean([r[key] for r in ev[s][grp].values()])
                     for s in range(4)])


def ms(k, grp, key='ate', how='subj'):
    """Mean over the 4 seeds and the sample standard deviation over seeds."""
    v = per_seed(k, grp, key, how)
    return float(v.mean()), float(v.std(ddof=1))


def paired_subj(t, c, grp, key='ate', q=Q95):
    """Conditional subject bootstrap (CI-A): per-sequence values averaged over seeds, then within subjects."""
    et, ec = frozen(t), frozen(c)
    subj = {q_: r['subject'] for q_, r in et[0][grp].items()}
    st = {q_: np.mean([et[s][grp][q_][key] for s in range(4)]) for q_ in subj}
    sc = {q_: np.mean([ec[s][grp][q_][key] for s in range(4)]) for q_ in subj}
    by = defaultdict(list)
    for q_, u in subj.items():
        by[u].append(q_)
    d = np.array([np.mean([st[x] for x in by[u]]) - np.mean([sc[x] for x in by[u]]) for u in sorted(by)])
    lo, hi = boot(d, q)
    return float(d.mean()), (lo, hi), int((d < 0).sum()), len(d)


def ranges(sw, grp, key='ate'):
    """seed -> sequence -> (cross-angle range over the 8 headings, value at psi = 0)."""
    return {s: {q: (max(v[a][key] for a in range(8)) - min(v[a][key] for a in range(8)), v[0][key])
                for q, v in sw[s][grp].items()} for s in range(4)}
