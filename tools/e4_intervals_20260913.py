"""表 9 / 图 6 的区间：独立受试者组上 HN − 偏航增强的起点锚定误差与固定跨度位移误差（只统计已有结果，不跑模型）。

口径与 tools/e4_error_timescale.py 的 summarise() 相同：每个时间点只用"该时间点有值"的序列（与原表同一序列集合），
4 个种子逐序列平均 → 受试者内对序列平均 → 受试者等权；差值的区间为受试者整群配对百分位 bootstrap，
10000 次，种子 20260906（与 docs/40 的统一评测显著性同一口径）。
自检：各前端的点估计必须与 results/e4_error_timescale/analysis.json 逐位一致（≤ 1e-9）。
输出：results/e4_error_timescale/intervals.json
用法：.venv/bin/python tools/e4_intervals_20260913.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E4 = ROOT / 'results/e4_error_timescale'
SEED, NB = 20260906, 10000
HN, YAW = 'GN + HeadingNorm', 'GN + yaw augmentation'
FAMS = ('GN only', YAW, 'GN + mixed-sensor PCA (adapted)', HN)
LABEL = 'independent-subject group'


def boot(x):
    rng = np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(0, len(x), (NB, len(x)))].mean(axis=1), [.025, .975]).tolist()


def cell(scoped, kind, key):
    usable = {r['seq'] for r in scoped if r[kind].get(key) is not None}
    if not usable:
        return None
    subj_of = {r['seq']: r['subject'] for r in scoped}
    seqvals = {f: defaultdict(list) for f in FAMS}
    for r in scoped:
        if r['seq'] in usable:
            seqvals[r['family']][r['seq']].append(r[kind][key])
    subjects = sorted({subj_of[q] for q in usable})
    means = {}
    for f in FAMS:
        assert {len(v) for v in seqvals[f].values()} == {4}, (f, key)
        sm = {q: float(np.mean(v)) for q, v in seqvals[f].items()}
        means[f] = np.array([np.mean([sm[q] for q in usable if subj_of[q] == s]) for s in subjects])
    d = means[HN] - means[YAW]
    return dict(key=key, sequences=len(usable), subjects=len(subjects),
                subject_macro={f: float(means[f].mean()) for f in FAMS},
                hn_minus_yaw=float(d.mean()), ci95=boot(d), hn_better_subjects=int((d < 0).sum()))


def main():
    rows = []
    for p in sorted(E4.glob('*.json')):
        if p.name in ('analysis.json', 'intervals.json'):
            continue
        d = json.loads(p.read_text())
        assert d.get('complete'), p.name
        rows += d['rows']
    scoped = [r for r in rows if r['dataset'] == 'ronin' and r['split'] == 'unseen']
    ref = json.loads((E4 / 'analysis.json').read_text())
    out = {'anchored': [], 'curve': []}
    for kind, refkey, tkey in (('anchored', 'anchored', 'elapsed_seconds'), ('curve', 'rows', 'horizon_seconds')):
        keys = sorted({k for r in scoped for k in r[kind]}, key=float)
        for k in keys:
            c = cell(scoped, kind, k)
            if c is None:
                continue
            for f in FAMS:                                   # 自检：与原分析逐位一致
                hit = [x for x in ref[refkey] if x['group'] == LABEL and x['family'] == f and float(x[tkey]) == float(k)]
                assert len(hit) == 1 and abs(hit[0]['subject_macro'] - c['subject_macro'][f]) < 1e-9, (kind, k, f)
                assert hit[0]['sequences'] == c['sequences'], (kind, k, f)
            out[kind].append(c)
    json.dump(out, open(E4 / 'intervals.json', 'w'), ensure_ascii=False, indent=1)
    print('自检：各前端点估计与 analysis.json 逐位一致')
    for kind, name in (('anchored', '起点锚定误差（经过时间 t, s）'), ('curve', '固定跨度位移误差（跨度 h, s）')):
        print(f'\n{name}：HN − 偏航增强（受试者宏平均，m）')
        for c in out[kind]:
            print(f"  {c['key']:>4s}  序列 {c['sequences']:2d}  受试者 {c['subjects']:2d}  {c['hn_minus_yaw']:+.3f}  "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  HN 更好 {c['hn_better_subjects']}/{c['subjects']}")
    print(f'\n[写出] {(E4 / "intervals.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
