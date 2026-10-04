"""E9 判定（docs/83 §5，写于运行前；只统计已有结果，不跑模型）。

统计口径与 tools/unified_significance_20260913.py 相同（导入其 subject / boot）：4 个种子先平均 → 受试者内对序列平均 → 受试者等权；
10,000 次受试者整群配对 bootstrap（种子 20260906）；手机数据要求受试者区间与序列区间同时排除零才算显著。
输入：results/unified_eval_20260913/（已冻结的 ours_*）与 results/eqnio_retrain_20260919/unified/（E9）。
输出：results/eqnio_retrain_20260919/verdict.json
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import unified_significance_20260913 as S

DIRS = [ROOT / 'results/unified_eval_20260913', ROOT / 'results/eqnio_retrain_20260919/unified']
OUT = ROOT / 'results/eqnio_retrain_20260919/verdict.json'
PRIMARY = ('ours_hn', 'e9_eqnio_so2')
SECONDARY = [('ours_yaw_hn', 'e9_eqnio_so2'), ('ours_hn', 'e9_eqnio_o2'), ('e9_eqnio_so2', 'ours_yaw'),
             ('ours_yaw_hn', 'e9_eqnio_o2'), ('e9_eqnio_o2', 'ours_yaw')]
NEEDED = {'ours_hn', 'ours_yaw_hn', 'ours_yaw', 'e9_eqnio_so2', 'e9_eqnio_o2'}
METRICS = ('ate_u', 'rte_u', 'ate_lit')


def load():
    cell = defaultdict(lambda: defaultdict(dict))
    for d in DIRS:
        for p in d.glob('*.json'):
            j = json.loads(p.read_text())
            if not isinstance(j, dict) or 'rows' not in j or not j.get('complete'):
                continue
            if j['provenance'].get('limit') is not None:
                raise RuntimeError(f'冒烟结果混入：{p}')
            for r in j['rows']:
                if r['model'] in NEEDED:
                    cell[(r['model'], r['dataset'])][r['sequence']][r['seed'] if r['seed'] is not None else 0] = r
    return cell


def compare(cell, t, c, ds, m):
    A, B = cell[(t, ds)], cell[(c, ds)]
    if not A or set(A) != set(B):
        raise RuntimeError(f'未配对或缺失 {t}/{c}/{ds}')
    for X, name in ((A, t), (B, c)):
        if any(len(v) != 4 for v in X.values()):
            raise RuntimeError(f'{name}/{ds} 不是 4 个种子')
    seq_a = {s: np.mean([r[m] for r in A[s].values()]) for s in A}
    seq_b = {s: np.mean([r[m] for r in B[s].values()]) for s in B}
    subj = defaultdict(list)
    for s in A:
        subj[S.subject(ds, s)].append(s)
    keys = sorted(subj)
    sa = np.array([np.mean([seq_a[s] for s in subj[k]]) for k in keys])
    sb = np.array([np.mean([seq_b[s] for s in subj[k]]) for k in keys])
    d_sub = sa - sb
    d_seq = np.array([seq_a[s] - seq_b[s] for s in sorted(A)])
    lo, hi = S.boot(d_sub); slo, shi = S.boot(d_seq)
    if ds == 'imunet':
        call = 'treatment_better' if (hi < 0 and shi < 0) else ('control_better' if (lo > 0 and slo > 0) else 'ns')
    else:
        call = 'treatment_better' if hi < 0 else ('control_better' if lo > 0 else 'ns')
    return dict(treatment=t, control=c, dataset=ds, metric=m, n_subjects=len(keys), n_sequences=len(A),
                treatment_mean=float(sa.mean()), control_mean=float(sb.mean()), difference=float(d_sub.mean()),
                rel_pct=float(100 * d_sub.mean() / sb.mean()), subject_ci95=[lo, hi], seq_ci95=[slo, shi],
                upper_rel_pct=float(100 * hi / sb.mean()), call=call, improved_subjects=int((d_sub < 0).sum()))


def overall(rows):
    n_h = sum(r['call'] == 'treatment_better' for r in rows); n_e = sum(r['call'] == 'control_better' for r in rows)
    if n_h >= 2 and n_e == 0: return 'A_hn_more_accurate'
    if n_e >= 2 and n_h == 0: return 'C_eqnio_more_accurate'
    if n_h and n_e: return 'D_mixed'
    return 'B_no_difference_detected'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=('primary', 'all'), default='all',
                    help='primary：只算 P1（E9-SO2 跑完即可）；all：连同次要比较（需 E9-O2 也跑完）')
    scope = ap.parse_args().scope
    cell = load()
    res = {'primary': [compare(cell, *PRIMARY, ds, 'ate_u') for ds in S.DATASETS]}
    res['primary_verdict'] = overall(res['primary'])
    res['secondary'] = [] if scope == 'primary' else [
        compare(cell, t, c, ds, m) for t, c in [PRIMARY] + SECONDARY for ds in S.DATASETS for m in METRICS
        if not (m == 'ate_u' and (t, c) == PRIMARY)]
    res['scope'] = scope
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print('P1 ours_hn − E9-SO2（统一口径 ATE）：', res['primary_verdict'])
    for r in res['primary']:
        print(f"  {r['dataset']:13s} {r['treatment_mean']:.3f} vs {r['control_mean']:.3f}  {r['difference']:+.3f} ({r['rel_pct']:+.1f}%) "
              f"受试者[{r['subject_ci95'][0]:+.3f}, {r['subject_ci95'][1]:+.3f}] 序列[{r['seq_ci95'][0]:+.3f}, {r['seq_ci95'][1]:+.3f}] {r['call']}")
    for r in res['secondary']:
        if r['metric'] == 'ate_u':
            print(f"  [次] {r['treatment']} − {r['control']} {r['dataset']:13s} {r['difference']:+.3f} ({r['rel_pct']:+.1f}%) {r['call']}")


if __name__ == '__main__':
    main()
