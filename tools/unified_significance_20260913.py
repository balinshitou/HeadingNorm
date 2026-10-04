"""统一评测的配对显著性统计（2026-09-13，只统计已有结果，不跑模型）。

统计口径与稿子表 3、表 4 完全相同（tools/e3_analyze.py）：4 个种子先平均 → 受试者内对序列平均 → 受试者等权；
区间为受试者整群配对百分位 bootstrap，10000 次，种子 20260906；不报 p 值。另报序列级 bootstrap 作补充。
受试者：RoNIN、RIDI 取序列名第一个下划线前；IMUNet 取 Subject_N（原文件名有 Subjetc 拼写）；TLIO 无受试者信息，按序列。
输入：results/unified_eval_20260913/*.json；输出：results/unified_eval_20260913/significance.json
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/unified_eval_20260913'
SEED, NB = 20260906, 10000
DATASETS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')
PAIRS = [('ours_hn', 'ours_yaw'), ('ours_hn', 'ours_gn'), ('ours_hn', 'ours_pca'), ('ext_resnet_hn', 'ext_resnet'),
         ('ours_yaw_hn', 'ours_yaw'), ('ours_yaw_hn', 'ours_hn'),          # docs/45
         ('ours_yaw_hn', 'ours_yaw_tta8'), ('ours_yaw_tta8', 'ours_yaw'), ('ours_yaw_tta8', 'ours_hn'),   # docs/47
         ('ours_yaw_hn', 'ours_yaw_fa2'), ('ours_yaw_fa2', 'ours_yaw'), ('ours_yaw_fa2', 'ours_yaw_tta8'),
         ('ours_yaw_fa2', 'ours_hn')]   # docs/50
METRICS = ('ate_u', 'rte_u', 'ate_lit')


def subject(ds, seq):
    if ds == 'imunet':
        return re.search(r'Subje[ct]+_(\d+)', seq).group(1)
    if ds == 'tlio':
        return seq
    return seq.split('_')[0]


def load():
    cell = defaultdict(lambda: defaultdict(dict))        # (model, ds) -> seq -> seed -> row
    for p in RES.glob('*.json'):
        if p.name in ('summary.json', 'significance.json', 'yawhn_verdict.json', 'yawhn_summary.json'):
            continue
        d = json.loads(p.read_text())
        for r in d['rows']:
            cell[(r['model'], r['dataset'])][r['sequence']][r['seed'] if r['seed'] is not None else 0] = r
    return cell


def boot(x):
    rng = np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(0, len(x), (NB, len(x)))].mean(axis=1), [.025, .975]).tolist()


def main():
    cell = load(); out = []
    for t, c in PAIRS:
        for ds in DATASETS:
            A, B = cell[(t, ds)], cell[(c, ds)]
            if set(A) != set(B):
                raise RuntimeError(f'未配对 {t}/{c}/{ds}')
            for m in METRICS:
                seq_a = {s: np.mean([r[m] for r in A[s].values()]) for s in A}
                seq_b = {s: np.mean([r[m] for r in B[s].values()]) for s in B}
                subj = defaultdict(list)
                for s in A:
                    subj[subject(ds, s)].append(s)
                keys = sorted(subj)
                sa = np.array([np.mean([seq_a[s] for s in subj[k]]) for k in keys])
                sb = np.array([np.mean([seq_b[s] for s in subj[k]]) for k in keys])
                d_sub = sa - sb
                d_seq = np.array([seq_a[s] - seq_b[s] for s in sorted(A)])
                lo, hi = boot(d_sub); slo, shi = boot(d_seq)
                out.append(dict(treatment=t, control=c, dataset=ds, metric=m, n_subjects=len(keys), n_sequences=len(A),
                                treatment_mean=float(sa.mean()), control_mean=float(sb.mean()), difference=float(d_sub.mean()),
                                rel_pct=float(100 * d_sub.mean() / sb.mean()), subject_ci95=[lo, hi],
                                sig='better' if hi < 0 else ('worse' if lo > 0 else 'ns'), improved_subjects=int((d_sub < 0).sum()),
                                seq_difference=float(d_seq.mean()), seq_ci95=[slo, shi],
                                seq_sig='better' if shi < 0 else ('worse' if slo > 0 else 'ns')))
    json.dump(out, open(RES / 'significance.json', 'w'), ensure_ascii=False, indent=1)
    for m in METRICS:
        print(f'\n== {m}（受试者宏平均；区间：受试者整群 bootstrap / 序列 bootstrap）')
        print('| 比较 | 数据集 | 受试者 | 处理组 | 对照组 | 差值 | 受试者区间 | 判定 | 变好受试者 | 序列区间 | 判定 |')
        for r in out:
            if r['metric'] != m:
                continue
            print(f"| {r['treatment']} − {r['control']} | {r['dataset']} | {r['n_subjects']} | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | "
                  f"{r['difference']:+.3f} ({r['rel_pct']:+.1f}%) | [{r['subject_ci95'][0]:+.3f}, {r['subject_ci95'][1]:+.3f}] | {r['sig']} | "
                  f"{r['improved_subjects']}/{r['n_subjects']} | [{r['seq_ci95'][0]:+.3f}, {r['seq_ci95'][1]:+.3f}] | {r['seq_sig']} |")


if __name__ == '__main__':
    main()
