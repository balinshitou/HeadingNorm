"""docs/45 新模型 ours_yaw_hn（偏航增强训练 + 测试时外挂 HN）的统一评测汇总。

聚合方式与 tools/unified_analyze_20260913.py 完全相同（4 个种子逐序列平均标量指标 → 序列宏平均）；
单独写出，不改动已冻结的 14 模型 summary.json。
输出：results/unified_eval_20260913/yawhn_summary.json
用法：.venv/bin/python tools/yawhn_summary_20260913.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/unified_eval_20260913'
DATASETS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')
KEYS = ('ate_u', 'rte_u', 'tlr', 'mcs', 'eps_s', 'eps_theta_deg', 'ate_shape', 'ate_scale_only', 'ate_heading_only',
        'ate_lit', 'rte_lit')
MODELS = ('ours_yaw_hn', 'ours_yaw_tta8', 'ours_yaw_fa2', 'ours_yaw', 'ours_hn')   # 后两者用于与 summary.json 逐位核对聚合方式；tta8 见 docs/47，fa2 见 docs/50
NEW = ('ours_yaw_hn', 'ours_yaw_tta8', 'ours_yaw_fa2')


def summarize(model):
    per = defaultdict(list)
    for p in sorted(RES.glob(f'{model}_s*.json')):
        d = json.loads(p.read_text())
        assert d.get('complete') and not d['provenance'].get('limit'), p.name
        for r in d['rows']:
            per[(r['dataset'], r['sequence'])].append(r)
    assert {len(v) for v in per.values()} == {4}, model
    out = {}
    for ds in DATASETS:
        rows = [{k: float(np.mean([x[k] for x in v])) for k in KEYS} for (d, _), v in per.items() if d == ds]
        g = lambda k: np.array([r[k] for r in rows]); A = g('ate_u')
        out[ds] = dict(n=len(rows), n_seeds=4, ate_lit=float(g('ate_lit').mean()), ate_u=float(A.mean()),
                       rte_u=float(g('rte_u').mean()), tlr=float(g('tlr').mean()), mcs=float(g('mcs').mean()),
                       eps_s_median=float(np.median(g('eps_s'))), eps_theta_abs_median=float(np.median(abs(g('eps_theta_deg')))),
                       ate_shape=float(g('ate_shape').mean()),
                       red_scale_pct=float(100 * (1 - g('ate_scale_only').mean() / A.mean())),
                       red_heading_pct=float(100 * (1 - g('ate_heading_only').mean() / A.mean())),
                       red_gain_pct=float(100 * (1 - g('ate_shape').mean() / A.mean())))
    return out


def main():
    frozen = json.loads((RES / 'summary.json').read_text())['table']
    res = {m: summarize(m) for m in MODELS}
    for m in ('ours_yaw', 'ours_hn'):                         # 聚合方式自检
        for ds in DATASETS:
            for k in ('ate_u', 'ate_shape', 'eps_s_median'):
                assert abs(res[m][ds][k] - frozen[f'{m}|{ds}'][k]) < 1e-9, (m, ds, k)
    print('聚合自检：ours_yaw、ours_hn 与 summary.json 逐位一致')
    print('| 模型 | 数据集 | ATE 文献 | ATE 统一 | RTE | TLR | ε_s 中位 | |ε_θ| 中位 | 形状 ATE | 复增益上限 |')
    for m in MODELS:
        for ds, v in res[m].items():
            print(f"| {m} | {ds} | {v['ate_lit']:.3f} | {v['ate_u']:.3f} | {v['rte_u']:.3f} | {v['tlr']:.3f} | {v['eps_s_median']:+.3f} | "
                  f"{v['eps_theta_abs_median']:.1f}° | {v['ate_shape']:.3f} | −{v['red_gain_pct']:.1f}% |")
    json.dump({m: res[m] for m in NEW}, open(RES / 'yawhn_summary.json', 'w'), ensure_ascii=False, indent=1)
    print(f'[写出] {(RES / "yawhn_summary.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
