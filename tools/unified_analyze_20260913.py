"""统一评测汇总与判定：只实现 docs/36 的规则。

输入：results/unified_eval_20260913/*.json（complete=True）
输出：results/unified_eval_20260913/summary.json，并打印表格。
本文模型先对 4 个种子逐序列取平均（只平均标量指标），再做序列宏平均。
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/unified_eval_20260913'
DATASETS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')
CROSS = ('ridi', 'tlio', 'imunet')
KEYS = ('ate_u', 'rte_u', 'tlr', 'mcs', 'eps_s', 'eps_theta_deg', 'ate_shape', 'ate_scale_only', 'ate_heading_only',
        'ate_lit', 'rte_lit')
ORDER = ('ext_resnet', 'ext_resnet_hn', 'ext_lstm', 'ext_lstm_hnseq', 'ext_tcn', 'ext_tcn_hnseq', 'eqnio_so2', 'eqnio_o2',
         'ours_gn', 'ours_yaw', 'ours_pca', 'ours_hn', 'ours_transformer_hn', 'ours_imunet_hn')


def main():
    per = defaultdict(lambda: defaultdict(list))            # model -> (ds,seq) -> [rows over seeds]
    for p in sorted(RES.glob('*.json')):
        if p.name in ('summary.json', 'significance.json', 'yawhn_verdict.json', 'yawhn_summary.json'):
            continue
        d = json.loads(p.read_text())
        if not d.get('complete') or d['provenance'].get('limit'):
            raise RuntimeError(f'{p.name} 未完成或为冒烟结果')
        for r in d['rows']:
            per[r['model']][(r['dataset'], r['sequence'])].append(r)
    table = {}
    for m in ORDER:
        if m not in per:
            continue
        seeds = {len(v) for v in per[m].values()}
        if len(seeds) != 1:
            raise RuntimeError(f'{m} 种子数不一致 {seeds}')
        n_seeds = next(iter(seeds))
        for ds in DATASETS:
            rows = [{k: float(np.mean([x[k] for x in v])) for k in KEYS} for (d, _), v in per[m].items() if d == ds]
            g = lambda k: np.array([r[k] for r in rows])
            A = g('ate_u')
            table[f'{m}|{ds}'] = dict(
                n=len(rows), n_seeds=n_seeds,
                ate_lit=float(g('ate_lit').mean()), ate_u=float(A.mean()), rte_u=float(g('rte_u').mean()),
                tlr=float(g('tlr').mean()), mcs=float(g('mcs').mean()),
                eps_s_median=float(np.median(g('eps_s'))), eps_s_abs_median=float(np.median(abs(g('eps_s')))),
                eps_theta_abs_median=float(np.median(abs(g('eps_theta_deg')))),
                ate_shape=float(g('ate_shape').mean()),
                red_scale_pct=float(100 * (1 - g('ate_scale_only').mean() / A.mean())),
                red_heading_pct=float(100 * (1 - g('ate_heading_only').mean() / A.mean())),
                red_gain_pct=float(100 * (1 - g('ate_shape').mean() / A.mean())))
    passed = [ds for ds in CROSS if table[f'ours_hn|{ds}']['red_scale_pct'] >= 10.0]
    verdict = dict(object='ours_hn（4 种子逐序列平均）', cross_domain=list(CROSS),
                   red_scale_pct={ds: table[f'ours_hn|{ds}']['red_scale_pct'] for ds in CROSS},
                   passed=passed, go_second_innovation=len(passed) >= 2,
                   reference_ext_resnet={ds: table[f'ext_resnet|{ds}']['red_scale_pct'] for ds in CROSS})
    json.dump({'table': table, 'verdict': verdict}, open(RES / 'summary.json', 'w'), ensure_ascii=False, indent=1)
    print('| 模型 | 数据集 | n | ATE 文献协议 | ATE 统一口径 | RTE | TLR | ε_s 中位 | \|ε_θ\| 中位 | 形状 ATE | 只修尺度上限 | 只修航向上限 | 复增益上限 |')
    for k, v in table.items():
        m, ds = k.split('|')
        print(f"| {m} | {ds} | {v['n']}×{v['n_seeds']} | {v['ate_lit']:.3f} | {v['ate_u']:.3f} | {v['rte_u']:.3f} | {v['tlr']:.3f} | "
              f"{v['eps_s_median']:+.3f} | {v['eps_theta_abs_median']:.1f}° | {v['ate_shape']:.3f} | −{v['red_scale_pct']:.1f}% | "
              f"−{v['red_heading_pct']:.1f}% | −{v['red_gain_pct']:.1f}% |")
    print('\n判定'); print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
