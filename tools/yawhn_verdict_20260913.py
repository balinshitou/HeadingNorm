"""docs/45 的判定：增强训练 + 外挂 HN（ours_yaw_hn）。只实现 docs/45 写定的规则，运行评测前写成。

输入：results/unified_eval_20260913/ours_yaw_hn_s*.json（自检）与 significance.json（先运行 unified_significance_20260913.py）
输出：results/unified_eval_20260913/yawhn_verdict.json，并打印表格。
用法：.venv/bin/python tools/yawhn_verdict_20260913.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/unified_eval_20260913'
DOCS24 = {0: 4.9934, 1: 5.0153, 2: 5.2829, 3: 5.0503}     # docs/24 §5.3，RoNIN unseen 文献协议序列宏平均
CONFIRM = ('ronin_seen', 'ridi', 'tlio', 'imunet')          # 检验集；ronin_unseen 为发现集，不计入判定
ALL = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')


def self_check():
    out = {}
    for s, ref in DOCS24.items():
        rows = json.loads((RES / f'ours_yaw_hn_s{s}.json').read_text())['rows']
        v = float(np.mean([r['ate_lit'] for r in rows if r['dataset'] == 'ronin_unseen']))
        out[s] = dict(ate_lit=v, docs24=ref, abs_diff=abs(v - ref), ok=abs(v - ref) <= 1e-3)
    return out


def call(r):
    """显著性：受试者区间；IMUNet 手机数据要求受试者区间与序列区间同时排除 0。"""
    lo, hi = r['subject_ci95']; slo, shi = r['seq_ci95']
    if r['dataset'] == 'imunet':
        return 'better' if (hi < 0 and shi < 0) else ('worse' if (lo > 0 and slo > 0) else 'ns')
    return 'better' if hi < 0 else ('worse' if lo > 0 else 'ns')


def main():
    sc = self_check()
    for s, v in sc.items():
        print(f"自检 s{s}: ate_lit {v['ate_lit']:.4f} vs docs/24 {v['docs24']:.4f}  差 {v['abs_diff']:.5f}  {'通过' if v['ok'] else '不通过'}")
    if not all(v['ok'] for v in sc.values()):
        json.dump({'self_check': sc, 'verdict': '自检不通过，停止'}, open(RES / 'yawhn_verdict.json', 'w'), ensure_ascii=False, indent=1)
        raise SystemExit('自检不通过：按 docs/45 §4 停止，不进入判定')
    sig = json.loads((RES / 'significance.json').read_text())
    res = {'self_check': sc}
    for key, (t, c) in {'A': ('ours_yaw_hn', 'ours_yaw'), 'B': ('ours_yaw_hn', 'ours_hn')}.items():
        rows = {r['dataset']: r for r in sig if r['treatment'] == t and r['control'] == c and r['metric'] == 'ate_u'}
        cells = {}
        print(f'\n比较 {key}：{t} − {c}（统一口径 ATE，受试者宏平均）')
        for ds in ALL:
            r = rows[ds]; v = call(r)
            cells[ds] = dict(difference=r['difference'], rel_pct=r['rel_pct'], subject_ci95=r['subject_ci95'],
                             seq_ci95=r['seq_ci95'], improved=f"{r['improved_subjects']}/{r['n_subjects']}", call=v,
                             role='discovery' if ds == 'ronin_unseen' else 'confirmatory')
            print(f"  {ds:13s} {r['treatment_mean']:.3f} vs {r['control_mean']:.3f}  {r['difference']:+.3f} ({r['rel_pct']:+.1f}%)  "
                  f"受试者 [{r['subject_ci95'][0]:+.3f}, {r['subject_ci95'][1]:+.3f}]  序列 [{r['seq_ci95'][0]:+.3f}, {r['seq_ci95'][1]:+.3f}]  "
                  f"{r['improved_subjects']}/{r['n_subjects']}  {v}{'  (发现集，不计入)' if ds == 'ronin_unseen' else ''}")
        nb = sum(cells[d]['call'] == 'better' for d in CONFIRM); nw = sum(cells[d]['call'] == 'worse' for d in CONFIRM)
        passed = nb >= 2 and nw == 0
        res[key] = dict(treatment=t, control=c, cells=cells, n_better=nb, n_worse=nw, passed=passed)
        print(f'  判定：检验集 {nb}/4 显著更好，{nw}/4 显著更差 → {"成立" if passed else "不成立"}')
    json.dump(res, open(RES / 'yawhn_verdict.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[写出] {(RES / "yawhn_verdict.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
