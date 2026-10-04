"""docs/50 的判定：外挂 HN 与两方向帧平均（偏航增强训练的 ResNet18）。只实现 docs/50 写定的规则，运行评测前写成。

输入：results/unified_eval_20260913/significance.json、ours_yaw_fa2_s*.json；results/fa2_repeat_20260913/rows.json
输出：results/fa2_repeat_20260913/verdict.json
用法：.venv/bin/python tools/fa2_verdict_20260913.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UE = ROOT / 'results/unified_eval_20260913'
FR = ROOT / 'results/fa2_repeat_20260913'
ALL = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')
MODELS = ('ours_yaw_hn', 'ours_yaw_fa2')
PSI = (0.0, 11.25, 22.5, 33.75)


def call(r):
    lo, hi = r['subject_ci95']; slo, shi = r['seq_ci95']
    if r['dataset'] == 'imunet':
        return 'better' if (hi < 0 and shi < 0) else ('worse' if (lo > 0 and slo > 0) else 'ns')
    return 'better' if hi < 0 else ('worse' if lo > 0 else 'ns')


def q1():
    sig = json.loads((UE / 'significance.json').read_text()); out = {}
    for t, c in (('ours_yaw_hn', 'ours_yaw_fa2'), ('ours_yaw_fa2', 'ours_yaw'), ('ours_yaw_fa2', 'ours_yaw_tta8'),
                 ('ours_yaw_fa2', 'ours_hn')):
        print(f'\n{t} − {c}（统一口径 ATE，受试者宏平均）')
        cells = {}
        for ds in ALL:
            r = next(x for x in sig if x['treatment'] == t and x['control'] == c and x['dataset'] == ds and x['metric'] == 'ate_u')
            v = call(r)
            cells[ds] = dict(treatment_mean=r['treatment_mean'], control_mean=r['control_mean'], difference=r['difference'],
                             rel_pct=r['rel_pct'], subject_ci95=r['subject_ci95'], seq_ci95=r['seq_ci95'],
                             improved=f"{r['improved_subjects']}/{r['n_subjects']}", call=v)
            print(f"  {ds:13s} {r['treatment_mean']:.3f} vs {r['control_mean']:.3f}  {r['difference']:+.3f} ({r['rel_pct']:+.1f}%)  "
                  f"受试者 [{r['subject_ci95'][0]:+.3f}, {r['subject_ci95'][1]:+.3f}]  序列 [{r['seq_ci95'][0]:+.3f}, {r['seq_ci95'][1]:+.3f}]  "
                  f"{r['improved_subjects']}/{r['n_subjects']}  {v}")
        out[f'{t}-{c}'] = cells
    D = out['ours_yaw_hn-ours_yaw_fa2']
    nb = sum(v['call'] == 'better' for v in D.values()); nw = sum(v['call'] == 'worse' for v in D.values())
    if nb >= 2 and nw == 0:
        verdict = 'HN 更准'
    elif nw >= 2 and nb == 0:
        verdict = '两方向帧平均更准'
    elif nb >= 1 and nw >= 1:
        verdict = '不一致'
    else:
        verdict = '未检出差异'
    print(f'\n主比较 D：{nb}/5 HN 显著更好，{nw}/5 HN 显著更差 → {verdict}')
    return out, dict(n_better=nb, n_worse=nw, verdict=verdict)


def q2():
    d = json.loads((FR / 'rows.json').read_text())['rows']
    ate = defaultdict(dict)
    for r in d:
        ate[(r['model'], r['seed'], r['sequence'])][r['psi_deg']] = r['ate_lit']
    q1_ref = {}
    for s in range(4):
        for r in json.loads((UE / f'ours_yaw_fa2_s{s}.json').read_text())['rows']:
            if r['dataset'] == 'ronin_unseen':
                q1_ref[(s, r['sequence'])] = r['ate_lit']
    dmax = max(abs(ate[('ours_yaw_fa2', s, q)][0.0] - v) for (s, q), v in q1_ref.items())
    print(f'\n自检：ψ=0 处两方向帧平均与 Q1 评测逐序列最大差 {dmax:.2e} m（≤1e-5 {"通过" if dmax <= 1e-5 else "不通过"}）')
    out = dict(self_check_fa2_psi0_max_abs_diff=dmax, self_check_ok=dmax <= 1e-5)
    print('\nQ2：4 个角度（0°, 11.25°, 22.5°, 33.75°）上的逐序列文献协议 ATE 极差（32 条 × 4 种子）')
    for m in MODELS:
        rng = np.array([max(v.values()) - min(v.values()) for k, v in ate.items() if k[0] == m])
        rel = np.array([(max(v.values()) - min(v.values())) / v[0.0] for k, v in ate.items() if k[0] == m])
        mean_psi = {deg: float(np.mean([v[deg] for k, v in ate.items() if k[0] == m])) for deg in PSI}
        out[m] = dict(n=len(rng), range_median=float(np.median(rng)), range_rel_median_pct=float(100 * np.median(rel)),
                      range_max=float(rng.max()), mean_ate_by_psi=mean_psi)
        print(f"  {m:14s} 极差中位 {np.median(rng):.3g} m，占 ATE 中位 {100*np.median(rel):.2g}%，最大 {rng.max():.3g} m；"
              f"各角度平均 ATE " + ', '.join(f'{k:g}°:{v:.3f}' for k, v in mean_psi.items()))
    out['criterion_ok'] = all(out[m]['range_max'] < 1e-4 for m in MODELS)
    print(f"  判据（两者逐序列极差最大 < 1e-4 m）：{'通过' if out['criterion_ok'] else '不通过'}")
    return out


def main():
    cells, d = q1(); rep = q2()
    json.dump(dict(q1=cells, q1_verdict=d, q2=rep), open(FR / 'verdict.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[写出] {(FR / "verdict.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
