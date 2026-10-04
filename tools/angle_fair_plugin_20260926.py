"""V24（2026-09-26；运行前判定标准 results/angle_fair_20260926/PLAN_运行前判定标准.md）。原 omnisci 案例 A5（移植自 hn_plugin_angle 案例，判定标准见 PLAN_运行前判定标准.md，先于首次运行写成）。外挂HN收益在"方向约定平均"下是否仍存在（判定标准见 PLAN_运行前判定标准.md，先于本脚本写成）。

记号（文献协议 ATE，m）：U(ψ) 原样网络在初始参考方向 ψ 下；H(α) 外挂HN、规范系朝向约定 α 下。
U0=U(0)，Ubar=8 个 ψ 的均值，Umin=8 个 ψ 的最小值；H0=H(0)，Hbar=8 个 α 的均值。
聚合：序列（本文网络先对 4 种子平均）→ 受试者内平均 → 受试者等权；TLIO 按序列。
区间：10,000 次受试者整群 percentile bootstrap，种子 20260906。
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / 'results'

B, SEED = 10000, 20260906
LV_MAIN = 1 - 0.05 / 3          # Bonferroni 3 项
DS_NAME = {'ronin_seen': 'RoNIN-seen', 'ronin_unseen': 'RoNIN-unseen', 'ridi': 'RIDI', 'tlio': 'TLIO-test'}
PRIMARY = ('ronin_unseen', 'ridi', 'tlio')


def subject(ds, seq):
    return seq if ds == 'tlio' else seq.split('_')[0]


def boot(subj, d, level):
    """受试者等权均值及其整群 bootstrap 区间。subj: 每条序列的受试者；d: 每条序列的值。"""
    us = sorted(set(subj)); idx = {u: i for i, u in enumerate(us)}
    g = np.array([idx[s] for s in subj])
    m = np.array([d[g == i].mean() for i in range(len(us))])
    rng = np.random.default_rng(SEED)
    bs = m[rng.integers(0, len(us), (B, len(us)))].mean(1)
    a = (1 - level) / 2
    return m.mean(), np.quantile(bs, a), np.quantile(bs, 1 - a), len(us)


def smean(subj, d):
    return boot(subj, d, 0.95)[0]


def report(tag, subj, name, d, base, level):
    m, lo, hi, n = boot(subj, d, level)
    rel = 100 * m / smean(subj, base)
    print(f'{tag}.{name}.diff_m = {m:.3f}')
    print(f'{tag}.{name}.rel_pct = {rel:.1f}')
    print(f'{tag}.{name}.ci_lo = {lo:.3f}')
    print(f'{tag}.{name}.ci_hi = {hi:.3f}')
    print(f'{tag}.{name}.ci_level_pct = {100*level:.2f}')
    print(f'{tag}.{name}.seq_improved = {int((d < 0).sum())}')
    return m, lo, hi, rel


# ------------------------------------------------------------------ 主分析：原作者 RoNIN ResNet
rows = json.load(open(R / 'hn_plugin_p0_20260912/ext_resnet.json'))['rows']
print('## released RoNIN ResNet (ext_resnet), literature protocol')
print(f'ext.n_angles = {len(rows[0]["orig_ate_by_psi"])}')
print(f'ext.max_abs_U0_minus_orig = {max(abs(r["orig_ate_by_psi"][0]-r["orig"]["ate"]) for r in rows):.1e}')
print(f'ext.max_abs_H0_minus_Hpsi90 = {max(abs(r["hn_ate_by_alpha"][0]-r["hn_ate_psi90"]) for r in rows):.1e}')
res, curves = {}, {}
for ds in DS_NAME:
    rr = [r for r in rows if r['dataset'] == ds]
    subj = [subject(ds, r['sequence']) for r in rr]
    U = np.array([r['orig_ate_by_psi'] for r in rr]); H = np.array([r['hn_ate_by_alpha'] for r in rr])
    U0, Ub, Um, H0, Hb = U[:, 0], U.mean(1), U.min(1), H[:, 0], H.mean(1)
    t = f'ext.{ds}'
    print(f'{t}.n_seq = {len(rr)}')
    print(f'{t}.n_subj = {len(set(subj))}')
    for nm, v in (('U0', U0), ('Ubar', Ub), ('Umin', Um), ('H0', H0), ('Hbar', Hb), ('Hmin', H.min(1))):
        print(f'{t}.{nm}_m = {smean(subj, v):.3f}')
    print(f'{t}.U_range_over_psi_median_m = {np.median(U.max(1)-U.min(1)):.3f}')
    print(f'{t}.H_range_over_alpha_median_m = {np.median(H.max(1)-H.min(1)):.3f}')
    print(f'{t}.H_range_over_alpha_median_pct_of_H0 = {100*np.median((H.max(1)-H.min(1))/H0):.1f}')
    lv = LV_MAIN if ds in PRIMARY else 0.95
    res[ds] = {
        'paper': report(t, subj, 'H0_minus_U0', H0 - U0, U0, 0.95),
        'vsUbar': report(t, subj, 'H0_minus_Ubar', H0 - Ub, Ub, 0.95),
        'fair': report(t, subj, 'Hbar_minus_Ubar', Hb - Ub, Ub, lv),
        'conv_luck': report(t, subj, 'H0_minus_Hbar', H0 - Hb, Hb, 0.95),
        'native_luck': report(t, subj, 'U0_minus_Ubar', U0 - Ub, Ub, 0.95),
        'vsUmin': report(t, subj, 'Hbar_minus_Umin', Hb - Um, Um, 0.95),
    }
    print(f'{t}.frac_seq_Hbar_below_Ubar = {(Hb < Ub).mean():.3f}')
    curves[ds] = ([smean(subj, U[:, k]) for k in range(8)], [smean(subj, H[:, k]) for k in range(8)])

npass = sum(res[d]['fair'][2] < 0 for d in PRIMARY)
nrev = sum(res[d]['fair'][1] > 0 for d in PRIMARY)
print(f'verdict.primary_pass_count = {npass}')
print(f'verdict.primary_reverse_count = {nrev}')
print(f'verdict.primary_n = {len(PRIMARY)}')
print('verdict.result = ' + ('PASS' if npass >= 2 and nrev == 0 else ('REVERSED' if nrev else 'NOT_PASS')))

# ------------------------------------------------------------------ 次要：本文 ResNet18-YawAug（4 种子，无 α 扫描）
print('## ResNet18-YawAug (ours, 4 seeds), literature protocol')
U = {}; H = {}
for s in range(4):
    for r in json.load(open(R / 'e1_heading_repeatability' / f'Sensors_v4_resnet_yaw_s{s}.json'))['rows']:
        ds = 'ridi' if r['dataset'] == 'ridi' else f'ronin_{r["split"]}'
        U.setdefault((ds, r['seq']), np.zeros((4, 8)))[s, r['angle_index']] = r['ate']
    for r in json.load(open(R / 'unified_eval_20260913' / f'ours_yaw_hn_s{s}.json'))['rows']:
        if r['dataset'] in ('ronin_seen', 'ronin_unseen', 'ridi'):
            H.setdefault((r['dataset'], r['sequence']), np.zeros(4))[s] = r['ate_lit']
print(f'ours.n_seq_total = {len(U)}')
res2 = {}
for ds in ('ronin_seen', 'ronin_unseen', 'ridi'):
    ks = sorted(k for k in U if k[0] == ds)
    subj = [subject(ds, k[1]) for k in ks]
    A = np.array([U[k] for k in ks])                 # seq × seed × angle
    U0, Ub, Um = A[:, :, 0].mean(1), A.mean((1, 2)), A.min(2).mean(1)
    H0 = np.array([H[k].mean() for k in ks])
    t = f'ours.{ds}'
    print(f'{t}.n_seq = {len(ks)}')
    print(f'{t}.n_subj = {len(set(subj))}')
    for nm, v in (('U0', U0), ('Ubar', Ub), ('Umin', Um), ('H0', H0)):
        print(f'{t}.{nm}_m = {smean(subj, v):.3f}')
    res2[ds] = {'paper': report(t, subj, 'H0_minus_U0', H0 - U0, U0, 0.95),
                'vsUbar': report(t, subj, 'H0_minus_Ubar', H0 - Ub, Ub, 0.95),
                'native_luck': report(t, subj, 'U0_minus_Ubar', U0 - Ub, Ub, 0.95),
                'vsUmin': report(t, subj, 'H0_minus_Umin', H0 - Um, Um, 0.95)}
    print(f'{t}.frac_seq_H0_below_Ubar = {(H0 < Ub).mean():.3f}')

