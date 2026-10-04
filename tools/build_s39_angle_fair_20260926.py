"""补充材料 S39 / 表S61：从 V24 输出（results/angle_fair_20260926/output.txt）生成 Markdown 表，不手抄数字。
用法：.venv/bin/python tools/build_s39_angle_fair_20260926.py > results/angle_fair_20260926/table_S61.md"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
V = {}
for line in open(ROOT / 'results/angle_fair_20260926/output.txt'):
    m = re.match(r'^(\S+) = (\S+)$', line.strip())
    if m:
        V[m.group(1)] = m.group(2)

NAME = {'ronin_seen': 'RoNIN-seen', 'ronin_unseen': 'RoNIN-unseen', 'ridi': 'RIDI', 'tlio': 'TLIO-test'}


def f(x):
    return x.replace('-', '−')


def sg(x):
    return f(x) if x.startswith('-') else '+' + x


def diff(p, name, bold_if_excl=True):
    d, r, lo, hi = (V[f'{p}.{name}.{k}'] for k in ('diff_m', 'rel_pct', 'ci_lo', 'ci_hi'))
    s = f'{sg(d)} ({sg(r)}%) [{sg(lo)}, {sg(hi)}]'
    excl = float(hi) < 0 or float(lo) > 0
    return f'**{s}**' if (excl and bold_if_excl) else s


print('| Network | Test set | Seqs / subjects | Unmodified, ψ=0 | Unmodified, mean over 8 ψ | Plug-in HN, α=0 | Plug-in HN, mean over 8 α '
      '| Mean-α HN − mean-ψ Unmodified [CI] | Seqs improved (mean vs. mean) | α=0 − mean-α HN [95% CI] | Per-seq. range over α, median (% of ATE) |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for ds in ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio'):
    p = f'ext.{ds}'
    lv = V[f'{p}.Hbar_minus_Ubar.ci_level_pct']
    n = V[f'{p}.n_seq']
    k = round(float(V[f'{p}.frac_seq_Hbar_below_Ubar']) * int(n))
    assert int(V[f'{p}.Hbar_minus_Ubar.seq_improved']) == k
    subj = V[f'{p}.n_subj'] if ds != 'tlio' else '—'
    print(f'| RoNIN ResNet (released) | {NAME[ds]} | {n} / {subj} | {V[p + ".U0_m"]} | {V[p + ".Ubar_m"]} | {V[p + ".H0_m"]} | {V[p + ".Hbar_m"]} '
          f'| {diff(p, "Hbar_minus_Ubar")} ({lv}%) | {k}/{n} | {diff(p, "H0_minus_Hbar")} '
          f'| {V[p + ".H_range_over_alpha_median_m"]} m ({V[p + ".H_range_over_alpha_median_pct_of_H0"]}%) |')
for ds in ('ronin_seen', 'ronin_unseen', 'ridi'):
    p = f'ours.{ds}'
    n = V[f'{p}.n_seq']
    k = round(float(V[f'{p}.frac_seq_H0_below_Ubar']) * int(n))
    print(f'| ResNet18-YawAug (ours, 4 seeds) | {NAME[ds]} | {n} / {V[p + ".n_subj"]} | {V[p + ".U0_m"]} | {V[p + ".Ubar_m"]} | {V[p + ".H0_m"]} | not run '
          f'| α=0: {diff(p, "H0_minus_Ubar")} (95.00%) | {k}/{n} | — | — |')
print()
print(f'verdict = {V["verdict.result"]} ({V["verdict.primary_pass_count"]}/{V["verdict.primary_n"]} primary sets pass, '
      f'{V["verdict.primary_reverse_count"]} reversed)')
