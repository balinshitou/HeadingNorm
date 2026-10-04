"""由 V25 判定结果生成正文表7、表8 的数据行与补充材料表S62（markdown），不手抄数字。
输入：results/confirm_anglefair_20260927/verdict.json、results/confirm_20260913/verdict.json（固定约定原判定）
输出：results/confirm_anglefair_20260927/tables.md
用法：.venv/bin/python tools/build_anglefair_tables_20260927.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V = json.loads((ROOT / 'results/confirm_anglefair_20260927/verdict.json').read_text())
C = json.loads((ROOT / 'results/confirm_20260913/verdict.json').read_text())
DSN = {'tlio_c': 'TLIO-confirm (318)', 'imunet_c': 'Phone-confirm (87, 4 subj.)'}
CALL = {'better': '**Pass**', 'noninferior': '**Non-inferior**', 'ns': 'Not shown', 'not_shown': 'Non-inferiority not shown',
        'worse': 'Worse'}
OLD = {'H1a′': 'H1a', 'H1b′': 'H1b', 'H2′': 'H2', 'H3′': 'H3'}
CMP = {'H1a′': ('Trained by us', 'ResNet18-YawAug: Plug-in HN (8 α) − Unmodified (8 ψ)'),
       'H1b′': ('Trained by us', 'ResNet18-YawAug: FA-2 (4 α) − Unmodified (8 ψ)'),
       'H2′': ('Released', 'RoNIN ResNet: Plug-in HN (8 α) − Unmodified (8 ψ)'),
       'H3′': ('Trained by us (training-time)', 'GN + HN − GN + MixPCA (8 ψ)')}


def m(x):
    return f'{x:.3f}'.replace('-', '−')


def ci(c):
    return f'[{c[0]:+.3f}, {c[1]:+.3f}]'.replace('-', '−')


def diff(r):
    return f"{m(r['difference'])} ({r['rel_pct']:+.1f}%)".replace('-', '−')


out = []
# ---------------- 表7
out.append('## 表7 数据行')
out.append('| ID | Provenance | Comparison (treatment − control) | Dataset | Treatment | Control | Difference (rel.) | Subject CI | Sequence CI | Improved | Verdict | Fixed convention (α = 0, ψ = 0): difference (rel.), verdict |')
out.append('|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|')
for r in V['main']:
    h = r['hypothesis']; ds = r['dataset']
    o = next(x for x in C['main'] if x['hypothesis'] == OLD[h] and x['dataset'] == ds)
    tl = ds == 'tlio_c'
    imp = f"{r['improved_sequences']}/{r['n_sequences']} seqs" if tl else \
        f"{r['improved_subjects']}/{r['n_subjects']} subj., {r['improved_sequences']}/{r['n_sequences']} seqs"
    verdict = CALL[r['call']] if tl else {'better': 'Pass by rule†', 'noninferior': 'Non-inferior by rule†'}.get(r['call'], CALL[r['call']])
    out.append(f"| {h} | {CMP[h][0]} | {CMP[h][1]} | {DSN[ds]} | {m(r['treatment_mean'])} | {m(r['control_mean'])} | {diff(r)} | "
               f"{'—' if tl else ci(r['subject_ci'])} | {ci(r['seq_ci'])} | {imp} | {verdict} | "
               f"{diff(o)}, {CALL[o['call']].replace('**', '')} |")
# ---------------- 表8
out.append('\n## 表8 数据行')
out.append('| Provenance | Frozen network | Unmodified ATE (8 ψ) | Plug-in HN ATE (8 α) | ATE difference (rel.) | ATE CI (level, family) | Improved seqs | Shape-error difference (rel.) | Shape-error CI (95%, descriptive) |')
out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
desc = {(d['model'], d['treat'], d['dataset'], d['name']): d for d in V['descriptive']}
rows8 = [('Trained by us', 'ResNet18-YawAug', next(r for r in V['main'] if r['hypothesis'] == 'H1a′' and r['dataset'] == 'tlio_c'),
          '99.375% (H1a′)', 'ours_yaw'),
         ('Trained by us', 'Transformer-YawAug', next(r for r in V['secondary'] if r['model'] == 'int_transformer_yaw' and r['dataset'] == 'tlio_c'),
          '97.5% (cross-architecture)', 'int_transformer_yaw'),
         ('Trained by us', 'IMUNet-YawAug', next(r for r in V['secondary'] if r['model'] == 'int_imunet_yaw' and r['dataset'] == 'tlio_c'),
          '97.5% (cross-architecture)', 'int_imunet_yaw'),
         ('Released', 'RoNIN ResNet', next(r for r in V['main'] if r['hypothesis'] == 'H2′' and r['dataset'] == 'tlio_c'),
          '99.375% (H2′)', 'ext_resnet')]
for prov, net, r, lv, key in rows8:
    s = desc[(key, 'H', 'tlio_c', 'fair_ate_shape')]
    out.append(f"| {prov} | {net} | {m(r['control_mean'])} | {m(r['treatment_mean'])} | {diff(r)} | {ci(r['seq_ci'])} {lv} | "
               f"{r['improved_sequences']}/{r['n_sequences']} | {diff(s)} | {ci(s['seq_ci'])} |")
# ---------------- 表S62
out.append('\n## 表S62 数据行')
NAMES = [('fixed_T0_minus_U0', 'Fixed: T(α=0) − U(ψ=0)'), ('T0_minus_Ubar', 'T(α=0) − Ū'), ('fair_Tbar_minus_Ubar', 'T̄ − Ū (primary)'),
         ('worstalpha_minus_Ubar', 'Worst α − Ū'), ('native_U0_minus_Ubar', 'U(ψ=0) − Ū'), ('conv_T0_minus_Tbar', 'T(α=0) − T̄'),
         ('tta8_minus_Tbar', 'TTA-8 − T̄'), ('fair_rte_u', 'T̄ − Ū, RTE'), ('fair_ate_lit', 'T̄ − Ū, literature ATE'),
         ('fair_ate_shape', 'T̄ − Ū, shape error')]
MOD = [('ours_yaw', 'H', 'ResNet18-YawAug, Plug-in HN'), ('ours_yaw', 'F', 'ResNet18-YawAug, FA-2'),
       ('int_transformer_yaw', 'H', 'Transformer-YawAug, Plug-in HN'), ('int_imunet_yaw', 'H', 'IMUNet-YawAug, Plug-in HN'),
       ('ext_resnet', 'H', 'RoNIN ResNet (released), Plug-in HN')]
out.append('| Frozen network, test-time method | Dataset | Comparison | Treatment | Control | Difference (rel.) | Subject CI | Sequence CI | Improved seqs |')
out.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
for key, t, lab in MOD:
    for ds in ('tlio_c', 'imunet_c'):
        for nm, nlab in NAMES:
            r = desc[(key, t, ds, nm)]
            out.append(f"| {lab} | {DSN[ds]} | {nlab} | {m(r['treatment_mean'])} | {m(r['control_mean'])} | {diff(r)} | "
                       f"{'—' if ds == 'tlio_c' else ci(r['subject_ci'])} | {ci(r['seq_ci'])} | {r['improved_sequences']}/{r['n_sequences']} |")
out.append('\n## 各角度均值（序列均值，m；本文网络先对4种子平均）')
out.append('| Frozen network, method | Dataset | Unmodified by ψ = 0°, 45°, …, 315° | Test-time method by α |')
out.append('|---|---|---|---|')
for key, t, lab in MOD:
    for ds in ('tlio_c', 'imunet_c'):
        b = V['by_angle'][f'{key}|{t}|{ds}']
        out.append(f"| {lab} | {DSN[ds]} | {' / '.join(f'{x:.3f}' for x in b['unmod_by_psi'])} | {' / '.join(f'{x:.3f}' for x in b['treat_by_alpha'])} |")
(ROOT / 'results/confirm_anglefair_20260927/tables.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
