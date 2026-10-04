"""v3 稿（paper/merged/manuscript_zh_v3.md）新画或重画的图。只读已有结果文件，不跑模型。

  v3fig_error_composition   正文图 8：12 个模型 × 5 个测试集的复增益上限与尺度误差
                            （v2 图 8 含两个"整段 HN"序列模型变体，该实验已判为不成立并删除，此处去掉）
  v3fig_confirmatory        正文图 10：确认性检验（从未评测过的 TLIO 318 条、IMUNet 手机 87 条）的主要比较
  另把补充材料图 S1、S3 两张由 priority 修订稿构建脚本生成的图复制到 figures/v3/（原文件随旧稿删除），manifest 记录原始出处。

风格与 tools/make_v2_figures_20260913.py 相同。每张图写 source_data CSV，figures/v3/manifest.csv 记录来源与 SHA。
用法：.venv/bin/python tools/make_v3_figures_20260915.py
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import make_v2_figures_20260913 as V2          # 复用配色、字体与 DATASETS
plt = V2.plt
from matplotlib.lines import Line2D

OUT = ROOT / 'figures' / 'v3'
MANIFEST: list[dict] = []
DROP = {'ext_lstm_hnseq', 'ext_tcn_hnseq'}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def save(fig, stem, header, rows, description, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    with open(OUT / f'{stem}_source_data.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    MANIFEST.append(dict(name=stem, description=description, script='tools/make_v3_figures_20260915.py',
                         sources='; '.join(f'{s} (sha {sha(ROOT / s)})' for s in sources)))
    print(f'[写出] figures/v3/{stem}.png  ({len(rows)} 行源数据)')


def fig_error_composition():
    src = 'results/unified_eval_20260913/summary.json'
    tab = json.loads((ROOT / src).read_text())['table']
    models = sorted({k.split('|')[0] for k in tab} - DROP - {'ours_yaw_hn', 'ours_yaw_fa2', 'ours_yaw_tta8'})
    assert len(models) == 12, models
    hi = {'ours_hn': (V2.C_HN, 'D', 'ResNet18 + GN + HN (ours)'), 'ext_resnet': (V2.C_OFFICIAL, 'o', 'Official RoNIN ResNet')}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    rows = []
    for j, (ds, _) in enumerate(V2.DATASETS):
        others = [m for m in models if m not in hi]
        for m, o in zip(others, np.linspace(-.22, .22, len(others))):
            r = tab[f'{m}|{ds}']
            axes[0].plot(j + o, r['red_gain_pct'], 'o', ms=3.2, color=V2.GREY, zorder=2)
            axes[1].plot(j + o, r['eps_s_median'], 'o', ms=3.2, color=V2.GREY, zorder=2)
        for m, (col, mk, _) in hi.items():
            r = tab[f'{m}|{ds}']
            axes[0].plot(j, r['red_gain_pct'], mk, ms=6.5, color=col, mec='white', mew=.8, zorder=4)
            axes[1].plot(j, r['eps_s_median'], mk, ms=6.5, color=col, mec='white', mew=.8, zorder=4)
        for m in models:
            r = tab[f'{m}|{ds}']
            rows.append([m, ds, r['n'], r['n_seeds'], f"{r['ate_u']:.4f}", f"{r['red_scale_pct']:.2f}",
                         f"{r['red_heading_pct']:.2f}", f"{r['red_gain_pct']:.2f}", f"{r['eps_s_median']:.4f}",
                         f"{r['eps_theta_abs_median']:.3f}", f"{r['tlr']:.4f}"])
    for ax in axes:
        ax.set_xticks(range(len(V2.DATASETS)))
        ax.set_xticklabels(['RoNIN\ntrain subj.', 'RoNIN\nindep. subj.', 'RIDI', 'TLIO', 'IMUNet\nphones'], fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('ATE reduction if the complex gain\nwere corrected with ground truth (%)')
    axes[0].set_ylim(0, None)
    axes[0].set_title('(a) Share of error that is global scale + yaw', fontsize=8.5)
    axes[1].axhline(0, color=V2.INK, lw=.8)
    axes[1].set_ylabel(r'scale error $\varepsilon_s=|a^*|-1$' '\n(median over sequences)')
    axes[1].set_title('(b) Positive = speed under-estimated', fontsize=8.5)
    handles = [Line2D([], [], marker=mk, ls='', color=col, ms=6, label=lab) for col, mk, lab in hi.values()]
    handles.append(Line2D([], [], marker='o', ls='', color=V2.GREY, ms=3.5, label='other 10 models'))
    fig.tight_layout(w_pad=2.5)
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, 0))
    save(fig, 'v3fig_error_composition',
         ['model', 'dataset', 'n_sequences', 'n_seeds', 'ate_u_m', 'red_scale_pct', 'red_heading_pct',
          'red_gain_pct', 'eps_s_median', 'eps_theta_abs_median_deg', 'tlr'], rows,
         'Start-centred complex-gain decomposition for 12 models on five test sets (the two whole-sequence-frame '
         'variants of the official LSTM/TCN are excluded). Same source as main-text Table 14.', [src])


def fig_confirmatory():
    src = 'results/confirm_20260913/verdict.json'
    v = json.loads((ROOT / src).read_text())
    order = [('H1a', 'Augmented net:\nbolt-on HN − as is'), ('H1b', 'Augmented net:\ntwo-direction FA − as is'),
             ('H3', 'Trained with HN −\nmixed PCA frame'), ('H2', 'Official ResNet:\nbolt-on HN − as is')]
    cols = {'H1a': V2.C_HN, 'H1b': V2.C_HN, 'H3': V2.C_PCA, 'H2': V2.C_OFFICIAL}
    dsl = [('tlio_c', 'TLIO train+val (318)'), ('imunet_c', 'IMUNet phones, train split (87 seq., 4 subj.)')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    rows = []
    for ax, (ds, title) in zip(axes, dsl):
        for i, (h, lab) in enumerate(order):
            r = next(x for x in v['main'] if x['hypothesis'] == h and x['dataset'] == ds)
            base = r['control_mean']
            d = 100 * r['difference'] / base
            ci = r['subject_ci'] if ds == 'tlio_c' else [max(r['subject_ci'][0], r['seq_ci'][0]), min(r['subject_ci'][1], r['seq_ci'][1])]
            lo, hi = (100 * c / base for c in ci)
            y = len(order) - 1 - i
            ok = r['call'] in ('better', 'noninferior')
            ax.plot([lo, hi], [y, y], color=cols[h], lw=2.0, solid_capstyle='butt')
            ax.plot(d, y, 'D', ms=5.5, mfc=cols[h] if ok else 'white', mec=cols[h], mew=1.3)
            rows.append([h, ds, r['n_subjects'], r['n_sequences'], f"{r['treatment_mean']:.4f}", f'{base:.4f}',
                         f"{r['difference']:.4f}", f'{d:.2f}', f'{lo:.2f}', f'{hi:.2f}', r['call']])
        ax.axvline(0, color=V2.INK, lw=.8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('unified-protocol ATE difference (% of control)\n99.375% interval (Bonferroni, 8 tests)')
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([lab for _, lab in order][::-1], fontsize=7.8)
    handles = [Line2D([], [], marker='D', ls='', mfc=V2.INK, mec=V2.INK, label='criterion met'),
               Line2D([], [], marker='D', ls='', mfc='white', mec=V2.INK, label='criterion not met')]
    fig.tight_layout(w_pad=1.5)
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(.55, 0))
    save(fig, 'v3fig_confirmatory',
         ['hypothesis', 'dataset', 'n_subjects', 'n_sequences', 'treatment_mean_m', 'control_mean_m', 'difference_m',
          'difference_pct', 'ci_low_pct', 'ci_high_pct', 'call'], rows,
         'Confirmatory test on never-evaluated data (docs/53 criteria, docs/54 results): H1a, H1b, H3 superiority, '
         'H2 non-inferiority (margin +2% of control). Phone intervals are the intersection of subject and sequence intervals.',
         [src])


def copy_supplement_figures():
    """补充材料图 S1、S3：原由 tools/build_priority_revision_20260909.py 生成，随旧稿删除；此处只登记已复制到 figures/v3/ 的文件。"""
    for stem, desc, srcs in [
        ('figure_S01_elapsed_error', 'Supplementary Figure S1: instantaneous position error vs elapsed time (ResNet18, independent subjects).',
         ['results/e4_error_timescale/analysis.json', 'paper/priority_revision_20260909/source_data/fig4_elapsed_error.csv']),
        ('figure_02_trajectories', 'Supplementary Figure S3: one full sequence at four initial reference directions (seed 0).',
         ['verification/mst_revision_20260909/illustration', 'paper/priority_revision_20260909/source_data/trajectory_illustration_scores.csv'])]:
        if (OUT / f'{stem}.png').exists():
            MANIFEST.append(dict(name=stem, description=desc, script='tools/build_priority_revision_20260909.py（复制自已删除的旧稿目录）',
                                 sources='; '.join(srcs)))


def main():
    fig_error_composition()
    fig_confirmatory()
    copy_supplement_figures()
    with open(OUT / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'description', 'script', 'sources']); w.writeheader(); w.writerows(MANIFEST)
    print('[写出] figures/v3/manifest.csv')


if __name__ == '__main__':
    main()
