"""v2 稿子（paper/merged/manuscript_zh_v2.md）§4.8、§4.9 新增的四张图。

只读已有结果文件，不跑任何模型；每张图同时写出 source_data CSV，manifest.csv 记录来源与脚本 SHA。
风格与 tools/mst_figures.py 相同（Okabe-Ito 配色、Helvetica、无上右框线）。

  v2fig_unified_paired      表 14：HN 与三个对照、官方 ResNet 外挂 HN 的配对差（统一口径 ATE，受试者整群 bootstrap）
  v2fig_error_composition   表 15 与 §4.8 正文：14 个模型 × 5 个测试集的复增益上限与尺度误差
  v2fig_phone_trajectory    IMUNet 手机数据上 HN 模型 ATE 居中位的一条序列：轨迹与累计路程
  v2fig_bolt_on             表 16：外部来源权重外挂 HN 的相对变化（逐窗网络；序列网络两种参考方向）

用法：.venv/bin/python tools/make_v2_figures_20260913.py
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures' / 'v2'
UE = ROOT / 'results' / 'unified_eval_20260913'

INK, GREY = '#333333', '#BBBBBB'
C_YAW, C_GN, C_PCA, C_HN, C_NOSIGN = '#E69F00', '#999999', '#56B4E9', '#009E73', '#D55E00'
C_OFFICIAL, C_MOBILE = '#0072B2', '#CC79A7'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#ECECEC', 'grid.linewidth': .6,
    'axes.axisbelow': True,
    'legend.frameon': False, 'lines.linewidth': 1.5, 'lines.markersize': 4,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

DATASETS = [('ronin_seen', 'RoNIN training subjects'), ('ronin_unseen', 'RoNIN independent subjects'),
            ('ridi', 'RIDI'), ('tlio', 'TLIO'), ('imunet', 'IMUNet phones')]
DS_LABEL = dict(DATASETS)
MANIFEST: list[dict] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def save(fig, stem, header, rows, description, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    with open(OUT / f'{stem}_source_data.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    MANIFEST.append(dict(name=stem, description=description,
                         sources='; '.join(f'{s} (sha {sha(ROOT / s)})' for s in sources)))
    print(f'[写出] figures/v2/{stem}.png  ({len(rows)} 行源数据)')


def excludes_zero(lo, hi):
    return hi < 0 or lo > 0


# --------------------------------------------------------------------------- 表 14
def fig_unified_paired():
    src = 'results/unified_eval_20260913/significance.json'
    sig = json.loads((ROOT / src).read_text())
    comps = [('ours_hn', 'ours_yaw', 'HN −\nyaw augmentation', C_YAW),
             ('ours_hn', 'ours_gn', 'HN −\nstandardisation only', C_GN),
             ('ours_hn', 'ours_pca', 'HN −\nPCA frame', C_PCA),
             ('ext_resnet_hn', 'ext_resnet', 'Official ResNet:\nbolt-on HN − as is', C_OFFICIAL)]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
    rows = []
    for ax, (t, c, title, col) in zip(axes, comps):
        for i, (ds, lab) in enumerate(DATASETS):
            r = next(x for x in sig if x['treatment'] == t and x['control'] == c
                     and x['dataset'] == ds and x['metric'] == 'ate_u')
            base = r['control_mean']
            d, lo, hi = (100 * v / base for v in (r['difference'], *r['subject_ci95']))
            y = len(DATASETS) - 1 - i
            filled = excludes_zero(lo, hi)
            ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle='butt')
            ax.plot(d, y, 'D', ms=5.5, mfc=col if filled else 'white', mec=col, mew=1.3)
            rows.append([title.replace('\n', ' '), ds, r['n_subjects'], f"{r['treatment_mean']:.4f}",
                         f'{base:.4f}', f"{r['difference']:.4f}", f"{r['subject_ci95'][0]:.4f}",
                         f"{r['subject_ci95'][1]:.4f}", f'{d:.2f}', f'{lo:.2f}', f'{hi:.2f}', int(filled)])
        ax.axvline(0, color=INK, lw=.8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('ATE difference\n(% of control)')
    n_sub = {ds: next(x['n_subjects'] for x in sig if x['dataset'] == ds) for ds, _ in DATASETS}
    axes[0].set_yticks(range(len(DATASETS)))
    axes[0].set_yticklabels([f'{lab} ({n_sub[ds]})' for ds, lab in DATASETS][::-1])
    axes[0].set_ylim(-.6, len(DATASETS) - .4)
    handles = [Line2D([], [], marker='D', ls='', mfc=INK, mec=INK, label='interval excludes 0'),
               Line2D([], [], marker='D', ls='', mfc='white', mec=INK, label='interval includes 0')]
    fig.tight_layout(w_pad=1.2)
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(.55, 0))
    save(fig, 'v2fig_unified_paired',
         ['comparison', 'dataset', 'n_subjects', 'treatment_mean_m', 'control_mean_m', 'difference_m',
          'ci_low_m', 'ci_high_m', 'difference_pct', 'ci_low_pct', 'ci_high_pct', 'excludes_zero'], rows,
         'Unified-protocol ATE, paired subject-macro differences with 95% subject-cluster bootstrap '
         '(10000 draws); difference and interval divided by the control mean. Same numbers as Table 14.',
         [src])


# --------------------------------------------------------------------------- 表 15
def fig_error_composition():
    src = 'results/unified_eval_20260913/summary.json'
    tab = json.loads((ROOT / src).read_text())['table']
    models = sorted({k.split('|')[0] for k in tab})
    assert len(models) == 14, models
    hi = {'ours_hn': (C_HN, 'D', 'ResNet18 + GN + HN (ours)'), 'ext_resnet': (C_OFFICIAL, 'o', 'Official RoNIN ResNet')}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    rows = []
    for j, (ds, _) in enumerate(DATASETS):
        others = [m for m in models if m not in hi]
        offs = np.linspace(-.22, .22, len(others))
        for m, o in zip(others, offs):
            r = tab[f'{m}|{ds}']
            axes[0].plot(j + o, r['red_gain_pct'], 'o', ms=3.2, color=GREY, zorder=2)
            axes[1].plot(j + o, r['eps_s_median'], 'o', ms=3.2, color=GREY, zorder=2)
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
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(['RoNIN\ntrain subj.', 'RoNIN\nindep. subj.', 'RIDI', 'TLIO', 'IMUNet\nphones'],
                           fontsize=7.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('ATE reduction if the complex gain\nwere corrected with ground truth (%)')
    axes[0].set_ylim(0, None)
    axes[0].set_title('(a) Share of error that is global scale + yaw', fontsize=8.5)
    axes[1].axhline(0, color=INK, lw=.8)
    axes[1].set_ylabel(r'scale error $\varepsilon_s=|a^*|-1$' '\n(median over sequences)')
    axes[1].set_title('(b) Positive = speed under-estimated', fontsize=8.5)
    handles = [Line2D([], [], marker=mk, ls='', color=col, ms=6, label=lab) for col, mk, lab in hi.values()]
    handles.append(Line2D([], [], marker='o', ls='', color=GREY, ms=3.5, label='other 12 models'))
    fig.tight_layout(w_pad=2.5)
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, 0))
    save(fig, 'v2fig_error_composition',
         ['model', 'dataset', 'n_sequences', 'n_seeds', 'ate_u_m', 'red_scale_pct', 'red_heading_pct',
          'red_gain_pct', 'eps_s_median', 'eps_theta_abs_median_deg', 'tlr'], rows,
         'Start-centred complex-gain decomposition for all 14 models on the five test sets: oracle ATE '
         'reduction from correcting scale and yaw together, and median scale error. Same source as Table 15.',
         [src])


# --------------------------------------------------------------------------- 手机轨迹
def fig_phone_trajectory():
    ds = 'imunet'
    srcs = ['results/unified_eval_20260913/ours_hn_s0.json', 'results/unified_eval_20260913/ours_yaw_s0.json',
            'results/unified_eval_20260913/ext_resnet.json']
    # 以 HN 模型 4 个种子逐序列平均的统一口径 ATE 取中位序列（与表 13 同一聚合）
    per = {}
    for s in range(4):
        for r in json.loads((UE / f'ours_hn_s{s}.json').read_text())['rows']:
            if r['dataset'] == ds:
                per.setdefault(r['sequence'], []).append(r['ate_u'])
    mean = {k: float(np.mean(v)) for k, v in per.items() if len(v) == 4}
    assert len(mean) == 36
    med = float(np.median(list(mean.values())))
    seq = min(mean, key=lambda k: abs(mean[k] - med))
    print(f'  手机数据：HN 4 种子平均 ATE 中位 {med:.3f} m → 选 {seq}（{mean[seq]:.3f} m）')
    cache = ROOT / 'data/eval/imunet_owndata' / f'imunet__{seq}.npz'
    with np.load(cache) as z:
        gt_full, te_full = np.asarray(z['gt'], float), np.asarray(z['te'], float)
    runs = [('ours_hn_s0', 'ResNet18 + GN + HN (ours, seed 0)', C_HN, '-'),
            ('ours_yaw_s0', 'ResNet18 + GN + yaw aug. (seed 0)', C_YAW, '-'),
            ('ext_resnet', 'Official RoNIN ResNet', C_OFFICIAL, '--')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    rows, ends = [], []
    gt = gt_full[::20]; t = te_full[::20] - te_full[0]
    L_gt = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    axes[0].plot(gt[:, 0] - gt[0, 0], gt[:, 1] - gt[0, 1], color=GREY, lw=3.2, label='ground truth', zorder=1)
    axes[1].plot(t, L_gt, color=GREY, lw=3.2, label='ground truth', zorder=1)
    for stem, lab, col, ls in runs:
        xy = np.load(UE / f'{stem}.npz')[f'{ds}__{seq}'].astype(float)
        row = next(r for r in json.loads((UE / f'{stem}.json').read_text())['rows']
                   if r['dataset'] == ds and r['sequence'] == seq)
        n = min(len(xy), len(gt))
        # 与 src/pkg_common.ate_rte 同一定义（逐坐标 RMSE，复刻 RoNIN metric.py），在 10 Hz 降采样轨迹上复算自检
        rmse10 = float(np.sqrt(np.mean((xy[:n] - gt[:n]) ** 2)))
        assert abs(rmse10 - row['ate_u']) / row['ate_u'] < .02, (stem, rmse10, row['ate_u'])
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy[:n], axis=0), axis=1))])
        axes[0].plot(xy[:n, 0] - gt[0, 0], xy[:n, 1] - gt[0, 1], color=col, ls=ls, lw=1.3, zorder=3,
                     label=f"{lab}: ATE {row['ate_u']:.2f} m, " + r'$\varepsilon_s$' + f" {row['eps_s']:+.2f}")
        axes[1].plot(t[:n], L, color=col, ls=ls, lw=1.3, zorder=3)
        ends.append((L[-1], t[n - 1], row['tlr'], col))
        for k in range(n):
            rows.append([stem, seq, f'{t[k]:.2f}', f'{xy[k, 0]:.4f}', f'{xy[k, 1]:.4f}', f'{gt[k, 0]:.4f}',
                         f'{gt[k, 1]:.4f}', f"{row['ate_u']:.4f}", f"{row['eps_s']:.4f}", f"{row['tlr']:.4f}"])
    for k, (Lend, tend, tlr, col) in enumerate(sorted(ends)):      # 按终点路程由低到高错开标注
        axes[1].annotate(f'TLR {tlr:.2f}', (tend, Lend), xytext=(5, 11 * (k - 1)), textcoords='offset points',
                         va='center', fontsize=7.5, color=INK,
                         arrowprops=dict(arrowstyle='-', color=col, lw=.8, shrinkA=0, shrinkB=1))
    axes[0].plot(0, 0, 'o', color=INK, ms=4, zorder=5)
    axes[0].annotate('start', (0, 0), xytext=(6, -10), textcoords='offset points', fontsize=7.5, color=INK)
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    axes[0].set_title('(a) Unified protocol: start and first-10 m heading fixed', fontsize=8.5)
    axes[1].set_xlabel('time (s)'); axes[1].set_ylabel('cumulative path length (m)')
    axes[1].set_title('(b) Path length: predictions fall short', fontsize=8.5)
    axes[1].set_xlim(0, t[-1] * 1.18)
    fig.suptitle(f'IMUNet phone sequence {seq} (median HN error of 36 test sequences)', fontsize=9.5)
    fig.tight_layout(w_pad=2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(.5, 0),
               fontsize=7.5)
    save(fig, 'v2fig_phone_trajectory',
         ['run', 'sequence', 'time_s', 'pred_x_m', 'pred_y_m', 'gt_x_m', 'gt_y_m', 'ate_u_m', 'eps_s', 'tlr'], rows,
         f'Unified-protocol trajectories (10 Hz) on IMUNet phone sequence {seq}, the sequence whose HN ATE '
         '(four-seed mean) is closest to the median of the 36 phone test sequences.',
         srcs + [f'data/eval/imunet_owndata/imunet__{seq}.npz'])
    return seq


# --------------------------------------------------------------------------- 表 16
def fig_bolt_on():
    s0, s1, s2 = ('results/hn_plugin_p0_20260912/summary.json', 'results/p1_summary_20260912.json',
                  'results/imunet_retrain_p2_20260913/summary.json')
    p0, p1, p2 = (json.loads((ROOT / s).read_text()) for s in (s0, s1, s2))
    dsets = [('ronin_seen', 'RoNIN training subj.'), ('ronin_unseen', 'RoNIN indep. subj.'),
             ('ridi', 'RIDI'), ('tlio', 'TLIO')]
    rows = []

    def pct(cell):
        return tuple(100 * v / cell['a'] for v in (cell['diff'], *cell['ci95']))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    # (a) 逐窗网络
    ax = axes[0]
    groups = [('Official RoNIN ResNet', C_OFFICIAL, 'o', lambda d: p0['cells'][f'ext_resnet|{d}']['ate_hn_vs_orig']),
              ('MobileNet (official IMUNet code,\nretrained; passed self-check)', C_MOBILE, 's',
               lambda d: p2['cells'][f'MobileNet|{d}']['ate_hn_vs_orig'])]
    ylabels = []
    y = 0
    for gi, (name, col, mk, get) in enumerate(groups):
        for d, dl in dsets:
            c = get(d); v, lo, hi = pct(c); f = excludes_zero(lo, hi)
            ax.plot([lo, hi], [-y, -y], color=col, lw=2)
            ax.plot(v, -y, mk, ms=5.5, mfc=col if f else 'white', mec=col, mew=1.3)
            rows.append(['frame-wise', name.split('\n')[0], 'per-window frame', d, c['n'], f"{c['a']:.4f}",
                         f"{c['b']:.4f}", f"{c['diff']:.4f}", f"{c['ci95'][0]:.4f}", f"{c['ci95'][1]:.4f}",
                         f'{v:.2f}', f'{lo:.2f}', f'{hi:.2f}', int(f)])
            ylabels.append((-y, dl)); y += 1
        y += .8
    ax.set_yticks([p for p, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=7.5)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel('ATE change after bolt-on (%)')
    ax.set_title('(a) Frame-wise networks', fontsize=8.5)
    leg_a = [Line2D([], [], marker=mk, ls='', color=col, ms=5.5, label=name.replace('\n', ' '))
             for name, col, mk, _ in groups]
    # (b) 序列网络：逐窗参考方向 vs 整段恒定参考方向，均相对原样
    ax = axes[1]
    ylabels = []; y = 0
    for net, key in (('LSTM', 'ext_lstm'), ('TCN', 'ext_tcn')):
        for d, dl in dsets:
            for off, mode, col, cell in ((-.17, 'per-window frame', C_NOSIGN, p0['cells'][f'{key}|{d}']['ate_hn_vs_orig']),
                                         (.17, 'constant whole-sequence frame', C_HN,
                                          p1['A_hnseq']['cells'][f'{key}|{d}']['hnseq_vs_orig'])):
                v, lo, hi = pct(cell); f = excludes_zero(lo, hi)
                ax.plot([lo, hi], [-y - off, -y - off], color=col, lw=2)
                ax.plot(v, -y - off, 'D', ms=5, mfc=col if f else 'white', mec=col, mew=1.3)
                rows.append(['sequence', f'Official RoNIN {net}', mode, d, cell['n'], f"{cell['a']:.4f}",
                             f"{cell['b']:.4f}", f"{cell['diff']:.4f}", f"{cell['ci95'][0]:.4f}",
                             f"{cell['ci95'][1]:.4f}", f'{v:.2f}', f'{lo:.2f}', f'{hi:.2f}', int(f)])
            ylabels.append((-y, f'{net} · {dl}')); y += 1
        y += .6
    ax.set_yticks([p for p, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=7.5)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel('ATE change vs. as released (%)')
    ax.set_title('(b) Sequence networks (official LSTM / TCN)', fontsize=8.5)
    leg_b = [Line2D([], [], marker='D', ls='', color=C_NOSIGN, ms=5, label='LSTM/TCN: per-window frame'),
             Line2D([], [], marker='D', ls='', color=C_HN, ms=5, label='LSTM/TCN: constant whole-sequence frame'),
             Line2D([], [], marker='D', ls='', mfc='white', mec=INK, ms=5, label='open marker: interval includes 0')]
    fig.tight_layout(w_pad=2)
    fig.legend(handles=leg_a + leg_b, loc='upper center', ncol=2, bbox_to_anchor=(.5, 0), fontsize=7.5)
    save(fig, 'v2fig_bolt_on',
         ['network_type', 'network', 'frame', 'dataset', 'n_sequences', 'orig_ate_m', 'bolt_on_ate_m', 'diff_m',
          'ci_low_m', 'ci_high_m', 'diff_pct', 'ci_low_pct', 'ci_high_pct', 'excludes_zero'], rows,
         'Bolt-on HN on externally sourced weights, official protocol ATE, relative to the unmodified network; '
         'difference and 95% paired sequence bootstrap interval divided by the original mean. Same numbers as Table 16.',
         [s0, s1, s2])


# --------------------------------------------------------------------------- 图 2（§3.3）
def fig_pipeline():
    """§3.3 的两个环节。G 与 h 位于 HN 旋转之后，对任意 ψ 收到同一输入，只需是确定性函数。示意图，无数据。"""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.axis('off'); ax.grid(False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    blocks = [(1.0, 'input\n$R_\\psi X$', '#F4F4F4', '#AAAAAA'),
              (20.5, '$R_{-\\phi}$', '#E4F3EE', C_HN),
              (40.0, 'standardise\n$G$', '#F4F4F4', '#AAAAAA'),
              (59.5, 'backbone\n$h$', '#F4F4F4', '#AAAAAA'),
              (79.0, '$R_{+\\phi}$', '#E4F3EE', C_HN)]
    for x, label, fc, ec in blocks:
        ax.add_patch(FancyBboxPatch((x, 26), 16, 11, boxstyle='round,pad=0.5,rounding_size=1.4',
                                    facecolor=fc, edgecolor=ec, linewidth=1.1))
        ax.text(x + 8, 31.5, label, ha='center', va='center', fontsize=8.6)
    ax.add_patch(Rectangle((38.2, 24.3), 39.6, 14.4, fill=False, edgecolor='#888888',
                           linewidth=.8, linestyle=(0, (3, 2))))
    for x in (17.5, 37.0, 56.5, 76.0):
        ax.annotate('', xy=(x + 3.0, 31.5), xytext=(x + .3, 31.5),
                    arrowprops=dict(arrowstyle='-|>', color='#999999', lw=1))
    ax.annotate('', xy=(98.5, 31.5), xytext=(95.8, 31.5),
                arrowprops=dict(arrowstyle='-|>', color='#999999', lw=1))
    ax.text(99.2, 31.5, '$R_\\psi\\hat v$', ha='left', va='center', fontsize=8.6)
    notes = [(28.5, 'link 1\n$\\lambda_1\\neq\\lambda_2$,  $m_3\\neq 0$', C_HN),
             (58.0, 'canonical frame: same input for every $\\psi$;\n'
                    'any deterministic $G$ and $h$', '#888888'),
             (87.0, 'link 2\nthe same $\\phi$', C_HN)]
    for x, txt, col in notes:
        ax.plot([x, x], [23.8, 21.0], color='#CCCCCC', lw=.8, ls=(0, (2, 2)))
        ax.text(x, 19.6, txt, ha='center', va='top', fontsize=7.6, color=col)
    ax.text(50, 3.0, '$\\hat v(R_\\psi X)=R_\\psi\\,\\hat v(X)$  holds exactly when both links hold',
            ha='center', va='bottom', fontsize=9)
    save(fig, 'v2fig_equivariance_pipeline', ['note'], [['schematic; no plotted data']],
         '§3.3 图2：精确偏航等变的两个环节（G 与 h 在规范坐标系内，无约束）', [])


def main():
    fig_pipeline()
    fig_unified_paired()
    fig_error_composition()
    seq = fig_phone_trajectory()
    fig_bolt_on()
    me = Path(__file__)
    with open(OUT / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'description', 'sources', 'script'])
        w.writeheader()
        for m in MANIFEST:
            w.writerow({**m, 'script': f'tools/{me.name} (sha {sha(me)})'})
    print(f'[写出] figures/v2/manifest.csv；手机示例序列 {seq}')


if __name__ == '__main__':
    main()
