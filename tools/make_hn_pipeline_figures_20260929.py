"""正文图3、图4：把 HN 的推理流程（不含 ψ）与参考方向实验（含 ψ）分成两张示意图。

  fig3_hn_inference      推理：只用当前窗口 X，逐窗口求 φ，R_{-φ} → f → R_{+φ}；全图不出现 ψ
  fig4_psi_experiment    实验（缩略图读 results/e0_plugin_overlay_20260929，V27）：把同一记录整体旋转 ψ_k=2πk/8，普通网络与加 HN 两行对照；ψ 只在实验中出现

图3为示意图，无数据；图4缩略图为真实轨迹；风格与 tools/make_v2_figures_20260913.py 相同。取代 figures/v2/v2fig_equivariance_pipeline。
用法：.venv/bin/python tools/make_hn_pipeline_figures_20260929.py
"""
from __future__ import annotations
import csv, hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures' / 'v13'

INK = '#333333'
C_HN, C_PSI = '#009E73', '#D55E00'
FC_HN, FC_PSI, FC_NET = '#E4F3EE', '#FBEBDD', '#F4F4F4'
EC_NET = '#AAAAAA'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def box(ax, x, y, w, h, label, fc=FC_NET, ec=EC_NET, ls='-', fs=8.6, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                facecolor=fc, edgecolor=ec, linewidth=1.1, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color='#999999', ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1, linestyle=ls, shrinkA=0, shrinkB=0))


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis('off'); ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    return fig, ax


def save(fig, stem, description, sources=''):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    with open(OUT / f'{stem}_source_data.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([['note'], ['schematic; thumbnails from ' + sources if sources else 'schematic; no plotted data']])
    return dict(name=stem, description=description, sources=sources,
                script=f'tools/{Path(__file__).name} (sha {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]})')


# --------------------------------------------------------------------------- 图3：推理（无 ψ）
def hn_frame(ax, x, y, w, h, title, fs=8.6):
    """HeadingNorm 外框：浅绿底、绿边，左上角标题。框内的网络 f 另画灰框，表示 f 本身不变。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.6',
                                facecolor='#F2FAF7', edgecolor=C_HN, linewidth=1.3, zorder=0))
    ax.text(x + 1.2, y + h - 1.0, title, ha='left', va='top', fontsize=fs, color=C_HN, fontweight='bold')


# --------------------------------------------------------------------------- 图3：推理（无 ψ）
def fig3():
    fig, ax = canvas(7.2, 3.3, 100, 50)
    y, h = 25, 11
    hn_frame(ax, 20.5, 2.5, 64, 44, 'HeadingNorm (HN)')
    box(ax, 0.5, y, 15, h, 'IMU window $X$\n(world frame)', fs=8.2)
    box(ax, 23, y, 14, h, 'rotate input\nby $-\\phi$', FC_HN, C_HN, fs=8.2)
    box(ax, 43, y, 14, h, 'network $f$\n(per-window)', '#FFFFFF', EC_NET, fs=8.2)
    box(ax, 63, y, 19, h, 'rotate output\nback by $+\\phi$', FC_HN, C_HN, fs=8.2)
    ax.text(90, y + h / 2, 'velocity $\\hat v$', ha='left', va='center', fontsize=8.6)
    ym = y + h / 2
    for x0, x1 in ((16.3, 22.6), (37.8, 42.6), (57.8, 62.6), (82.8, 89.5)):
        arrow(ax, x0, ym, x1, ym)
    box(ax, 23, 5, 59, 10,
        'estimate $\\phi$ from the horizontal acceleration of $X$:\n'
        'principal axis (Eq. 4) + third-moment sign (Eq. 5)', FC_HN, C_HN, fs=7.9)
    ax.plot([8, 8], [y - .4, 10], color='#999999', lw=1)
    arrow(ax, 8, 10, 22.6, 10)
    arrow(ax, 30, 15.6, 30, y - .4, color=C_HN)
    arrow(ax, 72.5, 15.6, 72.5, y - .4, color=C_HN)
    ax.text(73.8, 20.3, 'same $\\phi$', ha='left', va='center', fontsize=7.6, color=C_HN)
    ax.text(50, -3.5, '$f=h\\circ G$ when HN is used in training; $f$ = a frozen network when HN is plugged in.\n'
                      'Only the current window $X$ is used; no learnable parameters.',
            ha='center', va='top', fontsize=7.6, color='#666666')
    return save(fig, 'fig3_hn_inference',
                '正文图3：HN 推理流程（与算法1对应）。HeadingNorm 外框包住求φ、旋入−φ、旋出+φ，网络f不变；只用当前窗口 X；不含 ψ')


# --------------------------------------------------------------------------- 图4：参考方向实验（含 ψ）
V27 = ROOT / 'results/e0_plugin_overlay_20260929'


def load_v27():
    import json
    tr = {}
    with open(V27 / 'trajectories.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tr.setdefault((r['arm'], int(r['angle_index'])), []).append((float(r['x']), float(r['y'])))
    with open(V27 / 'ground_truth.csv', encoding='utf-8') as f:
        gt = [(float(r['x']), float(r['y'])) for r in csv.DictReader(f)]
    return tr, gt, json.loads((V27 / 'summary.json').read_text())


def thumb(fig, ax, x0, y0, x1, y1, tr, gt, arm):
    import numpy as np
    to_fig = ax.transData + fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = to_fig.transform((x0, y0)), to_fig.transform((x1, y1))
    a = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    g = np.asarray(gt); a.plot(g[:, 0], g[:, 1], color='#C8C8C8', lw=2.2, zorder=0)
    cmap = plt.get_cmap('viridis')
    for k in range(8):
        t = np.asarray(tr[(arm, k)]); a.plot(t[:, 0], t[:, 1], color=cmap(k / 7), lw=.7)
    a.set_aspect('equal', adjustable='datalim'); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color('#DDDDDD'); sp.set_linewidth(.6)
    return a


def fig4():
    tr, gt, sm = load_v27()
    fig, ax = canvas(8.2, 4.6, 140, 74)
    h = 9
    ya, yb = 51, 19            # 两行的框底
    X_IN, X_RM, X_F, X_RP, X_RB = (37, 10), (53, 11), (70, 12), (87, 11), (104, 12)   # (x, 宽)，两行对齐
    X_TH = 121                 # 缩略图左缘
    yc = (ya + yb + h) / 2
    box(ax, 0.5, yc - h / 2, 13, h, 'one\nrecording $X$', fs=8.0)
    box(ax, 17, yc - h / 2, 14, h, 'rotate by $\\psi$\n$\\psi=2\\pi k/8$', FC_PSI, C_PSI, ls='--', fs=8.0)
    arrow(ax, 14.1, yc, 16.6, yc)
    ax.plot([31.6, 33.5, 33.5], [yc, yc, ya + h / 2], color=C_PSI, lw=1)
    ax.plot([33.5, 33.5], [yc, yb + h / 2], color=C_PSI, lw=1)
    arrow(ax, 33.5, ya + h / 2, 36.6, ya + h / 2, color=C_PSI)
    arrow(ax, 33.5, yb + h / 2, 36.6, yb + h / 2, color=C_PSI)

    def row(yy, items):
        for (x, w), lab, fc, ec, ls in items:
            box(ax, x, yy, w, h, lab, fc, ec, ls, fs=8.2)
        for ((x0, w0), *_), ((x1, _), *_) in zip(items, items[1:]):
            arrow(ax, x0 + w0 + .6, yy + h / 2, x1 - .4, yy + h / 2)

    inp = (X_IN, '$R_\\psi X$', FC_NET, EC_NET, '-')
    net = (X_F, 'network $f$', '#FFFFFF', EC_NET, '-')
    back = (X_RB, 'rotate\nby $-\\psi$', FC_PSI, C_PSI, '--')

    # (a) 普通网络：直接把 R_ψX 送进 f
    ax.text(37, ya + h + 6.5, '(a) network $f$ alone', ha='left', va='bottom', fontsize=9.2, color=INK)
    row(ya, [inp, net, back])
    ax.text(X_F[0] + X_F[1] / 2, ya + h + 1.6, '$f$ sees a different input for each $\\psi$',
            ha='center', va='bottom', fontsize=7.4, color=C_PSI)

    # (b) 同一个 f，外面套 HeadingNorm
    hn_frame(ax, 50.5, yb - 6.5, 50, h + 15, 'HeadingNorm', fs=8.2)
    ax.text(37, yb + h + 10.5, '(b) the same network $f$ with HeadingNorm', ha='left', va='bottom', fontsize=9.2, color=INK)
    row(yb, [inp, (X_RM, 'rotate\nby $-\\phi$', FC_HN, C_HN, '-'), net,
             (X_RP, 'rotate\nback $+\\phi$', FC_HN, C_HN, '-'), back])
    ax.text(X_F[0] + X_F[1] / 2, yb + h + 1.6, '$f$ sees the same input for every $\\psi$',
            ha='center', va='bottom', fontsize=7.4, color=C_HN)
    ax.text(X_RM[0], yb - 1.4, 'link 1: $\\phi$ turns with $\\psi$', ha='left', va='top', fontsize=7.0, color=C_HN)
    ax.text(X_RP[0] + X_RP[1], yb - 1.4, 'link 2: the same $\\phi$', ha='right', va='top', fontsize=7.0, color=C_HN)

    # 缩略图：V27，同一冻结网络（ResNet18，YawAug，seed 0），序列 a058_2
    ua, hb = sm['unmodified'], sm['plug-in HN']
    for yy, arm, col, txt in ((ya, 'unmodified', C_PSI,
                               f'8 different trajectories\nATE {ua["ate_min"]:.3f}–{ua["ate_max"]:.3f} m'),
                              (yb, 'plug-in HN', C_HN,
                               f'8 trajectories coincide\nATE {hb["ate_min"]:.3f} m')):
        thumb(fig, ax, X_TH, yy - 6, X_TH + 21, yy + h + 6, tr, gt, arm)
        arrow(ax, X_RB[0] + X_RB[1] + .6, yy + h / 2, X_TH - .6, yy + h / 2)
        ax.text(X_TH + 10.5, yy - 7, txt, ha='center', va='top', fontsize=7.4, color=col)

    ax.text(70, 3.0, 'Orange dashed steps exist only in the experiment; at inference there is no $\\psi$ (Fig. 3).\n'
                      '$\\hat v(R_\\psi X)=R_\\psi\\,\\hat v(X)$ holds exactly when '
                      '$\\lambda_1\\neq\\lambda_2$ and $m_3\\neq 0$.',
            ha='center', va='top', fontsize=8)
    return save(fig, 'fig4_psi_experiment',
                '正文图4：参考方向实验。同一记录旋转 ψ_k=2πk/8；(a) 网络 f 单独使用，轨迹随 ψ 变；(b) 同一个 f 套 HeadingNorm，'
                '8 条轨迹重合；缩略图为 V27（Sensors_v4_resnet_yaw_s0，a058_2）；ψ 只在实验中出现',
                sources=f'results/e0_plugin_overlay_20260929/trajectories.csv (sha {hashlib.sha256((V27 / "trajectories.csv").read_bytes()).hexdigest()[:16]})')


def main():
    rows = [fig3(), fig4()]
    with open(OUT / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'description', 'sources', 'script'])
        w.writeheader(); w.writerows(rows)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
