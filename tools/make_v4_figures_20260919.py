"""v4 正文的两张新图（2026-09-19，docs/82 §3 第 2 项）。只读取已保存的结果，不运行模型。

  figures/v4/v4fig_problem_overview.png   正文图1：问题与方法一图说明
      (a) 同一段运动在两个水平参考方向下的表示（真值轨迹 + 两组坐标轴，示意）
      (b) 偏航增强模型对同一记录在8个参考方向下的预测，旋回后不重合
      (c) HN 模型对同一记录的8个预测，旋回后重合
      (b)(c) 数据与 v3 图5 相同：results/e0_trajectory_overlay/（序列 a058_2，seed 0）
  figures/v4/v4fig_repeatability.png      正文图3：左、中为 v3 图4 的受试者平均 ATE 随施加偏航角的变化（同一数据），
      右栏改为逐序列跨角 ATE 极差占该序列 ψ=0 处 ATE 的比例的经验累积分布（与正文表4同一定义），
      增加"去掉定符号"与原作者发布的 RoNIN ResNet（results/hn_plugin_p0_20260912/ext_resnet.json）。
每张图同时写出 *_source_data.csv。
用法：.venv/bin/python tools/make_v4_figures_20260919.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mst_figures as M  # 复用配色、字体与数据读取
import matplotlib.pyplot as plt

ROOT = M.ROOT
OUT = ROOT / 'figures/v4'
SEQ = 'a058_2'
OFFICIAL = 'Original RoNIN ResNet (released weights)'


def save(fig, name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=300)
    fig.savefig(OUT / f'{name}.pdf')
    with (OUT / f'{name}_source_data.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    plt.close(fig)
    print('[写出]', (OUT / f'{name}.png').relative_to(ROOT), len(rows), '行源数据')


def problem_overview():
    traj, gt = M._overlay_rows()
    traj = [r for r in traj if r['seq'] == SEQ]
    gx = np.array([float(r['x']) for r in gt if r['seq'] == SEQ])
    gy = np.array([float(r['y']) for r in gt if r['seq'] == SEQ])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
    rows = []

    # (a) 示意：同一条真值轨迹，两组水平坐标轴
    ax = axes[0]
    ax.plot(gx, gy, color='#777777', lw=2.2, solid_capstyle='round')
    span = max(np.ptp(gx), np.ptp(gy)); L = .28 * span
    ox, oy = gx[0], gy[0]
    psi = np.deg2rad(55)
    for ang, col, ls, lab in ((0., '#333333', '-', ('x', 'y')), (psi, '#CC5500', '--', ("x'", "y'"))):
        for k, t in enumerate(lab):
            a = ang + k * np.pi / 2
            ax.annotate('', xy=(ox + L * np.cos(a), oy + L * np.sin(a)), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.4, linestyle=ls))
            ax.text(ox + 1.12 * L * np.cos(a), oy + 1.12 * L * np.sin(a), t, color=col,
                    ha='center', va='center', fontsize=9)
    ax.text(.5, -.14, 'one motion, two horizontal frames\n'
            r'$X$ and $R_\psi X$ describe the same walk', transform=ax.transAxes,
            ha='center', va='top', fontsize=8.5)
    ax.set_title('(a) representation', pad=6)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x, y in zip(gx, gy):
        rows.append(['a_ground_truth', '', '', x, y])

    # (b)(c) 同一记录、同一冻结模型、8个参考方向
    arms = [('GN + yaw augmentation', '(b) ordinary network (yaw-augmented)'),
            ('GN + HeadingNorm', '(c) with HeadingNorm')]
    for ax, (arm, title) in zip(axes[1:], arms):
        sub = [r for r in traj if r['arm'] == arm]
        ates = {int(r['angle_index']): float(r['ate']) for r in sub}
        ax.plot(gx, gy, color='#BBBBBB', lw=3, solid_capstyle='round', zorder=1, label='ground truth')
        n = len(ates)
        for k in sorted(ates):
            xs = [float(r['x']) for r in sub if int(r['angle_index']) == k]
            ys = [float(r['y']) for r in sub if int(r['angle_index']) == k]
            ax.plot(xs, ys, color=plt.cm.viridis(k / max(n - 1, 1)), lw=1.0, alpha=.9, zorder=2)
            rows += [[arm, k, ates[k], x, y] for x, y in zip(xs, ys)]
        lo, hi = min(ates.values()), max(ates.values())
        note = (f'8 directions: ATE {lo:.3f}–{hi:.3f} m' if hi - lo > .01
                else f'8 directions coincide: ATE {lo:.3f} m')
        ax.text(.5, -.14, r'recovered $R_{-\psi}F(R_\psi X)$' + '\n' + note, transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, color='#B33A00' if hi - lo > .01 else '#00674F')
        ax.set_title(title, pad=6)
        ax.set_aspect('equal'); ax.set_xlabel(''); ax.tick_params(labelsize=7.5)
    axes[1].legend(loc='best', fontsize=7.5)
    fig.text(.5, .99, 'Horizontal coordinate rotation changes representation, not motion.',
             ha='center', va='top', fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.2, wspace=.25)
    save(fig, 'v4fig_problem_overview', ['series', 'angle_index', 'ate_m', 'x_m', 'y_m'], rows)


def relative_ranges():
    """每条（种子，序列）的跨角ATE极差 / ψ=0 处ATE，RoNIN独立受试者组；与正文表4同一定义。"""
    rows = M.sweep_rows()
    cell = {}
    for r in rows:
        if r['dataset'] == 'ronin' and r['split'] == 'unseen' and r['family'] in M.ORDER:
            cell.setdefault((r['family'], r['seed'], r['seq']), {})[r['angle_index']] = float(r['ate'])
    out = {}
    for (fam, _, _), v in cell.items():
        if len(v) >= 8:
            a = np.array([v[i] for i in sorted(v)])
            out.setdefault(fam, []).append(100 * (a.max() - a.min()) / a[0])
    ext = json.loads((ROOT / 'results/hn_plugin_p0_20260912/ext_resnet.json').read_text())['rows']
    P = np.array([r['orig_ate_by_psi'] for r in ext if r['dataset'] == 'ronin_unseen'])
    out[OFFICIAL] = list(100 * (P.max(1) - P.min(1)) / P[:, 0])
    return out


def repeatability():
    rows_all = M.sweep_rows()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    src = []
    for ax, (ds, sp, title) in zip(axes[:2], (('ronin', 'unseen', 'RoNIN independent-subject group'),
                                              ('ridi', '', 'RIDI'))):
        curves = M.angle_curves(rows_all, ds, sp)
        for fam in M.FOUR:
            ang, mean, sd = curves[fam]
            ang = list(ang) + [360.]; mean = np.append(mean, mean[0]); sd = np.append(sd, sd[0])
            ax.plot(ang, mean, color=M.COLOUR[fam], marker=M.MARKER[fam], label=M.LABEL[fam], zorder=3)
            ax.fill_between(ang, mean - sd, mean + sd, color=M.COLOUR[fam], alpha=.15, lw=0, zorder=2)
            src += [[title, M.LABEL[fam], a, m, s] for a, m, s in zip(ang[:-1], mean[:-1], sd[:-1])]
        ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlabel('yaw applied to the test input (deg)')
        ax.set_title(title, pad=6)
    axes[0].set_ylabel('subject-macro ATE (m)')
    axes[0].legend(loc='upper left', fontsize=7, handlelength=1.3)

    ax = axes[2]
    rel = relative_ranges()
    order = M.ORDER[:-1] + [OFFICIAL, M.ORDER[-1]]
    for fam in order:
        v = np.sort(np.array(rel[fam])); y = np.arange(1, len(v) + 1) / len(v)
        col = '#0072B2' if fam == OFFICIAL else M.COLOUR[fam]
        ls = ':' if fam == OFFICIAL else '-'
        ax.step(np.r_[0, v], np.r_[0, y], where='post', color=col, ls=ls,
                label=('Original RoNIN ResNet' if fam == OFFICIAL else M.LABEL[fam]))
        src.append(['ecdf', M.LABEL.get(fam, fam), 'median_pct', float(np.median(v)), len(v)])
    ax.set_xlim(-3, 150); ax.set_ylim(0, 1.02)
    ax.set_xlabel('per-recording ATE range / ATE at $\\psi=0$ (%)')
    ax.set_ylabel('empirical CDF')
    ax.set_title('What a single recording sees (independent group)', pad=6)
    ax.legend(loc='lower right', fontsize=6.8, handlelength=1.6)
    fig.subplots_adjust(wspace=.3)
    save(fig, 'v4fig_repeatability', ['panel', 'series', 'x', 'y', 'extra'], src)
    return {k: float(np.median(v)) for k, v in rel.items()}


if __name__ == '__main__':
    problem_overview()
    med = repeatability()
    for k, v in med.items():
        print(f'  中位极差占比 {k}: {v:.1f}%')
