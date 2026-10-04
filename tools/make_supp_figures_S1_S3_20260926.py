"""补充材料图 S1、S3（2026-09-26 重写；原生成脚本 tools/build_priority_revision_20260909.py 所依赖的模块已于 09-15 删除）。
两步：extract 从结果文件写出作图源数据 CSV；plot 只读 CSV 作图。复现包只附 CSV，因此只运行 plot 即可重画。
  图S1：results/supplement_sources_20260909/fig4_elapsed_error.csv（ResNet18，独立受试者组，受试者与4种子平均的瞬时坐标RMSE）
  图S3：verification/mst_revision_20260909/illustration/{trajectories.npz,report.json}（序列 a006_2，seed 0，4个初始参考方向）
用法：python tools/make_supp_figures_S1_S3_20260926.py [extract] plot"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/v11'
S1_SRC = OUT / 'figS01_elapsed_error_source_data.csv'
S3_SRC = OUT / 'figS03_trajectories_source_data.csv'
S3_ATE = OUT / 'figS03_trajectories_ate.csv'
ARMS = [('yaw', 'Yaw augmentation'), ('ns', 'HN without sign rule'), ('hn', 'HN')]
ANG = [(0, 0), (2, 90), (4, 180), (6, 270)]
DECIM = 20                      # 200 Hz → 10 Hz，与论文轨迹保存频率相同
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})


def extract():
    rows = list(csv.DictReader(open(ROOT / 'results/supplement_sources_20260909/fig4_elapsed_error.csv')))
    with open(S1_SRC, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = ROOT / 'verification/mst_revision_20260909/illustration'
    with np.load(d / 'trajectories.npz') as f:          # 一次读入内存（逐次访问 npz 会反复解压）
        z = {k: f[k] for k in f.files}
    rep = json.load(open(d / 'report.json'))
    idx = np.arange(0, len(z['time_s']), DECIM)
    with open(S3_SRC, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['series', 'angle_deg', 'time_s', 'x_m', 'y_m'])
        for i in idx:
            w.writerow(['reference', '', f'{z["time_s"][i]:.3f}', *(f'{v:.4f}' for v in z['ground_truth_m'][i])])
        for arm, _ in ARMS:
            for k, deg in ANG:
                p = z[f'{arm}_{k}_position_m']
                for i in idx:
                    w.writerow([arm, deg, f'{z["time_s"][i]:.3f}', f'{p[i, 0]:.4f}', f'{p[i, 1]:.4f}'])
    with open(S3_ATE, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['sequence', 'arm', 'model', 'angle_deg', 'ate_m', 'rte_m'])
        for r in rep['rows']:
            w.writerow([r['sequence'], r['arm'], r['model'], r['angle_degrees'], r['ate_m'], r['rte_m']])
    print(f'[extract] {S1_SRC.name}, {S3_SRC.name} ({len(idx)} samples per series), {S3_ATE.name}')


def plot():
    rows = list(csv.DictReader(open(S1_SRC)))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for arm, lab, c, m in [('yaw', 'Yaw augmentation', '#d55e00', 's'), ('hn', 'HN', '#0072b2', 'o')]:
        rr = [r for r in rows if r['arm'] == arm]
        ax.plot([float(r['seconds']) for r in rr], [float(r['mean_m']) for r in rr], marker=m, color=c, label=lab, ms=5)
    ax.set_xlabel('Elapsed time (s)'); ax.set_ylabel('Instantaneous coordinate RMSE (m)'); ax.set_ylim(bottom=0)
    ax.grid(color='#e5e5e5'); ax.legend(frameon=True)
    fig.tight_layout(); fig.savefig(OUT / 'figS01_elapsed_error.png'); fig.savefig(OUT / 'figS01_elapsed_error.pdf'); plt.close(fig)

    tr = {}
    for r in csv.DictReader(open(S3_SRC)):
        tr.setdefault((r['series'], r['angle_deg']), []).append((float(r['x_m']), float(r['y_m'])))
    tr = {k: np.array(v) for k, v in tr.items()}
    styles = {0: ('#1f77b4', '-'), 90: ('#d55e00', '--'), 180: ('#009e73', ':'), 270: ('#cc79a7', '-.')}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for ax, (arm, title) in zip(axs, ARMS):
        g = tr[('reference', '')]; ax.plot(g[:, 0], g[:, 1], color='k', lw=1.4, label='Reference')
        for _, deg in ANG:
            c, ls = styles[deg]; p = tr[(arm, str(deg))]
            ax.plot(p[:, 0], p[:, 1], color=c, ls=ls, lw=0.9, label=f'{deg}°')
        ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_aspect('equal', adjustable='box'); ax.grid(color='#e5e5e5')
    axs[0].set_ylabel('y (m)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(OUT / 'figS03_trajectories.png'); fig.savefig(OUT / 'figS03_trajectories.pdf'); plt.close(fig)
    print('[plot] figures/v11/figS01_elapsed_error.{png,pdf}, figS03_trajectories.{png,pdf}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[1:] or ['plot']
    if 'extract' in steps:
        extract()
    if 'plot' in steps:
        plot()
