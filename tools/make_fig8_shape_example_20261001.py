"""正文图8：TLIO确认集一条序列上，四个冻结网络外挂HN前后的轨迹形状（2026-10-01）。

选序列规则（作者要求取效果最明显的一条以便审稿人观察；图注须写明它不代表平均效果）：
  只读 results/confirm_anglefair_20260927/ 的固定约定结果（原网络 ψ=0 → U0，外挂HN α=0 → H0；本文网络取种子0），
  在真值路程不少于100 m、且四个网络形状误差均下降的TLIO确认集序列中，取四个网络形状误差相对下降的平均值最大者。
推理：冻结权重，每个网络 U0 与 H0 各一次全序列前向，实现同 tools/confirm_anglefair_eval_20260927.py；
  逐项核对 ate_u、ate_shape 与已存结果一致（差 < 1e-4 m）。
作图：统一评价方式对齐后再按式(12)用真值修正整体尺度与整体转角，只剩形状差异。
输出：figures/v14/fig8_shape_example.{png,pdf}、_source_data.csv；缓存 results/fig8_shape_example_20261001/
用法：.venv/bin/python tools/make_fig8_shape_example_20261001.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import hn_plugin_p0_20260912 as H  # noqa: E402
import unified_eval_20260913 as UE  # noqa: E402
import confirm_eval_20260913 as CE  # noqa: E402
UE.H, UE.P = H, H.P

SRC = ROOT / 'results/confirm_anglefair_20260927'
CACHE = ROOT / 'results/fig8_shape_example_20261001'
OUT = ROOT / 'figures/v14'
STEM = 'fig8_shape_example'
NETS = [('ResNet18-YawAug', 'ours_yaw_s0'), ('Transformer-YawAug', 'int_transformer_yaw_s0'),
        ('IMUNet-YawAug', 'int_imunet_yaw_s0'), ('RoNIN ResNet (released)', 'ext_resnet')]
MIN_PATH = 100.0
C_GT, C_U, C_HN = '#BBBBBB', '#D55E00', '#009E73'
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})


def select():
    R = {}
    for name, f in NETS:
        for r in json.load(open(SRC / f'{f}.json'))['rows']:
            if r['dataset'] == 'tlio_c':
                R.setdefault(r['sequence'], {})[name] = {k: r['v'][k] for k in ('U0', 'H0')}
    best = None
    for p in CE.files('tlio_c'):
        z = CE.read(p); s = str(z['seq']); g = np.asarray(z['gt'], float)
        if np.linalg.norm(np.diff(g, axis=0), axis=1).sum() < MIN_PATH:
            continue
        red = [(d['U0']['ate_shape'] - d['H0']['ate_shape']) / d['U0']['ate_shape'] for d in R[s].values()]
        if min(red) > 0 and (best is None or np.mean(red) > best[1]):
            best = (s, float(np.mean(red)), p, R[s])
    return best


def load_net(f, dev):
    if f.startswith('ours_yaw'):
        from sensors_evaluate import load
        return load(UE.OURS['ours_yaw'].format(int(f[-1])), dev)[0]
    return H.load_model(f, dev)[0]


def shape_corrected(xy, g, te):
    n = min(len(xy), len(g)); xu, _ = UE.unify(xy[:n], g[:n], te[:n])
    w = UE.cplx(xu - g[0]); q = UE.cplx(g[:n] - g[0]); a = np.vdot(w, q) / np.vdot(w, w).real
    c = a * w
    return np.stack([c.real, c.imag], 1) + g[0]


def main():
    seq, red, path, ref = select()
    z = CE.read(path); g = np.asarray(z['gt'], float); te = np.asarray(z['te'], float)
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    npz = CACHE / f'tlio_c_{seq}.npz'
    if npz.exists():
        d = np.load(npz); tr = {k: d[k] for k in d.files}
    else:
        import torch
        dev = torch.device('cpu'); feat = np.asarray(z['feat'], np.float32)
        n = len(range(0, len(feat) - H.WINDOW, H.STRIDE)); th = H.headings(feat)[1].astype(np.float32)
        tr = {}
        for name, f in NETS:
            net = load_net(f, dev)
            ids, v = H.run_window(net, feat, np.zeros(n, np.float32), dev); tr[f'{f}_U0'] = np.asarray(H.traj_window(v, ids, z), float)
            ids, v = H.run_window(net, feat, th, dev); tr[f'{f}_H0'] = np.asarray(H.traj_window(v, ids, z), float)
        np.savez(npz, **tr)
    M = {}
    for name, f in NETS:
        for k in ('U0', 'H0'):
            m = UE.metrics(tr[f'{f}_{k}'], z, 'tlio')[0]; M[(name, k)] = m
            for q in ('ate_u', 'ate_shape'):
                assert abs(m[q] - ref[name][k][q]) < 1e-4, (name, k, q, m[q], ref[name][k][q])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    rows = []
    C = {k: shape_corrected(v, g, te) for k, v in tr.items()}
    n = min(len(v) for v in C.values()); gg = g[:n]; sl = slice(None, None, 20)
    allxy = np.concatenate([gg] + [v[:n] for v in C.values()])
    lo, hi = allxy.min(0) - 1.5, allxy.max(0) + 1.5
    for j, (ax, (name, f)) in enumerate(zip(axes.flat, NETS)):
        u, h = C[f'{f}_U0'][:n], C[f'{f}_H0'][:n]
        ax.plot(*gg[sl].T, color=C_GT, lw=3.2, solid_capstyle='round', zorder=1, label='ground truth')
        ax.plot(*u[sl].T, color=C_U, lw=1.2, zorder=2, label='unmodified')
        ax.plot(*h[sl].T, color=C_HN, lw=1.4, zorder=3, label='plug-in HN')
        ax.plot(*gg[0], 'o', color='#555555', ms=4, zorder=5)
        su, sh = M[(name, 'U0')]['ate_shape'], M[(name, 'H0')]['ate_shape']
        au, ah = M[(name, 'U0')]['ate_u'], M[(name, 'H0')]['ate_u']
        ax.set_title(f'({chr(97 + j)}) {name}', fontsize=9, pad=4)
        ax.text(0.98, 0.03, f'shape error: {su:.2f} → {sh:.2f} m ({100 * (sh / su - 1):+.0f}%)'.replace('-', '\u2212') + f'\n'
                             f'ATE: {au:.2f} → {ah:.2f} m ({100 * (ah / au - 1):+.0f}%)'.replace('-', '\u2212'),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.6,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#dddddd'))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#eeeeee', lw=0.6)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        for i in range(0, n, 20):
            rows.append([name, f'{te[i] - te[0]:.3f}', *gg[i].round(4), *u[i].round(4), *h[i].round(4)])
    axes[0, 1].legend(loc='upper left', fontsize=7.5, frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight', pad_inches=0.03)
    with open(OUT / f'{STEM}_source_data.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['network', 't_s', 'gt_x', 'gt_y', 'unmod_x', 'unmod_y', 'hn_x', 'hn_y']); w.writerows(rows)
    info = dict(sequence=seq, mean_rel_shape_reduction=red, min_path_m=MIN_PATH,
                rule='TLIO-confirm, GT path >= 100 m, all 4 networks reduce shape error (psi=0, alpha=0, seed 0); max mean relative reduction',
                metrics={f'{name}|{k}': {q: M[(name, k)][q] for q in ('ate_u', 'ate_shape')} for name, _ in NETS for k in ('U0', 'H0')},
                check='ate_u, ate_shape match results/confirm_anglefair_20260927 within 1e-4 m',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    json.dump(info, open(CACHE / 'provenance.json', 'w'), indent=1, ensure_ascii=False)
    print(seq, f'mean reduction {100 * red:.1f}%')
    for name, _ in NETS:
        print(f"  {name:25s} shape {M[(name, 'U0')]['ate_shape']:.3f} -> {M[(name, 'H0')]['ate_shape']:.3f}   ATE {M[(name, 'U0')]['ate_u']:.3f} -> {M[(name, 'H0')]['ate_u']:.3f}")


if __name__ == '__main__':
    main()
