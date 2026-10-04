"""统一评测（判定标准见 docs/36，先于本脚本运行写成）。

口径（所有数据集相同）：①平移使预测起点与真值起点重合；②以起点为中心、只用真值路程前 10 m 拟合一个旋转角并固定（docs/36 §6）；
③在对齐后的轨迹上算 ATE / RTE / TLR / MCS、以起点为中心的复增益分解（ε_s = |a*|−1，ε_θ = arg a*，形状 ATE），
以及只修尺度 / 只修航向 / 两者都修的真值上限。另报文献协议（RoNIN 只平移；RIDI、TLIO、IMUNet 前 10 s 刚体对齐）。
只用加速度计与陀螺仪（世界系来自既有缓存或游戏旋转向量）；真值只用于评测。

分组运行（避免 EqNIO 与官方 RoNIN 源码的同名 model_resnet1d 冲突，也控制同时占用 MPS 的进程数）：
  --group window : 官方 ResNet（原样、+HN）与本文 6 类冻结模型 × 4 种子（MPS）
  --group seq    : 官方 LSTM / TCN（原样、HN-seq，CPU），然后 EqNIO SO(2)/O(2)（MPS）
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
OUT = ROOT / 'results' / 'unified_eval_20260913'
DATASETS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet')
EXPECTED = {'ronin_seen': 32, 'ronin_unseen': 32, 'ridi': 94, 'tlio': 36, 'imunet': 36}
OURS = {'ours_gn': 'Sensors_v4_resnet_gn_s{}', 'ours_yaw': 'Sensors_v4_resnet_yaw_s{}',
        'ours_pca': 'Sensors_v4_resnet_mixed_pca_s{}', 'ours_hn': 'ResNet18_v3_hn_gn_s{}',
        'ours_transformer_hn': 'A6_v3_nopre_s{}', 'ours_imunet_hn': 'IMUNet2024_v3_hn_gn_s{}'}
PREFIX_S, PREFIX_M, DS_SAVE = 10.0, 10.0, 20      # 文献协议的时间前缀 / 统一口径的路程前缀 / 轨迹保存降采样
H = P = None


def files(ds):
    if ds == 'imunet':
        return sorted((ROOT / 'data/eval/imunet_owndata').glob('imunet__*.npz'))
    return H.dataset_files(ds)


def cplx(v):
    return v[:, 0] + 1j * v[:, 1]


def unify(xy, gt, te):
    """①起点重合；②用真值路程走满前 PREFIX_M 米的那一段拟合一个旋转角（以起点为中心，不缩放）。
    docs/36 §6 运行前修订：原为前 10 s，RoNIN 开头多静止，改为按路程截取。返回对齐轨迹与该角（度）。"""
    w = cplx(xy - xy[0]); q = cplx(gt - gt[0])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    m = cum <= PREFIX_M
    if m.sum() < 2 or cum[-1] < PREFIX_M:
        m = np.ones(len(gt), bool)
    c = np.vdot(w[m], q[m])
    th = float(np.angle(c)) if abs(c) > 1e-12 else 0.0
    wz = w * np.exp(1j * th)
    return np.stack([wz.real, wz.imag], 1) + gt[0], float(np.degrees(th))


def metrics(xy_raw, z, ds):
    gt, te = np.asarray(z['gt'], float), np.asarray(z['te'], float)
    n = min(len(xy_raw), len(gt)); xy_raw, gt, te = xy_raw[:n], gt[:n], te[:n]
    xu, th0 = unify(xy_raw, gt, te)
    ate = lambda x: float(P.ate_rte(x, gt, 200.0)[0])
    a_u, r_u = P.ate_rte(xu, gt, 200.0)
    tlr, mcs = P.shape_metrics(xu, gt)
    w = cplx(xu - gt[0]); q = cplx(gt - gt[0])
    S = float(np.vdot(w, w).real); c = np.vdot(w, q); a = c / S
    back = lambda zz: np.stack([zz.real, zz.imag], 1) + gt[0]
    if ds.startswith('ronin'):
        lit = P.align_registered('ronin', xy_raw, gt, te)[0]
    elif ds == 'ridi':
        lit = P.align_registered('ridi', xy_raw, gt, te)[0]
    else:
        lit = P.align_prefix_rigid_2d(xy_raw, gt, te, PREFIX_S)[0]
    a_l, r_l = P.ate_rte(lit, gt, 200.0)
    return dict(ate_u=float(a_u), rte_u=float(r_u), tlr=float(tlr), mcs=float(mcs), prefix_heading_deg=th0,
                eps_s=float(abs(a) - 1), eps_theta_deg=float(np.degrees(np.angle(a))),
                ate_shape=ate(back(a * w)), ate_scale_only=ate(back((c.real / S) * w)),
                ate_heading_only=ate(back(np.exp(1j * np.angle(c)) * w)),
                ate_lit=float(a_l), rte_lit=float(r_l)), xu


def run_model(name, seed, predict, dev, out_dir, limit, prov_extra):
    dest = out_dir / f'{name}{"" if seed is None else f"_s{seed}"}.json'
    if dest.exists() and json.loads(dest.read_text()).get('complete'):
        print(f'[已完成，跳过] {dest.name}', flush=True); return
    rows, trajs = [], {}
    for ds in DATASETS:
        fl = files(ds)
        if limit is None and len(fl) != EXPECTED[ds]:
            raise RuntimeError(f'{ds} 条数 {len(fl)} ≠ {EXPECTED[ds]}')
        fl = fl[:limit] if limit else fl
        t0 = time.time()
        for path in fl:
            with np.load(path, allow_pickle=False) as f:
                z = {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt', 'seq')}
            xy = predict(np.asarray(z['feat'], np.float32), z)
            m, xu = metrics(np.asarray(xy, float), z, ds)
            m.update(model=name, seed=seed, dataset=ds, sequence=str(z['seq']))
            rows.append(m); trajs[f'{ds}|{z["seq"]}'] = xu[::DS_SAVE].astype(np.float32)
        print(f'[{name} s{seed} {ds}] {len(fl)} 条  ATE_u 均值 {np.mean([r["ate_u"] for r in rows if r["dataset"]==ds]):.3f}  '
              f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(dest.with_suffix('.npz'), **{k.replace('|', '__'): v for k, v in trajs.items()})
    json.dump({'provenance': dict(model=name, seed=seed, script_sha256=H.sha(__file__), device=str(dev), limit=limit,
                                  **prov_extra), 'complete': True, 'rows': rows}, open(dest, 'w'), ensure_ascii=False)
    print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


def main():
    global H, P
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=('window', 'seq', 'yawhn', 'yawtta', 'yawfa2'), required=True)
    # yawhn：docs/45，增强训练 + 外挂 HN；yawtta：docs/47，增强训练 + C8 测试时平均；yawfa2：docs/50，两方向帧平均
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out_dir = OUT if a.limit is None else ROOT / 'verification/generated/unified_eval_smoke'
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.group == 'seq':
        import eqnio_models_20260912 as EM           # 须先于 hn_plugin_p0 导入
    import hn_plugin_p0_20260912 as H_; H = H_; P = H.P
    mps = P.pick_device(verbose=False)
    cpu = torch.device('cpu')

    if a.group == 'yawhn':
        from sensors_evaluate import load
        tag = OURS['ours_yaw']
        for s in range(4):
            net_o, _, _, path = load(tag.format(s), mps)
            run_model('ours_yaw_hn', s, lambda f, z, n=net_o: _win(n, f, z, H.headings(f)[1], mps), mps, out_dir, a.limit,
                      dict(tag=tag.format(s), weight_sha256=H.sha(path), bolt_on='HN theta (models.HeadingNorm)'))
    elif a.group == 'yawtta':
        from sensors_evaluate import load
        tag = OURS['ours_yaw']
        for s in range(4):
            net_o, _, _, path = load(tag.format(s), mps)
            run_model('ours_yaw_tta8', s, lambda f, z, n=net_o: _tta(n, f, z, mps), mps, out_dir, a.limit,
                      dict(tag=tag.format(s), weight_sha256=H.sha(path), tta='C8: mean_k R(+psi_k) f(R(-psi_k) x), psi_k=2*pi*k/8'))
    elif a.group == 'yawfa2':
        from sensors_evaluate import load
        tag = OURS['ours_yaw']
        for s in range(4):
            net_o, _, _, path = load(tag.format(s), mps)
            run_model('ours_yaw_fa2', s, lambda f, z, n=net_o: _fa2(n, f, z, mps), mps, out_dir, a.limit,
                      dict(tag=tag.format(s), weight_sha256=H.sha(path),
                           fa='2-direction: mean over phi in {theta_HN, theta_HN + pi}; sign-free'))
    elif a.group == 'window':
        net, _, wpath = H.load_model('ext_resnet', mps)
        prov = dict(weight_sha256=H.sha(wpath))
        run_model('ext_resnet', None, lambda f, z: _win(net, f, z, None, mps), mps, out_dir, a.limit, prov)
        run_model('ext_resnet_hn', None, lambda f, z: _win(net, f, z, H.headings(f)[1], mps), mps, out_dir, a.limit, prov)
        from sensors_evaluate import load
        for name, tag in OURS.items():
            for s in range(4):
                net_o, _, _, path = load(tag.format(s), mps)
                run_model(name, s, lambda f, z, n=net_o: _win(n, f, z, None, mps), mps, out_dir, a.limit,
                          dict(tag=tag.format(s), weight_sha256=H.sha(path)))
    else:
        import hn_seq_p1_20260912 as HS
        for w in ('ext_lstm', 'ext_tcn'):
            net, _, wpath = H.load_model(w, cpu); lstm = w == 'ext_lstm'
            prov = dict(weight_sha256=H.sha(wpath))
            run_model(w, None, lambda f, z, n=net, l=lstm: H.traj_seq(H.run_seq(n, [f], None, cpu, l)[0], z),
                      cpu, out_dir, a.limit, prov)
            run_model(w + '_hnseq', None, lambda f, z, n=net, l=lstm: H.traj_seq(
                H.run_seq(n, [f], [np.full(len(f), HS.theta_seq(f), np.float32)], cpu, l)[0], z), cpu, out_dir, a.limit, prov)
        for w in ('eqnio_so2', 'eqnio_o2'):
            net, wpath = EM.load_eqnio(w, mps)
            run_model(w, None, lambda f, z, n=net: _win(n, f, z, None, mps), mps, out_dir, a.limit,
                      dict(weight_sha256=H.sha(wpath)))


def _win(net, f, z, phi, dev):
    ids, v = H.run_window(net, f, phi, dev)
    return H.traj_window(v, ids, z)


def _tta(net, f, z, dev, k=8):
    """C8 测试时平均（docs/47）：每个分支旋入 −ψ_k、旋出 +ψ_k（与外挂 HN 同一旋转实现），速度取平均后积分。"""
    n = len(range(0, len(f) - H.WINDOW, H.STRIDE))
    outs = [H.run_window(net, f, np.full(n, 2 * np.pi * j / k, np.float32), dev) for j in range(k)]
    return H.traj_window(np.mean([v for _, v in outs], axis=0), outs[0][0], z)


def _fa2(net, f, z, dev):
    """两方向帧平均（docs/50）：主轴两个方向 θ 与 θ+π 各前向一次、旋回后取平均。{θ, θ+π} 与定符号结果无关。"""
    th = H.headings(f)[1].astype(np.float64)
    outs = [H.run_window(net, f, (th + d).astype(np.float32), dev) for d in (0.0, np.pi)]
    return H.traj_window(np.mean([v for _, v in outs], axis=0), outs[0][0], z)


if __name__ == '__main__':
    main()
