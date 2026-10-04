"""P0 插件实验：把 HN / HN-S 外挂到外部公开权重与本项目两个冻结网络上。

预注册：docs/27_P0插件实验预注册_20260912.md（先于本脚本的任何结果写成）。
不改动：RoNIN 官方源码与权重、src/models.py、tools/e1_heading_repeatability.py。

窗口模型（官方 ResNet、本项目 IMUNet/Transformer）：逐窗旋转，与
models.NormalizedBackbone(f, hnorm=True, gnorm=False) 数值相同。
序列模型（官方 LSTM/TCN）：逐窗算 θ，每帧取中心最近窗的 θ，逐帧旋入旋出。
HN-S：弱三阶矩窗（|标准化三阶矩| < TAU）在 {θ, θ+π} 中选与上一窗更近者。
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src')); import hn_paths   # 外部资料路径表
RONIN_SRC = hn_paths.RONIN_SRC
WEIGHTS = hn_paths.RONIN_WEIGHTS
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
sys.path.append(str(RONIN_SRC))          # 放最后，避免遮蔽本项目模块
import pkg_common as P
import models
from e1_heading_repeatability import rotate_horizontal, rotate_2d
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
from model_temporal import BilinearLSTMSeqNetwork, TCNSeqNetwork

OUT = ROOT / 'results' / 'hn_plugin_p0_20260912'
WINDOW, STRIDE, TAU, N_ANG = P.WINDOW, 10, 0.1, 8
ANGLES = [2 * np.pi * k / N_ANG for k in range(N_ANG)]
DATASETS = ('ronin_unseen', 'ronin_seen', 'ridi', 'tlio')
EXTERNAL = ('ext_resnet', 'ext_lstm', 'ext_tcn')
INTERNAL = tuple(f'int_{a}_yaw_s{s}' for a in ('imunet', 'transformer') for s in range(4))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- 模型
def load_model(name, dev):
    """返回 (net, kind, 权重路径)。kind ∈ {'window', 'seq'}。"""
    if name == 'ext_resnet':
        path = WEIGHTS / 'ronin_resnet/checkpoint_gsn_latest.pt'
        net = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64, output_block=FCOutputModule,
                       kernel_size=3, fc_dim=512, in_dim=7, dropout=.5, trans_planes=128)
        kind = 'window'
    elif name == 'ext_lstm':
        path = WEIGHTS / 'ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt'
        net = BilinearLSTMSeqNetwork(6, 2, 1, dev, lstm_size=100, lstm_layers=3, dropout=0.2)
        kind = 'seq'
    elif name == 'ext_tcn':
        path = WEIGHTS / 'ronin_tcn/checkpoints/ronin_tcn_checkpoint.pt'
        net = TCNSeqNetwork(6, 2, 3, [32, 64, 128, 256, 72, 36], dropout=0.2)
        kind = 'seq'
    elif name.startswith('int_'):
        arch, seed = name[4:].rsplit('_yaw_s', 1)
        path = ROOT / f'models/e3_backbone_frontend/E3_{arch}_yaw_s{seed}.pt'
        meta = json.loads(path.with_suffix('.json').read_text())
        if 'checkpoint_sha256' in meta and meta['checkpoint_sha256'] != sha(path):
            raise RuntimeError(f'权重摘要不符：{path}')
        o = torch.load(path, map_location='cpu', weights_only=False)
        if o.get('hnorm'):
            raise RuntimeError(f'{name} 训练时已含 HN，不能作外挂对象')
        if float(o.get('win_sec', 1.0)) != 1.0 or float(o.get('rate', 200.0)) != 200.0:
            raise RuntimeError(f'{name} 窗长/采样率不是 1 s / 200 Hz')
        net = models.build(o['arch'], pre=o.get('pre', True), width=o.get('width', 1.0),
                           hnorm=False, nout=o.get('nout', 2), gnorm=bool(o.get('gnorm', 0)),
                           gnorm_mu=o.get('gnorm_mu'), gnorm_sd=o.get('gnorm_sd'),
                           rate=float(o.get('rate', 200.0)))
        net.load_state_dict(o['model_state_dict'], strict=True)
        return net.eval().to(dev), 'window', path
    else:
        raise ValueError(name)
    obj = torch.load(path, map_location='cpu', weights_only=False)
    net.load_state_dict(obj['model_state_dict'], strict=True)
    return net.eval().to(dev), kind, path


# ---------------------------------------------------------------- 参考方向
def headings(feat):
    """逐窗参考方向。返回 (窗起点, θ_HN, θ_HN-S, |标准化三阶矩|)。θ_HN 由 models.HeadingNorm 原样算出。"""
    ids = np.arange(0, len(feat) - WINDOW, STRIDE, dtype=int)
    hn = models.HeadingNorm()
    th = np.empty(len(ids), np.float32); sk = np.empty(len(ids))
    with torch.no_grad():
        for i0 in range(0, len(ids), 4096):
            idx = ids[i0:i0 + 4096]
            x = torch.from_numpy(np.stack([feat[j:j + WINDOW].T for j in idx]).astype(np.float32))
            _, t = hn(x)
            ac = x[:, 3:5, :] - x[:, 3:5, :].mean(2, keepdim=True)
            proj = ac[:, 0, :] * torch.cos(t)[:, None] + ac[:, 1, :] * torch.sin(t)[:, None]
            s = (proj ** 3).mean(1) / (proj ** 2).mean(1).clamp_min(1e-12) ** 1.5
            th[i0:i0 + len(idx)] = t.numpy(); sk[i0:i0 + len(idx)] = np.abs(s.numpy())
    ths = th.astype(np.float64).copy()
    for k in range(1, len(ths)):
        if sk[k] < TAU and abs(wrap(ths[k] - ths[k - 1])) > np.pi / 2:
            ths[k] = ths[k] + np.pi
    return ids, th, ths.astype(np.float32), sk


def flip_pct(th):
    return float(100 * (np.degrees(np.abs(wrap(np.diff(th.astype(np.float64))))) > 150).mean())


# ---------------------------------------------------------------- 推理
def _rot_win(x, p):
    """与 models.HeadingNorm 的旋转写法相同：水平分量旋 −p。x: [B,6,T]。"""
    c, s = torch.cos(-p)[:, None], torch.sin(-p)[:, None]
    out = x.clone()
    for off in (0, 3):
        xx, yy = x[:, off, :], x[:, off + 1, :]
        out[:, off, :] = c * xx - s * yy
        out[:, off + 1, :] = s * xx + c * yy
    return out


def run_window(net, feat, phi, dev, chunk=512):
    """复刻 pkg_eval.predict；phi 非 None 时逐窗旋入 −phi、旋出 +phi。"""
    ids = np.arange(0, len(feat) - WINDOW, STRIDE, dtype=int)
    out = np.empty((len(ids), 2), np.float32)
    with torch.no_grad():
        for i0 in range(0, len(ids), chunk):
            idx = ids[i0:i0 + chunk]
            x = torch.from_numpy(np.stack([feat[j:j + WINDOW].T for j in idx]).astype(np.float32)).to(dev)
            if phi is None:
                y = net(x)
            else:
                p = torch.from_numpy(np.asarray(phi[i0:i0 + len(idx)], np.float32)).to(dev)
                y = models._unrotate(net(_rot_win(x, p)), p)
            out[i0:i0 + len(idx)] = y.cpu().numpy()[:, :2]
    return ids, np.asarray(out, float)


def frame_phi(win_phi, n):
    """每帧取中心最近窗的角度（窗 k 的中心 = k·STRIDE + WINDOW/2）。"""
    k = np.clip(np.rint((np.arange(n) - WINDOW / 2) / STRIDE).astype(int), 0, len(win_phi) - 1)
    return np.asarray(win_phi, np.float32)[k]


def run_seq(net, feats, phis, dev, is_lstm):
    """feats: B 条 [N,6]；phis: None 或 B 条逐帧角度。返回 [B,N,2] 世界系速度。"""
    x = torch.from_numpy(np.stack(feats).astype(np.float32)).to(dev)
    if phis is not None:
        p = torch.from_numpy(np.stack(phis).astype(np.float32)).to(dev)
        c, s = torch.cos(-p), torch.sin(-p)
        x = x.clone()
        for off in (0, 3):
            xx, yy = x[..., off].clone(), x[..., off + 1].clone()
            x[..., off] = c * xx - s * yy
            x[..., off + 1] = s * xx + c * yy
    if is_lstm:
        net.batch_size = x.shape[0]        # 官方实现按 batch_size 建零初始隐状态
    with torch.no_grad():
        y = net(x)[..., :2]
    if phis is not None:
        c, s = torch.cos(p), torch.sin(p)
        y = torch.stack([c * y[..., 0] - s * y[..., 1], s * y[..., 0] + c * y[..., 1]], -1)
    return y.cpu().numpy().astype(float)


# ---------------------------------------------------------------- 打分
def umeyama_ate(e, g):
    mu_e, mu_g = e.mean(0), g.mean(0); ec, gc = e - mu_e, g - mu_g
    U, S, Vt = np.linalg.svd(gc.T @ ec / len(e))
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / ec.var(0).sum()
    return P.ate_rte((s * (R @ ec.T)).T + mu_g, g, 200.0)[0], float(s)


def score(xy, z, dataset, extras=False):
    gt, te = z['gt'], z['te']
    if dataset.startswith('ronin'):
        al = P.align_registered('ronin', xy, gt, te)[0]
    elif dataset == 'ridi':
        al = P.align_registered('ridi', xy, gt, te)[0]
    else:
        al = P.align_prefix_rigid_2d(xy, gt, te, 10.0)[0]
    ate, rte = P.ate_rte(al, gt, 200.0)
    r = {'ate': float(ate), 'rte': float(rte)}
    if extras:
        r['tlr'], r['mcs'] = map(float, P.shape_metrics(al, gt))
        r['ate_startyaw'] = float(P.ate_rte(P.align_start_yaw(xy, gt)[0], gt, 200.0)[0])
        r['ate_umeyama_sim'], r['umeyama_scale'] = map(float, umeyama_ate(np.asarray(xy, float), gt))
    return r


def traj_window(vel, ids, z):
    return P.trajectory_from_velocity(vel, ids, z['tm'], z['te'])


def traj_seq(vel, z):
    return P.trajectory_from_velocity(vel, np.arange(len(vel)), z['tm'], z['te'])


# ---------------------------------------------------------------- 一条序列
def evaluate_sequence(name, net, kind, z, dataset, dev, sweeps):
    feat = np.asarray(z['feat'], np.float32)
    ids_w, th, ths, sk = headings(feat)
    row = {'weight': name, 'dataset': dataset, 'sequence': str(z['seq']),
           'n_frames': int(len(feat)), 'n_windows': int(len(ids_w)),
           'flip_pct_hn': flip_pct(th), 'flip_pct_hns': flip_pct(ths),
           'weak_skew_pct': float(100 * (sk < TAU).mean())}
    psis = ANGLES if sweeps else ANGLES[:1]
    alphas = ANGLES if sweeps else ANGLES[:1]

    if kind == 'window':
        orig = []
        for k, psi in enumerate(psis):
            f = rotate_horizontal(feat, psi) if psi else feat
            ids, v = run_window(net, f, None, dev)
            xy = traj_window(v, ids, z)
            orig.append(score(rotate_2d(xy, -psi) if psi else xy, z, dataset, extras=(k == 0)))
        hn = []
        for k, a in enumerate(alphas):
            ids, v = run_window(net, feat, th - np.float32(a), dev)
            hn.append(score(traj_window(v, ids, z), z, dataset, extras=(k == 0)))
        ids, v = run_window(net, feat, ths, dev)
        hns = score(traj_window(v, ids, z), z, dataset, extras=True)
        if sweeps:                                   # ψ=90° 不变性自检
            psi = ANGLES[2]; f = rotate_horizontal(feat, psi)
            _, th90, ths90, _ = headings(f)
            for key, t in (('hn', th90), ('hns', ths90)):
                ids, v = run_window(net, f, t, dev)
                row[f'{key}_ate_psi90'] = score(rotate_2d(traj_window(v, ids, z), -psi), z, dataset)['ate']
    else:
        is_lstm = name == 'ext_lstm'
        n = len(feat)
        fs = [rotate_horizontal(feat, psi) if psi else feat for psi in psis]
        V = run_seq(net, fs, None, dev, is_lstm)
        orig = [score(rotate_2d(traj_seq(V[k], z), -psi) if psi else traj_seq(V[k], z), z, dataset,
                      extras=(k == 0)) for k, psi in enumerate(psis)]
        V = run_seq(net, [feat] * len(alphas), [frame_phi(th - np.float32(a), n) for a in alphas],
                    dev, is_lstm)
        hn = [score(traj_seq(V[k], z), z, dataset, extras=(k == 0)) for k in range(len(alphas))]
        feats, phis = [feat], [frame_phi(ths, n)]
        if sweeps:
            psi = ANGLES[2]; f = rotate_horizontal(feat, psi)
            _, th90, ths90, _ = headings(f)
            feats += [f, f]; phis += [frame_phi(th90, n), frame_phi(ths90, n)]
        V = run_seq(net, feats, phis, dev, is_lstm)
        hns = score(traj_seq(V[0], z), z, dataset, extras=True)
        if sweeps:
            row['hn_ate_psi90'] = score(rotate_2d(traj_seq(V[1], z), -psi), z, dataset)['ate']
            row['hns_ate_psi90'] = score(rotate_2d(traj_seq(V[2], z), -psi), z, dataset)['ate']

    row['orig_ate_by_psi'] = [r['ate'] for r in orig]
    row['orig_rte_by_psi'] = [r['rte'] for r in orig]
    row['orig'] = orig[0]
    row['hn_ate_by_alpha'] = [r['ate'] for r in hn]
    row['hn_rte_by_alpha'] = [r['rte'] for r in hn]
    row['hn'] = hn[0]
    row['hns'] = hns
    return row


def dataset_files(name):
    if name == 'tlio':
        return sorted(P.CACHE_TLIO.glob('tlio__*.npz'))
    out = []
    for p in sorted(P.CACHE_EVAL.glob('*.npz')):
        with np.load(p, allow_pickle=False) as z:
            d, s = str(z['dataset']), str(z['split'])
        if (name == 'ridi' and d == 'ridi') or (name == f'ronin_{s}' and d == 'ronin'):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=','.join(EXTERNAL + INTERNAL))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--limit', type=int, help='每个测试集最多几条，仅冒烟测试')
    a = ap.parse_args()
    torch.set_num_threads(4)
    out_dir = OUT if a.limit is None else ROOT / 'verification/generated/hn_plugin_p0_smoke'
    out_dir.mkdir(parents=True, exist_ok=True)
    script_sha = sha(__file__)
    expected = {'ronin_unseen': 32, 'ronin_seen': 32, 'ridi': 94, 'tlio': 36}
    for name in a.weights.split(','):
        kind_dev = torch.device('cpu') if name in ('ext_lstm', 'ext_tcn') else P.pick_device(verbose=False)
        net, kind, wpath = load_model(name, kind_dev)
        sweeps = name in EXTERNAL
        dest = out_dir / f'{name}.json'
        prov = {'weight': name, 'weight_path': str(wpath), 'weight_sha256': sha(wpath),
                'script_sha256': script_sha, 'kind': kind, 'sweeps': sweeps, 'tau': TAU,
                'device': str(kind_dev), 'limit': a.limit}
        rows = []
        if dest.exists():
            prev = json.loads(dest.read_text())
            if all(prev['provenance'].get(k) == v for k, v in prov.items()):
                rows = prev['rows']
        done = {(r['dataset'], r['sequence']) for r in rows}
        for ds in a.datasets.split(','):
            files = dataset_files(ds)
            if a.limit is None and len(files) != expected[ds]:
                raise RuntimeError(f'{ds} 条数 {len(files)} ≠ {expected[ds]}')
            files = files[:a.limit] if a.limit else files
            t0 = time.time()
            for i, path in enumerate(files):
                with np.load(path, allow_pickle=False) as f:
                    z = {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt', 'seq')}
                if (ds, str(z['seq'])) in done:
                    continue
                r = evaluate_sequence(name, net, kind, z, ds, kind_dev, sweeps)
                rows.append(r)
                json.dump({'provenance': prov, 'complete': False, 'rows': rows},
                          open(dest, 'w'), ensure_ascii=False)
                print(f"[{name} {ds} {i+1}/{len(files)}] {r['sequence']}  原样 {r['orig']['ate']:.3f}  "
                      f"HN {r['hn']['ate']:.3f}  HN-S {r['hns']['ate']:.3f}  "
                      f"翻转 {r['flip_pct_hn']:.1f}%→{r['flip_pct_hns']:.1f}%  {time.time()-t0:.0f}s", flush=True)
        json.dump({'provenance': prov, 'complete': True, 'rows': rows},
                  open(dest, 'w'), ensure_ascii=False)
        print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
