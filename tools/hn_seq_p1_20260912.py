"""P1-A：HN-seq（整段恒定参考角）外挂到官方 RoNIN ResNet / LSTM / TCN。预注册 docs/29 §A。

θ_seq = models.HeadingNorm 在序列前 10 s（2000 点）水平加速度上的参考角；整条序列只旋这一个角。
原样与逐帧 HN 结果直接取自 P0 结果文件（同一代码、同一缓存），按 (测试集, 序列) 配对。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'tools')]
import hn_plugin_p0_20260912 as H

P, models = H.P, H.models
OUT = ROOT / 'results' / 'hn_seq_p1_20260912'
P0 = ROOT / 'results' / 'hn_plugin_p0_20260912'
SEQ_SAMPLES = 2000                      # 前 10 s @200 Hz


def theta_seq(feat):
    x = torch.from_numpy(np.ascontiguousarray(feat[:SEQ_SAMPLES].T[None]).astype(np.float32))
    with torch.no_grad():
        _, t = models.HeadingNorm()(x)
    return np.float32(t.item())


def velocities(name, net, kind, feats, thetas, dev):
    """feats/thetas：同一条序列的若干旋转版本及其 θ_seq。返回 [(ids 或 None, vel)]。"""
    if kind == 'window':
        out = []
        for f, t in zip(feats, thetas):
            k = len(np.arange(0, len(f) - H.WINDOW, H.STRIDE))
            out.append(H.run_window(net, f, np.full(k, t, np.float32), dev))
        return out
    n = len(feats[0])
    V = H.run_seq(net, feats, [np.full(n, t, np.float32) for t in thetas], dev, name == 'ext_lstm')
    return [(None, V[i]) for i in range(len(feats))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=','.join(H.EXTERNAL))
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out_dir = OUT if a.limit is None else ROOT / 'verification/generated/hn_seq_p1_smoke'
    out_dir.mkdir(parents=True, exist_ok=True)
    script_sha = H.sha(__file__)
    for name in a.weights.split(','):
        p0 = json.loads((P0 / f'{name}.json').read_text())
        if not p0['complete']:
            raise RuntimeError(f'P0 {name} 未完成')
        p0rows = {(r['dataset'], r['sequence']): r for r in p0['rows']}
        dev = torch.device('cpu') if name in ('ext_lstm', 'ext_tcn') else P.pick_device(verbose=False)
        net, kind, wpath = H.load_model(name, dev)
        rows = []
        for ds in H.DATASETS:
            files = H.dataset_files(ds)
            files = files[:a.limit] if a.limit else files
            t0 = time.time()
            for i, path in enumerate(files):
                with np.load(path, allow_pickle=False) as f:
                    z = {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt', 'seq')}
                feat = np.asarray(z['feat'], np.float32)
                if len(feat) < SEQ_SAMPLES + H.WINDOW:
                    raise RuntimeError(f'{ds}/{z["seq"]} 短于 10 s')
                psi = H.ANGLES[2]
                f90 = H.rotate_horizontal(feat, psi)
                th0, th90 = theta_seq(feat), theta_seq(f90)
                (i0, v0), (i9, v9) = velocities(name, net, kind, [feat, f90], [th0, th90], dev)
                tr = (lambda v, ids: H.traj_window(v, ids, z)) if kind == 'window' else (lambda v, ids: H.traj_seq(v, z))
                hs = H.score(tr(v0, i0), z, ds, extras=True)
                hs90 = H.score(H.rotate_2d(tr(v9, i9), -psi), z, ds)['ate']
                ref = p0rows[(ds, str(z['seq']))]
                rows.append({'weight': name, 'dataset': ds, 'sequence': str(z['seq']),
                             'theta_seq_deg': float(np.degrees(th0)), 'hnseq': hs, 'hnseq_ate_psi90': hs90,
                             'orig': ref['orig'], 'orig_ate_by_psi': ref['orig_ate_by_psi'], 'hn': ref['hn']})
                print(f"[{name} {ds} {i+1}/{len(files)}] {z['seq']}  原样 {ref['orig']['ate']:.3f}  "
                      f"逐帧HN {ref['hn']['ate']:.3f}  HN-seq {hs['ate']:.3f}  ψ90差 {abs(hs90-hs['ate']):.1e}  "
                      f"{time.time()-t0:.0f}s", flush=True)
        dest = out_dir / f'{name}.json'
        json.dump({'provenance': {'weight': name, 'weight_sha256': H.sha(wpath), 'script_sha256': script_sha,
                                  'p0_script_sha256': p0['provenance']['script_sha256'], 'device': str(dev),
                                  'seq_samples': SEQ_SAMPLES, 'limit': a.limit},
                   'complete': True, 'rows': rows}, open(dest, 'w'), ensure_ascii=False)
        print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
