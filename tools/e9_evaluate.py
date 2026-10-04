"""E9 评测（docs/83 §4）：统一口径 5 个测试集 + 跨角重复性 + 确认集（描述性）。

统一口径直接导入 tools/unified_eval_20260913.py 的 run_model / _win（不修改该脚本），结果格式与 results/unified_eval_20260913 相同；
跨角重复性沿用 tools/e6_evaluate.py 的协议（导入 e1_heading_repeatability 的旋转函数），只在 RoNIN 独立受试者组、只用 seed 0
（EqNIO 参考系按构造等变，此项只是核对，docs/83 附记）。确认集由 tools/confirm_eval_20260913.py 的数据读取函数提供（描述性）。
用法（HN_DEVICE=mps）：
  .venv/bin/python tools/e9_evaluate.py --family eqnio_so2 --what unified|sweep|confirm [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import e9_eqnio_frontend as F          # 须先于 hn_plugin_p0 导入（EqNIO 自带同名 model_resnet1d）
import hn_plugin_p0_20260912 as H
import unified_eval_20260913 as U
import pkg_common as P
import pkg_eval as E
from e1_heading_repeatability import rotate_horizontal, rotate_2d, sequences

U.H, U.P = H, H.P
MODELS = ROOT / 'models/e9_eqnio_retrain'
SMOKE = ROOT / 'verification/generated/e9_smoke'
OUT = ROOT / 'results/eqnio_retrain_20260919'
ANGLES = [2 * np.pi * k / 8 for k in range(8)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path_pt, device):
    meta = json.loads(path_pt.with_suffix('.json').read_text())
    if digest(path_pt) != meta['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint digest mismatch: {path_pt}')
    o = torch.load(path_pt, map_location='cpu', weights_only=False)
    spec = o['spec']
    net = F.build(spec['frontend'], o['gnorm_mu'], o['gnorm_sd'], spec['arch'], spec['pre'], spec['width'])
    net.load_state_dict(o['model_state_dict'], strict=True)
    return net.eval().to(device), spec


def checkpoints(family, smoke):
    if smoke:
        return sorted(SMOKE.glob(f'E9_{family}_*_s0.pt'))
    chosen = json.loads((OUT / 'lr_selection.json').read_text())[family]['chosen_lr']
    tmpl = json.loads((ROOT / 'config/e9_eqnio_retrain.json').read_text())['tag_template']
    return [MODELS / (tmpl.format(family=family, lr=chosen, seed=s) + '.pt') for s in range(4)]


def unified(family, smoke, device):
    out_dir = SMOKE / 'unified' if smoke else OUT / 'unified'
    out_dir.mkdir(parents=True, exist_ok=True)
    for pt in checkpoints(family, smoke):
        net, spec = load(pt, device)
        U.run_model(f'e9_{family}', spec['seed'], lambda f, z, n=net: U._win(n, f, z, None, device), device, out_dir,
                    2 if smoke else None, dict(tag=spec['tag'], weight_sha256=digest(pt), lr=spec['lr']))


def sweep(family, smoke, device):
    out_dir = SMOKE / 'sweep' if smoke else OUT / 'sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = checkpoints(family, smoke)[0]
    net, spec = load(pt, device)
    files = [f for f in sequences() if f[1] == 'ronin' and f[2] == 'unseen']
    files = files[:2] if smoke else files
    rows, t0 = [], time.monotonic()
    for path, dataset, split in files:
        with np.load(path, allow_pickle=False) as z:
            feat = np.asarray(z['feat'], np.float32)
            tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
            seq = str(z['seq'])
        for index, psi in enumerate(ANGLES):
            rotated = feat if index == 0 else rotate_horizontal(feat, psi)
            ids, velocity = E.predict(net, rotated, device, 1., 200., chunk=512)
            restored = velocity if index == 0 else rotate_2d(velocity, -psi)
            traj = P.trajectory_from_velocity(restored, ids, tm, te)
            aligned, _, _ = P.align_registered(dataset, traj, gt, te)
            ate, rte = P.ate_rte(aligned, gt, 200.)
            rows.append(dict(model=spec['tag'], family=family, seed=spec['seed'], dataset=dataset, split=split,
                             seq=seq, subject=seq.split('_')[0], angle_index=index, ate=float(ate), rte=float(rte)))
    dest = out_dir / f'{spec["tag"]}.json'
    dest.write_text(json.dumps(dict(model=spec['tag'], checkpoint_sha256=digest(pt), script_sha256=digest(__file__),
                                    complete=True, rows=rows, elapsed_seconds=time.monotonic() - t0), indent=1))
    rng = [max(r['ate'] for r in rows if r['seq'] == s) - min(r['ate'] for r in rows if r['seq'] == s)
           for s in sorted({r['seq'] for r in rows})]
    print(f'[sweep] {spec["tag"]}: {len(rows)} 行，逐序列跨角极差最大 {max(rng):.2e} m', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=('eqnio_so2', 'eqnio_o2'))
    ap.add_argument('--what', required=True, choices=('unified', 'sweep'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(4)
    device = P.pick_device()
    {'unified': unified, 'sweep': sweep}[a.what](a.family, a.smoke, device)


if __name__ == '__main__':
    main()
