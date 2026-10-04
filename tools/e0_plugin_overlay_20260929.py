"""V27：正文图4缩略图。同一冻结 YawAug 网络在 a058_2 上原样推理与外挂 HN 的 8 方向轨迹。

与 tools/e0_trajectory_overlay.py 同一评测流程（旋转输入 ψ → 预测 → 旋回 −ψ → 官方对齐 → ATE），
只把网络换成同一份权重的两种用法：原样推理（E.predict）与外挂 HN（hn_plugin_p0.run_window，保留训练时的 GN）。
只推理，不训练，不改任何已有结果文件。登记见 provenance/EXPERIMENT_REGISTRY_20260915.csv V27。
用法：.venv/bin/python tools/e0_plugin_overlay_20260929.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import pkg_common as P
import pkg_eval as E
from sensors_evaluate import load
from e1_heading_repeatability import rotate_horizontal, rotate_2d, sequences
import hn_plugin_p0_20260912 as H

OUT = ROOT / 'results/e0_plugin_overlay_20260929'
TAG, SEQ, STRIDE = 'Sensors_v4_resnet_yaw_s0', 'a058_2', 5


def main():
    dev = P.pick_device()
    torch.set_num_threads(4)
    net, win_sec, rate, ckpt = load(TAG, dev)
    path, dataset, split = [f for f in sequences() if f[0].stem.endswith(SEQ) or SEQ in f[0].stem][0]
    with np.load(path, allow_pickle=False) as z:
        assert str(z['seq']) == SEQ
        feat = np.asarray(z['feat'], np.float32)
        tm, te, gt = (np.asarray(z[k], float) for k in ['tm', 'te', 'gt'])
        score_rate = float(z['rate']) if 'rate' in z else 200.
    rows, summ = [], {}
    arms = {'unmodified': lambda f: E.predict(net, f, dev, win_sec, rate, chunk=512),
            'plug-in HN': lambda f: H.run_window(net, f, H.headings(f)[1], dev)}
    for arm, fn in arms.items():
        ates = []
        for k in range(8):
            psi = 2 * np.pi * k / 8
            rot = feat if k == 0 else rotate_horizontal(feat, psi)
            ids, v = fn(rot)
            v = v if k == 0 else rotate_2d(v, -psi)
            traj = P.trajectory_from_velocity(v, ids, tm, te)
            al, _, _ = P.align_registered(dataset, traj, gt, te)
            ate, rte = P.ate_rte(al, gt, score_rate)
            ates.append(ate)
            rows += [dict(arm=arm, model=TAG, seq=SEQ, angle_index=k, angle_deg=45 * k, ate=ate,
                          sample=i, x=float(al[i, 0]), y=float(al[i, 1])) for i in range(0, len(al), STRIDE)]
            print(f'[{arm}] psi={45 * k:3d} ate={ate:.4f}', flush=True)
        summ[arm] = dict(ate_by_psi=ates, ate_min=min(ates), ate_max=max(ates), range_m=max(ates) - min(ates))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'trajectories.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / 'ground_truth.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['sample', 'x', 'y'])
        w.writerows([i, float(gt[i, 0]), float(gt[i, 1])] for i in range(0, len(gt), STRIDE))
    summ['meta'] = dict(model=TAG, checkpoint=str(ckpt), seq=SEQ, dataset=dataset, split=split,
                        plugin='hn_plugin_p0_20260912.run_window with headings(feat)[1]; GN kept')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != 'meta'}, indent=1))


if __name__ == '__main__':
    main()
