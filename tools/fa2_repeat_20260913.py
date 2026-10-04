"""docs/50 Q2：任意初始朝向下的轨迹重复性——外挂 HN 与两方向帧平均（偏航增强训练的 ResNet18，4 个种子）。

RoNIN 独立受试者组 32 条；ψ ∈ {0°, 11.25°, 22.5°, 33.75°}（与 docs/47 同一组角度）。
做法与 E1 相同：输入整体旋转 ψ，预测轨迹旋回 −ψ 后计分（文献协议 ATE 与统一口径 ATE，均复用 unified_eval 的 metrics）。
启动自检：ψ = 0 处 ours_yaw_hn 的逐序列文献协议 ATE 与已有统一评测结果一致（≤ 1e-5 m），否则停止。
用法：HN_DEVICE=mps .venv/bin/python tools/fa2_repeat_20260913.py
输出：results/fa2_repeat_20260913/rows.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import hn_plugin_p0_20260912 as H
import unified_eval_20260913 as UE
from e1_heading_repeatability import rotate_horizontal, rotate_2d
from sensors_evaluate import load

OUT = ROOT / 'results/fa2_repeat_20260913'
PSI_DEG = (0.0, 11.25, 22.5, 33.75)
TAG = 'Sensors_v4_resnet_yaw_s{}'


def main():
    UE.H, UE.P = H, H.P
    torch.set_num_threads(4)
    dev = H.P.pick_device(verbose=False)
    if dev.type != 'mps':
        raise SystemExit(f'设备为 {dev}，请设 HN_DEVICE=mps')
    OUT.mkdir(parents=True, exist_ok=True)
    files = H.dataset_files('ronin_unseen')
    assert len(files) == 32, len(files)
    ref = {}
    for s in range(4):
        for r in json.loads((UE.OUT / f'ours_yaw_hn_s{s}.json').read_text())['rows']:
            if r['dataset'] == 'ronin_unseen':
                ref[(s, r['sequence'])] = r['ate_lit']
    rows = []
    for s in range(4):
        net, _, _, path = load(TAG.format(s), dev)
        t0 = time.time()
        for path_f in files:
            with np.load(path_f, allow_pickle=False) as f:
                z = {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt', 'seq')}
            feat0 = np.asarray(z['feat'], np.float32); seq = str(z['seq'])
            for deg in PSI_DEG:
                psi = np.radians(deg)
                feat = rotate_horizontal(feat0, psi) if deg else feat0
                preds = {'ours_yaw_hn': UE._win(net, feat, z, H.headings(feat)[1], dev),
                         'ours_yaw_fa2': UE._fa2(net, feat, z, dev)}
                for m, xy in preds.items():
                    xy = rotate_2d(xy, -psi) if deg else np.asarray(xy, float)
                    met, _ = UE.metrics(np.asarray(xy, float), z, 'ronin_unseen')
                    rows.append(dict(model=m, seed=s, sequence=seq, psi_deg=deg, ate_lit=met['ate_lit'], ate_u=met['ate_u']))
                    if deg == 0 and m == 'ours_yaw_hn':
                        d = abs(met['ate_lit'] - ref[(s, seq)])
                        if d > 1e-5:
                            raise SystemExit(f'自检不通过：{m} s{s} {seq} ψ=0 差 {d:.2e} m（docs/50 §4）')
        print(f'[s{s}] 32 条 × {len(PSI_DEG)} 角 × 2 模型  {time.time() - t0:.0f}s（ψ=0 自检通过）', flush=True)
    json.dump({'provenance': dict(script_sha256=H.sha(__file__), device=str(dev), psi_deg=PSI_DEG), 'rows': rows},
              open(OUT / 'rows.json', 'w'), ensure_ascii=False)
    print(f'[写出] {(OUT / "rows.json").relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
