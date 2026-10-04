"""V22 推理（判定标准 docs/91，先于本脚本运行写成）：Transformer / IMUNet 随机偏航增强网络，原样与外挂 HN。

权重：models/e3_backbone_frontend/E3_{arch}_yaw_s{0..3}.pt，经 hn_plugin_p0_20260912.load_model('int_<arch>_yaw_s<k>') 加载
（该函数拒绝训练时已含 HN、或窗长/采样率不是 1 s / 200 Hz 的权重）。
推理与计分完全复用 unified_eval_20260913 的 _win 与 metrics；数据：统一评测 5 个测试集 + 确认集 tlio_c、imunet_c。
输出：results/plugin_arch_20260923/int_<arch>_yaw[_hn]_s<k>.json（行格式与统一评测相同）
用法：HN_DEVICE=mps .venv/bin/python tools/plugin_arch_eval_20260923.py [--limit N]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import hn_plugin_p0_20260912 as H
import unified_eval_20260913 as UE
import confirm_eval_20260913 as CE

OUT = ROOT / 'results/plugin_arch_20260923'
ARCHS = ('transformer', 'imunet')


def datasets(limit):
    for ds in UE.DATASETS:
        fl = UE.files(ds)
        if limit is None and len(fl) != UE.EXPECTED[ds]:
            raise RuntimeError(f'{ds} 条数 {len(fl)} ≠ {UE.EXPECTED[ds]}')
        yield ds, ds, (fl[:limit] if limit else fl)
    for ds in CE.DATA:
        fl = CE.files(ds)
        yield ds, CE.METRIC_DS[ds], (fl[:limit] if limit else fl)


def run(name, seed, predict, dev, out_dir, limit, prov):
    dest = out_dir / f'{name}_s{seed}.json'
    if dest.exists() and json.loads(dest.read_text()).get('complete'):
        print(f'[已完成，跳过] {dest.name}', flush=True); return
    rows = []
    for ds, metric_ds, fl in datasets(limit):
        t0 = time.time()
        for p in fl:
            z = CE.read(p)
            xy = predict(np.asarray(z['feat'], np.float32), z)
            m, _ = UE.metrics(np.asarray(xy, float), z, metric_ds)
            m.update(model=name, seed=seed, dataset=ds, sequence=str(z['seq'])); rows.append(m)
        print(f'[{name} s{seed} {ds}] {len(fl)} 条  ATE_u 均值 '
              f'{np.mean([r["ate_u"] for r in rows if r["dataset"] == ds]):.3f}  {time.time() - t0:.0f}s', flush=True)
    json.dump({'provenance': dict(model=name, seed=seed, script_sha256=H.sha(__file__), device=str(dev), limit=limit, **prov),
               'complete': True, 'rows': rows}, open(dest, 'w'), ensure_ascii=False)
    print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    UE.H, UE.P = H, H.P
    torch.set_num_threads(4)
    dev = H.P.pick_device(verbose=False)
    if dev.type != 'mps':
        raise SystemExit(f'设备为 {dev}，请设 HN_DEVICE=mps')
    out_dir = OUT if a.limit is None else ROOT / 'verification/generated/plugin_arch_smoke'
    out_dir.mkdir(parents=True, exist_ok=True)
    for arch in ARCHS:
        for s in range(4):
            net, kind, path = H.load_model(f'int_{arch}_yaw_s{s}', dev)
            assert kind == 'window', kind
            prov = dict(weights=str(path.relative_to(ROOT)), weight_sha256=H.sha(path))
            run(f'int_{arch}_yaw', s, lambda f, z, n=net: UE._win(n, f, z, None, dev), dev, out_dir, a.limit, prov)
            run(f'int_{arch}_yaw_hn', s, lambda f, z, n=net: UE._win(n, f, z, H.headings(f)[1], dev), dev, out_dir, a.limit,
                dict(prov, bolt_on='HN theta (H.headings)'))


if __name__ == '__main__':
    main()
