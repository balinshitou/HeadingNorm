"""docs/53 确认性检验：在从未评测过的 TLIO train+val 与 IMUNet 训练划分上只评测冻结模型。判定标准见 docs/53（先于本脚本运行写成）。

推理与计分完全复用 unified_eval_20260913 的 _win、_fa2、metrics。
用法：
  HN_DEVICE=mps .venv/bin/python tools/confirm_eval_20260913.py --selfcheck
  HN_DEVICE=mps .venv/bin/python tools/confirm_eval_20260913.py --group yaw    # ours_yaw / ours_yaw_hn / ours_yaw_fa2
  HN_DEVICE=mps .venv/bin/python tools/confirm_eval_20260913.py --group rest   # ours_hn / ours_pca / ours_gn / ext_resnet(_hn)
输出：results/confirm_20260913/<模型>[_s<种子>].json
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
from sensors_evaluate import load

OUT = ROOT / 'results/confirm_20260913'
DATA = {'tlio_c': ROOT / 'data/eval/confirm_tlio', 'imunet_c': ROOT / 'data/eval/confirm_imunet'}
METRIC_DS = {'tlio_c': 'tlio', 'imunet_c': 'imunet'}     # 文献协议的对齐方式与原数据集相同


def files(ds):
    n = json.loads((OUT / 'cache_manifest.json').read_text())['n'][ds]
    fl = sorted(DATA[ds].glob('*.npz'))
    if len(fl) != n:
        raise RuntimeError(f'{ds} 缓存 {len(fl)} 条 ≠ 清单 {n} 条')
    return fl


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt', 'seq')}


def run(name, seed, predict, dev, prov):
    dest = OUT / f'{name}{"" if seed is None else f"_s{seed}"}.json'
    if dest.exists() and json.loads(dest.read_text()).get('complete'):
        print(f'[已完成，跳过] {dest.name}', flush=True); return
    rows = []
    for ds in DATA:
        t0 = time.time(); fl = files(ds)
        for p in fl:
            z = read(p)
            xy = predict(np.asarray(z['feat'], np.float32), z)
            m, _ = UE.metrics(np.asarray(xy, float), z, METRIC_DS[ds])
            m.update(model=name, seed=seed, dataset=ds, sequence=str(z['seq'])); rows.append(m)
        print(f'[{name} s{seed} {ds}] {len(fl)} 条  {time.time() - t0:.0f}s', flush=True)
    json.dump({'provenance': dict(model=name, seed=seed, script_sha256=H.sha(__file__), device=str(dev), **prov),
               'complete': True, 'rows': rows}, open(dest, 'w'), ensure_ascii=False)
    print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


def selfcheck(dev):
    """docs/53 §6.2：在已有测试缓存上复现 ours_yaw s0 的统一评测结果。"""
    net, _, _, _ = load(UE.OURS['ours_yaw'].format(0), dev)
    ref = {(r['dataset'], r['sequence']): r['ate_u']
           for r in json.loads((UE.OUT / 'ours_yaw_s0.json').read_text())['rows']}
    for ds, folder in (('tlio', 'tlio_posthoc'), ('imunet', 'imunet_owndata')):
        for p in sorted((ROOT / 'data/eval' / folder).glob('*.npz'))[:3]:
            z = read(p)
            m, _ = UE.metrics(np.asarray(UE._win(net, np.asarray(z['feat'], np.float32), z, None, dev), float), z, ds)
            d = abs(m['ate_u'] - ref[(ds, str(z['seq']))])
            if d > 1e-5:
                raise SystemExit(f'评测自检不通过：{ds}/{z["seq"]} 差 {d:.2e} m（docs/53 §6.2）')
            print(f'[评测自检通过] {ds}/{z["seq"]} 差 {d:.1e} m', flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selfcheck', action='store_true')
    g.add_argument('--group', choices=('yaw', 'rest'))
    a = ap.parse_args()
    UE.H, UE.P = H, H.P
    torch.set_num_threads(4)
    dev = H.P.pick_device(verbose=False)
    if dev.type != 'mps':
        raise SystemExit(f'设备为 {dev}，请设 HN_DEVICE=mps')
    OUT.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        selfcheck(dev); return
    if a.group == 'yaw':
        tag = UE.OURS['ours_yaw']
        for s in range(4):
            net, _, _, path = load(tag.format(s), dev)
            prov = dict(tag=tag.format(s), weight_sha256=H.sha(path))
            run('ours_yaw', s, lambda f, z, n=net: UE._win(n, f, z, None, dev), dev, prov)
            run('ours_yaw_hn', s, lambda f, z, n=net: UE._win(n, f, z, H.headings(f)[1], dev), dev, prov)
            run('ours_yaw_fa2', s, lambda f, z, n=net: UE._fa2(n, f, z, dev), dev, prov)
    else:
        for name in ('ours_hn', 'ours_pca', 'ours_gn'):
            tag = UE.OURS[name]
            for s in range(4):
                net, _, _, path = load(tag.format(s), dev)
                run(name, s, lambda f, z, n=net: UE._win(n, f, z, None, dev), dev,
                    dict(tag=tag.format(s), weight_sha256=H.sha(path)))
        net, _, wpath = H.load_model('ext_resnet', dev)
        prov = dict(weight_sha256=H.sha(wpath))
        run('ext_resnet', None, lambda f, z: UE._win(net, f, z, None, dev), dev, prov)
        run('ext_resnet_hn', None, lambda f, z: UE._win(net, f, z, H.headings(f)[1], dev), dev, prov)


if __name__ == '__main__':
    main()
