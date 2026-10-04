"""V25 推理（判定标准 docs/100，先于本脚本首次运行写成）：确认集上原样推理的 8ψ 扫描与外挂 HN 的 8α 扫描。

每条序列每个模型：U(ψ_k)（输入旋 ψ_k、输出旋回；与 TTA 同一旋转实现）、H(α_k)（φ = θ_HN − α_k）各 8 次前向；
FA-2(α_j) = H(α_j) 与 H(α_j+π) 的逐窗速度平均（j = 0..3）；TTA-8 = 8 个 U 分支速度平均。均不需要额外前向。
计分复用 unified_eval_20260913.metrics（统一评价方式 ate_u 等）。
用法：
  HN_DEVICE=mps .venv/bin/python tools/confirm_anglefair_eval_20260927.py --selfcheck
  HN_DEVICE=mps .venv/bin/python tools/confirm_anglefair_eval_20260927.py --group resnet   # ours_yaw ×4、ours_pca ×4、ext_resnet
  HN_DEVICE=mps .venv/bin/python tools/confirm_anglefair_eval_20260927.py --group arch     # transformer ×4、imunet ×4
输出：results/confirm_anglefair_20260927/<模型>[_s<种子>].json
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
from sensors_evaluate import load

OUT = ROOT / 'results/confirm_anglefair_20260927'
ANG = [2 * np.pi * k / 8 for k in range(8)]
KEEP = ('ate_u', 'rte_u', 'ate_lit', 'ate_shape')


def sweep(net, feat, z, metric_ds, dev, do_h=True):
    """返回 {变体: {指标: 值}}。变体：U0..U7、H0..H7、F0..F3、T8。"""
    n = len(range(0, len(feat) - H.WINDOW, H.STRIDE))
    th = H.headings(feat)[1].astype(np.float64)
    vu, vh = [], []
    for a in ANG:
        ids, v = H.run_window(net, feat, np.full(n, a, np.float32), dev); vu.append(v)
        if do_h:
            vh.append(H.run_window(net, feat, (th - a).astype(np.float32), dev)[1])
    var = {f'U{k}': vu[k] for k in range(8)}
    var['T8'] = np.mean(vu, 0)
    if do_h:
        var.update({f'H{k}': vh[k] for k in range(8)})
        var.update({f'F{j}': 0.5 * (vh[j] + vh[j + 4]) for j in range(4)})
    out = {}
    for k, v in var.items():
        m, _ = UE.metrics(np.asarray(H.traj_window(v, ids, z), float), z, metric_ds)
        out[k] = {q: m[q] for q in KEEP}
    return out


def run(name, seed, net, dev, do_h, prov, limit=None):
    dest = OUT / f'{name}{"" if seed is None else f"_s{seed}"}.json'
    if dest.exists() and json.loads(dest.read_text()).get('complete'):
        print(f'[已完成，跳过] {dest.name}', flush=True); return
    rows = []
    for ds in CE.DATA:
        t0 = time.time(); fl = CE.files(ds)
        for p in (fl[:limit] if limit else fl):
            z = CE.read(p)
            r = sweep(net, np.asarray(z['feat'], np.float32), z, CE.METRIC_DS[ds], dev, do_h)
            rows.append(dict(model=name, seed=seed, dataset=ds, sequence=str(z['seq']), v=r))
        print(f'[{name} s{seed} {ds}] {len(fl)} 条  {time.time() - t0:.0f}s', flush=True)
    json.dump({'provenance': dict(model=name, seed=seed, script_sha256=H.sha(__file__), device=str(dev), **prov),
               'complete': True, 'rows': rows}, open(dest, 'w'), ensure_ascii=False)
    print(f'[写出] {dest.relative_to(ROOT)}', flush=True)


def selfcheck(dev):
    """docs/100 §3：U0/H0/F0 与已有确认集结果逐条一致。"""
    def ref(path, key='ate_u'):
        return {r['sequence']: r[key] for r in json.loads(path.read_text())['rows'] if r['dataset'] == 'tlio_c'}
    C = ROOT / 'results/confirm_20260913'; A = ROOT / 'results/plugin_arch_20260923'
    cases = [('ours_yaw', lambda: load(UE.OURS['ours_yaw'].format(0), dev)[0],
              {'U0': C / 'ours_yaw_s0.json', 'H0': C / 'ours_yaw_hn_s0.json', 'F0': C / 'ours_yaw_fa2_s0.json'}),
             ('ext_resnet', lambda: H.load_model('ext_resnet', dev)[0],
              {'U0': C / 'ext_resnet.json', 'H0': C / 'ext_resnet_hn.json'}),
             ('int_imunet', lambda: H.load_model('int_imunet_yaw_s0', dev)[0],
              {'U0': A / 'int_imunet_yaw_s0.json', 'H0': A / 'int_imunet_yaw_hn_s0.json'})]
    worst = 0.0
    for name, mk, refs in cases:
        net = mk(); R = {k: ref(p) for k, p in refs.items()}
        for p in CE.files('tlio_c')[:3]:
            z = CE.read(p); s = str(z['seq'])
            r = sweep(net, np.asarray(z['feat'], np.float32), z, 'tlio', dev)
            for k in refs:
                d = abs(r[k]['ate_u'] - R[k][s]); worst = max(worst, d)
                print(f'[自检] {name} {k} {s[-10:]}  新 {r[k]["ate_u"]:.6f}  旧 {R[k][s]:.6f}  差 {d:.1e}', flush=True)
    if worst > 1e-4:
        raise SystemExit(f'自检不通过：最大差 {worst:.2e} m（docs/100 §3）')
    print(f'[自检通过] 最大差 {worst:.1e} m', flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selfcheck', action='store_true')
    g.add_argument('--group', choices=('resnet', 'arch'))
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    UE.H, UE.P = H, H.P
    torch.set_num_threads(4)
    dev = H.P.pick_device(verbose=False)
    if dev.type != 'mps':
        raise SystemExit(f'设备为 {dev}，请设 HN_DEVICE=mps')
    OUT.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        selfcheck(dev); return
    if a.group == 'resnet':
        net, _, wpath = H.load_model('ext_resnet', dev)
        run('ext_resnet', None, net, dev, True, dict(weight_sha256=H.sha(wpath)), a.limit)
        for name, do_h in (('ours_yaw', True), ('ours_pca', False)):
            for s in range(4):
                net, _, _, path = load(UE.OURS[name].format(s), dev)
                run(name, s, net, dev, do_h, dict(tag=UE.OURS[name].format(s), weight_sha256=H.sha(path)), a.limit)
    else:
        for arch in ('transformer', 'imunet'):
            for s in range(4):
                net, kind, path = H.load_model(f'int_{arch}_yaw_s{s}', dev)
                assert kind == 'window', kind
                run(f'int_{arch}_yaw', s, net, dev, True,
                    dict(weights=str(path.relative_to(ROOT)), weight_sha256=H.sha(path)), a.limit)


if __name__ == '__main__':
    main()
