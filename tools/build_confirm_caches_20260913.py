"""docs/53：为确认性检验构建评测缓存（只做数据转换，不跑任何模型）。

TLIO：code-data/tlio_golden 的 train_list（284）+ val_list（36），处理与 tools/rebuild_tlio_cache.py 相同（raw_data.load_tlio_processed）。
IMUNet：code-data/IMUNet_dataset/list_train.txt（90），处理与 tools/build_imunet_cache_20260913.py 相同
（游戏旋转向量把陀螺与加计旋到世界系，只用加计与陀螺；真值 pos_x、pos_y 只用于评测）。
启动自检：用同一函数重建已有测试缓存中的 1 条 TLIO 与 1 条 IMUNet 序列，与已有缓存逐元素一致，否则停止。
不按任何性能指标挑选或剔除；读入检查失败的序列记录原因后剔除，剔除超过 10% 则停止（docs/53 §2）。
用法：.venv/bin/python tools/build_confirm_caches_20260913.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import hn_paths  # noqa: E402  外部资料路径表
sys.path.append(str(hn_paths.PYLIB_IMUNET))
import raw_data as R  # noqa: E402
import quaternion  # noqa: E402

TLIO = hn_paths.TLIO_RAW
IMU = hn_paths.IMUNET_RAW
OUT_T = ROOT / 'data/eval/confirm_tlio'
OUT_I = ROOT / 'data/eval/confirm_imunet'
MAN = ROOT / 'results/confirm_20260913/cache_manifest.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_list(p):
    return [s.strip() for s in open(p) if s.strip()]


def tlio_arrays(seq):
    d = R.load_tlio_processed(R.tlio_records(TLIO, [seq])[0])
    return dict(feat=np.asarray(d['features'], np.float32), tm=np.asarray(d['ts_model'], np.float64),
                te=np.asarray(d['ts_eval'], np.float64), gt=np.asarray(d['gt'], np.float32), rate=float(d['rate']),
                input_attitude_source=d['input_attitude_source'])


def imunet_arrays(seq):
    """与 build_imunet_cache_20260913.build 逐行相同的处理。"""
    d = pd.read_csv(IMU / seq / 'processed/data.csv')
    ts = d['time'].values * 1e-9
    ori = quaternion.from_float_array(d[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
    nz = np.zeros((len(d), 1))
    rot = lambda v: quaternion.as_float_array(ori * quaternion.from_float_array(np.concatenate([nz, v], 1)) * ori.conj())[:, 1:]
    g = rot(d[['gyro_x', 'gyro_y', 'gyro_z']].values); a = rot(d[['acce_x', 'acce_y', 'acce_z']].values)
    feat = np.concatenate([g, a], 1).astype(np.float32)
    gt = d[['pos_x', 'pos_y']].values.astype(float)
    if np.any(np.diff(ts) <= 0) or not (np.isfinite(feat).all() and np.isfinite(gt).all()):
        raise ValueError('时间戳不递增或含非有限值')
    return dict(feat=feat, tm=ts, te=ts, gt=gt, rate=np.float64(200.0))


def selfcheck():
    for folder, fn in (('tlio_posthoc', tlio_arrays), ('imunet_owndata', imunet_arrays)):
        ref_path = sorted((ROOT / 'data/eval' / folder).glob('*.npz'))[0]
        with np.load(ref_path, allow_pickle=False) as f:
            ref = {k: np.asarray(f[k]) for k in ('feat', 'tm', 'te', 'gt')}; seq = str(f['seq'])
        new = fn(seq)
        for k, v in ref.items():
            if v.shape != new[k].shape or not np.array_equal(v, new[k]):
                raise SystemExit(f'缓存自检不通过：{folder}/{seq} 字段 {k}（docs/53 §6.1）')
        print(f'[缓存自检通过] {folder}/{seq}：feat/tm/te/gt 逐元素一致', flush=True)


def build(name, seqs, fn, out, extra):
    out.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for s in seqs:
        try:
            arr = fn(s)
        except Exception as e:  # 读入检查失败：记录原因后剔除
            failed.append(dict(seq=s, reason=f'{type(e).__name__}: {e}')); continue
        np.savez_compressed(out / f'{name}__{s}.npz', dataset=name, seq=s, **extra, **arr)
        ok.append(s)
    print(f'[{name}] 写出 {len(ok)} 条，剔除 {len(failed)} 条', flush=True)
    if len(failed) > 0.10 * len(seqs):
        raise SystemExit(f'{name} 剔除 {len(failed)}/{len(seqs)} 超过 10%，按 docs/53 §2 停止')
    return ok, failed


def main():
    selfcheck()
    t_ids = read_list(TLIO / 'train_list.txt') + read_list(TLIO / 'val_list.txt')
    i_ids = read_list(IMU / 'list_train.txt')
    # docs/53 写的 284 + 36 = 320 是 wc -l 行数；两份列表末行各为空行，非空 ID 为 283 + 35 = 318（运行前更正，记入 docs/54）
    assert len(t_ids) == 318 and len(set(t_ids)) == 318, len(t_ids)
    assert len(i_ids) == 90 and len(set(i_ids)) == 90, len(i_ids)
    assert not set(t_ids) & set(read_list(TLIO / 'test_list.txt'))
    assert not set(i_ids) & set(read_list(IMU / 'list_test.txt'))
    t_ok, t_fail = build('tlio_c', t_ids, tlio_arrays, OUT_T, dict(split='train+val (confirmatory)'))
    i_ok, i_fail = build('imunet_c', i_ids, imunet_arrays, OUT_I, dict(split='list_train (confirmatory)'))
    MAN.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(script_sha256=sha(__file__),
                   lists={'tlio_train': sha(TLIO / 'train_list.txt'), 'tlio_val': sha(TLIO / 'val_list.txt'),
                          'imunet_list_train': sha(IMU / 'list_train.txt')},
                   n={'tlio_c': len(t_ok), 'imunet_c': len(i_ok)}, failed={'tlio_c': t_fail, 'imunet_c': i_fail},
                   sequences={'tlio_c': t_ok, 'imunet_c': i_ok}),
              open(MAN, 'w'), ensure_ascii=False, indent=1)
    print(f'[写出] {MAN.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
