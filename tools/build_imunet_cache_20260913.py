"""把 IMUNet 作者数据集的测试划分（list_test.txt，36 条）转成本项目评测缓存格式。

只用加速度计与陀螺仪：只用游戏旋转向量（Android GAME_ROTATION_VECTOR，只融合加计与陀螺）把 gyro/acce
旋到世界系；该世界系 z 轴朝上、水平朝向任意。不用 ARCore 姿态：它是相机位姿，与 IMU 轴差一个固定旋转
（RoNIN 用 rot_imu_to_tango 补偿，作者代码没有），直接用会把重力转到水平面（2026-09-13 首版即如此，已弃）。
水平朝向任意，所以评测一律用前 10 s 真值确定初始航向（docs/36）。真值 gt = pos_x, pos_y（作者已换成 z 轴朝上）。
输出：data/eval/imunet_owndata/imunet__<序列>.npz，字段与 data/eval/benchmark 相同。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src')); import hn_paths   # 外部资料路径表
sys.path.append(str(hn_paths.PYLIB_IMUNET))
import quaternion

ROOT = Path(__file__).resolve().parent.parent
SRC = hn_paths.IMUNET_RAW
OUT = ROOT / 'data/eval/imunet_owndata'


def build(seq):
    d = pd.read_csv(SRC / seq / 'processed/data.csv')
    ts = d['time'].values * 1e-9
    ori = quaternion.from_float_array(d[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
    nz = np.zeros((len(d), 1))
    rot = lambda v: quaternion.as_float_array(ori * quaternion.from_float_array(np.concatenate([nz, v], 1)) * ori.conj())[:, 1:]
    g = rot(d[['gyro_x', 'gyro_y', 'gyro_z']].values); a = rot(d[['acce_x', 'acce_y', 'acce_z']].values)
    feat = np.concatenate([g, a], 1).astype(np.float32)
    gt = d[['pos_x', 'pos_y']].values.astype(float)
    np.savez_compressed(OUT / f'imunet__{seq}.npz', feat=feat, tm=ts, te=ts, gt=gt, rate=np.float64(200.0),
                        dataset='imunet', seq=seq, split='test')
    return feat, ts, gt


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = [s.strip() for s in open(SRC / 'list_test.txt') if s.strip()]
    stats = []
    for s in seqs:
        feat, ts, gt = build(s)
        stats.append((s, len(ts) / (ts[-1] - ts[0]), feat[:, 5].mean(), np.linalg.norm(feat[:, 3:5].mean(0)),
                      (ts[-1] - ts[0]) / 60, np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()))
    fs = np.array([x[1] for x in stats]); az = np.array([x[2] for x in stats]); ah = np.array([x[3] for x in stats])
    print(f'写出 {len(seqs)} 条；采样率 {fs.min():.1f}–{fs.max():.1f} Hz；加计 z 均值 {az.min():.2f}–{az.max():.2f}；'
          f'水平加计均值模长 最大 {ah.max():.2f}；时长 {sum(x[4] for x in stats):.1f} min；路长 {sum(x[5] for x in stats)/1000:.2f} km')
