"""把 RoNIN、RIDI、OxIOD、TLIO 官方文件解析为论文统一的 IMU 数据结构。

本模块与提交的派生 ``.npz`` 缓存刻意分离：合法取得原始数据的读者可在任意目录重建
缓存，不依赖作者电脑的绝对路径。所有函数只读第三方目录，不会向原数据写入文件。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation, Slerp


G_TO_MS2 = 9.80665
MODEL_RATE_HZ = 200.0
WINDOW = 200
GRAV_SCORE_MAX = 0.05


@dataclass(frozen=True)
class RawRoots:
    ronin: Path
    ronin_lists: Path | None
    ridi: Path
    oxiod: Path


def quat_apply(q, v):
    """用逐行四元数旋转逐行向量；四元数顺序固定为 (w,x,y,z)。"""
    q, v = np.asarray(q, float), np.asarray(v, float)
    w, xyz = q[:, 0], q[:, 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w[:, None] * t + np.cross(xyz, t)


def _gravity_score(q, gravity_device):
    unit = gravity_device / np.linalg.norm(gravity_device, axis=1, keepdims=True)
    world = quat_apply(q, unit)
    return float(np.mean(np.linalg.norm(world - np.array([0., 0., 1.]), axis=1)))


def _slerp(t, q, target_t):
    q = np.asarray(q, float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    target_t = np.clip(target_t, t[0], t[-1])
    out = Slerp(t, rotation)(target_t).as_quat()
    return out[:, [3, 0, 1, 2]]


def _uniform(t, arrays, rate, native=200.0):
    n = int(np.floor((float(t[-1]) - float(t[0])) * native)) + 1
    target_t = float(t[0]) + np.arange(n) / native
    values = [interp1d(t, a, axis=0, bounds_error=False,
                       fill_value="extrapolate")(target_t) for a in arrays]
    if rate < native:
        divisor = int(round(native / rate))
        if abs(native / divisor - rate) > 1e-6:
            raise ValueError(f"rate {rate} is not an integer division of {native}")
        values = [resample_poly(a, 1, divisor, axis=0) for a in values]
        target_t = float(t[0]) + np.arange(len(values[0])) / rate
    return target_t, values


def _gyro_bias(gyro, rate, seconds=2.0, threshold=0.05):
    n = int(seconds * rate)
    if len(gyro) >= n and float(np.mean(np.linalg.norm(gyro[:n], axis=1))) < threshold:
        return gyro[:n].mean(axis=0)
    return np.zeros(3)


def _wxyz_rotation(q):
    q = np.asarray(q, float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def load_ronin_official(sequence_dir: Path):
    """复现本文采用的 RoNIN ``GlobSpeedSequence`` 世界系特征构造。"""
    import h5py

    h5_path = sequence_dir / "data.hdf5"
    with (sequence_dir / "info.json").open(encoding="utf-8") as fh:
        info = json.load(fh)
    with h5py.File(h5_path, "r") as handle:
        ts = np.asarray(handle["synced/time"], float)
        gyro_uncalib = np.asarray(handle["synced/gyro_uncalib"], float)
        acce_uncalib = np.asarray(handle["synced/acce"], float)
        game_rv = np.asarray(handle["synced/game_rv"], float)
        gt = np.asarray(handle["pose/tango_pos"], float)[:, :2]
        tango0 = np.asarray(handle["pose/tango_ori"], float)[0]
    gyro = gyro_uncalib - np.asarray(info["imu_init_gyro_bias"], float)
    acce = np.asarray(info["imu_acce_scale"], float) * (
        acce_uncalib - np.asarray(info["imu_acce_bias"], float))
    n = min(map(len, (ts, gyro, acce, game_rv, gt)))
    ts, gyro, acce, game_rv, gt = (x[:n] for x in (ts, gyro, acce, game_rv, gt))
    orientation = _wxyz_rotation(game_rv)
    initial = (_wxyz_rotation(tango0) *
               _wxyz_rotation(np.asarray(info["start_calibration"], float)) *
               orientation[0].inv())
    orientation = initial * orientation
    features = np.concatenate([orientation.apply(gyro), orientation.apply(acce)], axis=1)
    start = int(info.get("start_frame", 0))
    if start < 0 or start >= n - WINDOW:
        raise ValueError(f"invalid start_frame={start} for n={n}")
    ts, features, gt = ts[start:], features[start:], gt[start:]
    if np.any(np.diff(ts) <= 0):
        raise ValueError("RoNIN timestamps are not strictly increasing")
    return {"features": features, "ts_model": ts, "ts_eval": ts, "gt": gt,
            "rate": MODEL_RATE_HZ, "start_frame": start}


def ronin_records(roots: RawRoots, splits, names_by_split=None):
    directories = {"test_seen": "seen_subjects_test_set",
                   "test_unseen": "unseen_subjects_test_set"}
    records = []
    for split in splits:
        if names_by_split is not None:
            names = names_by_split[split]
        else:
            if roots.ronin_lists is None:
                raise ValueError("RoNIN lists are required when no sequence manifest is supplied")
            names = [x for x in (roots.ronin_lists / f"list_{split}.txt").read_text().split()
                     if x]
        for name in names:
            if split in directories:
                candidates = [roots.ronin / directories[split] / name]
            else:
                candidates = [roots.ronin / folder / name for folder in
                              ("train_dataset_1", "train_dataset_2",
                               "seen_subjects_test_set", "unseen_subjects_test_set")]
            sequence_dir = next((p for p in candidates if (p / "data.hdf5").exists()), None)
            if sequence_dir is not None:
                records.append({"dataset": "ronin", "seq": name, "split": split,
                                "subject": name.split("_")[0], "dir": sequence_dir})
    return records


def _cross_features(q, gyro, acce, ts):
    world = np.concatenate([quat_apply(q, gyro), quat_apply(q, acce)], axis=1)
    n_out = int(round((float(ts[-1]) - float(ts[0])) * MODEL_RATE_HZ)) + 1
    if n_out == len(world):
        return world, ts
    target_t = np.linspace(float(ts[0]), float(ts[-1]), n_out)
    return interp1d(ts, world, axis=0, assume_sorted=True)(target_t), target_t


RIDI_MODES = ("bag", "body", "leg", "handheld", "lopata")


def ridi_records(roots: RawRoots):
    records = []
    for csv_path in sorted(roots.ridi.glob("*/processed/data.csv")):
        seq = csv_path.parents[1].name
        mode = next((m for m in RIDI_MODES if m in seq), "other")
        records.append({"dataset": "ridi", "seq": seq, "split": "", "mode": mode,
                        "subject": seq.split("_")[0], "csv": csv_path})
    return records


def load_ridi(record):
    import pandas as pd

    frame = pd.read_csv(record["csv"], index_col=0)
    t = frame["time"].to_numpy(float) * 1e-9
    gyro = frame[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float)
    acce = frame[["acce_x", "acce_y", "acce_z"]].to_numpy(float)
    gravity = frame[["grav_x", "grav_y", "grav_z"]].to_numpy(float)
    gt = frame[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    q = frame[["rv_w", "rv_x", "rv_y", "rv_z"]].to_numpy(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    score = _gravity_score(q, gravity)
    if not np.isfinite(score) or score > GRAV_SCORE_MAX:
        raise ValueError(f"{record['seq']}: gravity convention check failed ({score:.4f})")
    ts, (gyro, acce, gt, gravity) = _uniform(
        t, [gyro, acce, gt, gravity], MODEL_RATE_HZ, native=200.0)
    q = _slerp(t, q, ts)
    gyro -= _gyro_bias(gyro, MODEL_RATE_HZ)
    features, model_t = _cross_features(q, gyro, acce, ts)
    return {"features": features, "ts_model": model_t, "ts_eval": ts,
            "gt": gt[:, :2], "rate": MODEL_RATE_HZ}


def _axis_rotation(axis, angle):
    return Rotation.from_euler(axis, np.asarray(angle, float).reshape(-1, 1))


def _oxiod_quats(roll, pitch, yaw):
    rotation = (Rotation.from_euler("x", np.pi) * _axis_rotation("z", yaw) *
                _axis_rotation("x", pitch) * _axis_rotation("y", roll))
    q = rotation.as_quat()
    return q[:, [3, 0, 1, 2]]


def oxiod_records(roots: RawRoots, only_large=True):
    records = []
    for imu in sorted(roots.oxiod.glob("*/*/syn/imu*.csv")):
        mode, run = imu.parents[2].name, imu.parents[1].name
        if mode == "test" or (only_large and mode != "large scale"):
            continue
        index = imu.stem[3:]
        gt = imu.parent / f"vi{index}.csv"
        if not gt.exists():
            gt = imu.parent / f"tango{index}.csv"
        if gt.exists():
            records.append({"dataset": "oxiod", "seq": f"{mode}__{run}__{index}",
                            "split": "", "mode": mode, "imu": imu, "gt_file": gt})
    return records


def load_oxiod(record):
    imu = np.loadtxt(record["imu"], delimiter=",")
    if imu.ndim != 2 or imu.shape[1] < 16:
        raise ValueError(f"{record['seq']}: non-iPhone OxIOD schema")
    truth = np.loadtxt(record["gt_file"], delimiter=",")
    n = min(len(imu), len(truth))
    imu, truth = imu[:n], truth[:n]
    gyro = imu[:, 4:7]
    gravity = imu[:, 7:10] * G_TO_MS2
    acce = gravity + imu[:, 10:13] * G_TO_MS2
    q = _oxiod_quats(imu[:, 1], imu[:, 2], imu[:, 3])
    score = _gravity_score(q, gravity)
    if not np.isfinite(score) or score > GRAV_SCORE_MAX:
        raise ValueError(f"{record['seq']}: gravity convention check failed ({score:.4f})")
    # Rx(pi) maps ARKit's z-down world to the paper's z-up frame.
    gt = truth[:, 2:5] * np.array([1., -1., -1.])
    ts = np.arange(n) / 100.0
    gyro -= _gyro_bias(gyro, 100.0)
    features, model_t = _cross_features(q, gyro, acce, ts)
    return {"features": features, "ts_model": model_t, "ts_eval": ts,
            "gt": gt[:, :2], "rate": 100.0}


def load_cross(record):
    if record["dataset"] == "ridi":
        return load_ridi(record)
    if record["dataset"] == "oxiod":
        return load_oxiod(record)
    raise ValueError(record["dataset"])


TLIO_EXPECTED_COLUMNS = [
    "ts_us(1)",
    "gyr_compensated_rotated_in_World(3)",
    "acc_compensated_rotated_in_World(3)",
    "qxyzw_World_Device(4)",
    "pos_World_Device(3)",
    "vel_World(3)",
]


def tlio_records(root: Path, sequence_ids):
    """只按冻结清单定位 TLIO 序列，不扫描结果，也不按性能挑选。"""
    root = Path(root)
    records = []
    for seq in sequence_ids:
        directory = root / str(seq)
        records.append({"dataset": "tlio", "seq": str(seq),
                        "split": "official_test", "dir": directory})
    return records


def load_tlio_processed(record):
    """读取 TLIO 发布的 VIO 处理后世界系数组。

    名称特意保留 ``processed``：第 1:7 列已做偏置补偿并旋转到 VIO 世界系，适合检验
    世界系输入模型的条件跨形态迁移，但不能表述为只使用原始 IMU 的基准实验。
    """
    directory = Path(record["dir"])
    array_path = directory / "imu0_resampled.npy"
    description_path = directory / "imu0_resampled_description.json"
    if not array_path.exists() or not description_path.exists():
        raise FileNotFoundError(f"TLIO sequence incomplete: {directory}")
    description = json.load(open(description_path, encoding="utf-8"))
    if description.get("columns_name(width)") != TLIO_EXPECTED_COLUMNS:
        raise ValueError(f"unexpected TLIO processed schema: {directory.name}")
    values = np.load(array_path, mmap_mode="r")
    if values.ndim != 2 or values.shape[1] < 17 or len(values) < WINDOW + 2:
        raise ValueError(f"invalid TLIO processed array: {directory.name}/{values.shape}")
    ts = np.asarray(values[:, 0], np.float64) / 1e6
    if np.any(np.diff(ts) <= 0):
        raise ValueError(f"TLIO timestamps are not strictly increasing: {directory.name}")
    observed_rate = 1.0 / float(np.median(np.diff(ts)))
    if not np.isclose(observed_rate, MODEL_RATE_HZ, rtol=0, atol=0.1):
        raise ValueError(f"TLIO rate mismatch: {directory.name}/{observed_rate:.6f} Hz")
    features = np.concatenate([np.asarray(values[:, 1:4], np.float64),
                               np.asarray(values[:, 4:7], np.float64)], axis=1)
    gt = np.asarray(values[:, 11:13], np.float64)
    if not np.isfinite(features).all() or not np.isfinite(gt).all():
        raise ValueError(f"non-finite TLIO values: {directory.name}")
    return {"features": features, "ts_model": ts, "ts_eval": ts, "gt": gt,
            "rate": MODEL_RATE_HZ,
            "input_attitude_source": "VIO-derived processed world frame"}
