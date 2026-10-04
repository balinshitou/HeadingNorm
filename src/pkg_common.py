"""可移植包的公共部件：路径、设备、指标。零外部依赖（只需 numpy/torch）。

设计约束
  · 所有路径相对包根目录，**不含任何绝对路径**；
  · 默认使用 CPU 复算，减少不同加速器算子带来的末位浮点差异；也可显式选择
    cuda / mps / auto；纯 CPU 机器上按可用核数设线程；
  · 指标定义**逐行复刻**原项目（ATE/RTE 复刻 RoNIN metric.py；TLR/MCS；
    预先固定的主评价协议与 start_yaw 闭式对齐），保证结果可与本项目既有数字并排。
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_RONIN = ROOT / "data" / "train" / "ronin"
CACHE_OXIOD = ROOT / "data" / "train" / "oxiod"
CACHE_EVAL = ROOT / "data" / "eval" / "benchmark"
CACHE_TLIO = ROOT / "data" / "eval" / "tlio_posthoc"
SELF_DATA = ROOT / "data" / "eval" / "self_collected"
SELF_GT = ROOT / "data" / "eval" / "self_collected_ground_truth"
CKPT = ROOT / "models"
GENERATED_MODELS = CKPT / "generated"
SUBMISSION_MODELS = CKPT / "submission_frozen"
RESULTS = ROOT / "results" / "generated"
for _p in (CKPT, GENERATED_MODELS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

WINDOW = 200          # 200 点 @200 Hz = 1 s
RTE_SECONDS = 60.0


def pick_device(verbose=True):
    """选择计算设备。

    ``HN_DEVICE`` 可取 ``cpu``、``cuda``、``mps`` 或 ``auto``。投稿复算默认
    取 CPU，以便 Windows、macOS 和 Linux 使用同一数值路径；日常训练可设置
    ``HN_DEVICE=auto`` 自动使用 CUDA/MPS。这里选择的是一次前向/反向计算的
    执行设备，并不会改变模型结构或检查点参数。
    """
    import torch
    requested = os.environ.get("HN_DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("HN_DEVICE 只能是 cpu、cuda、mps 或 auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定 HN_DEVICE=cuda，但当前 PyTorch 未检测到 CUDA")
    if requested == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("已指定 HN_DEVICE=mps，但当前 PyTorch 未检测到 Apple MPS")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    use_mps = requested == "mps" or (
        requested == "auto" and not use_cuda and
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if use_cuda:
        dev = torch.device("cuda"); nm = torch.cuda.get_device_name(0)
    elif use_mps:
        dev = torch.device("mps"); nm = "Apple MPS"
    else:
        dev = torch.device("cpu")
        n = os.cpu_count() or 4
        # 物理核通常是逻辑核的一半；超订会因内存带宽争用而变慢
        torch.set_num_threads(max(1, min(n, int(os.environ.get("PKG_THREADS", n)))))
        nm = f"CPU × {torch.get_num_threads()} 线程"
    if verbose:
        print(f"[设备] {nm}", flush=True)
    return dev


# ---------------------------------------------------------------- 指标
def ate_rte(est, gt, rate_hz):
    """逐行复刻 RoNIN metric.py 的坐标级 ATE / RTE。"""
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    ate = float(np.sqrt(np.mean((est - gt) ** 2)))
    ppm = int(round(RTE_SECONDS * rate_hz))
    if len(est) < ppm:
        delta = len(est) - 1; f = ppm / len(est)
    else:
        delta = ppm; f = 1.0
    rel = est[delta:] + gt[:-delta] - est[:-delta] - gt[delta:]
    return ate, float(np.sqrt(np.mean(rel ** 2)) * f)


def shape_metrics(est, gt):
    """TLR 路长比、MCS 逐步方向余弦，均以 1.0 为理想。"""
    de = np.diff(est, axis=0); dg = np.diff(gt, axis=0)
    le = float(np.linalg.norm(de, axis=1).sum()); lg = float(np.linalg.norm(dg, axis=1).sum())
    tlr = le / lg if lg > 1e-9 else float("nan")
    ne = np.linalg.norm(de, axis=1); ng = np.linalg.norm(dg, axis=1)
    m = (ne > 1e-9) & (ng > 1e-9)
    mcs = float(((de[m] * dg[m]).sum(1) / (ne[m] * ng[m])).mean()) if m.sum() else float("nan")
    return tlr, mcs


def align_start_yaw(est, gt):
    """起点重合 + 绕起点最优旋转（扣掉不可辨识的 θ₀）。不缩放、不镜像。
    闭式：θ*=atan2(b,a)，a=Σ(x·u+y·v)，b=Σ(x·v−y·u)。"""
    e = np.asarray(est, float); g = np.asarray(gt, float)
    n = min(len(e), len(g))
    p = e[:n] - e[0]; q = g[:n] - g[0]
    a = float((p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]).sum())
    b = float((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum())
    th = np.arctan2(b, a) if (a * a + b * b) > 1e-20 else 0.0
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return ((np.asarray(est, float) - e[0]) @ R.T) + g[0], float(np.degrees(th))


def align_prefix_rigid_2d(est, gt, ts, seconds):
    """仅用序列开头 ``seconds`` 秒估计 SE(2) 刚体对齐。

    这是 RIDI/OxIOD 注册评测协议：平移和旋转只由前缀确定，随后冻结；
    不使用尺度、不允许镜像，也不使用整条轨迹的信息。
    返回对齐轨迹、用于配准的样本数和旋转角（度）。
    """
    e = np.asarray(est, float)
    g = np.asarray(gt, float)
    t = np.asarray(ts, float)
    n = min(len(e), len(g), len(t))
    e, g, t = e[:n], g[:n], t[:n]
    mask = t <= t[0] + float(seconds) + 1e-9
    if int(mask.sum()) < 2:
        raise ValueError(f"配准前缀不足：{int(mask.sum())} 个样本")
    ep, gp = e[mask], g[mask]
    mu_e, mu_g = ep.mean(axis=0), gp.mean(axis=0)
    ec, gc = ep - mu_e, gp - mu_g
    if float(np.sum(ec * ec)) <= 1e-12:
        rotation = np.eye(2)
    else:
        u, _, vt = np.linalg.svd(ec.T @ gc)
        sign = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
        rotation = vt.T @ np.diag([1.0, sign]) @ u.T
    translation = mu_g - rotation @ mu_e
    aligned = (rotation @ e.T).T + translation
    theta = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return aligned, int(mask.sum()), theta


def align_registered(dataset, est, gt, ts):
    """应用论文预先固定的数据集特定主评价协议（代码标签 registered）。

    RoNIN 已在公共世界坐标系中给出预测输入和真值，只平移到真值起点；
    RIDI 使用前 10 s、OxIOD 使用前 5 s 做无尺度/无镜像 SE(2) 配准。
    """
    dataset = str(dataset).lower()
    e, g = np.asarray(est, float), np.asarray(gt, float)
    n = min(len(e), len(g), len(ts))
    if dataset == "ronin":
        return e[:n] + g[0], 1, 0.0
    seconds = {"ridi": 10.0, "oxiod": 5.0}.get(dataset)
    if seconds is None:
        raise ValueError(f"未注册的数据集：{dataset}")
    return align_prefix_rigid_2d(e[:n], g[:n], np.asarray(ts)[:n], seconds)


def trajectory_from_velocity(vel, ids, tm, te):
    """复刻 RoNIN recon_traj_with_preds：零起点积分后插到评测栅格。"""
    dt = float(np.mean(tm[ids[1:]] - tm[ids[:-1]]))
    pos = np.zeros((len(vel) + 2, 2), float)
    pos[1:-1] = np.cumsum(np.asarray(vel, float) * dt, axis=0)
    pos[-1] = pos[-2]
    ts = np.concatenate([[tm[0] - 1e-6], tm[ids], [tm[-1] + 1e-6]])
    return np.stack([np.interp(te, ts, pos[:, 0]), np.interp(te, ts, pos[:, 1])], 1)


def rectilinearity(vel, q=30):
    """R₄ = |E[e^{4iθ}]|，方向通道指标：不需时间对应、不受 θ₀ 与尺度影响。"""
    sp = np.linalg.norm(vel, axis=1)
    m = sp > np.percentile(sp, q)
    if m.sum() < 50:
        return float("nan")
    th = np.arctan2(vel[m, 1], vel[m, 0])
    return float(np.abs(np.mean(np.exp(4j * th))))


def residual_channels(vel, vg):
    """把速度残差沿真值方向拆成速率分量与方向分量（先闭式扣 θ₀）。"""
    s = np.linalg.norm(vg, axis=1); m = s > 0.3
    if m.sum() < 50:
        return {}
    vp, vq = vel[m], vg[m]
    th = np.arctan2(np.sum(vp[:, 0]*vq[:, 1] - vp[:, 1]*vq[:, 0]),
                    np.sum(vp[:, 0]*vq[:, 0] + vp[:, 1]*vq[:, 1]))
    c, sn = np.cos(th), np.sin(th)
    vp = np.stack([c*vp[:, 0] - sn*vp[:, 1], sn*vp[:, 0] + c*vp[:, 1]], 1)
    u = vq / s[m, None]
    r = vp - vq
    rp = np.sum(r * u, 1); rq = np.linalg.norm(r - rp[:, None] * u, axis=1)
    dth = np.degrees(np.arctan2(rq, np.maximum(s[m] + rp, 1e-6)))
    return dict(n=int(m.sum()), rms_speed=float(np.sqrt(np.mean(rp**2))),
                rms_dir=float(np.sqrt(np.mean(rq**2))),
                dtheta_med=float(np.median(np.abs(dth))))
