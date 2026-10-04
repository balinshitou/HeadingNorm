"""起点固定时的复增益上限（2026-09-13，只读已有轨迹，不跑网络）。

轨迹写成复数：z = 预测 − 起点，q = 真值 − 起点。最优复增益 a* = Σ conj(z)·q / Σ|z|²（闭式最小二乘）。
分别计算：只修尺度（实数 s）、只修航向（e^{iθ}）、两者都修（a*）后的 ATE（RoNIN 逐坐标 RMSE）。
这是用真值拟合的上限（oracle），不可部署。
输入：results/hn_on_official_ronin_20260910/*__traj.npz（官方 ResNet，RoNIN unseen 32 条，原样与外挂 HN）
输出：results/complex_gain_oracle_20260913.json
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import pkg_common as P

rows = []
for f in sorted(glob.glob(str(ROOT / 'results/hn_on_official_ronin_20260910/*__traj.npz'))):
    z = np.load(f); g = z['gt'].astype(float)
    for arm in ('official', 'official_hn'):
        e = z[arm].astype(float)
        q = (g[:, 0] - g[0, 0]) + 1j * (g[:, 1] - g[0, 1]); w = (e[:, 0] - e[0, 0]) + 1j * (e[:, 1] - e[0, 1])
        S = np.vdot(w, w).real; c = np.vdot(w, q)
        a = c / S
        ate = lambda zz: float(P.ate_rte(np.stack([zz.real, zz.imag], 1) + g[0], g, 200.0)[0])
        rows.append(dict(sequence=Path(f).name.split('__')[0], arm=arm, ate_start=ate(w), ate_scale=ate((c.real / S) * w),
                         ate_heading=ate(np.exp(1j * np.angle(c)) * w), ate_gain=ate(a * w),
                         gain_abs=float(abs(a)), gain_deg=float(np.degrees(np.angle(a)))))
summ = {}
for arm in ('official', 'official_hn'):
    r = [x for x in rows if x['arm'] == arm]; g = lambda k: np.array([x[k] for x in r])
    T = g('ate_start'); fr = lambda b: float(np.median(1 - (b / T) ** 2))
    summ[arm] = dict(ate_start=float(T.mean()), ate_scale=float(g('ate_scale').mean()), ate_heading=float(g('ate_heading').mean()),
                     ate_gain=float(g('ate_gain').mean()), expl_scale=fr(g('ate_scale')), expl_heading=fr(g('ate_heading')),
                     expl_gain=fr(g('ate_gain')), gain_abs_median=float(np.median(g('gain_abs'))),
                     gain_abs_p10=float(np.percentile(g('gain_abs'), 10)), gain_abs_p90=float(np.percentile(g('gain_abs'), 90)),
                     gain_deg_abs_median=float(np.median(abs(g('gain_deg')))), gain_deg_abs_p90=float(np.percentile(abs(g('gain_deg')), 90)))
json.dump({'summary': summ, 'per_sequence': rows}, open(ROOT / 'results/complex_gain_oracle_20260913.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
