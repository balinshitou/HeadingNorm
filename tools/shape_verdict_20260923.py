"""V21 判定：外挂 HN 是否使轨迹形状更接近真值（规则见 docs/89，运行前写定）。

只读已有结果：results/unified_eval_20260913/*.json 与 results/confirm_20260913/*.json 的逐（序列，种子）行，
指标 ate_shape（tools/unified_eval_20260913.py::metrics()：统一口径对齐后，以起点为中心的最优复增益修正后剩余的 ATE）。
聚合同 tools/unified_significance_20260913.py：种子平均 → 受试者内序列平均 → 受试者等权；TLIO 按序列。
主要检验族 2 比较 × 4 数据集 = 8 项，Bonferroni 99.375% 百分位 bootstrap（10000 次，种子 20260906）。
判定区间：RIDI 受试者整群；TLIO、TLIO 确认集按序列；IMUNet 手机测试划分只用序列区间（n=4 受试者区间退化）。
输出：results/shape_verdict_20260923/verdict.json
用法：.venv/bin/python tools/shape_verdict_20260923.py
"""
import json, re, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / 'results/unified_eval_20260913', ROOT / 'results/confirm_20260913']
SKIP = {'summary.json', 'significance.json', 'yawhn_verdict.json', 'yawhn_summary.json', 'verdict.json', 'cache_manifest.json'}
OUT = ROOT / 'results/shape_verdict_20260923'
SEED, NB = 20260906, 10000
Q_MAIN = [0.05 / 8 / 2, 1 - 0.05 / 8 / 2]
Q_DESC = [.025, .975]
FAMILY = [('P1', 'ours_yaw_hn', 'ours_yaw'), ('P2', 'ext_resnet_hn', 'ext_resnet')]
JUDGE_DS = ('ridi', 'tlio', 'tlio_c', 'imunet')
DECIDE_ON = {'ridi': 'subject', 'tlio': 'seq', 'tlio_c': 'seq', 'imunet': 'seq'}
DESC_DS_FAMILY = ('ronin_seen', 'ronin_unseen', 'imunet_c')
ALL_DS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet', 'tlio_c', 'imunet_c')
DESC_PAIRS = [('D1', 'ours_yaw_fa2', 'ours_yaw'), ('D2', 'ours_hn', 'ours_yaw')]


def subject(ds, seq):
    if ds.startswith('imunet'):
        return re.search(r'Subje[ct]+_(\d+)', seq).group(1)
    if ds.startswith('tlio'):
        return seq
    return seq.split('_')[0]


def load():
    cell = defaultdict(lambda: defaultdict(dict))
    for d in SRC:
        for p in sorted(d.glob('*.json')):
            if p.name in SKIP:
                continue
            for r in json.loads(p.read_text())['rows']:
                key = (r['model'], r['dataset'])
                seed = r['seed'] if r['seed'] is not None else 0
                assert seed not in cell[key][r['sequence']], ('重复行', key, r['sequence'], seed)
                cell[key][r['sequence']][seed] = r
    return cell


def boot(x, q):
    rng = np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(0, len(x), (NB, len(x)))].mean(axis=1), q).tolist()


def per_seq(C, metric, f=np.mean):
    return {s: float(f([r[metric] for r in C[s].values()])) for s in C}


def stats(cell, t, c, ds, q, metric='ate_shape'):
    A, B = cell[(t, ds)], cell[(c, ds)]
    if not A or set(A) != set(B):
        raise RuntimeError(f'未配对或缺失：{t}/{c}/{ds}')
    sa, sb = per_seq(A, metric), per_seq(B, metric)
    grp = defaultdict(list)
    for s in A:
        grp[subject(ds, s)].append(s)
    keys = sorted(grp)
    ua = np.array([np.mean([sa[s] for s in grp[k]]) for k in keys])
    ub = np.array([np.mean([sb[s] for s in grp[k]]) for k in keys])
    d_sub = ua - ub
    d_seq = np.array([sa[s] - sb[s] for s in sorted(A)])
    return dict(treatment=t, control=c, dataset=ds, metric=metric, n_subjects=len(keys), n_sequences=len(A),
                n_seeds_t=len(next(iter(A.values()))), treatment_mean=float(ua.mean()), control_mean=float(ub.mean()),
                difference=float(d_sub.mean()), rel_pct=float(100 * d_sub.mean() / ub.mean()),
                subject_ci=boot(d_sub, q), seq_ci=boot(d_seq, q), q=q,
                improved_subjects=int((d_sub < 0).sum()), improved_sequences=int((d_seq < 0).sum()))


def sig(ci):
    return 'lower' if ci[1] < 0 else ('higher' if ci[0] > 0 else 'ns')


def extras(cell, t, c, ds):
    """描述性：同比较的统一口径 ATE 差、形状占比、|ε_θ| 与 |ε_s| 的逐序列中位数。"""
    a = stats(cell, t, c, ds, Q_DESC, 'ate_u')
    s = stats(cell, t, c, ds, Q_DESC, 'ate_shape')
    A, B = cell[(t, ds)], cell[(c, ds)]
    med = lambda C, m: float(np.median(list(per_seq(C, m, lambda v: np.mean(np.abs(v))).values())))
    return dict(ate_u_diff=a['difference'], ate_u_rel_pct=a['rel_pct'],
                shape_share=(s['difference'] / a['difference']) if a['difference'] != 0 else None,
                abs_eps_theta_med=[med(A, 'eps_theta_deg'), med(B, 'eps_theta_deg')],
                abs_eps_s_med=[med(A, 'eps_s'), med(B, 'eps_s')])


def main():
    cell = load()
    fam, verdicts = [], {}
    for pid, t, c in FAMILY:
        calls = {}
        for ds in JUDGE_DS:
            r = stats(cell, t, c, ds, Q_MAIN)
            ci = r['subject_ci'] if DECIDE_ON[ds] == 'subject' else r['seq_ci']
            r.update(id=pid, decide_on=DECIDE_ON[ds], call=sig(ci), **extras(cell, t, c, ds))
            calls[ds] = r['call']; fam.append(r)
        n_low = sum(v == 'lower' for v in calls.values())
        n_high = sum(v == 'higher' for v in calls.values())
        verdicts[pid] = dict(calls=calls, n_lower=n_low, n_higher=n_high,
                             verdict='有害' if n_high else ('成立' if n_low >= 2 else '不成立'))
    desc = []
    for pid, t, c in FAMILY:
        for ds in DESC_DS_FAMILY:
            r = stats(cell, t, c, ds, Q_DESC); r.update(id=pid, call_subject=sig(r['subject_ci']),
                                                         call_seq=sig(r['seq_ci']), **extras(cell, t, c, ds))
            desc.append(r)
    for pid, t, c in DESC_PAIRS:
        for ds in ALL_DS:
            r = stats(cell, t, c, ds, Q_DESC); r.update(id=pid, call_subject=sig(r['subject_ci']),
                                                         call_seq=sig(r['seq_ci']), **extras(cell, t, c, ds))
            desc.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(criteria='docs/89', registry='V21', q_main=Q_MAIN, n_tests=8, bootstrap=dict(n=NB, seed=SEED),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               family=fam, verdicts=verdicts, descriptive=desc)
    (OUT / 'verdict.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print('## 主要检验族（形状ATE，99.375%，处理 − 对照）')
    print('| 编号 | 数据集 | 单位 | 处理 | 对照 | 差（相对） | 判定所用区间 | 另一区间 | 改善 | 判定 | ΔATE（相对） | 形状占比 |')
    for r in fam:
        ci, other = (r['subject_ci'], r['seq_ci']) if r['decide_on'] == 'subject' else (r['seq_ci'], r['subject_ci'])
        print(f"| {r['id']} | {r['dataset']} | {r['n_subjects']}人/{r['n_sequences']}条 | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | "
              f"{r['difference']:+.3f}（{r['rel_pct']:+.1f}%） | {r['decide_on']} [{ci[0]:+.3f}, {ci[1]:+.3f}] | [{other[0]:+.3f}, {other[1]:+.3f}] | "
              f"{r['improved_subjects']}/{r['n_subjects']}人、{r['improved_sequences']}/{r['n_sequences']}条 | {r['call']} | "
              f"{r['ate_u_diff']:+.3f}（{r['ate_u_rel_pct']:+.1f}%） | {r['shape_share']:.2f} |")
    for pid, v in verdicts.items():
        print(pid, v)
    print('\n## 描述性（95%）')
    for r in desc:
        print(f"| {r['id']} | {r['dataset']} | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | {r['difference']:+.3f}（{r['rel_pct']:+.1f}%） | "
              f"subj [{r['subject_ci'][0]:+.3f}, {r['subject_ci'][1]:+.3f}] {r['call_subject']} | seq [{r['seq_ci'][0]:+.3f}, {r['seq_ci'][1]:+.3f}] {r['call_seq']} | "
              f"{r['improved_subjects']}/{r['n_subjects']} | ΔATE {r['ate_u_rel_pct']:+.1f}% | 占比 {r['shape_share']:.2f} | "
              f"|εθ| {r['abs_eps_theta_med'][0]:.1f}°/{r['abs_eps_theta_med'][1]:.1f}° | |εs| {r['abs_eps_s_med'][0]:.3f}/{r['abs_eps_s_med'][1]:.3f} |")


if __name__ == '__main__':
    main()
