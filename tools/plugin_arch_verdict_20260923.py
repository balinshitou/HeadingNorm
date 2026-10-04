"""V22 判定（规则见 docs/91，运行前写定）：外挂 HN 能否推广到 Transformer / IMUNet 增强网络。

主要检验族：A1 int_transformer_yaw_hn − int_transformer_yaw、A2 int_imunet_yaw_hn − int_imunet_yaw，
数据 tlio_c（318 条，按序列），统一口径 ATE；序列级百分位 bootstrap 10000 次、种子 20260906，Bonferroni 2 项 → 97.5%。
判定前自检：原样臂在 RoNIN 独立受试者组上的文献协议 ATE（4 种子逐序列平均后序列等权）须复现 P0 汇总值（差 ≤ 0.001 m）。
输入：results/plugin_arch_20260923/*.json；输出：results/plugin_arch_20260923/verdict.json
用法：.venv/bin/python tools/plugin_arch_verdict_20260923.py
"""
import json, re, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/plugin_arch_20260923'
P0 = ROOT / 'results/hn_plugin_p0_20260912/summary.json'
SEED, NB = 20260906, 10000
Q_MAIN = [0.05 / 2 / 2, 1 - 0.05 / 2 / 2]
Q_DESC = [.025, .975]
FAMILY = [('A1', 'int_transformer_yaw_hn', 'int_transformer_yaw'), ('A2', 'int_imunet_yaw_hn', 'int_imunet_yaw')]
DESC_DS = ('imunet', 'imunet_c', 'ronin_seen', 'ronin_unseen', 'ridi', 'tlio')
P0_KEY = {'int_transformer_yaw': 'int_transformer_yaw(4 seeds)|ronin_unseen', 'int_imunet_yaw': 'int_imunet_yaw(4 seeds)|ronin_unseen'}


def subject(ds, seq):
    if ds.startswith('imunet'):
        return re.search(r'Subje[ct]+_(\d+)', seq).group(1)
    if ds.startswith('tlio'):
        return seq
    return seq.split('_')[0]


def load():
    cell = defaultdict(lambda: defaultdict(dict))
    for p in sorted(RES.glob('int_*.json')):
        d = json.loads(p.read_text())
        assert d['complete'] and d['provenance']['limit'] is None, p
        for r in d['rows']:
            cell[(r['model'], r['dataset'])][r['sequence']][r['seed']] = r
    for k, v in cell.items():
        assert all(len(s) == 4 for s in v.values()), ('种子不全', k)
    return cell


def boot(x, q):
    rng = np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(0, len(x), (NB, len(x)))].mean(axis=1), q).tolist()


def stats(cell, t, c, ds, q, metric='ate_u'):
    A, B = cell[(t, ds)], cell[(c, ds)]
    if not A or set(A) != set(B):
        raise RuntimeError(f'未配对或缺失：{t}/{c}/{ds}')
    sa = {s: np.mean([r[metric] for r in A[s].values()]) for s in A}
    sb = {s: np.mean([r[metric] for r in B[s].values()]) for s in B}
    grp = defaultdict(list)
    for s in A:
        grp[subject(ds, s)].append(s)
    keys = sorted(grp)
    ua = np.array([np.mean([sa[s] for s in grp[k]]) for k in keys]); ub = np.array([np.mean([sb[s] for s in grp[k]]) for k in keys])
    d_sub = ua - ub; d_seq = np.array([sa[s] - sb[s] for s in sorted(A)])
    return dict(treatment=t, control=c, dataset=ds, metric=metric, n_subjects=len(keys), n_sequences=len(A),
                treatment_mean=float(ua.mean()), control_mean=float(ub.mean()), difference=float(d_sub.mean()),
                rel_pct=float(100 * d_sub.mean() / ub.mean()), subject_ci=boot(d_sub, q), seq_ci=boot(d_seq, q), q=q,
                improved_subjects=int((d_sub < 0).sum()), improved_sequences=int((d_seq < 0).sum()))


def sig(ci):
    return 'lower' if ci[1] < 0 else ('higher' if ci[0] > 0 else 'ns')


def selfcheck(cell):
    p0 = json.loads(P0.read_text())['cells']
    out = {}
    for name, key in P0_KEY.items():
        C = cell[(name, 'ronin_unseen')]
        ours = float(np.mean([np.mean([r['ate_lit'] for r in C[s].values()]) for s in C]))
        ref = p0[key]['ate_hn_vs_orig']['a']
        out[name] = dict(ours=ours, p0=ref, diff=ours - ref)
        if abs(ours - ref) > 1e-3:
            raise SystemExit(f'自检不通过：{name} 文献协议 ATE {ours:.4f} ≠ P0 {ref:.4f}（docs/91 §5）')
        print(f'[自检通过] {name} 独立受试者组文献协议 ATE {ours:.4f} vs P0 {ref:.4f}')
    return out


def main():
    cell = load()
    chk = selfcheck(cell)
    fam, calls = [], {}
    for hid, t, c in FAMILY:
        r = stats(cell, t, c, 'tlio_c', Q_MAIN)
        r.update(id=hid, call=sig(r['seq_ci'])); fam.append(r); calls[hid] = r['call']
    n_low = sum(v == 'lower' for v in calls.values()); n_high = sum(v == 'higher' for v in calls.values())
    verdict = '有害' if n_high else ('成立' if n_low == 2 else ('部分成立' if n_low == 1 else '不成立'))
    desc = []
    for hid, t, c in FAMILY:
        for ds in ('tlio_c',) + DESC_DS:
            for metric in ('ate_u', 'ate_shape'):
                if ds == 'tlio_c' and metric == 'ate_u':
                    continue
                r = stats(cell, t, c, ds, Q_DESC, metric)
                r.update(id=hid, call_subject=sig(r['subject_ci']), call_seq=sig(r['seq_ci'])); desc.append(r)
    harm = [f"{r['id']}|{r['dataset']}|{r['metric']}" for r in desc if 'higher' in (r['call_seq'], r['call_subject'])]
    out = dict(criteria='docs/91', registry='V22', q_main=Q_MAIN, bootstrap=dict(n=NB, seed=SEED), selfcheck=chk,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               family=fam, calls=calls, verdict=verdict, descriptive=desc, harm_descriptive=harm)
    (RES / 'verdict.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print('## 主要检验族（tlio_c，统一口径ATE，97.5%，序列级）')
    for r in fam:
        print(f"| {r['id']} | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | {r['difference']:+.3f}（{r['rel_pct']:+.1f}%） | "
              f"[{r['seq_ci'][0]:+.3f}, {r['seq_ci'][1]:+.3f}] | {r['improved_sequences']}/{r['n_sequences']} | {r['call']} |")
    print('判定：', verdict, calls)
    print('\n## 描述性（95%）')
    for r in desc:
        print(f"| {r['id']} | {r['dataset']} | {r['metric']} | {r['treatment_mean']:.3f} | {r['control_mean']:.3f} | "
              f"{r['difference']:+.3f}（{r['rel_pct']:+.1f}%） | subj [{r['subject_ci'][0]:+.3f}, {r['subject_ci'][1]:+.3f}] {r['call_subject']} | "
              f"seq [{r['seq_ci'][0]:+.3f}, {r['seq_ci'][1]:+.3f}] {r['call_seq']} | {r['improved_subjects']}/{r['n_subjects']}人 {r['improved_sequences']}/{r['n_sequences']}条 |")
    print('描述性中显著更高：', harm or '无')


if __name__ == '__main__':
    main()
