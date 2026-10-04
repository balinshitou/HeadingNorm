"""docs/53 的判定：只实现 docs/53 写定的规则，评测运行前写成。

主要检验族 4 个假设 × 2 个数据集 = 8 项，Bonferroni：99.375% 百分位 bootstrap 区间（10000 次，种子 20260906）。
TLIO 按序列；IMUNet 受试者整群区间与序列区间须同时满足。描述性比较报 95% 区间，不参与判定。
输入：results/confirm_20260913/*.json；输出：results/confirm_20260913/verdict.json
用法：.venv/bin/python tools/confirm_verdict_20260913.py
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results/confirm_20260913'
SEED, NB = 20260906, 10000
DS = ('tlio_c', 'imunet_c')
ALPHA = 0.05 / 8
Q_MAIN, Q_DESC = [ALPHA / 2, 1 - ALPHA / 2], [.025, .975]
MARGIN = 0.02
FAMILY = [('H1a', 'ours_yaw_hn', 'ours_yaw', 'sup'), ('H1b', 'ours_yaw_fa2', 'ours_yaw', 'sup'),
          ('H2', 'ext_resnet_hn', 'ext_resnet', 'noninf'), ('H3', 'ours_hn', 'ours_pca', 'sup')]
DESC = [('ours_hn', 'ours_yaw'), ('ours_yaw_hn', 'ours_yaw_fa2'), ('ours_yaw_fa2', 'ours_hn'), ('ours_hn', 'ours_gn')]
SEEDED = ('ours_yaw', 'ours_yaw_hn', 'ours_yaw_fa2', 'ours_hn', 'ours_pca', 'ours_gn')


def subject(ds, seq):
    return re.search(r'Subje[ct]+_(\d+)', seq).group(1) if ds == 'imunet_c' else seq


def load():
    cell = defaultdict(lambda: defaultdict(dict))
    for p in RES.glob('*.json'):
        if p.name in ('cache_manifest.json', 'verdict.json'):
            continue
        for r in json.loads(p.read_text())['rows']:
            cell[(r['model'], r['dataset'])][r['sequence']][r['seed'] if r['seed'] is not None else 0] = r
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
    ua = np.array([np.mean([sa[s] for s in grp[k]]) for k in keys])
    ub = np.array([np.mean([sb[s] for s in grp[k]]) for k in keys])
    d_sub = ua - ub; d_seq = np.array([sa[s] - sb[s] for s in sorted(A)])
    return dict(treatment=t, control=c, dataset=ds, metric=metric, n_subjects=len(keys), n_sequences=len(A),
                treatment_mean=float(ua.mean()), control_mean=float(ub.mean()), difference=float(d_sub.mean()),
                rel_pct=float(100 * d_sub.mean() / ub.mean()), subject_ci=boot(d_sub, q), seq_ci=boot(d_seq, q),
                improved_subjects=int((d_sub < 0).sum()), improved_sequences=int((d_seq < 0).sum()), q=q)


def call(r, kind):
    """TLIO：只看序列区间；IMUNet：受试者区间与序列区间同时满足。"""
    cis = [r['seq_ci']] if r['dataset'] == 'tlio_c' else [r['subject_ci'], r['seq_ci']]
    worse = all(lo > 0 for lo, _ in cis)
    if kind == 'sup':
        return 'better' if all(hi < 0 for _, hi in cis) else ('worse' if worse else 'ns')
    bound = MARGIN * r['control_mean']
    return 'noninferior' if all(hi < bound for _, hi in cis) else ('worse' if worse else 'not_shown')


def claim(cells_by_ds, ok_word):
    ok = {ds: any(c == ok_word for c in v) for ds, v in cells_by_ds.items()}
    bad = {ds: any(c == 'worse' for c in v) for ds, v in cells_by_ds.items()}
    if all(ok.values()):
        return '确认'
    if any(ok[d] and not any(bad[e] for e in ok if e != d) for d in ok):
        return '部分确认'
    return '未确认'


def main():
    cell = load()
    # docs/53 §6.3：序列集合一致、本文模型 4 个种子
    man = json.loads((RES / 'cache_manifest.json').read_text())
    for (m, ds), seqs in cell.items():
        if len(seqs) != man['n'][ds]:
            raise SystemExit(f'序列数不符：{m}/{ds} {len(seqs)} ≠ {man["n"][ds]}')
        if m in SEEDED and any(len(v) != 4 for v in seqs.values()):
            raise SystemExit(f'种子数不符：{m}/{ds}')
    main_rows, calls = [], defaultdict(dict)
    print(f'主要检验族（Bonferroni，{100 * (1 - ALPHA):.3f}% 区间；统一口径 ATE）')
    for hid, t, c, kind in FAMILY:
        for ds in DS:
            r = stats(cell, t, c, ds, Q_MAIN); r.update(hypothesis=hid, kind=kind); r['call'] = call(r, kind)
            main_rows.append(r); calls[hid][ds] = r['call']
            print(f"  {hid} {t} − {c}  {ds:9s} {r['treatment_mean']:.3f} vs {r['control_mean']:.3f}  {r['difference']:+.3f} "
                  f"({r['rel_pct']:+.1f}%)  受试者 [{r['subject_ci'][0]:+.3f}, {r['subject_ci'][1]:+.3f}]  "
                  f"序列 [{r['seq_ci'][0]:+.3f}, {r['seq_ci'][1]:+.3f}]  {r['call']}")
    claims = {'B': claim({ds: [calls['H1a'][ds], calls['H1b'][ds]] for ds in DS}, 'better'),
              'A': claim({ds: [calls['H3'][ds]] for ds in DS}, 'better'),
              'H2_official_plugin': claim({ds: [calls['H2'][ds]] for ds in DS}, 'noninferior')}
    print(f"\n判定：主张B {claims['B']}；主张A {claims['A']}；官方权重外挂非劣 {claims['H2_official_plugin']}")
    desc = []
    print('\n描述性（95% 区间，不参与判定）')
    for t, c in DESC:
        for ds in DS:
            for metric in ('ate_u', 'rte_u', 'ate_lit'):
                r = stats(cell, t, c, ds, Q_DESC, metric); desc.append(r)
                if metric == 'ate_u':
                    print(f"  {t} − {c}  {ds:9s} {r['difference']:+.3f} ({r['rel_pct']:+.1f}%)  "
                          f"受试者 [{r['subject_ci'][0]:+.3f}, {r['subject_ci'][1]:+.3f}]  序列 [{r['seq_ci'][0]:+.3f}, {r['seq_ci'][1]:+.3f}]")
    for hid, t, c, kind in FAMILY:          # 主要比较的 RTE 与文献协议 ATE，也作描述性报告
        for ds in DS:
            for metric in ('rte_u', 'ate_lit'):
                desc.append({**stats(cell, t, c, ds, Q_DESC, metric), 'hypothesis': hid})
    dev = defaultdict(list)                  # IMUNet 按设备的均值（描述性）
    for m in ('ours_yaw', 'ours_yaw_hn', 'ours_yaw_fa2', 'ours_hn', 'ours_pca'):
        for s, v in cell[(m, 'imunet_c')].items():
            dev[(m, re.search(r'(S10|S21|Tango|Xiaomi)', s).group(1))].append(np.mean([r['ate_u'] for r in v.values()]))
    by_device = {f'{m}|{d}': dict(n=len(x), mean=float(np.mean(x))) for (m, d), x in sorted(dev.items())}
    json.dump(dict(alpha_family=0.05, n_tests=8, q_main=Q_MAIN, margin=MARGIN, main=main_rows, claims=claims,
                   descriptive=desc, imunet_by_device=by_device, manifest_n=man['n'], failed=man['failed']),
              open(RES / 'verdict.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[写出] {(RES / "verdict.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
