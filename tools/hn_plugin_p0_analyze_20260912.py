"""P0 插件实验汇总与判定：只实现 docs/27 第 6–8 节，不另加规则。

输入：results/hn_plugin_p0_20260912/<模型>.json（complete=True）
输出：results/hn_plugin_p0_20260912/summary.json，并打印表格。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'hn_plugin_p0_20260912'
DATASETS = ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio')
EXTERNAL = ('ext_resnet', 'ext_lstm', 'ext_tcn')
PUBLISHED = {('ext_lstm', 'ronin_seen'): 4.18, ('ext_lstm', 'ronin_unseen'): 5.32,
             ('ext_tcn', 'ronin_seen'): 4.38, ('ext_tcn', 'ronin_unseen'): 5.70}
N_BOOT = 20000


def ci(d):
    d = np.asarray(d, float)
    idx = np.random.default_rng(0).integers(0, len(d), (N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return [float(lo), float(hi)]


def paired(a, b):
    """b − a 的配对统计。"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a
    lo, hi = ci(d)
    return dict(a=float(a.mean()), b=float(b.mean()), rel_pct=float(100 * (b.mean() / a.mean() - 1)),
                diff=float(d.mean()), ci95=[lo, hi], n_better=int((d < 0).sum()), n=len(d),
                sig='better' if hi < 0 else ('worse' if lo > 0 else 'ns'))


def load(name):
    p = RES / f'{name}.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not d['complete']:
        raise RuntimeError(f'{name} 未跑完')
    return d


def cell(rows):
    """一个 (模型, 测试集) 格子。rows 已按序列对齐（内部模型为种子平均后的行）。"""
    g = lambda f: np.array([f(r) for r in rows], float)
    o0, h0, s0 = g(lambda r: r['orig']['ate']), g(lambda r: r['hn']['ate']), g(lambda r: r['hns']['ate'])
    out = {'n': len(rows),
           'ate_hn_vs_orig': paired(o0, h0),
           'rte_hn_vs_orig': paired(g(lambda r: r['orig']['rte']), g(lambda r: r['hn']['rte'])),
           'ate_hns_vs_hn': paired(h0, s0),
           'ate_hns_vs_orig': paired(o0, s0),
           'flip_pct_hn': float(g(lambda r: r['flip_pct_hn']).mean()),
           'flip_pct_hns': float(g(lambda r: r['flip_pct_hns']).mean())}
    for k in ('tlr', 'mcs', 'ate_startyaw', 'ate_umeyama_sim', 'umeyama_scale'):
        out[f'{k}_hn_vs_orig'] = paired(g(lambda r: r['orig'][k]), g(lambda r: r['hn'][k]))
    if len(rows[0]['orig_ate_by_psi']) > 1:
        P = np.array([r['orig_ate_by_psi'] for r in rows]); A = np.array([r['hn_ate_by_alpha'] for r in rows])
        out['ate_hn_vs_orig_mean8'] = paired(P.mean(1), h0)
        out['orig_psi_range_pct_median'] = float(100 * np.median((P.max(1) - P.min(1)) / P.mean(1)))
        out['orig_mean8'] = float(P.mean())
        out['hn_by_alpha'] = [float(x) for x in A.mean(0)]
        out['alpha_all_below_orig_mean8'] = bool((A.mean(0) < P.mean()).all())
        out['alpha_per_seq_range_pct_median'] = float(100 * np.median((A.max(1) - A.min(1)) / A.mean(1)))
        out['max_invariance_gap'] = float(max(max(abs(r['hn_ate_psi90'] - r['hn']['ate']),
                                                  abs(r['hns_ate_psi90'] - r['hns']['ate'])) for r in rows))
    return out


def seed_average(datas):
    """内部模型：同一序列对 4 个种子取平均（只平均用得到的标量）。"""
    by = {}
    for d in datas:
        for r in d['rows']:
            by.setdefault((r['dataset'], r['sequence']), []).append(r)
    rows = []
    for (ds, seq), rs in by.items():
        if len(rs) != len(datas):
            raise RuntimeError(f'{ds}/{seq} 种子不全')
        m = {'dataset': ds, 'sequence': seq, 'orig_ate_by_psi': [0.0],
             'flip_pct_hn': rs[0]['flip_pct_hn'], 'flip_pct_hns': rs[0]['flip_pct_hns']}
        for arm in ('orig', 'hn', 'hns'):
            m[arm] = {k: float(np.mean([r[arm][k] for r in rs])) for k in rs[0][arm]}
        rows.append(m)
    return rows


def main():
    summary = {'cells': {}, 'checks': {}, 'verdict': {}}
    ext = {w: load(w) for w in EXTERNAL}
    for w, d in ext.items():
        if d is None:
            raise RuntimeError(f'缺 {w}')
        for ds in DATASETS:
            rows = sorted([r for r in d['rows'] if r['dataset'] == ds], key=lambda r: r['sequence'])
            summary['cells'][f'{w}|{ds}'] = cell(rows)
    for arch in ('imunet', 'transformer'):
        datas = [load(f'int_{arch}_yaw_s{s}') for s in range(4)]
        if any(x is None for x in datas):
            continue
        avg = seed_average(datas)
        for ds in DATASETS:
            rows = sorted([r for r in avg if r['dataset'] == ds], key=lambda r: r['sequence'])
            summary['cells'][f'int_{arch}_yaw(4 seeds)|{ds}'] = cell(rows)

    C = summary['cells']
    # ---- 链路自检（docs/27 §8）
    c = C['ext_resnet|ronin_unseen']['ate_hn_vs_orig']
    summary['checks']['1_resnet_reproduces_docs24'] = dict(orig=c['a'], hn=c['b'],
        ok=abs(c['a'] - 5.1404) < 1e-3 and abs(c['b'] - 4.8040) < 1e-3)
    chk2 = {}
    for (w, ds), pub in PUBLISHED.items():
        v = C[f'{w}|{ds}']['ate_hn_vs_orig']['a']
        chk2[f'{w}|{ds}'] = dict(ours=v, published=pub, rel_pct=100 * (v / pub - 1), ok=abs(v / pub - 1) <= 0.10)
    summary['checks']['2_lstm_tcn_vs_published'] = chk2
    gap = max(C[f'{w}|{ds}']['max_invariance_gap'] for w in EXTERNAL for ds in DATASETS)
    summary['checks']['3_invariance'] = dict(max_gap=gap, ok=gap < 1e-3)
    fl = {k: (v['flip_pct_hn'], v['flip_pct_hns']) for k, v in C.items()}
    summary['checks']['4_flip_reduced'] = dict(ok=all(b < a for a, b in fl.values()))

    # ---- 判定（docs/27 §7）
    sig = {k: C[f'{w}|{ds}']['ate_hn_vs_orig']['sig'] for w in EXTERNAL for ds in DATASETS for k in [f'{w}|{ds}']}
    per_weight = {w: sum(sig[f'{w}|{ds}'] == 'better' for ds in DATASETS) for w in EXTERNAL}
    any_worse = [k for k, s in sig.items() if s == 'worse']
    h1 = all(n >= 3 for n in per_weight.values()) and not any_worse
    s_sig = {f'{w}|{ds}': C[f'{w}|{ds}']['ate_hns_vs_hn']['sig'] for w in EXTERNAL for ds in DATASETS}
    n_s_better = sum(s == 'better' for s in s_sig.values()); s_worse = [k for k, s in s_sig.items() if s == 'worse']
    summary['verdict'] = dict(
        per_cell=sig, better_per_weight=per_weight, significantly_worse_cells=any_worse,
        H1_plugin_and_more_accurate=h1, H1prime_no_accuracy_loss=not any_worse,
        hns_vs_hn=s_sig, hns_better_cells=n_s_better, hns_worse_cells=s_worse,
        hns_decision=('abandon' if s_worse else ('default' if n_s_better >= 4 else 'robustness_option')))
    json.dump(summary, open(RES / 'summary.json', 'w'), ensure_ascii=False, indent=1)

    # ---- 打印
    print('链路自检')
    for k, v in summary['checks'].items():
        print(' ', k, json.dumps(v, ensure_ascii=False))
    print('\n主终点：ATE（ψ=0），HN − 原样')
    print('| 模型 | 测试集 | n | 原样 | +HN | 变化 | 95% 区间 | 变好 | 原样 8 朝向均值 | vs 8 朝向均值 区间 | 跨角极差中位 | +HN-S | HN-S−HN 区间 | 翻转率 HN→HN-S |')
    for k, v in C.items():
        w, ds = k.split('|'); a = v['ate_hn_vs_orig']; s = v['ate_hns_vs_hn']
        m8 = v.get('ate_hn_vs_orig_mean8')
        print(f"| {w} | {ds} | {v['n']} | {a['a']:.3f} | {a['b']:.3f} | {a['rel_pct']:+.1f}% | "
              f"[{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] {a['sig']} | {a['n_better']}/{a['n']} | "
              + (f"{m8['a']:.3f} | [{m8['ci95'][0]:+.3f}, {m8['ci95'][1]:+.3f}] {m8['sig']} | "
                 f"{v['orig_psi_range_pct_median']:.1f}% | " if m8 else '— | — | — | ')
              + f"{s['b']:.3f} | [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] {s['sig']} | "
              f"{v['flip_pct_hn']:.2f}%→{v['flip_pct_hns']:.2f}% |")
    print('\n次要：RTE / TLR / MCS / 带尺度 ATE（HN − 原样）')
    for k, v in C.items():
        f = lambda n: f"{v[n]['a']:.3f}→{v[n]['b']:.3f} [{v[n]['ci95'][0]:+.3f},{v[n]['ci95'][1]:+.3f}]"
        print(f"| {k} | RTE {f('rte_hn_vs_orig')} | TLR {f('tlr_hn_vs_orig')} | MCS {f('mcs_hn_vs_orig')} | "
              f"ATE起点旋转 {f('ate_startyaw_hn_vs_orig')} | ATE带尺度 {f('ate_umeyama_sim_hn_vs_orig')} |")
    print('\nα 扫描（外部权重）')
    for k, v in C.items():
        if 'hn_by_alpha' in v:
            print(f"| {k} | 原样8朝向 {v['orig_mean8']:.3f} | α: " + ' '.join(f'{x:.3f}' for x in v['hn_by_alpha'])
                  + f" | 全部低于原样: {v['alpha_all_below_orig_mean8']} | 逐序列 α 极差中位 {v['alpha_per_seq_range_pct_median']:.1f}% |")
    print('\n判定'); print(json.dumps(summary['verdict'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
