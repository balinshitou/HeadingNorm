"""V25 判定（docs/100 写定的规则；bootstrap、call、claim 原样复用 tools/confirm_verdict_20260913.py）。

终点：Ū = 8 个 ψ 的原样推理平均；H̄ = 8 个 α 的外挂 HN 平均；F̄ = 4 个 α 的 FA-2 平均；训练时 HN 精确不变，直接取 confirm_20260913。
本文网络每条序列先对 4 种子平均；TLIO 按序列，imunet_c 受试者区间与序列区间须同时满足。
输入：results/confirm_anglefair_20260927/*.json、results/confirm_20260913/ours_hn_s*.json
输出：results/confirm_anglefair_20260927/verdict.json 与 output.txt（逐行 key = value）
用法：.venv/bin/python tools/confirm_anglefair_verdict_20260927.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import confirm_verdict_20260913 as CV

RES = ROOT / 'results/confirm_anglefair_20260927'
DS = ('tlio_c', 'imunet_c')
MODELS = {'ours_yaw': 4, 'ours_pca': 4, 'ext_resnet': 1, 'int_transformer_yaw': 4, 'int_imunet_yaw': 4}
METRICS = ('ate_u', 'rte_u', 'ate_lit', 'ate_shape')
Q_SEC = [0.0125, 0.9875]                        # 次要：每项 97.5%
lines = []


def say(k, v):
    lines.append(f'{k} = {v}'); print(f'{k} = {v}')


def load():
    """返回 V[(model, ds)][seq][variant][metric] = 种子平均值。"""
    V = {}
    for m, ns in MODELS.items():
        files = [RES / f'{m}.json'] if ns == 1 else [RES / f'{m}_s{s}.json' for s in range(ns)]
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for f in files:
            d = json.loads(f.read_text())
            if not d.get('complete'):
                raise SystemExit(f'未完成：{f.name}')
            for r in d['rows']:
                for var, mm in r['v'].items():
                    for q in METRICS:
                        acc[r['dataset']][r['sequence']][(var, q)].append(mm[q])
        for ds in DS:
            seqs = acc[ds]
            if len(seqs) != json.loads((ROOT / 'results/confirm_20260913/cache_manifest.json').read_text())['n'][ds]:
                raise SystemExit(f'序列数不符：{m}/{ds}')
            V[(m, ds)] = {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in seqs.items()}
            if any(len(v) != ns for d in seqs.values() for v in d.values()):
                raise SystemExit(f'种子数不符：{m}/{ds}')
    hn = defaultdict(lambda: defaultdict(list))
    for s in range(4):
        for r in json.loads((ROOT / f'results/confirm_20260913/ours_hn_s{s}.json').read_text())['rows']:
            for q in METRICS:
                if q in r:
                    hn[r['dataset']][(r['sequence'], q)].append(r[q])
    return V, hn


def ep(V, m, ds, kind, q='ate_u'):
    """某模型在某数据集上某终点的逐序列值 {seq: value}。"""
    n = {'U': 8, 'H': 8, 'F': 4}
    out = {}
    for s, d in V[(m, ds)].items():
        if kind.endswith('bar'):                     # Ubar / Hbar / Fbar：对全部角度取平均
            t = kind[0]
            out[s] = float(np.mean([d[(f'{t}{k}', q)] for k in range(n[t])]))
        else:                                        # U0..U7、H0..H7、F0..F3、T8：单个变体
            out[s] = d[(kind, q)]
    return out


def stats(a, b, ds, qtl):
    """a、b：{seq: 值}。与 CV.stats 同一聚合（受试者内平均 → 受试者等权；序列区间）。"""
    if set(a) != set(b):
        raise RuntimeError('未配对')
    grp = defaultdict(list)
    for s in a:
        grp[CV.subject(ds, s)].append(s)
    keys = sorted(grp)
    ua = np.array([np.mean([a[s] for s in grp[k]]) for k in keys])
    ub = np.array([np.mean([b[s] for s in grp[k]]) for k in keys])
    d_sub = ua - ub; d_seq = np.array([a[s] - b[s] for s in sorted(a)])
    return dict(dataset=ds, n_subjects=len(keys), n_sequences=len(a), treatment_mean=float(ua.mean()),
                control_mean=float(ub.mean()), difference=float(d_sub.mean()), rel_pct=float(100 * d_sub.mean() / ub.mean()),
                subject_ci=CV.boot(d_sub, qtl), seq_ci=CV.boot(d_seq, qtl),
                improved_subjects=int((d_sub < 0).sum()), improved_sequences=int((d_seq < 0).sum()), q=qtl)


def fmt(r):
    return (f"{r['treatment_mean']:.3f} vs {r['control_mean']:.3f}  {r['difference']:+.3f} ({r['rel_pct']:+.1f}%)  "
            f"受试者[{r['subject_ci'][0]:+.3f},{r['subject_ci'][1]:+.3f}]  序列[{r['seq_ci'][0]:+.3f},{r['seq_ci'][1]:+.3f}]  "
            f"改善 {r['improved_sequences']}/{r['n_sequences']}")


def main():
    V, hn = load()
    out = dict(main=[], secondary=[], descriptive=[], by_angle={})
    # ---- 主要检验族（docs/100 §4）
    fam = [('H1a′', 'ours_yaw', 'Hbar', 'ours_yaw', 'sup'), ('H1b′', 'ours_yaw', 'Fbar', 'ours_yaw', 'sup'),
           ('H2′', 'ext_resnet', 'Hbar', 'ext_resnet', 'noninf'), ('H3′', 'ours_hn', None, 'ours_pca', 'sup')]
    calls = defaultdict(dict)
    for hid, tm, tk, cm, kind in fam:
        for ds in DS:
            b = ep(V, cm, ds, 'Ubar')
            a = ({s: float(np.mean(hn[ds][(s, 'ate_u')])) for s in b} if tm == 'ours_hn' else ep(V, tm, ds, tk))
            r = stats(a, b, ds, CV.Q_MAIN); r.update(hypothesis=hid, kind=kind, treatment=f'{tm}:{tk}', control=f'{cm}:Ubar')
            r['call'] = CV.call(r, kind); calls[hid][ds] = r['call']; out['main'].append(r)
            say(f'main.{hid}.{ds}', fmt(r) + f'  → {r["call"]}')
    out['claims'] = {'B′': CV.claim({ds: [calls['H1a′'][ds], calls['H1b′'][ds]] for ds in DS}, 'better'),
                     'A′': CV.claim({ds: [calls['H3′'][ds]] for ds in DS}, 'better'),
                     'H2′': CV.claim({ds: [calls['H2′'][ds]] for ds in DS}, 'noninferior')}
    say('claims', out['claims'])
    # ---- 次要（docs/100 §5）
    for m in ('int_transformer_yaw', 'int_imunet_yaw'):
        for ds in DS:
            r = stats(ep(V, m, ds, 'Hbar'), ep(V, m, ds, 'Ubar'), ds, Q_SEC); r.update(model=m)
            r['call'] = CV.call(r, 'sup'); out['secondary'].append(r)
            say(f'sec.{m}.{ds}', fmt(r) + f'  → {r["call"]}')
    # ---- 描述（docs/100 §6；95%）
    plug = [('ours_yaw', 'H'), ('ours_yaw', 'F'), ('ext_resnet', 'H'), ('int_transformer_yaw', 'H'), ('int_imunet_yaw', 'H')]
    for m, t in plug:
        for ds in DS:
            nA = 8 if t == 'H' else 4
            per = [ep(V, m, ds, f'{t}{k}') for k in range(nA)]
            means = [float(np.mean(list(p.values()))) for p in per]
            worst = int(np.argmax(means)); best = int(np.argmin(means))
            Ub, U0 = ep(V, m, ds, 'Ubar'), ep(V, m, ds, 'U0')
            Tb = ep(V, m, ds, f'{t}bar'); T0 = per[0]
            out['by_angle'][f'{m}|{t}|{ds}'] = dict(treat_by_alpha=means, unmod_by_psi=[float(np.mean(list(ep(V, m, ds, f'U{k}').values()))) for k in range(8)])
            for name, a, b in (('fixed_T0_minus_U0', T0, U0), ('T0_minus_Ubar', T0, Ub), ('fair_Tbar_minus_Ubar', Tb, Ub),
                               ('worstalpha_minus_Ubar', per[worst], Ub), ('bestalpha_minus_Ubar', per[best], Ub),
                               ('native_U0_minus_Ubar', U0, Ub), ('conv_T0_minus_Tbar', T0, Tb),
                               ('tta8_minus_Tbar', ep(V, m, ds, 'T8'), Tb)):
                r = stats(a, b, ds, CV.Q_DESC); r.update(model=m, treat=t, name=name); out['descriptive'].append(r)
                say(f'desc.{m}.{t}.{ds}.{name}', fmt(r))
            for q in ('rte_u', 'ate_shape', 'ate_lit'):
                r = stats(ep(V, m, ds, f'{t}bar', q), ep(V, m, ds, 'Ubar', q), ds, CV.Q_DESC); r.update(model=m, treat=t, name=f'fair_{q}')
                out['descriptive'].append(r); say(f'desc.{m}.{t}.{ds}.fair_{q}', fmt(r))
            say(f'desc.{m}.{t}.{ds}.frac_seq_Tbar_below_Ubar', f'{np.mean([Tb[s] < Ub[s] for s in Ub]):.3f}')
            say(f'desc.{m}.{t}.{ds}.worst_alpha_deg', 45 * worst)
    json.dump(out, open(RES / 'verdict.json', 'w'), ensure_ascii=False, indent=1)
    (RES / 'output.txt').write_text('\n'.join(lines) + '\n')
    print(f'[写出] {(RES / "verdict.json").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
