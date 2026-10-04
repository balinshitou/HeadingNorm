"""The nine numerical tables that accompany the Supplementary Materials as CSV files (supp_data/*.csv)."""
from collections import defaultdict

import numpy as np

import tables_lib as L
from tables_lib import R, ci, num, pct, sci, sgn
from tables_main import af
from tables_registry import table
from tables_supp import DSL, TEST, composition

SEEDS = lambda m, ds: {len(v) for v in L.cells()[(m, ds)].values()}


def metrics_row(m, ds):
    cl = L.cells()[(m, ds)]
    ns = SEEDS(m, ds); assert len(ns) == 1
    sm = lambda k: L.seq_weighted(m, ds, k)
    e, th, tlr, rs, rh, rg = composition(m, ds)
    return dict(n=f'{len(cl)}×{ns.pop()}', lit=num(sm('ate_lit')), u=num(sm('ate_u')), rte=num(sm('rte_u')), tlr=num(tlr), mcs=num(sm('mcs')),
                eps=sgn(e), th=f'{th:.1f}°', shape=num(sm('ate_shape')), rs=pct(rs), rh=pct(rh), rg=pct(rg))


@table('supp_data/unified_eval_full_metrics.csv')
def full_metrics():
    head = ['Model', 'Test set', 'Sequences × seeds', 'ATE, official protocol', 'ATE, unified protocol', 'RTE, unified protocol', 'TLR', 'MCS',
            'ε_s median', 'ε_θ median', 'Shape error', 'Scale-only bound', 'Heading-only bound', 'Complex-gain bound']
    rows = []
    for lab, m in (('Ours, ResNet18 + GN', 'ours_gn'), ('Ours, ResNet18 + GN + YawAug', 'ours_yaw'), ('Ours, ResNet18 + GN + PCA frame', 'ours_pca'),
                   ('Ours, ResNet18 + GN + HN', 'ours_hn'), ('Ours, Transformer + GN + HN', 'ours_transformer_hn'),
                   ('Ours, IMUNet + GN + HN', 'ours_imunet_hn'), ('RoNIN ResNet', 'ext_resnet'), ('RoNIN ResNet + plug-in HN', 'ext_resnet_hn')):
        for ds in TEST:
            x = metrics_row(m, ds)
            rows.append([lab, DSL[ds]] + [x[k] for k in ('n', 'lit', 'u', 'rte', 'tlr', 'mcs', 'eps', 'th', 'shape', 'rs', 'rh', 'rg')])
    return head, rows


def test_metrics(m):
    head = ['Test set', 'Sequences × seeds', 'ATE, official protocol', 'ATE, unified protocol', 'RTE, unified protocol', 'TLR', 'ε_s median',
            'ε_θ median', 'Shape error', 'Complex-gain bound']
    rows = []
    for ds in TEST:
        x = metrics_row(m, ds)
        rows.append([DSL[ds]] + [x[k] for k in ('n', 'lit', 'u', 'rte', 'tlr', 'eps', 'th', 'shape', 'rg')])
    return head, rows


@table('supp_data/testtime_plugin_metrics.csv')
def plugin_metrics():
    return test_metrics('ours_yaw_hn')


@table('supp_data/testtime_fa2_metrics.csv')
def fa2_metrics():
    return test_metrics('ours_yaw_fa2')


def paired_rows(comparisons, metrics, with_metric):
    """Subject-level paired comparison (CI-B); 'Result' follows the subject interval; TLIO has sequences as units."""
    rows = []
    for lab, t, c in comparisons:
        for metric, mlab in metrics:
            for ds in TEST:
                r = L.paired(t, c, ds, metric)
                res = 'lower' if r['cs'][1] < 0 else ('higher' if r['cs'][0] > 0 else 'ns')
                rows.append([lab] + ([mlab] if with_metric else []) + [DSL[ds], str(r['nsub']), num(r['t']), num(r['c']),
                            f"{sgn(r['d'])} ({pct(r['rel'], sign=True)})", ci(*r['cs']), res, f"{r['imp_sub']}/{r['nsub']}",
                            'same as subject CI' if ds == 'tlio' else ci(*r['cq'])])
    return rows


HEAD_P = ['Test set', 'Subjects', 'Treatment', 'Control', 'Difference (rel.)', 'Subject cluster CI', 'Result', 'Subjects improved', 'Sequence CI']
CMP_TRAIN = [('HN − YawAug', 'ours_hn', 'ours_yaw'), ('HN − GN only', 'ours_hn', 'ours_gn'), ('HN − PCA frame', 'ours_hn', 'ours_pca'),
             ('RoNIN ResNet: Plug-in HN − Unmodified', 'ext_resnet_hn', 'ext_resnet')]
MET3 = [('ate_u', 'Unified-protocol ATE'), ('rte_u', 'Unified-protocol RTE'), ('ate_lit', 'Official-protocol ATE')]


@table('supp_data/unified_paired_ate.csv')
def unified_paired():
    return ['Comparison'] + HEAD_P, paired_rows(CMP_TRAIN, [('ate_u', '')], False)


@table('supp_data/literature_paired_ate.csv')
def literature_paired():
    return ['Comparison'] + HEAD_P, paired_rows(CMP_TRAIN, [('ate_lit', '')], False)


@table('supp_data/testtime_plugin_paired.csv')
def plugin_paired():
    cmp = [('ResNet18-YawAug plug-in HN − ResNet18-YawAug', 'ours_yaw_hn', 'ours_yaw'),
           ('ResNet18-YawAug plug-in HN − training-time HN', 'ours_yaw_hn', 'ours_hn')]
    return ['Comparison', 'Metric'] + HEAD_P, paired_rows(cmp, MET3, True)


@table('supp_data/testtime_fa2_paired.csv')
def fa2_paired():
    cmp = [('ResNet18-YawAug plug-in HN − ResNet18-YawAug FA-2', 'ours_yaw_hn', 'ours_yaw_fa2'),
           ('ResNet18-YawAug FA-2 − ResNet18-YawAug', 'ours_yaw_fa2', 'ours_yaw'),
           ('ResNet18-YawAug FA-2 − training-time HN', 'ours_yaw_fa2', 'ours_hn')]
    return ['Comparison', 'Metric'] + HEAD_P, paired_rows(cmp, MET3, True)


@table('supp_data/testtime_fa2_repeatability.csv')
def fa2_repeatability():
    rows_all = L.rows_of(R / 'fa2_repeat_20260913/rows.json')
    psis = sorted({r['psi_deg'] for r in rows_all})
    head = ['Method', 'Sequences × seeds', 'Per-sequence range, median (m)', 'Per-sequence range, max (m)'] + [f'ψ={p:g}°' for p in psis]
    rows = []
    for lab, m in (('Two-way frame averaging (2 forward passes)', 'ours_yaw_fa2'), ('Plug-in HN (1 forward pass)', 'ours_yaw_hn')):
        by = defaultdict(dict)
        for r in rows_all:
            if r['model'] == m:
                by[(r['seed'], r['sequence'])][r['psi_deg']] = r['ate_lit']
        rng = np.array([max(v.values()) - min(v.values()) for v in by.values()])
        rows.append([lab, str(len(by)), sci(np.median(rng)), sci(rng.max())] + [num(np.mean([v[p] for v in by.values()])) for p in psis])
    return head, rows


@table('supp_data/confirm_anglefair_per_angle.csv')
def anglefair_per_angle():
    head = ['Frozen network, method', 'Dataset', 'Unmodified by ψ = 0°, 45°, …, 315°', 'Test-time method by α']
    rows = []
    for m, t, lab in (('ours_yaw', 'H', 'ResNet18-YawAug, Plug-in HN'), ('ours_yaw', 'F', 'ResNet18-YawAug, FA-2'),
                      ('int_transformer_yaw', 'H', 'Transformer-YawAug, Plug-in HN'), ('int_imunet_yaw', 'H', 'IMUNet-YawAug, Plug-in HN'),
                      ('ext_resnet', 'H', 'RoNIN ResNet (released), Plug-in HN')):
        for ds in ('tlio_c', 'imunet_c'):
            U = [np.mean(list(af(m, ds, f'U{k}').values())) for k in range(8)]
            T = [np.mean(list(af(m, ds, f'{t}{k}').values())) for k in range(8 if t == 'H' else 4)]
            n = af(m, ds, 'U0')
            nsub = len({L.subject(ds, s) for s in n})
            dsl = f'TLIO-confirm ({len(n)})' if ds == 'tlio_c' else f'Phone-confirm ({len(n)}, {nsub} subj.)'
            rows.append([lab, dsl, ' / '.join(f'{x:.3f}' for x in U), ' / '.join(f'{x:.3f}' for x in T)])
    return head, rows
