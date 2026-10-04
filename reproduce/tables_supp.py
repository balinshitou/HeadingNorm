"""Tables S1-S28 of the Supplementary Materials."""
import csv
import json
import re
from collections import defaultdict

import numpy as np

import tables_lib as L
from tables_lib import R, ROOT, Q95, bold, ci, num, pct, pm, sci, sgn, thousands
from tables_main import FRONT, Q_MAIN, Q_SEC, af, diff_rel, meta
from tables_registry import table

DSL = {'ronin_seen': 'RoNIN-seen', 'ronin_unseen': 'RoNIN-unseen', 'ridi': 'RIDI', 'tlio': 'TLIO-test', 'imunet': 'Phone-test',
       'tlio_c': 'TLIO-confirm', 'imunet_c': 'Phone-confirm'}
TEST = ['ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet']
BACKBONES = [('ResNet18 / GN only', 'gn'), ('ResNet18 / YawAug', 'yaw'), ('ResNet18 / PCA frame', 'pca'),
             ('ResNet18 / HN w/o sign', 'nosign'), ('ResNet18 / HN', 'hn'), ('Transformer / YawAug', 'tf_yaw'),
             ('Transformer / HN', 'tf_hn'), ('IMUNet / YawAug', 'imu_yaw'), ('IMUNet / HN', 'imu_hn')]
SEEN_NOTE = 'RoNIN-seen (in-sample at the subject level)'


def pm_s(m, s, d=3):
    return f'{num(m, d)} ± {num(s, d)}'


def tex_sci(x):
    """$2.5\\times10^{-7}$ as written in the manuscript."""
    mant, ex = f'{x:.1e}'.split('e')
    return f'${mant}\\times10^{{{int(ex)}}}$'


def imp_both(r):
    return f"{r['imp_sub']}/{r['nsub']}; {r['imp_seq']}/{r['nseq']}"


# ------------------------------------------------------------------------------------------------------------- S1
@table('S1')
def s1():
    head = ['Backbone / front end', 'RoNIN-unseen, sequence ATE', 'RIDI, sequence ATE', 'RoNIN-seen, subject ATE', 'RoNIN-seen, subject RTE']
    rows = [[lab, num(L.per_seed(k, 'unseen', how='seq').mean()), num(L.per_seed(k, 'ridi', how='seq').mean()),
             pm_s(*L.ms(k, 'seen')), pm_s(*L.ms(k, 'seen', 'rte'))] for lab, k in BACKBONES]
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S2
@table('S2')
def s2():
    m = meta('ResNet18_v3_hn_gn_s0')
    mu, sd = m['gnorm_mu'], m['gnorm_sd']
    assert mu[0] == mu[1] == mu[3] == mu[4] == 0 and sd[0] == sd[1] and sd[3] == sd[4]
    head = ['Training fraction', '$\\mu_{\\omega z}$', '$\\mu_{az}$', '$s_{\\omega xy}$', '$s_{\\omega z}$', '$s_{axy}$', '$s_{az}$']
    return head, [[f"{round(100 * m['data_frac_requested'])}%"] + [num(x, 6) for x in (mu[2], mu[5], sd[0], sd[2], sd[3], sd[5])]]


# ------------------------------------------------------------------------------------------------------------- S3
@table('S3')
def s3():
    import sys
    import torch
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
    from sensors_evaluate import load
    net = load('A6_v3_nopre_s0', torch.device('cpu'))[0]
    grp = defaultdict(int)
    for n, p in net.named_parameters():
        grp[n.split('.')[0]] += p.numel()
    T = 200
    k, st = net.embed.kernel_size[0], net.embed.stride[0]
    d, nl = net.embed.out_channels, len(net.enc.layers)
    lay = net.enc.layers[0]
    nh, dff = lay.self_attn.num_heads, lay.linear1.out_features
    P = (T - k) // st + 1
    h1, h2 = net.head[1], net.head[4]
    assert k == st and grp['enc'] + grp['embed'] + grp['cls'] + grp['head'] == sum(grp.values())
    head = ['Module', 'Output shape', 'Configuration', 'Trainable params']
    rows = [['Input window', f'$B\\times6\\times{T}$', f'$T={T}$, 200 Hz', '—'],
            ['Patch embedding', f'$B\\times{d}\\times{P}$', f'kernel = stride = {k}, $P={P}$', thousands(grp['embed'])],
            ['CLS and positional encoding', f'$B\\times{P + 1}\\times{d}$', 'sinusoidal, not learnable', thousands(grp['cls'])],
            ['Transformer encoder', f'$B\\times{P + 1}\\times{d}$', f'$L={nl}$, $d={d}$, $H={nh}$, $d_{{\\mathrm{{ff}}}}={dff}$',
             thousands(grp['enc'])],
            ['Regression head (CLS only)', '$B\\times2$', f'{h1.in_features}→{h1.out_features}→{h2.out_features}, GELU', thousands(grp['head'])],
            ['**Total**', '', '', bold(thousands(sum(grp.values())))]]
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S4
@table('S4')
def s4():
    j = json.load(open(R / 'data_stats_20261004/phone_test_devices.json'))
    name = {'S10': 'Galaxy S10', 'S21': 'Galaxy S21', 'Tango': 'Tango', 'Xiaomi': 'Xiaomi'}
    head = ['Device', 'Sequences', 'Duration (min)', 'Ground-truth path length (km)']
    rows = [[name[k], str(v['sequences']), num(v['minutes'], 1), num(v['km'], 2)] for k, v in j['devices'].items()]
    t = j['devices'].values()
    rows.append(['Total', str(sum(v['sequences'] for v in t)), num(sum(v['minutes'] for v in t), 1), num(sum(v['km'] for v in t), 2)])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S5
@table('S5')
def s5():
    head = ['Front end', 'RoNIN-unseen ATE', 'RoNIN-unseen RTE', 'RIDI ATE', 'RIDI RTE']
    lab = {'gn': 'GN only', 'yaw': 'YawAug', 'pca': 'PCA frame', 'nosign': 'HN w/o sign', 'hn': 'HN'}
    rows = [[lab[k]] + [num(np.sqrt(2) * L.ms(k, g, key)[0]) for g, key in (('unseen', 'ate'), ('unseen', 'rte'), ('ridi', 'ate'), ('ridi', 'rte'))]
            for _, k in FRONT]
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S6
@table('S6')
def s6():
    m = meta('Sensors_v4_resnet_yaw_s0'); spec = m['spec']
    ntr, nva = len(m['train_files']), len(m['val_files'])
    str_, sva = len({f['subject'] for f in m['train_files']}), len({f['subject'] for f in m['val_files']})
    delta = re.search(r'"--delta", type=float, default=([0-9.]+)', (ROOT / 'src/pkg_train.py').read_text()).group(1)
    wd = float(re.search(r'AdamW\(net\.parameters\(\), lr=spec\[.lr.\], weight_decay=([0-9.e-]+)\)', (ROOT / 'src/sensors_train.py').read_text()).group(1))
    seeds = sorted({meta(f'Sensors_v4_resnet_yaw_s{s}')['spec']['seed'] for s in range(4)})
    head = ['Setting', 'Published RoNIN training setup', 'Our controlled front-end experiments']
    rows = [['Training source', 'Released pretrained checkpoint', f'{ntr} sequences from {str_} subjects'],
            ['Validation', 'Original validation protocol', f'{nva} sequences from {sva} disjoint subjects'],
            ['Loss', 'Per-coordinate MSE', f'Per-coordinate Huber, threshold {delta}'],
            ['Optimizer', 'Adam', f'AdamW, weight decay {wd:.4f}'],
            ['Learning rate', 'Initial 0.0001, reduced on validation plateau', f"OneCycle, maximum {spec['lr']:.3f}"],
            ['Budget', 'Epoch-based, typically about 100 epochs', f"{spec['steps']:,} updates"],
            ['Batch size', '128', str(spec['bs'])],
            ['Window sampling', 'Shuffled strided windows with random shifts', f'Random windows matched across the {len(seeds)} seeds'],
            ['Fixed input normalization', 'No GN', 'Same GN constants for all configurations'],
            ['Yaw augmentation', 'Used in the original training', 'Only as the yaw baseline'],
            ['Main reporting', 'Published sequence aggregates', 'Subject-mean metrics and seed dispersion'],
            ['Role of RIDI', 'Separate published benchmark protocol', 'Transfer from RoNIN without fine-tuning']]
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S7
@table('S7')
def s7():
    off = np.mean([float(r['ate']) for r in L.csv_rows(R / 'supplement_sources_20260909/official_replay.csv')])
    yh = np.array([np.mean([r['ate_lit'] for r in L.rows_of(R / f'unified_eval_20260913/ours_yaw_hn_s{s}.json')
                            if r['dataset'] == 'ronin_unseen']) for s in range(4)])
    hours = f"{num(meta('Sensors_v4_resnet_yaw_s0')['train_hours'], 2)} h"
    head = ['Configuration', 'Training data', 'RoNIN-unseen ATE', 'Relative to official']
    rows = [['RoNIN ResNet (weights released by the original authors)', 'Full training set', num(off), '—']]
    for lab, v in (('ResNet18 + GN + YawAug, plug-in HN at test time', (yh.mean(), yh.std(ddof=1))),
                   ('Transformer + HN + GN', L.ms('tf_hn', 'unseen', how='seq')), ('IMUNet + HN + GN', L.ms('imu_hn', 'unseen', how='seq')),
                   ('ResNet18 + HN + GN', L.ms('hn', 'unseen', how='seq')), ('ResNet18 + GN + YawAug', L.ms('yaw', 'unseen', how='seq')),
                   ('ResNet18 + GN', L.ms('gn', 'unseen', how='seq'))):
        rows.append([lab, hours, pm(*v), pct(100 * (v[0] / off - 1), sign=True)])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S8
@table('S8')
def s8():
    head = ['Model', 'RoNIN-seen', 'RoNIN-unseen', 'RIDI', 'TLIO-test', 'Phone-test']
    models = [('Ours, ResNet18 + GN', 'ours_gn'), ('Ours, ResNet18 + GN + YawAug', 'ours_yaw'),
              ('Ours, ResNet18 + GN + YawAug, plug-in HN at test time (Section 5.4 of the main text)', 'ours_yaw_hn'),
              ('Ours, ResNet18 + GN + PCA frame', 'ours_pca'), ('Ours, ResNet18 + GN + HN', 'ours_hn'),
              ('Ours, Transformer + GN + HN', 'ours_transformer_hn'), ('Ours, IMUNet + GN + HN', 'ours_imunet_hn'),
              ('RoNIN ResNet', 'ext_resnet'), ('RoNIN ResNet + plug-in HN', 'ext_resnet_hn')]
    return head, [[lab] + [num(L.seq_weighted(m, ds, 'ate_u')) for ds in TEST] for lab, m in models]


# ------------------------------------------------------------------------------------------------------------- S9
@table('S9')
def s9():
    head = ['Comparison', 'RoNIN-seen', 'RoNIN-unseen', 'RIDI', 'TLIO-test', 'Phone-test']
    rows = []
    for lab, t, c in (('HN − YawAug', 'ours_hn', 'ours_yaw'), ('HN − GN only', 'ours_hn', 'ours_gn'), ('HN − PCA frame', 'ours_hn', 'ours_pca'),
                      ('RoNIN ResNet: Plug-in HN − Unmodified', 'ext_resnet_hn', 'ext_resnet')):
        row = [lab]
        for ds in TEST:
            r = L.paired(t, c, ds)
            row.append(bold(f"{sgn(r['d'])} {ci(*r['cs'])}", r['cs'][1] < 0 or r['cs'][0] > 0))
        rows.append(row)
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S10
@table('S10')
def s10():
    head = ['Backbone / front end', 'RoNIN-unseen median range', 'RIDI median range', 'RoNIN-seen median range']
    rows = []
    for lab, k in BACKBONES:
        row = [lab]
        for g in ('unseen', 'ridi', 'seen'):
            rg = L.ranges(L.sweep(k), g)
            row.append(sci(np.mean([np.median([a for a, _ in rg[s].values()]) for s in range(4)])))
        rows.append(row)
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S11
def pooled(sw, grp, key):
    rg = L.ranges(sw, grp, key)
    a = np.array([v[0] for s in range(4) for v in rg[s].values()])
    r = np.array([v[0] / v[1] for s in range(4) for v in rg[s].values()])
    return np.median(a), 100 * np.median(r), a.max()


@table('S11')
def s11():
    head = ['Front end', 'RoNIN-unseen, official protocol', 'RoNIN-unseen, optimal rotation', 'RIDI, official protocol', 'RIDI, optimal rotation']
    rows = []
    for lab, k in (('GN only', 'gn'), ('GN + YawAug', 'yaw'), ('GN + PCA frame', 'pca'), ('GN + HN', 'hn')):
        sw = L.sweep(k, 'e1b_yaw_aligned')
        row = [lab]
        for g, key in (('unseen', 'ate'), ('unseen', 'ate_yaw'), ('ridi', 'ate'), ('ridi', 'ate_yaw')):
            med, ratio, mx = pooled(sw, g, key)
            row.append(f'$<10^{{-5}}$ m ({pct(ratio)})' if mx < 1e-5 else f'{num(med)} m ({pct(ratio)})')
        rows.append(row)
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S12
@table('S12')
def s12():
    j = json.load(open(R / 'e6_sign_ablation/analytic_frame_check.json'))
    n = j['rows'][0]['windows']
    head = ['Yaw (°)', f'Without sign: failures / {n}', f'With sign: failures / {n}', 'Modulo-π deviation without sign (rad)']
    rows = [[str(int(r['angle_deg'])), str(r['unsigned_inconsistent']), str(r['signed_inconsistent']), sci(r['unsigned_max_error_mod_pi_rad'])]
            for r in sorted(j['rows'], key=lambda r: r['angle_deg'])]
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S13
@table('S13')
def s13():
    d = json.load(open(R / 'sensors_v4/diagnostics.json'))
    rt = {(r['model'], round(np.degrees(r['angle_rad']))): r for r in d['rotation_tests']}
    head = ['Checkpoint (seed 0)', 'Rotation (°)', 'Windows', 'Mean inconsistency (m/s)', 'Max inconsistency (m/s)']
    rows = []
    for lab, tag in (('ResNet18 / GN only', 'Sensors_v4_resnet_gn_s0'), ('ResNet18 / GN + YawAug', 'Sensors_v4_resnet_yaw_s0'),
                     ('ResNet18 / GN + PCA frame', 'Sensors_v4_resnet_mixed_pca_s0'), ('ResNet18 / GN + HN', 'ResNet18_v3_hn_gn_s0'),
                     ('Transformer / GN + HN', 'A6_v3_nopre_s0'), ('IMUNet / GN + HN', 'IMUNet2024_v3_hn_gn_s0')):
        for ang in (30, 90, 180, 300):
            r = rt[(tag, ang)]
            rows.append([lab, str(ang), str(r['windows']), sci(r['mean_error_mps']), sci(r['maximum_error_mps'])])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S14
@table('S14')
def s14():
    head = ['Test set', 'Unmodified', 'Plug-in HN', 'Difference [subject CI]', 'Sequence CI', 'Subjects improved']
    rows = []
    for ds, lab in (('ronin_unseen', 'RoNIN-unseen'), ('ronin_seen', SEEN_NOTE), ('ridi', 'RIDI'), ('tlio', 'TLIO-test'), ('imunet', 'Phone-test')):
        r = L.paired('ours_yaw_hn', 'ours_yaw', ds)
        sig = r['cs'][1] < 0 and (ds != 'imunet' or r['cq'][1] < 0)
        rows.append([lab, num(r['c']), num(r['t']), f"{bold(sgn(r['d']) + ' ' + ci(*r['cs']), sig)} ({pct(r['rel'], sign=True)})",
                     'same' if ds == 'tlio' else ci(*r['cq']), f"{r['imp_sub']}/{r['nsub']}"])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S15
def repeat(path, model):
    rows = [r for r in L.rows_of(R / path) if r['model'] == model]
    by = defaultdict(dict)
    for r in rows:
        by[(r['seed'], r['sequence'])][r['psi_deg']] = r['ate_lit']
    rng = np.array([max(v.values()) - min(v.values()) for v in by.values()])
    ratio = np.array([(max(v.values()) - min(v.values())) / v[0.0] for v in by.values()])
    return np.median(rng), 100 * np.median(ratio)


@table('S15')
def s15():
    head = ['Method', 'Forward passes per window', 'RoNIN-seen', 'RoNIN-unseen', 'RIDI', 'TLIO-test', 'Phone-test',
            'Per-sequence range under arbitrary headings']
    rows = []
    for lab, m, npass, rep in (('Unmodified', 'ours_yaw', 1, ('tta_repeat_20260913/rows.json', 'ours_yaw')),
                               ('Plug-in HN', 'ours_yaw_hn', 1, ('tta_repeat_20260913/rows.json', 'ours_yaw_hn')),
                               ('Two-way frame averaging', 'ours_yaw_fa2', 2, ('fa2_repeat_20260913/rows.json', 'ours_yaw_fa2'))):
        vals = [num(L.paired('ours_yaw_hn', 'ours_yaw', ds)['c'] if m == 'ours_yaw' else L.paired(m, 'ours_yaw', ds)['t']) for ds in TEST]
        med, ratio = repeat(*rep)
        rows.append([lab, str(npass)] + vals + [f"{num(med) if med > 1e-5 else tex_sci(med)} m ({pct(ratio)})"])
    row = ['Plug-in HN − two-way frame averaging', '']
    for ds in TEST:
        r = L.paired('ours_yaw_hn', 'ours_yaw_fa2', ds)
        sig = (r['cs'][1] < 0 or r['cs'][0] > 0) and (ds != 'imunet' or r['cq'][1] < 0 or r['cq'][0] > 0)
        row.append(bold(f"{sgn(r['d'])} {ci(*r['cs'])}", sig))
    rows.append(row + [''])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S16
@table('S16')
def s16():
    head = ['Model', 'Dataset', 'Metric', 'Plug-in HN', 'Unmodified', 'Difference (rel.)', 'Subject CI', 'Sequence CI',
            'Improved (subjects; sequences)']
    order = [('Transformer-YawAug', 'int_transformer_yaw', 'tlio_c', 'ate_u', 'main'), ('IMUNet-YawAug', 'int_imunet_yaw', 'tlio_c', 'ate_u', 'main')]
    for lab, base in (('Transformer-YawAug', 'int_transformer_yaw'), ('IMUNet-YawAug', 'int_imunet_yaw')):
        order.append((lab, base, 'tlio_c', 'ate_shape', ''))
        for ds in ('imunet', 'imunet_c', 'ronin_seen', 'ronin_unseen', 'ridi', 'tlio'):
            order += [(lab, base, ds, 'ate_u', ''), (lab, base, ds, 'ate_shape', '')]
    rows = []
    for lab, base, ds, metric, tag in order:
        r = L.paired(base + '_hn', base, ds, metric, Q_SEC if tag == 'main' else Q95)
        dl = {'tlio_c': 'TLIO-confirm (main)' if tag else 'TLIO-confirm', 'ronin_seen': SEEN_NOTE}.get(ds, DSL[ds])
        rows.append([lab, dl, 'ATE' if metric == 'ate_u' else 'Shape error', num(r['t']), num(r['c']), diff_rel(r),
                     '—' if ds.startswith('tlio') else ci(*r['cs']), ci(*r['cq']), imp_both(r)])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S17
@table('S17')
def s17():
    head = ['Network', 'Test set', 'Seqs / subjects', 'Unmodified, ψ=0', 'Unmodified, mean over 8 ψ', 'Plug-in HN, α=0',
            'Plug-in HN, mean over 8 α', 'Mean-α HN − mean-ψ Unmodified [CI]', 'Seqs improved (mean vs. mean)', 'α=0 − mean-α HN [95% CI]']
    lv_main = 1 - 0.05 / 3                      # Bonferroni over the three primary test sets
    primary = ('ronin_unseen', 'ridi', 'tlio')
    smean = lambda ds, v: L.compare(v, v, ds)['c']

    def cmp_txt(r, lv):
        sig = r['cs'][1] < 0 or r['cs'][0] > 0
        return bold(f"{sgn(r['d'])} ({pct(r['rel'], sign=True)}) {ci(*r['cs'])}", sig), f'({100 * lv:.2f}%)'

    rows = []
    rr = L.rows_of(R / 'hn_plugin_p0_20260912/ext_resnet.json')
    for ds in ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio'):
        x = [r for r in rr if r['dataset'] == ds]
        U = {r['sequence']: np.array(r['orig_ate_by_psi']) for r in x}
        H = {r['sequence']: np.array(r['hn_ate_by_alpha']) for r in x}
        U0, Ub = {s: v[0] for s, v in U.items()}, {s: v.mean() for s, v in U.items()}
        H0, Hb = {s: v[0] for s, v in H.items()}, {s: v.mean() for s, v in H.items()}
        lv = lv_main if ds in primary else 0.95
        fair = L.compare(Hb, Ub, ds, L.level_q(lv)); conv = L.compare(H0, Hb, ds)
        a, b = cmp_txt(fair, lv)
        nsub = '—' if ds == 'tlio' else str(fair['nsub'])
        rows.append(['RoNIN ResNet (released)', DSL[ds], f'{len(x)} / {nsub}', num(smean(ds, U0)), num(smean(ds, Ub)), num(smean(ds, H0)),
                     num(smean(ds, Hb)), f'{a} {b}', f"{sum(Hb[s] < Ub[s] for s in Ub)}/{len(x)}", cmp_txt(conv, 0.95)[0]])
    U, H = defaultdict(lambda: np.zeros((4, 8))), defaultdict(lambda: np.zeros(4))
    for s in range(4):
        for r in L.rows_of(R / f'e1_heading_repeatability/Sensors_v4_resnet_yaw_s{s}.json'):
            ds = 'ridi' if r['dataset'] == 'ridi' else f"ronin_{r['split']}"
            U[(ds, r['seq'])][s, r['angle_index']] = r['ate']
        for r in L.rows_of(R / f'unified_eval_20260913/ours_yaw_hn_s{s}.json'):
            if r['dataset'] in ('ronin_seen', 'ronin_unseen', 'ridi'):
                H[(r['dataset'], r['sequence'])][s] = r['ate_lit']
    for ds in ('ronin_seen', 'ronin_unseen', 'ridi'):
        ks = sorted(k for k in U if k[0] == ds)
        U0 = {k[1]: U[k][:, 0].mean() for k in ks}; Ub = {k[1]: U[k].mean() for k in ks}; H0 = {k[1]: H[k].mean() for k in ks}
        r = L.compare(H0, Ub, ds)
        a, b = cmp_txt(r, 0.95)
        rows.append(['ResNet18-YawAug (ours, 4 seeds)', DSL[ds], f"{len(ks)} / {r['nsub']}", num(smean(ds, U0)), num(smean(ds, Ub)),
                     num(smean(ds, H0)), 'not run', f'α=0: {a} {b}', f"{sum(H0[s] < Ub[s] for s in Ub)}/{len(ks)}", '—'])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S18-S20
CMPL = {('ours_yaw_hn', 'ours_yaw'): 'ResNet18-YawAug plug-in HN − ResNet18-YawAug', ('ours_yaw_fa2', 'ours_yaw'): 'ResNet18-YawAug FA-2 − ResNet18-YawAug',
        ('ext_resnet_hn', 'ext_resnet'): 'RoNIN ResNet plug-in HN − RoNIN ResNet', ('ours_hn', 'ours_pca'): 'Training-time HN − PCA frame',
        ('ours_hn', 'ours_yaw'): 'Training-time HN − ResNet18-YawAug', ('ours_yaw_hn', 'ours_yaw_fa2'): 'ResNet18-YawAug plug-in HN − ResNet18-YawAug FA-2',
        ('ours_yaw_fa2', 'ours_hn'): 'ResNet18-YawAug FA-2 − training-time HN', ('ours_hn', 'ours_gn'): 'Training-time HN − GN only'}
MARGIN = 0.02


@table('S18')
def s18():
    head = ['Hypothesis', 'Comparison', 'Dataset', 'Treatment', 'Control', 'Difference (rel.)', 'Subject CI', 'Sequence CI',
            'Improved (subjects; sequences)', 'Decision']
    rows = []
    for hyp, t, c, kind in (('H1a', 'ours_yaw_hn', 'ours_yaw', 'sup'), ('H1b', 'ours_yaw_fa2', 'ours_yaw', 'sup'),
                            ('H2', 'ext_resnet_hn', 'ext_resnet', 'noninf'), ('Training-time front end', 'ours_hn', 'ours_pca', 'sup')):
        for ds in ('tlio_c', 'imunet_c'):
            r = L.paired(t, c, ds, 'ate_u', Q_MAIN)
            his = [r['cq'][1]] + ([r['cs'][1]] if ds == 'imunet_c' else [])
            if kind == 'sup':
                dec = 'Supported' if all(h < 0 for h in his) else 'Not supported'
            else:
                dec = 'Non-inferior (supported)' if all(h < MARGIN * r['c'] for h in his) else 'Non-inferiority not demonstrated'
            rows.append([hyp, CMPL[(t, c)], DSL[ds], num(r['t']), num(r['c']), diff_rel(r), ci(*(r['cq'] if ds == 'tlio_c' else r['cs'])),
                         ci(*r['cq']), imp_both(r), dec])
    return head, rows


MET = [('ate_u', 'Unified-protocol ATE'), ('rte_u', 'Unified-protocol RTE'), ('ate_lit', 'Official-protocol ATE')]


@table('S19')
def s19():
    head = ['Comparison', 'Dataset', 'Metric', 'Treatment', 'Control', 'Difference (rel.)', 'Subject CI', 'Sequence CI', 'Improved']
    spec = [(p, MET) for p in (('ours_hn', 'ours_yaw'), ('ours_yaw_hn', 'ours_yaw_fa2'), ('ours_yaw_fa2', 'ours_hn'), ('ours_hn', 'ours_gn'))]
    spec += [(p, MET[1:]) for p in (('ours_yaw_hn', 'ours_yaw'), ('ours_yaw_fa2', 'ours_yaw'), ('ext_resnet_hn', 'ext_resnet'), ('ours_hn', 'ours_pca'))]
    rows = []
    for (t, c), mets in spec:
        for ds in ('tlio_c', 'imunet_c'):
            for metric, mlab in mets:
                r = L.paired(t, c, ds, metric)
                imp = f"{r['imp_seq']}/{r['nseq']}" if ds == 'tlio_c' else f"{r['imp_sub']}/{r['nsub']}"
                rows.append([CMPL[(t, c)], DSL[ds], mlab, num(r['t']), num(r['c']), diff_rel(r),
                             ci(*(r['cq'] if ds == 'tlio_c' else r['cs'])), ci(*r['cq']), imp])
    return head, rows


@table('S20')
def s20():
    models = ['ours_yaw_hn', 'ours_yaw_fa2', 'ours_yaw', 'ours_hn', 'ours_pca']
    sm = {m: L.seqmeans(L.cells(), m, 'imunet_c', 'ate_u') for m in models}
    head = ['Device (sequences)', 'ResNet18-YawAug plug-in HN', 'ResNet18-YawAug FA-2', 'ResNet18-YawAug (unmodified)', 'Training-time HN', 'PCA frame']
    rows = []
    for dev, name in (('S10', 'Galaxy S10'), ('S21', 'Galaxy S21'), ('Tango', 'Tango'), ('Xiaomi', 'Xiaomi')):
        seqs = [s for s in sm['ours_yaw'] if re.search(rf'_{dev}(_|$)', s)]
        rows.append([f'{name} ({len(seqs)})'] + [num(np.mean([sm[m][s] for s in seqs])) for m in models])
    assert sum(int(re.search(r'\((\d+)\)', r[0]).group(1)) for r in rows) == len(sm['ours_yaw'])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S21
@table('S21')
def s21():
    head = ['Frozen network, test-time method', 'Dataset', 'Comparison', 'Treatment', 'Control', 'Difference (rel.)', 'Subject CI',
            'Sequence CI', 'Improved seqs']
    rows = []
    for m, t, lab in (('ours_yaw', 'H', 'ResNet18-YawAug, Plug-in HN'), ('ours_yaw', 'F', 'ResNet18-YawAug, FA-2'),
                      ('int_transformer_yaw', 'H', 'Transformer-YawAug, Plug-in HN'), ('int_imunet_yaw', 'H', 'IMUNet-YawAug, Plug-in HN'),
                      ('ext_resnet', 'H', 'RoNIN ResNet (released), Plug-in HN')):
        for ds in ('tlio_c', 'imunet_c'):
            nA = 8 if t == 'H' else 4
            per = [af(m, ds, f'{t}{k}') for k in range(nA)]
            worst = int(np.argmax([np.mean(list(p.values())) for p in per]))
            Ub, U0, Tb, T0 = af(m, ds, 'Ubar'), af(m, ds, 'U0'), af(m, ds, f'{t}bar'), per[0]
            items = [('Fixed: T(α=0) − U(ψ=0)', T0, U0), ('T(α=0) − Ū', T0, Ub), ('T̄ − Ū (primary)', Tb, Ub), ('Worst α − Ū', per[worst], Ub),
                     ('U(ψ=0) − Ū', U0, Ub), ('T(α=0) − T̄', T0, Tb)]
            items += [(f'T̄ − Ū, {ql}', af(m, ds, f'{t}bar', q), af(m, ds, 'Ubar', q))
                      for q, ql in (('rte_u', 'RTE'), ('ate_lit', 'official-protocol ATE'), ('ate_shape', 'shape error'))]
            for name, a, b in items:
                r = L.compare(a, b, ds)
                dsl = f"TLIO-confirm ({r['nseq']})" if ds == 'tlio_c' else f"Phone-confirm ({r['nseq']}, {r['nsub']} subj.)"
                rows.append([lab, dsl, name, num(r['t']), num(r['c']), diff_rel(r), '—' if ds == 'tlio_c' else ci(*r['cs']), ci(*r['cq']),
                             f"{r['imp_seq']}/{r['nseq']}"])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S22
def composition(m, ds):
    cl = L.cells()[(m, ds)]
    keys = ('ate_u', 'tlr', 'eps_s', 'eps_theta_deg', 'ate_shape', 'ate_scale_only', 'ate_heading_only')
    rows = [{k: np.mean([r[k] for r in v.values()]) for k in keys} for v in cl.values()]
    g = lambda k: np.array([r[k] for r in rows])
    au = g('ate_u').mean()
    return (np.median(g('eps_s')), np.median(np.abs(g('eps_theta_deg'))), g('tlr').mean(), -100 * (1 - g('ate_scale_only').mean() / au),
            -100 * (1 - g('ate_heading_only').mean() / au), -100 * (1 - g('ate_shape').mean() / au))


@table('S22')
def s22():
    head = ['Model', 'Test set', 'Scale error $\\varepsilon_s$', 'Yaw error $\\lvert\\varepsilon_\\theta\\rvert$', 'TLR', 'Scale-only bound',
            'Heading-only bound', 'Complex-gain bound']
    rows = []
    for lab, m in (('Ours, ResNet18 + GN + HN', 'ours_hn'), ('RoNIN ResNet', 'ext_resnet')):
        for k, ds in enumerate(TEST):
            e, th, tlr, rs, rh, rg = composition(m, ds)
            rows.append([lab if k == 0 else '', DSL[ds], sgn(e), f'{th:.1f}°', num(tlr), pct(rs), pct(rh), pct(rg)])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S23
@table('S23')
def s23():
    head = ['Comparison', 'Dataset', 'Plug-in HN', 'Unmodified', 'Difference (rel.)', 'Interval used for the decision',
            'Improved (subjects; sequences)', 'Result', 'Unified-protocol ATE difference for the same comparison']
    rows = []
    for lab, t, c in (('ResNet18-YawAug', 'ours_yaw_hn', 'ours_yaw'), ('RoNIN ResNet', 'ext_resnet_hn', 'ext_resnet')):
        for ds, kind in (('ridi', 'Subject'), ('tlio', 'Sequence'), ('tlio_c', 'Sequence'), ('imunet', 'Sequence')):
            r = L.paired(t, c, ds, 'ate_shape', Q_MAIN)
            iv = r['cs'] if kind == 'Subject' else r['cq']
            res = '**Significant decrease**' if iv[1] < 0 else ('**Significant increase**' if iv[0] > 0 else 'Not significant')
            a = L.paired(t, c, ds)
            rows.append([lab, DSL[ds], num(r['t']), num(r['c']), diff_rel(r), f'{kind} {ci(*iv)}', imp_both(r), res, diff_rel(a)])
    return head, rows


# ------------------------------------------------------------------------------------------------------------- S24-S28
@table('S24')
def s24():
    head = ['Front end', 'RoNIN-unseen ATE', 'RIDI ATE', 'RoNIN-seen ATE']
    v = {k: [L.ms(k, g) for g in ('unseen', 'ridi', 'seen')] for _, k in FRONT}
    lab = {'nosign': 'GN + HN w/o sign (Section 5.3 of the main text)'}
    rows = [[lab.get(k, n)] + [bold(pm(*v[k][j]), min(v, key=lambda kk: v[kk][j][0]) == k) for j in range(3)] for n, k in FRONT]
    return head, rows


@table('S25')
def s25():
    head = ['Test group', 'Baseline', 'HN − baseline ATE (m)', 'Conditional 95% CI']
    rows = []
    for c, lab in (('yaw', 'GN + YawAug'), ('pca', 'GN + PCA frame'), ('nosign', 'GN + HN w/o sign')):
        for g, gl in (('unseen', 'RoNIN-unseen'), ('ridi', 'RIDI')):
            d, (lo, hi), _, _ = L.paired_subj('hn', c, g)
            rows.append([gl, lab, sgn(d), ci(lo, hi)])
    return head, rows


@table('S26')
def s26():
    head = ['Configuration', 'RoNIN-unseen', 'RIDI', 'RoNIN-seen']
    return head, [[lab] + [pm(*L.ms(k, g)) for g in ('unseen', 'ridi', 'seen')]
                  for lab, k in (('No GN, no HN', 'raw'), ('GN only', 'gn'), ('GN + HN', 'hn'))]


@table('S27')
def s27():
    head = ['Backbone', 'Parameters', 'RoNIN-unseen', 'RIDI', 'RoNIN-seen']
    bb = [('Transformer', 'tf_hn', 'A6_v3_nopre_s0'), ('ResNet18', 'hn', 'ResNet18_v3_hn_gn_s0'), ('IMUNet', 'imu_hn', 'IMUNet2024_v3_hn_gn_s0')]
    v = {k: [L.ms(k, g) for g in ('unseen', 'ridi', 'seen')] for _, k, _ in bb}
    return head, [[lab, thousands(meta(tag)['parameters'])] + [bold(pm(*v[k][j]), min(v, key=lambda kk: v[kk][j][0]) == k) for j in range(3)]
                  for lab, k, tag in bb]


@table('S28')
def s28():
    d = json.load(open(R / 'sensors_v4/diagnostics.json'))
    t = {(r['model'], r['device']): r for r in d['timings'] if r['batch'] == 1}
    head = ['HN backbone', 'Parameters', 'CPU median / P95', 'MPS median / P95']
    rows = []
    for lab, tag in (('ResNet18', 'ResNet18_v3_hn_gn_s0'), ('Transformer', 'A6_v3_nopre_s0'), ('IMUNet', 'IMUNet2024_v3_hn_gn_s0')):
        c, g = t[(tag, 'cpu')], t[(tag, 'mps')]
        assert c['parameters'] == meta(tag)['parameters']
        rows.append([lab, thousands(c['parameters']), f"{num(c['median_ms'])} / {num(c['p95_ms'])}", f"{num(g['median_ms'])} / {num(g['p95_ms'])}"])
    return head, rows
