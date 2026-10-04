"""Tables 1-9 of the main text."""
import json
from collections import defaultdict

import numpy as np

import tables_lib as L
from tables_lib import R, ROOT, ci, num, pct, pm, sci, sgn, bold, thousands
from tables_registry import table

META = {}


def meta(tag):
    """Training metadata stored next to each checkpoint (split files, hours, parameters, GN constants, settings)."""
    if tag not in META:
        for d in ('sensors_v4', 'submission_frozen', 'e3_backbone_frontend', 'e6_sign_ablation'):
            p = ROOT / f'models/{d}/{tag}.json'
            if p.exists():
                META[tag] = json.load(open(p))
                break
    return META[tag]


# --------------------------------------------------------------------------------------------------------- Table 1
@table('1')
def t1():
    st = json.load(open(R / 'data_stats_20260929/stats.json'))
    m = meta('Sensors_v4_resnet_yaw_s0')
    tr, va = {f['subject'] for f in m['train_files']}, {f['subject'] for f in m['val_files']}
    assert not tr & va and abs(m['train_hours'] - st['ronin_train_hours']) < 1e-9 and abs(m['val_hours'] - st['ronin_val_hours']) < 1e-9
    seen = st['ronin_seen']['per_group']
    n_tr = sum(n for g, n in seen.items() if g in tr); n_va = sum(n for g, n in seen.items() if g in va)
    assert n_tr + n_va == st['ronin_seen']['sequences']
    assert not set(st['ronin_unseen']['per_group']) & (tr | va)
    assert set(st['phone_confirm']['per_group']) == set(st['phone_test']['per_group'])
    h = lambda k: num(st[k]['hours'], 2)
    head = ['Dataset', 'Source split', 'Sequences', 'Subjects', 'Duration (h)', 'Use']
    blank = lambda t: [f'**{t}**', '', '', '', '', '']
    rows = [blank('Training data'),
            ['RoNIN-train', 'RoNIN official train/val, re-split by subject', str(len(m['train_files'])), str(len(tr)),
             num(st['ronin_train_hours'], 2), 'Training; GN constants'],
            ['RoNIN-val', 'RoNIN official train/val, re-split by subject', str(len(m['val_files'])), f'{len(va)}, disjoint from training',
             num(st['ronin_val_hours'], 2), 'Checkpoint selection'],
            blank('Test sets'),
            ['RoNIN-seen', 'RoNIN official test, seen', str(st['ronin_seen']['sequences']),
             f"{st['ronin_seen']['groups']} ({n_tr} seqs from training subjects, {n_va} from validation subjects)", h('ronin_seen'),
             'Descriptive only (within-subject)'],
            ['RoNIN-unseen', 'RoNIN official test, unseen', str(st['ronin_unseen']['sequences']),
             f"{st['ronin_unseen']['groups']}, disjoint from training and validation", h('ronin_unseen'), 'Main cross-subject test'],
            ['RIDI', 'All sequences', str(st['ridi']['sequences']), f"{st['ridi']['groups']} groups by collector name", h('ridi'),
             'Cross-dataset test'],
            ['TLIO-test', 'TLIO official test', str(st['tlio_test']['sequences']), 'not released (counted per sequence)', h('tlio_test'),
             'Cross-device test (head-mounted)'],
            ['Phone-test', 'IMUNet phone data, test split', str(st['phone_test']['sequences']), str(st['phone_test']['groups']),
             h('phone_test'), 'Cross-device phone test'],
            blank('Confirmation sets'),
            ['TLIO-confirm', 'TLIO official train + val', str(st['tlio_confirm']['sequences']), 'not released (counted per sequence)',
             h('tlio_confirm'), 'Confirmatory test'],
            ['Phone-confirm', 'IMUNet phone data, training split', str(st['phone_confirm']['sequences']),
             f"the same {st['phone_confirm']['groups']} as Phone-test", h('phone_confirm'), 'Confirmatory test; direction only']]
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 2
@table('2')
def t2():
    p = {k: thousands(meta(t)['parameters']) for k, t in (('resnet', 'ResNet18_v3_hn_gn_s0'), ('tf', 'A6_v3_nopre_s0'),
                                                          ('imu', 'IMUNet2024_v3_hn_gn_s0'))}
    for t in ('Sensors_v4_resnet_gn_s0', 'Sensors_v4_resnet_yaw_s0', 'Sensors_v4_resnet_mixed_pca_s0', 'E6_resnet_heading_nosign_s0'):
        assert thousands(meta(t)['parameters']) == p['resnet'], t
    assert thousands(meta('E3_transformer_yaw_s0')['parameters']) == p['tf']
    assert thousands(meta('E3_imunet_yaw_s0')['parameters']) == p['imu']
    head = ['Provenance', 'Network', 'Orientation handling in training', 'Input normalization', 'Trainable params',
            'Experiments without plug-in', 'Plug-in experiments']
    rows = [['Trained by us', 'ResNet18', 'GN only / YawAug / PCA frame / HN w/o sign / HN', 'GN', p['resnet'],
             'Sensitivity; sign check; front-end comparison', 'ResNet18-YawAug: Plug-in HN, FA-2'],
            ['Trained by us', 'Transformer', 'YawAug / HN', 'GN', p['tf'], 'Invariance check (HN version)', 'Transformer-YawAug: Plug-in HN'],
            ['Trained by us', 'IMUNet', 'YawAug / HN', 'GN', p['imu'], 'Invariance check (HN version)', 'IMUNet-YawAug: Plug-in HN'],
            ['Released', 'RoNIN ResNet (per-window)', "Authors' setting", 'No GN', '—', 'Sensitivity', 'Plug-in HN'],
            ['Released', 'RoNIN LSTM, RoNIN TCN (sequence)', "Authors' setting", 'No GN', '—', 'Sensitivity',
             'Not applicable (sequence networks)']]
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 3
FRONT = [('GN only', 'gn'), ('GN + YawAug', 'yaw'), ('GN + PCA frame', 'pca'), ('GN + HN w/o sign', 'nosign'), ('GN + HN', 'hn')]


def repeatability(k, grp):
    """Subject-mean ATE per heading and seed: its range is averaged over seeds and divided by the seed-mean ATE at psi = 0;
    the worst heading is taken per seed and averaged. Per-sequence ranges are pooled over all (seed, sequence) pairs."""
    sw = L.sweep(k)
    curve = np.array([[L.subj_macro({q: {'ate': v[a]['ate'], 'subject': v[a]['subject']} for q, v in sw[s][grp].items()})
                       for a in range(8)] for s in range(4)])                        # seed x heading
    rng_ = (curve.max(1) - curve.min(1)).mean()
    base = curve[:, 0].mean()
    rel, worst = 100 * rng_ / base, 100 * (curve.max(1).mean() / base - 1)       # tools/e1_analyze.py definitions
    rg = L.ranges(sw, grp)
    a = np.array([v[0] for s in range(4) for v in rg[s].values()])
    r = np.array([v[0] / v[1] for s in range(4) for v in rg[s].values()])
    return rng_, rel, worst, np.median(a), 100 * np.median(r), a.max()


@table('3')
def t3():
    head = ['Training front-end', 'Range of subject-mean ATE (m)', 'Range / ATE', 'Worst angle vs. $\\psi=0$',
            'Per-seq. range, median (m)', 'Per-seq. range / ATE, median', 'Per-seq. range, max (m)']
    rows = []
    for grp, lab in (('unseen', 'RoNIN-unseen'), ('ridi', 'RIDI')):
        rows.append([f'**{lab}**'] + [''] * 6)
        for name, k in FRONT:
            a, b, c, d, e, f = repeatability(k, grp)
            if k == 'hn':
                rows.append([name] + [bold(x) for x in (num(a, 4), pct(b), pct(c, sign=True), sci(d, 1), pct(e), sci(f, 1))])
            else:
                rows.append([name, num(a), pct(b), pct(c, sign=True), num(d), pct(e), num(f, 2)])
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 4
@table('4')
def t4():
    head = ['Released network', 'RoNIN-seen', 'RoNIN-unseen', 'RIDI', 'TLIO-test']
    rows = []
    for f, lab in (('ext_resnet', 'RoNIN ResNet (per-window)'), ('ext_lstm', 'RoNIN LSTM (sequence)'), ('ext_tcn', 'RoNIN TCN (sequence)')):
        rr = L.rows_of(R / f'hn_plugin_p0_20260912/{f}.json')
        row = [lab]
        for ds in ('ronin_seen', 'ronin_unseen', 'ridi', 'tlio'):
            U = np.array([r['orig_ate_by_psi'] for r in rr if r['dataset'] == ds])
            assert U.shape[1] == 8
            row.append(pct(100 * np.median((U.max(1) - U.min(1)) / U[:, 0])))
        rows.append(row)
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 5
@table('5')
def t5():
    j = json.load(open(R / 'e6_sign_ablation/analytic_frame_check.json'))
    rr = sorted(j['rows'], key=lambda r: r['angle_deg'])
    head = ['Yaw $\\psi$'] + [f"{int(r['angle_deg'])}°" for r in rr] + ['Total']
    rows = []
    for lab, key in (('HN (with third-moment sign)', 'signed_inconsistent'), ('HN w/o sign', 'unsigned_inconsistent')):
        rows.append([lab] + [f"{r[key]}/{r['windows']}" for r in rr]
                    + [bold(f"{sum(r[key] for r in rr)}/{sum(r['windows'] for r in rr)}")])
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 6
@table('6')
def t6():
    head = ['Training front-end', 'RoNIN-unseen ATE', 'RIDI ATE', '(GN + HN) − this row, RoNIN-unseen', '(GN + HN) − this row, RIDI']
    means = {k: [L.ms(k, g) for g in ('unseen', 'ridi')] for _, k in FRONT}
    rows = []
    for name, k in FRONT:
        best = [min(means, key=lambda kk: means[kk][j][0]) == k for j in range(2)]
        row = [name] + [bold(pm(*means[k][j]), best[j]) for j in range(2)]
        for g in ('unseen', 'ridi'):
            if k in ('gn', 'hn'):
                row.append('—')
            else:
                d, (lo, hi), _, _ = L.paired_subj('hn', k, g)
                row.append(bold(f'{sgn(d)} {ci(lo, hi)}', hi < 0 or lo > 0))
        rows.append(row)
    return head, rows


# --------------------------------------------------------------------------------------------------------- Table 7
TEST = ['ronin_seen', 'ronin_unseen', 'ridi', 'tlio', 'imunet']
CONF = ['tlio_c', 'imunet_c']


def excludes_zero(r, ds):
    """TLIO: sequence interval; phone data: subject and sequence intervals both; otherwise the subject interval."""
    if ds.startswith('tlio'):
        lo, hi = r['cq']
    elif ds.startswith('imunet'):
        lo, hi = max(r['cs'][0], r['cq'][0]), min(r['cs'][1], r['cq'][1])
    else:
        lo, hi = r['cs']
    return hi < 0 or lo > 0, lo, hi


@table('7')
def t7():
    n_tc = len(L.seqmeans(L.cells(), 'ours_yaw', 'tlio_c', 'ate_u')); n_pc = len(L.seqmeans(L.cells(), 'ours_yaw', 'imunet_c', 'ate_u'))
    head = ['Frozen network', 'Test-time method', 'RoNIN-seen', 'RoNIN-unseen', 'RIDI', 'TLIO-test', 'Phone-test',
            f'**TLIO-confirm ({n_tc})**', f'**Phone-confirm ({n_pc})**']
    rows = []
    blocks = [('Trained by us', [('ResNet18-YawAug', 'ours_yaw', [('Plug-in HN', 'ours_yaw_hn'), ('FA-2', 'ours_yaw_fa2')]),
                                 ('Transformer-YawAug', 'int_transformer_yaw', [('Plug-in HN', 'int_transformer_yaw_hn')]),
                                 ('IMUNet-YawAug', 'int_imunet_yaw', [('Plug-in HN', 'int_imunet_yaw_hn')])]),
              ('Released by original authors', [('RoNIN ResNet', 'ext_resnet', [('Plug-in HN', 'ext_resnet_hn')])])]
    for grp, nets in blocks:
        rows.append([f'**{grp}**'] + [''] * 8)
        for lab, base, methods in nets:
            first = L.paired(methods[0][1], base, TEST[0])
            rows.append([lab, 'Unmodified ATE (m)'] + [num(L.paired(methods[0][1], base, ds)['c']) for ds in TEST + CONF])
            assert first
            for mlab, t in methods:
                cells = []
                for ds in TEST + CONF:
                    r = L.paired(t, base, ds)
                    cells.append(pct(r['rel'], sign=True) + ('\\*' if excludes_zero(r, ds)[0] else ''))
                rows.append(['', mlab] + cells)
    return head, rows


# ------------------------------------------------------------------------------------------------ Tables 8, 9 (V25)
AF = R / 'confirm_anglefair_20260927'
AF_MODELS = {'ours_yaw': 4, 'ours_pca': 4, 'ext_resnet': 1, 'int_transformer_yaw': 4, 'int_imunet_yaw': 4}
_AF = {}


def af(m, ds, kind, q='ate_u'):
    """Per-sequence endpoint on the confirmation sets (seeds averaged): 'Ubar'/'Hbar'/'Fbar' average over the headings
    or conventions; 'U0', 'H3', ... a single one."""
    if m not in _AF:
        ns = AF_MODELS[m]
        files = [AF / f'{m}.json'] if ns == 1 else [AF / f'{m}_s{s}.json' for s in range(ns)]
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for f in files:
            d = json.load(open(f))
            assert d.get('complete')
            for r in d['rows']:
                for var, mm in r['v'].items():
                    for k, v in mm.items():
                        acc[r['dataset']][r['sequence']][(var, k)].append(v)
        _AF[m] = {ds: {s: {k: float(np.mean(v)) for k, v in dd.items()} for s, dd in seqs.items()} for ds, seqs in acc.items()}
    n = {'U': 8, 'H': 8, 'F': 4}
    out = {}
    for s, d in _AF[m][ds].items():
        out[s] = float(np.mean([d[(f'{kind[0]}{k}', q)] for k in range(n[kind[0]])])) if kind.endswith('bar') else d[(kind, q)]
    return out


Q_MAIN = L.level_q(1 - 0.05 / 8)          # 99.375%: 8 tests of the main family
Q_SEC = L.level_q(0.975)                  # 97.5%: 2 cross-architecture tests


def ds_label(ds, r):
    return f"TLIO-confirm ({r['nseq']})" if ds == 'tlio_c' else f"Phone-confirm ({r['nseq']}, {r['nsub']} subj.)"


def diff_rel(r, d=3):
    return f"{sgn(r['d'], d)} ({pct(r['rel'], sign=True)})"


@table('8')
def t8():
    head = ['Hypothesis, frozen network: method', 'Dataset', 'Unmodified (8 ψ)', 'Treatment (8 α)', 'Difference (rel.)', 'Subject CI',
            'Sequence CI', 'Improved', 'Result']
    spec = [('Trained by us', [('H1a, ResNet18-YawAug: Plug-in HN', 'ours_yaw', 'Hbar', CONF, Q_MAIN),
                               ('H1b, ResNet18-YawAug: FA-2', 'ours_yaw', 'Fbar', CONF, Q_MAIN),
                               ('Cross-arch., Transformer-YawAug: Plug-in HN', 'int_transformer_yaw', 'Hbar', ['tlio_c'], Q_SEC),
                               ('Cross-arch., IMUNet-YawAug: Plug-in HN', 'int_imunet_yaw', 'Hbar', ['tlio_c'], Q_SEC)]),
            ('Released by original authors', [('H2, RoNIN ResNet: Plug-in HN', 'ext_resnet', 'Hbar', CONF, Q_MAIN)])]
    rows = []
    for grp, items in spec:
        rows.append([f'**{grp}**'] + [''] * 8)
        for lab, m, kind, dss, q in items:
            for ds in dss:
                r = L.compare(af(m, ds, kind), af(m, ds, 'Ubar'), ds, q)
                tl = ds == 'tlio_c'
                down = r['cq'][1] < 0 and (tl or r['cs'][1] < 0)
                res = ('Significant decrease' if tl else 'Decrease†') if down else 'Not significant'
                imp = f"{r['imp_seq']}/{r['nseq']} seqs" if tl else f"{r['imp_sub']}/{r['nsub']} subj., {r['imp_seq']}/{r['nseq']} seqs"
                rows.append([lab, ds_label(ds, r), num(r['c']), num(r['t']), diff_rel(r), '—' if tl else ci(*r['cs']), ci(*r['cq']),
                             imp, res])
    return head, rows


@table('9')
def t9():
    head = ['Frozen network', 'Unmodified (8 ψ)', 'Plug-in HN (8 α)', 'Difference (rel.)', '95% CI', 'Improved seqs',
            'ATE difference (rel.)']
    rows = []
    for lab, m in (('ResNet18-YawAug', 'ours_yaw'), ('Transformer-YawAug', 'int_transformer_yaw'), ('IMUNet-YawAug', 'int_imunet_yaw'),
                   ('RoNIN ResNet', 'ext_resnet')):
        s = L.compare(af(m, 'tlio_c', 'Hbar', 'ate_shape'), af(m, 'tlio_c', 'Ubar', 'ate_shape'), 'tlio_c')
        a = L.compare(af(m, 'tlio_c', 'Hbar'), af(m, 'tlio_c', 'Ubar'), 'tlio_c')
        rows.append([lab, num(s['c']), num(s['t']), diff_rel(s), ci(*s['cq']), f"{s['imp_seq']}/{s['nseq']}", pct(a['rel'], sign=True)])
    return head, rows
