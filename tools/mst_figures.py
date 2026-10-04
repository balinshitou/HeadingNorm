#!/usr/bin/env python3
"""Build every main-text and supplementary figure, one panel per file.

Each figure is a single panel written to its own file, so no panel can crowd
another and every figure can be regenerated, inspected or replaced on its own.
Nothing here recomputes a model or an evaluation: the script only reads frozen
artefacts under results/ and data/, aggregates them, and draws. Every figure also
writes a source-data CSV containing exactly the values that were plotted, so a
reader can reproduce the picture without running any model.

Requirements: python >= 3.9, numpy, matplotlib. No other dependency.

Usage
-----
    python tools/mst_figures.py                 # build every figure
    python tools/mst_figures.py --list          # list figure ids and what they show
    python tools/mst_figures.py --only fig03    # build one figure (repeatable)
    python tools/mst_figures.py --outdir /tmp/f --formats pdf png

Outputs go to figures/mst/ by default:
    <id>_<name>.pdf / .svg / .png     the figure
    <id>_<name>_source_data.csv       the plotted values
    manifest.csv                      id, name, description, source artefacts

Data sources
------------
    results/sensors_v4/analysis.json          frontend summaries, paired contrasts,
                                              conditioning bins
    results/sensors_v4/diagnostics.json       window-level rotation consistency
    results/e1_heading_repeatability/*.json   yaw sweep, four frontends and three
                                              HeadingNorm backbones
    results/e6_sign_ablation/*.json           yaw sweep for the sign-rule ablation,
                                              plus the network-free frame check
    results/e3_backbone_frontend/*.json       yaw sweep and analysis for the
                                              backbone x frontend factorial
    results/e4_error_timescale/analysis.json  error against time scale
    models/**/<tag>.json                      validation loss of each training run
    data/eval/benchmark/ronin__*.npz          one real window for the method figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'figures/mst'

# --------------------------------------------------------------------------- style
# Okabe-Ito, chosen for colour-vision deficiency; every series also has a marker,
# so the figures survive greyscale printing.
COLOUR = {
    'GN only': '#999999',
    'GN + yaw augmentation': '#E69F00',
    'GN + mixed-sensor PCA (adapted)': '#56B4E9',
    'GN + HeadingNorm without sign rule': '#D55E00',
    'GN + HeadingNorm': '#009E73',
}
LABEL = {
    'GN only': 'Standardisation only',
    'GN + yaw augmentation': 'Random yaw augmentation',
    'GN + mixed-sensor PCA (adapted)': 'Mixed-sensor PCA frame',
    'GN + HeadingNorm without sign rule': 'HN without sign rule',
    'GN + HeadingNorm': 'HeadingNorm',
}
MARKER = {
    'GN only': 's', 'GN + yaw augmentation': 'o',
    'GN + mixed-sensor PCA (adapted)': '^',
    'GN + HeadingNorm without sign rule': 'v', 'GN + HeadingNorm': 'D',
}
ORDER = list(COLOUR)
FOUR = [f for f in ORDER if f != 'GN + HeadingNorm without sign rule']
GREY, INK = '#BBBBBB', '#333333'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#ECECEC', 'grid.linewidth': .6,
    'axes.axisbelow': True,
    'legend.frameon': False, 'lines.linewidth': 1.5, 'lines.markersize': 4,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

SIZE = (4.7, 3.3)          # default single-panel size, comfortable at one column
WIDE = (6.6, 2.6)          # for the pipeline schematic

# --------------------------------------------------------------------------- registry
REGISTRY: list[dict] = []


def figure(fid, name, description, sources):
    def wrap(fn):
        REGISTRY.append(dict(id=fid, name=name, description=description,
                             sources=sources, fn=fn))
        return fn
    return wrap


# --------------------------------------------------------------------------- helpers
def _json(rel):
    path = ROOT / rel
    return json.loads(path.read_text()) if path.exists() else None


def sweep_rows():
    """Every completed row of the yaw-sweep evaluations."""
    rows = []
    for d in ('results/e1_heading_repeatability', 'results/e6_sign_ablation',
              'results/e3_backbone_frontend'):
        directory = ROOT / d
        if not directory.exists():
            continue
        for p in sorted(directory.glob('*.json')):
            if p.name in {'analysis.json', 'analytic_frame_check.json',
                          'table_per_seed_convention.json'}:
                continue
            payload = json.loads(p.read_text())
            if payload.get('complete') and 'rows' in payload:
                rows.extend(payload['rows'])
    return rows


def subject_macro(bucket, metric='ate'):
    by = defaultdict(list)
    for r in bucket:
        by[r['subject']].append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in by.values()]))


def angle_curves(rows, dataset, split):
    """family -> (angles in degrees, mean over seeds, seed standard deviation)."""
    cell = defaultdict(list)
    for r in rows:
        if r['dataset'] == dataset and r['split'] == split:
            cell[(r['family'], r['seed'], r['angle_index'])].append(r)
    out = {}
    for fam in sorted({k[0] for k in cell}):
        seeds = sorted({k[1] for k in cell if k[0] == fam})
        idx = sorted({k[2] for k in cell if k[0] == fam})
        arr = np.array([[subject_macro(cell[(fam, s, i)]) for i in idx] for s in seeds])
        sd = arr.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(idx))
        out[fam] = ([360 * i / len(idx) for i in idx], arr.mean(axis=0), sd)
    return out


def sequence_ranges(rows, dataset, split):
    """family -> list of per-(seed, sequence) cross-angle ATE ranges."""
    cell = defaultdict(dict)
    for r in rows:
        if r['dataset'] == dataset and r['split'] == split:
            cell[(r['family'], r['seed'], r['seq'])][r['angle_index']] = float(r['ate'])
    out = defaultdict(list)
    for (fam, _, _), v in cell.items():
        if len(v) >= 8:
            out[fam].append(max(v.values()) - min(v.values()))
    return out


def frontend_summaries():
    a = _json('results/sensors_v4/analysis.json')
    fix = {'GN + mixed-sensor PCA': 'GN + mixed-sensor PCA (adapted)'}
    return a, {(fix.get(s['family'], s['family']), s['dataset'], s['split'], s['metric']): s
               for s in a['summaries']}


def real_window(index_from_middle=0):
    """One deterministic 1 s window of horizontal acceleration from a RoNIN sequence."""
    for path in sorted((ROOT / 'data/eval/benchmark').glob('ronin__*.npz')):
        with np.load(path, allow_pickle=False) as z:
            if str(z['split']) != 'unseen':
                continue
            feat = np.asarray(z['feat'], float)
            seq = str(z['seq'])
        j = len(feat) // 2 + index_from_middle
        w = feat[j:j + 200, 3:5]
        return feat, w - w.mean(axis=0), seq, j
    return None, None, None, None


def frame_angles(ac):
    """Unsigned and signed frame angle of one centred horizontal-acceleration window."""
    cov = ac.T @ ac / len(ac)
    phi0 = .5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    proj = ac @ np.array([np.cos(phi0), np.sin(phi0)])
    m3 = float((proj ** 3).mean())
    return phi0, phi0 + (np.pi if m3 < 0 else 0.), m3, proj


def bar_group(ax, groups, series, values, errors=None, width=.8, colours=None,
              labels=None):
    """Grouped bars; returns the x positions of the group centres."""
    n = len(series)
    w = width / n
    centres = np.arange(len(groups), dtype=float)
    for si, s in enumerate(series):
        x = centres + (si - (n - 1) / 2) * w
        ax.bar(x, [values[(g, s)] for g in groups], width=w * .88,
               color=(colours or {}).get(s, GREY), edgecolor='white', linewidth=.5,
               label=(labels or {}).get(s, s), zorder=2)
        if errors:
            ax.errorbar(x, [values[(g, s)] for g in groups],
                        yerr=[errors[(g, s)] for g in groups], color=INK, lw=.8,
                        capsize=2, ls='none', zorder=3)
    return centres


# =========================================================================== method
@figure('fig01', 'reference_frames',
        'One motion described in two horizontal reference frames gives two different '
        'sets of channel values and two different velocity labels.',
        ['schematic, no data'])
def fig_reference_frames():
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.grid(False)
    v = np.array([.60, .78])
    ax.annotate('', xy=tuple(v), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='#111111', lw=2, shrinkA=0,
                                shrinkB=0), zorder=5)
    ax.text(v[0] + .04, v[1] + .04, 'velocity $v$', fontsize=9, ha='left', va='bottom')
    rows = []
    for psi, col, ls, tag in [(0., INK, '-', 'A'), (np.deg2rad(40), '#0072B2', (0, (5, 3)), 'B')]:
        R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        for e, nm in ((np.array([.90, 0.]), 'x'), (np.array([0., .90]), 'y')):
            tip = R @ e
            ax.annotate('', xy=tuple(tip), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1, ls=ls,
                                        shrinkA=0, shrinkB=0), zorder=3)
            lab = R @ (e * 1.14)
            ax.text(lab[0], lab[1], f'${nm}_{tag}$', color=col, fontsize=9,
                    ha='center', va='center')
        comp = R.T @ v
        foot = R @ np.array([comp[0], 0.])
        ax.plot([v[0], foot[0]], [v[1], foot[1]], color=col, lw=.8, ls=(0, (2, 2)), zorder=2)
        ax.plot([0, foot[0]], [0, foot[1]], color=col, lw=3, alpha=.45,
                solid_capstyle='butt', zorder=2)
        pos = foot * .5 + (np.array([0, -.09]) if psi == 0 else np.array([-.16, .02]))
        ax.text(pos[0], pos[1], f'$v_x^{{{tag}}}={comp[0]:.2f}$', color=col, fontsize=8.5,
                ha='center', va='center')
        rows += [[f'frame {tag}', 'yaw_deg', float(np.degrees(psi))],
                 [f'frame {tag}', 'v_x', float(comp[0])],
                 [f'frame {tag}', 'v_y', float(comp[1])]]
    ax.set_aspect('equal')
    ax.set_xlim(-.75, 1.20); ax.set_ylim(-.30, 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.set_title('The horizontal reference direction is free', pad=8)
    ax.text(.5, -.06, 'the motion is identical; the numbers the network sees are not',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color='#666666')
    return fig, rows, ['frame', 'quantity', 'value']


@figure('fig02', 'window_axis_and_sign',
        'One real 1 s window of horizontal acceleration: the covariance principal axis '
        'is a line defined modulo pi, and the third central moment selects one of its '
        'two directions.',
        ['data/eval/benchmark/ronin__*.npz'])
def fig_window_axis():
    feat, ac, seq, j = real_window()
    if ac is None:
        return None
    phi0, phi, m3, proj = frame_angles(ac)
    lam = np.linalg.eigvalsh(ac.T @ ac / len(ac))
    u = np.array([np.cos(phi), np.sin(phi)])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.grid(False)
    ax.scatter(ac[:, 0], ac[:, 1], s=6, c='#C8C8C8', linewidths=0, zorder=1,
               label='samples in the window')
    lim = 1.35 * np.abs(ac).max()
    ax.plot([-lim * np.cos(phi0), lim * np.cos(phi0)],
            [-lim * np.sin(phi0), lim * np.sin(phi0)],
            color='#555555', lw=1, ls=(0, (5, 4)), zorder=2,
            label='principal axis (defined mod $\\pi$)')
    ax.annotate('', xy=(.62 * lim * u[0], .62 * lim * u[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=COLOUR['GN + HeadingNorm'],
                                lw=2, shrinkA=0, shrinkB=0), zorder=3)
    ax.annotate('', xy=(-.62 * lim * u[0], -.62 * lim * u[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='#D55E00', lw=1.2,
                                shrinkA=0, shrinkB=0), zorder=3)
    perp = np.array([-u[1], u[0]])
    tag = .74 * lim * u + .14 * lim * perp
    ax.text(tag[0], tag[1], '$+u$  selected', color=COLOUR['GN + HeadingNorm'],
            fontsize=9, ha='center', va='center', fontweight='bold')
    tag2 = -.74 * lim * u - .12 * lim * perp
    ax.text(tag2[0], tag2[1], '$-u$', color='#D55E00', fontsize=9, ha='center',
            va='center')
    ax.set_aspect('equal'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('centred $a_x$  (m s$^{-2}$)')
    ax.set_ylabel('centred $a_y$  (m s$^{-2}$)')
    ax.set_title('The axis is a line; the sign is a choice', pad=8)
    ax.legend(loc='upper left', bbox_to_anchor=(-.02, 1.0), handlelength=1.6,
              labelspacing=.3, fontsize=7.6)
    ax.text(.98, .02, f'$\\lambda_1/\\lambda_2$ = {lam[1] / max(lam[0], 1e-12):.1f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#555555')
    rows = [[seq, j, float(x), float(y)] for x, y in ac]
    return fig, rows, ['sequence', 'window_start_sample', 'centred_ax', 'centred_ay']


@figure('fig03', 'frame_equivariance_real_windows',
        'Change of the reference direction under an applied yaw, on twelve real '
        'windows, with and without the third-moment sign rule. No network weights '
        'are involved.',
        ['data/eval/benchmark/ronin__*.npz'])
def fig_frame_equivariance():
    feat, _, seq, _ = real_window()
    if feat is None:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    psis = np.linspace(0, 2 * np.pi, 181)
    starts = np.linspace(200, len(feat) - 400, 12).astype(int)
    rows = []
    for n, j0 in enumerate(starts):
        w = feat[j0:j0 + 200, 3:5]
        w = w - w.mean(axis=0)
        base0, base1, _, _ = frame_angles(w)
        signed, unsigned = [], []
        for psi in psis:
            R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            p0, p1, _, _ = frame_angles(w @ R.T)
            signed.append(np.angle(np.exp(1j * (p1 - base1))) % (2 * np.pi))
            unsigned.append(np.angle(np.exp(1j * (p0 - base0))) % (2 * np.pi))
        ax.plot(np.degrees(psis), np.degrees(unsigned),
                color=COLOUR['GN + HeadingNorm without sign rule'], lw=.9, alpha=.55,
                zorder=2, label='without the sign rule' if n == 0 else None)
        ax.plot(np.degrees(psis), np.degrees(signed), color=COLOUR['GN + HeadingNorm'],
                lw=1.1, alpha=.85, zorder=3,
                label='with the third-moment sign rule' if n == 0 else None)
        rows += [[seq, int(j0), float(np.degrees(p)), float(np.degrees(s)),
                  float(np.degrees(u))]
                 for p, s, u in zip(psis[::10], signed[::10], unsigned[::10])]
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlim(0, 360); ax.set_ylim(-15, 375)
    ax.set_xlabel('yaw applied to the window, $\\psi$ (deg)')
    ax.set_ylabel('$\\phi(R_\\psi X) - \\phi(X)$  (deg)')
    ax.set_title('Only the sign rule keeps the frame on the diagonal', pad=8)
    ax.legend(loc='upper left', handlelength=1.6, labelspacing=.3, fontsize=7.8)
    return fig, rows, ['sequence', 'window_start_sample', 'applied_yaw_deg',
                       'with_sign_rule_deg', 'without_sign_rule_deg']


@figure('fig04', 'equivariance_pipeline',
        'The three links whose conjunction gives exact yaw equivariance of the whole '
        'frontend; the backbone is unconstrained.',
        ['schematic, no data'])
def fig_pipeline():
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axis('off'); ax.grid(False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    green = COLOUR['GN + HeadingNorm']
    blocks = [(1.0, 'input\n$R_\\psi X$', '#F4F4F4', '#AAAAAA'),
              (20.5, '$R_{-\\phi}$', '#E4F3EE', green),
              (40.0, 'standardise\n$G$', '#E4F3EE', green),
              (59.5, 'backbone\n$h$', '#F4F4F4', '#AAAAAA'),
              (79.0, '$R_{+\\phi}$', '#E4F3EE', green)]
    for x, label, fc, ec in blocks:
        ax.add_patch(FancyBboxPatch((x, 26), 16, 11,
                                    boxstyle='round,pad=0.5,rounding_size=1.4',
                                    facecolor=fc, edgecolor=ec, linewidth=1.1))
        ax.text(x + 8, 31.5, label, ha='center', va='center', fontsize=8.6)
    for x in (17.5, 37.0, 56.5, 76.0):
        ax.annotate('', xy=(x + 3.0, 31.5), xytext=(x + .3, 31.5),
                    arrowprops=dict(arrowstyle='-|>', color='#999999', lw=1))
    ax.annotate('', xy=(98.5, 31.5), xytext=(95.8, 31.5),
                arrowprops=dict(arrowstyle='-|>', color='#999999', lw=1))
    ax.text(99.2, 31.5, '$R_\\psi\\hat v$', ha='left', va='center', fontsize=8.6)
    notes = [(28.5, 'link 1\n$\\lambda_1\\neq\\lambda_2$,  $m_3\\neq 0$', green),
             (48.0, 'link 2\nzero horizontal mean,\nshared horizontal scale', green),
             (67.5, 'unconstrained;\nonly deterministic', '#888888'),
             (87.0, 'link 3\nthe same $\\phi$', green)]
    for x, txt, col in notes:
        ax.plot([x, x], [25.4, 21.0], color='#CCCCCC', lw=.8, ls=(0, (2, 2)))
        ax.text(x, 19.6, txt, ha='center', va='top', fontsize=7.6, color=col)
    ax.text(50, 3.0, '$\\hat v(R_\\psi X)=R_\\psi\\,\\hat v(X)$  holds exactly '
                     'when all three links hold',
            ha='center', va='bottom', fontsize=9)
    return fig, [], ['(schematic; no plotted data)']


# ====================================================================== comparison
GROUPS3 = [('ronin', 'unseen', 'Independent-\nsubject group'),
           ('ridi', '', 'RIDI'), ('ronin', 'seen', 'Training-\nsubject group')]


@figure('fig05', 'frontend_ate',
        'Subject-macro absolute trajectory error of the four frontends on a shared '
        'ResNet18 backbone, mean over four seeds.',
        ['results/sensors_v4/analysis.json'])
def fig_frontend_ate():
    _, summ = frontend_summaries()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    groups = [g[2] for g in GROUPS3]
    values, errors, rows = {}, {}, []
    for ds, sp, g in GROUPS3:
        for fam in FOUR:
            s = summ[(fam, ds, sp, 'ate')]
            values[(g, fam)] = s['subject_macro_mean']
            errors[(g, fam)] = s['subject_macro_seed_sd']
            rows.append([g.replace('\n', ' '), LABEL[fam], s['subject_macro_mean'],
                         s['subject_macro_seed_sd'], s['subjects']])
    centres = bar_group(ax, groups, FOUR, values, errors, colours=COLOUR, labels=LABEL)
    ax.set_xticks(centres); ax.set_xticklabels(groups)
    ax.set_ylabel('Subject-macro ATE (m)')
    ax.set_ylim(0, 10.6)
    ax.legend(loc='upper right', handlelength=1.2, labelspacing=.3, fontsize=7.8)
    ax.set_title('One backbone, one budget, four frontends', pad=8)
    ax.xaxis.grid(False)
    return fig, rows, ['group', 'frontend', 'subject_macro_ate_m', 'seed_sd_m', 'subjects']


@figure('fig06', 'paired_differences',
        'Paired subject differences, HeadingNorm minus each control, for ATE and for '
        'the 60 s relative error, with 95 per cent subject-cluster bootstrap intervals.',
        ['results/sensors_v4/analysis.json'])
def fig_paired_differences():
    a, _ = frontend_summaries()
    prim = [('GN + yaw augmentation', 'ronin', 'unseen', 'independent-subject'),
            ('GN + yaw augmentation', 'ridi', '', 'RIDI'),
            ('GN + mixed-sensor PCA', 'ronin', 'unseen', 'independent-subject'),
            ('GN + mixed-sensor PCA', 'ridi', '', 'RIDI')]
    fix = {'GN + mixed-sensor PCA': 'GN + mixed-sensor PCA (adapted)'}
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    labels, ticks, rows = [], [], []
    for i, (ctrl, ds, sp, glab) in enumerate(prim):
        col = COLOUR[fix.get(ctrl, ctrl)]
        for metric, marker, face, dy in (('ate', 'D', col, .16), ('rte', 'o', 'white', -.16)):
            c = next((c for c in a['contrasts'] if c['control'] == ctrl
                      and c['dataset'] == ds and c['split'] == sp
                      and c['metric'] == metric), None)
            if c is None:
                continue
            lo, hi = c['conditional_subject_bootstrap95']
            ax.plot([lo, hi], [-i + dy] * 2, color=col, lw=2, solid_capstyle='round',
                    zorder=2)
            ax.plot([c['difference']], [-i + dy], marker=marker, color=face,
                    markeredgecolor=col, markeredgewidth=1.1, markersize=5, zorder=3)
            rows.append(['RIDI' if ds == 'ridi' else 'independent-subject group',
                         LABEL[fix.get(ctrl, ctrl)], metric, c['difference'], lo, hi,
                         c['n_subjects']])
        labels.append('vs %s\n%s' % ('yaw augmentation' if 'yaw' in ctrl else 'PCA frame',
                                     glab))
        ticks.append(-i)
    ax.axvline(0, color=INK, lw=.9, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-len(prim) + .45, .62)
    ax.set_xlabel('HeadingNorm minus control (m)')
    ax.yaxis.grid(False)
    ax.plot([], [], marker='D', color=COLOUR['GN + HeadingNorm'], lw=0, markersize=5,
            label='ATE')
    ax.plot([], [], marker='o', color='white',
            markeredgecolor=COLOUR['GN + HeadingNorm'], markeredgewidth=1.1, lw=0,
            markersize=5, label='60 s RTE')
    ax.legend(loc='upper left', handlelength=.9, labelspacing=.3, fontsize=7.8,
              bbox_to_anchor=(-.01, 1.02))
    ax.set_title('Paired subject differences, 95 % bootstrap', pad=8)
    return fig, rows, ['group', 'control', 'metric', 'difference_m', 'ci_low_m',
                       'ci_high_m', 'subjects']


# ==================================================================== repeatability
def _repeatability_curve(dataset, split, title):
    rows_all = sweep_rows()
    if not rows_all:
        return None
    curves = angle_curves(rows_all, dataset, split)
    fig, ax = plt.subplots(figsize=(4.7, 3.4))
    rows = []
    for fam in ORDER:
        if fam not in curves:
            continue
        ang, mean, sd = curves[fam]
        ang = list(ang) + [360.]
        mean = np.append(mean, mean[0]); sd = np.append(sd, sd[0])
        ax.plot(ang, mean, color=COLOUR[fam], marker=MARKER[fam], label=LABEL[fam],
                zorder=3)
        ax.fill_between(ang, mean - sd, mean + sd, color=COLOUR[fam], alpha=.15, lw=0,
                        zorder=2)
        rows += [[LABEL[fam], a, m] for a, m in zip(ang[:-1], mean[:-1])]
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('yaw applied to the test input (deg)')
    ax.set_ylabel('Subject-macro ATE (m)')
    ax.set_title(title, pad=8)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, .5), handlelength=1.4,
              labelspacing=.4, fontsize=7.8)
    return fig, rows, ['frontend', 'applied_yaw_deg', 'subject_macro_ate_m']


@figure('fig07', 'repeatability_independent_subjects',
        'Subject-macro ATE against the yaw applied to the test input, RoNIN '
        'independent-subject group. The shaded band is the seed-to-seed standard '
        'deviation, that is, training noise.',
        ['results/e1_heading_repeatability/*.json', 'results/e6_sign_ablation/*.json'])
def fig_repeat_unseen():
    return _repeatability_curve('ronin', 'unseen', 'RoNIN independent-subject group')


@figure('fig08', 'repeatability_ridi',
        'Subject-macro ATE against the yaw applied to the test input, RIDI.',
        ['results/e1_heading_repeatability/*.json', 'results/e6_sign_ablation/*.json'])
def fig_repeat_ridi():
    return _repeatability_curve('ridi', '', 'RIDI')


@figure('fig09', 'repeatability_ecdf',
        'Empirical distribution of the per-sequence ATE range across the eight '
        'initial reference directions, RoNIN independent-subject group.',
        ['results/e1_heading_repeatability/*.json', 'results/e6_sign_ablation/*.json'])
def fig_repeat_ecdf():
    rows_all = sweep_rows()
    if not rows_all:
        return None
    ranges = sequence_ranges(rows_all, 'ronin', 'unseen')
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    rows = []
    for fam in ORDER:
        if fam not in ranges:
            continue
        v = np.sort(np.maximum(np.array(ranges[fam]), 1e-7))
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where='post', color=COLOUR[fam], label=LABEL[fam])
        rows.append([LABEL[fam], float(np.median(v)), float(v.max()), len(v)])
    ax.set_xscale('log'); ax.set_xlim(2e-7, 60); ax.set_ylim(0, 1.02)
    ax.set_xlabel('per-sequence ATE range over eight angles (m)')
    ax.set_ylabel('empirical CDF')
    ax.set_title('What a single recording sees', pad=8)
    ax.legend(loc='center left', handlelength=1.4, labelspacing=.4, fontsize=7.8,
              bbox_to_anchor=(.02, .60))
    ax.text(6e-6, .05, 'float32 rounding scale', fontsize=7.4, color='#777777',
            ha='left', va='bottom')
    return fig, rows, ['frontend', 'median_range_m', 'max_range_m', 'sequences']


# ======================================================================= sign rule
@figure('fig10', 'sign_rule_frame_failure',
        'Fraction of 72 fixed windows whose reference direction fails to follow the '
        'applied rotation, with and without the third-moment sign rule. No network '
        'weights are involved.',
        ['results/e6_sign_ablation/analytic_frame_check.json'])
def fig_sign_failure():
    chk = _json('results/e6_sign_ablation/analytic_frame_check.json')
    if chk is None:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    deg = [r['angle_deg'] for r in chk['rows']]
    n = chk['windows']
    bad = [100 * r['unsigned_inconsistent'] / n for r in chk['rows']]
    ok = [100 * r['signed_inconsistent'] / n for r in chk['rows']]
    x = np.arange(len(deg))
    ax.bar(x - .2, bad, width=.38, color=COLOUR['GN + HeadingNorm without sign rule'],
           edgecolor='white', linewidth=.5, zorder=2, label='without the sign rule')
    ax.bar(x + .2, np.maximum(ok, .8), width=.38, color=COLOUR['GN + HeadingNorm'],
           edgecolor='white', linewidth=.5, zorder=2,
           label='with the third-moment sign rule')
    for xi, b in zip(x, bad):
        ax.text(xi - .2, b + 2.5, f'{b:.0f}', ha='center', fontsize=7.4, color='#555555')
    ax.set_xticks(x); ax.set_xticklabels([f'{int(d)}' for d in deg])
    ax.set_xlabel('yaw applied to the test input (deg)')
    ax.set_ylabel('windows whose frame does not\nfollow the rotation (%)')
    ax.set_ylim(0, 128); ax.set_yticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(False)
    ax.legend(loc='upper left', handlelength=1.2, labelspacing=.3, fontsize=7.8,
              bbox_to_anchor=(-.01, 1.03))
    ax.annotate(f'0 at every angle\n(0 of {len(deg) * n} cases)', xy=(5.2, .8),
                xytext=(5.4, 34), fontsize=7.6, color=COLOUR['GN + HeadingNorm'],
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', color=COLOUR['GN + HeadingNorm'], lw=.9))
    ax.set_title(f'Frame equivariance on {n} fixed windows', pad=8)
    rows = [[int(d), 'without sign rule', b, n] for d, b in zip(deg, bad)]
    rows += [[int(d), 'with sign rule', o, n] for d, o in zip(deg, ok)]
    return fig, rows, ['applied_yaw_deg', 'variant', 'percent_windows_inconsistent',
                       'windows']


@figure('fig11', 'sign_rule_window_vs_trajectory',
        'Effect of removing the third-moment sign rule at the window level and at the '
        'trajectory level.',
        ['models/e6_sign_ablation/*.json', 'results/sensors_v4/analysis.json'])
def fig_sign_levels():
    fig, ax = plt.subplots(figsize=(4.7, 3.3))
    labels = ['Validation\nHuber loss', 'ATE\nindependent\nsubjects', 'ATE\nRIDI',
              'ATE\ntraining\nsubjects']
    base = [0.013779, 5.825, 3.099, 4.936]
    abl = [0.014040, 6.899, 3.502, 5.238]
    rel = [100 * (b - a) / a for a, b in zip(base, abl)]
    x = np.arange(len(labels))
    ax.bar(x, rel, width=.55, color=COLOUR['GN + HeadingNorm without sign rule'],
           edgecolor='white', linewidth=.5, zorder=2)
    for xi, r in zip(x, rel):
        ax.text(xi, r + .8, f'+{r:.1f} %', ha='center', fontsize=8.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel('change when the sign rule is removed (%)')
    ax.set_ylim(0, 23)
    ax.xaxis.grid(False)
    ax.set_title('Small at the window level, large at the trajectory level', pad=8)
    rows = [[l.replace('\n', ' '), a, b, r] for l, a, b, r in zip(labels, base, abl, rel)]
    return fig, rows, ['quantity', 'with_sign_rule', 'sign_rule_removed', 'percent_change']


# ====================================================================== time scales
@figure('fig12', 'anchored_error',
        'Start-anchored error against elapsed time on the independent-subject group; '
        'this is the quantity ATE averages.',
        ['results/e4_error_timescale/analysis.json'])
def fig_anchored():
    d = _json('results/e4_error_timescale/analysis.json')
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    rows = []
    for fam in ORDER:
        pts = sorted([(r['elapsed_seconds'], r['subject_macro'], r['sequences'])
                      for r in d['anchored'] if r['family'] == fam])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOUR[fam],
                marker=MARKER[fam], label=LABEL[fam])
        rows += [[LABEL[fam], a, b, c] for a, b, c in pts]
    ax.set_xscale('log')
    ax.set_xticks([10, 30, 60, 120, 300, 600])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('elapsed time (s)')
    ax.set_ylabel('start-anchored error (m)')
    ax.set_title('What ATE averages', pad=8)
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=.35, fontsize=7.8)
    return fig, rows, ['frontend', 'elapsed_seconds', 'subject_macro_error_m', 'sequences']


@figure('fig13', 'displacement_error',
        'Displacement error against span on the independent-subject group; the 60 s '
        'relative trajectory error is one sample of this curve.',
        ['results/e4_error_timescale/analysis.json'])
def fig_displacement():
    d = _json('results/e4_error_timescale/analysis.json')
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    rows = []
    for fam in ORDER:
        pts = sorted([(r['horizon_seconds'], r['subject_macro']) for r in d['rows']
                      if r['family'] == fam and r['group'] == 'independent-subject group'])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOUR[fam],
                marker=MARKER[fam], label=LABEL[fam])
        rows += [[LABEL[fam], a, b] for a, b in pts]
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 20, 60, 120])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    lo, hi = ax.get_ylim()
    ax.axvline(60, color=INK, lw=.9, ls=(0, (4, 3)))
    ax.text(56, lo + .04 * (hi - lo), '60 s RTE', fontsize=8, va='bottom', ha='right',
            rotation=90)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('span $h$ (s)')
    ax.set_ylabel('displacement error (m)')
    ax.set_title('What RTE samples', pad=8)
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=.35, fontsize=7.8)
    return fig, rows, ['frontend', 'span_seconds', 'subject_macro_error_m']


@figure('fig14', 'timescale_difference',
        'HeadingNorm minus random yaw augmentation for both error definitions; the '
        'displacement difference stays positive while the start-anchored difference '
        'crosses zero between 90 s and 120 s.',
        ['results/e4_error_timescale/analysis.json'])
def fig_timescale_difference():
    d = _json('results/e4_error_timescale/analysis.json')
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    green = COLOUR['GN + HeadingNorm']
    rows = []
    series = [
        ('start-anchored error', 'D', '-', [(r['elapsed_seconds'], r['family'],
                                             r['subject_macro']) for r in d['anchored']]),
        ('displacement error', 'o', (0, (4, 2)),
         [(r['horizon_seconds'], r['family'], r['subject_macro']) for r in d['rows']
          if r['group'] == 'independent-subject group']),
    ]
    for name, marker, ls, raw in series:
        ref = {t: v for t, f, v in raw if f == 'GN + yaw augmentation'}
        tgt = sorted([(t, v) for t, f, v in raw if f == 'GN + HeadingNorm' and t in ref])
        if not tgt:
            continue
        ax.plot([t for t, _ in tgt], [v - ref[t] for t, v in tgt], color=green,
                marker=marker, ls=ls, label=name,
                markerfacecolor=green if marker == 'D' else 'white',
                markeredgecolor=green)
        rows += [[name, t, v - ref[t]] for t, v in tgt]
    lo, hi = ax.get_ylim()
    ax.axhspan(lo, 0, color=green, alpha=.07, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=.9)
    ax.text(1.1, lo * .55, 'HeadingNorm better', fontsize=8, color=green, va='center',
            ha='left')
    ax.set_ylim(lo, hi)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 60, 300])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('time scale (s)')
    ax.set_ylabel('HeadingNorm minus yaw augmentation (m)')
    ax.set_title('The two metrics sample different time scales', pad=8)
    ax.legend(loc='upper left', handlelength=1.8, labelspacing=.35, fontsize=7.8)
    return fig, rows, ['error_definition', 'time_scale_seconds', 'difference_m']


# ===================================================================== conditioning
def _conditioning(variable, xlabel, title):
    a, _ = frontend_summaries()
    tmpl = {'ResNet18_v3_hn_gn': 'GN + HeadingNorm',
            'Sensors_v4_resnet_yaw': 'GN + yaw augmentation',
            'Sensors_v4_resnet_gn': 'GN only',
            'Sensors_v4_resnet_mixed_pca': 'GN + mixed-sensor PCA (adapted)'}
    agg = defaultdict(lambda: [0, 0.])
    for b in a['exploratory_geometry_bins']:
        if b['dataset'] != 'ronin' or b['split'] != 'unseen':
            continue
        fam = tmpl.get(b['model'].rsplit('_s', 1)[0])
        if fam is None or b['variable'] != variable:
            continue
        agg[(b['lower'], b['upper'], fam)][0] += b['windows']
        agg[(b['lower'], b['upper'], fam)][1] += b['sum_velocity_error']
    edges = sorted({(k[0], k[1]) for k in agg},
                   key=lambda e: (e[0] if e[0] is not None else -1))
    names = ['< %g' % hi if lo == 0 else ('> %g' % lo if hi is None else '%g–%g' % (lo, hi))
             for lo, hi in edges]
    total = sum(agg[(lo, hi, 'GN + HeadingNorm')][0] for lo, hi in edges)
    share = [100 * agg[(lo, hi, 'GN + HeadingNorm')][0] / total for lo, hi in edges]

    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    x = np.arange(len(edges))
    rows = []
    for fam in ORDER:
        vals = []
        for lo, hi in edges:
            c, s = agg.get((lo, hi, fam), (0, 0.))
            vals.append(s / c if c else np.nan)
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(x, vals, color=COLOUR[fam], marker=MARKER[fam], label=LABEL[fam])
        rows += [[n, LABEL[fam], v] for n, v in zip(names, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n({s:.1f} % of\nwindows)' for n, s in zip(names, share)],
                       fontsize=7.6)
    rows += [[n, 'window share (%)', s] for n, s in zip(names, share)]
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel('mean per-window velocity error (m s$^{-1}$)')
    ax.set_title(title, pad=8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + .32 * (hi - lo))
    ax.legend(loc='upper left', handlelength=1.4, labelspacing=.35, fontsize=7.6)
    ax.xaxis.grid(False)
    return fig, rows, ['bin', 'series', 'value']


@figure('fig15', 'conditioning_eigenvalue_gap',
        'Mean per-window velocity error against the relative eigenvalue gap of the '
        'horizontal covariance, with the share of windows in each bin.',
        ['results/sensors_v4/analysis.json'])
def fig_cond_gap():
    return _conditioning('relative_eigenvalue_gap',
                         'relative eigenvalue gap of the horizontal covariance',
                         'Near-isotropic windows are rare and not worse')


@figure('fig16', 'conditioning_third_moment',
        'Mean per-window velocity error against the absolute standardised third '
        'moment, with the share of windows in each bin.',
        ['results/sensors_v4/analysis.json'])
def fig_cond_m3():
    return _conditioning('absolute_standardized_third_moment',
                         'absolute standardised third moment  $|m_3|$',
                         'Windows with a near-zero third moment are rare')


# ========================================================================= backbone
@figure('fig17', 'backbone_frontend_ate',
        'Subject-macro ATE of three backbones under two frontends, completing the '
        'backbone x frontend factorial.',
        ['results/e3_backbone_frontend/analysis.json'])
def fig_backbone_ate():
    d = _json('results/e3_backbone_frontend/analysis.json')
    if d is None or d.get('missing_arms'):
        return None
    backbones = d['backbones']
    groups = [('ronin', 'unseen', 'Independent-subject'), ('ridi', '', 'RIDI')]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    rows, w, step = [], .34, len(backbones) + 1.0
    for gi, (ds, sp, glab) in enumerate(groups):
        for bi, bb in enumerate(backbones):
            for fi, (arm, col, lab) in enumerate([
                    ('yaw', COLOUR['GN + yaw augmentation'], 'Random yaw augmentation'),
                    ('hn', COLOUR['GN + HeadingNorm'], 'HeadingNorm')]):
                e = next((r for r in d['summaries'] if r['backbone'] == bb
                          and r['dataset'] == ds and r['split'] == sp
                          and r['metric'] == 'ate' and r['arm'] == arm), None)
                if e is None:
                    continue
                x = gi * step + bi + (fi - .5) * w
                ax.bar(x, e['subject_macro_mean'], width=w * .9, color=col,
                       edgecolor='white', linewidth=.5, zorder=2,
                       label=lab if (gi == 0 and bi == 0) else None)
                ax.errorbar(x, e['subject_macro_mean'], yerr=e['subject_macro_seed_sd'],
                            color=INK, lw=.8, capsize=2, zorder=3)
                rows.append([glab, bb, lab, e['subject_macro_mean'],
                             e['subject_macro_seed_sd']])
    ticks = [gi * step + bi for gi in range(len(groups)) for bi in range(len(backbones))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(backbones * len(groups), fontsize=7.8)
    for gi, (_, _, glab) in enumerate(groups):
        ax.text(gi * step + (len(backbones) - 1) / 2, -.19, glab,
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.4)
    ax.set_ylabel('Subject-macro ATE (m)')
    ax.xaxis.grid(False)
    ax.legend(loc='upper left', handlelength=1.2, labelspacing=.3, fontsize=7.8)
    ax.set_title('Three backbones, two frontends', pad=8)
    return fig, rows, ['group', 'backbone', 'frontend', 'subject_macro_ate_m', 'seed_sd_m']


@figure('fig18', 'backbone_paired_differences',
        'HeadingNorm minus random yaw augmentation per backbone, with 95 per cent '
        'subject-cluster bootstrap intervals.',
        ['results/e3_backbone_frontend/analysis.json'])
def fig_backbone_diff():
    d = _json('results/e3_backbone_frontend/analysis.json')
    if d is None or d.get('missing_arms'):
        return None
    groups = [('ronin', 'unseen', 'independent-subject'), ('ridi', '', 'RIDI')]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    green = COLOUR['GN + HeadingNorm']
    labels, ticks, rows, i = [], [], [], 0
    for ds, sp, glab in groups:
        for bb in d['backbones']:
            c = next((r for r in d['contrasts'] if r['backbone'] == bb
                      and r['dataset'] == ds and r['split'] == sp
                      and r['metric'] == 'ate'), None)
            if c is None:
                continue
            lo, hi = c['conditional_subject_bootstrap95']
            # Colour by sign: the direction of the effect is the finding here.
            col = green if c['difference'] < 0 else COLOUR['GN + yaw augmentation']
            ax.plot([lo, hi], [-i] * 2, color=col, lw=2, solid_capstyle='round')
            ax.plot([c['difference']], [-i], marker='D', color=col, markersize=5,
                    markeredgecolor='white', markeredgewidth=.7)
            labels.append(f'{bb}\n{glab}'); ticks.append(-i); i += 1
            rows.append([glab, bb, c['difference'], lo, hi, c['n_subjects']])
    ax.axvline(0, color=INK, lw=.9)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('HeadingNorm minus yaw augmentation, ATE (m)')
    ax.yaxis.grid(False)
    lo_x, hi_x = ax.get_xlim()
    ax.text(lo_x + .03 * (hi_x - lo_x), .5, 'HeadingNorm lower', fontsize=7.6, color=green,
            ha='left', va='center')
    ax.text(hi_x - .03 * (hi_x - lo_x), .5, 'augmentation lower', fontsize=7.6,
            color=COLOUR['GN + yaw augmentation'], ha='right', va='center')
    ax.set_ylim(-i + .5, .95)
    ax.set_title('The sign is not the same on all three backbones', pad=8)
    return fig, rows, ['group', 'backbone', 'difference_m', 'ci_low_m', 'ci_high_m',
                       'subjects']


# ============================================================================= main

def _overlay_rows():
    """Load the eight-angle trajectory dump produced by tools/e0_trajectory_overlay.py."""
    import csv as _csv
    base = ROOT / 'results/e0_trajectory_overlay'
    with (base / 'trajectories.csv').open() as fh:
        traj = list(_csv.DictReader(fh))
    with (base / 'ground_truth.csv').open() as fh:
        gt = list(_csv.DictReader(fh))
    return traj, gt


def _trajectory_overlay(seq, title_note):
    traj, gt = _overlay_rows()
    traj = [r for r in traj if r['seq'] == seq]
    gt = [r for r in gt if r['seq'] == seq]
    gx = [float(r['x']) for r in gt]
    gy = [float(r['y']) for r in gt]

    arms = ['GN + yaw augmentation', 'GN + HeadingNorm']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    rows = []
    for ax, arm in zip(axes, arms):
        sub = [r for r in traj if r['arm'] == arm]
        ates = {}
        for r in sub:
            ates[int(r['angle_index'])] = float(r['ate'])
        ax.plot(gx, gy, color='#BBBBBB', lw=3.0, solid_capstyle='round',
                zorder=1, label='ground truth')
        n = len(ates)
        for k in sorted(ates):
            xs = [float(r['x']) for r in sub if int(r['angle_index']) == k]
            ys = [float(r['y']) for r in sub if int(r['angle_index']) == k]
            shade = plt.cm.viridis(k / max(n - 1, 1))
            ax.plot(xs, ys, color=shade, lw=1.0, alpha=.9, zorder=2)
            for x, y in zip(xs, ys):
                rows.append([seq, arm, k, round(360. * k / n, 1), ates[k], x, y])
        lo, hi = min(ates.values()), max(ates.values())
        spread = hi - lo
        ax.set_title(arm.replace('GN + ', ''), pad=6)
        ax.text(.5, -.26, f'ATE {lo:.3f}-{hi:.3f} m,  range {spread:.3f} m',
                transform=ax.transAxes, ha='center', va='top', fontsize=9,
                color=('#B33A00' if spread > .01 else '#00674F'))
        if spread <= .01:
            ax.text(.5, -.38, 'the eight curves coincide', transform=ax.transAxes,
                    ha='center', va='top', fontsize=8, color='#666666')
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].legend(loc='best', frameon=False, fontsize=8)
    fig.suptitle(f'One recording, eight initial reference directions ({title_note})', y=1.0)
    fig.subplots_adjust(bottom=.28)
    return fig, rows, ['sequence', 'frontend', 'angle_index', 'angle_deg', 'ate_m', 'x_m', 'y_m']


@figure('fig19', 'trajectory_overlay_median_sequence',
        'The same recording is presented to the same frozen model at eight initial '
        'reference directions. Random yaw augmentation returns eight different '
        'trajectories. HeadingNorm returns one. Sequence a058_2 is the '
        'independent-subject sequence whose cross-angle ATE range is closest to the '
        'median of that group.',
        ['results/e0_trajectory_overlay/trajectories.csv',
         'results/e0_trajectory_overlay/ground_truth.csv'])
def fig_overlay_median():
    return _trajectory_overlay('a058_2', 'sequence a058_2')


@figure('fig20', 'trajectory_overlay_second_sequence',
        'The companion of figure 19 on sequence a050_1, the other sequence equally '
        'close to the median cross-angle range. HeadingNorm is exactly repeatable '
        'here as well, and on this recording it is the less accurate of the two, '
        'which shows that repeatability is a property of the construction and not a '
        'consequence of being more accurate.',
        ['results/e0_trajectory_overlay/trajectories.csv',
         'results/e0_trajectory_overlay/ground_truth.csv'])
def fig_overlay_second():
    return _trajectory_overlay('a050_1', 'sequence a050_1')


def build(entry, outdir, formats):
    result = entry['fn']()
    if result is None:
        print(f"[skip] {entry['id']}_{entry['name']}: required artefacts not available")
        return False
    fig, rows, header = result
    stem = f"{entry['id']}_{entry['name']}"
    for ext in formats:
        fig.savefig(outdir / f'{stem}.{ext}', dpi=600 if ext == 'png' else None)
    with open(outdir / f'{stem}_source_data.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    plt.close(fig)
    print(f"[ok]   {stem}  ({len(rows)} source rows)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--outdir', default=str(DEFAULT_OUT))
    ap.add_argument('--only', action='append', default=[],
                    help='figure id, e.g. fig03; may be repeated')
    ap.add_argument('--formats', nargs='+', default=['pdf', 'svg', 'png'])
    ap.add_argument('--list', action='store_true', help='list the figures and exit')
    args = ap.parse_args()

    if args.list:
        for e in REGISTRY:
            print(f"{e['id']}  {e['name']:38s} {e['description']}")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = [e for e in REGISTRY if not args.only or e['id'] in args.only]
    if not wanted:
        print(f'no figure matches {args.only}; use --list to see the ids')
        return 1
    built = [e for e in wanted if build(e, outdir, args.formats)]
    with open(outdir / 'manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'name', 'description', 'source_artefacts', 'built'])
        for e in REGISTRY:
            w.writerow([e['id'], e['name'], e['description'], '; '.join(e['sources']),
                        'yes' if e in built else 'no'])
    print(f'\n{len(built)}/{len(wanted)} figures written to {outdir}')
    print(f'manifest: {outdir / "manifest.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
