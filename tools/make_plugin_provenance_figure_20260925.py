"""外挂HN按模型来源分组的森林图（HeadingNorm-Transformer-20260925 稿图6）。
数字与区间直接调用 plugin_by_provenance_recheck_20260925.cmp（逐序列结果复算，docs/107）。
用法：.venv/bin/python tools/make_plugin_provenance_figure_20260925.py
"""
import contextlib, csv, importlib.util, io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rc', ROOT / 'tools/plugin_by_provenance_recheck_20260925.py')
rc = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rc)

OUT = ROOT / 'figures/v10'; OUT.mkdir(parents=True, exist_ok=True)
NETS = [  # (处理, 对照, 图例, 来源, 颜色)
    ('ours_yaw_hn', 'ours_yaw', 'ResNet18-YawAug (trained by us)', 'ours', '#1f77b4'),
    ('int_transformer_yaw_hn', 'int_transformer_yaw', 'Transformer-YawAug (trained by us)', 'ours', '#2ca02c'),
    ('int_imunet_yaw_hn', 'int_imunet_yaw', 'IMUNet-YawAug (trained by us)', 'ours', '#9467bd'),
    ('ext_resnet_hn', 'ext_resnet', 'RoNIN ResNet (released by authors)', 'released', '#d62728'),
]
DS = [('ronin_seen', 'RoNIN-seen'), ('ronin_unseen', 'RoNIN-unseen\n(discovery)'),
      ('ridi', 'RIDI'), ('tlio', 'TLIO-test'), ('imunet', 'Phone-test'),
      ('tlio_c', 'TLIO-confirm\n(318, never evaluated)'), ('imunet_c', 'Phone-confirm\n(87, never evaluated)')]

rows = []
for ds, _ in DS:
    for t, c, lab, src, col in NETS:
        r = rc.cmp(t, c, ds)
        if ds.startswith('tlio'): lo, hi = r['ci_q']
        elif ds.startswith('imunet'): lo, hi = max(r['ci_s'][0], r['ci_q'][0]), min(r['ci_s'][1], r['ci_q'][1])
        else: lo, hi = r['ci_s']
        rows.append(dict(dataset=ds, network=lab, provenance=src, control_ate_m=round(r['cm'], 3),
                         plugin_ate_m=round(r['tm'], 3), rel_pct=round(r['rel'], 1),
                         ci95_lo_pct=round(100 * lo / r['cm'], 1), ci95_hi_pct=round(100 * hi / r['cm'], 1),
                         excludes_zero=bool(r['sig']), color=col))

with open(OUT / 'v10fig_plugin_by_provenance_source_data.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'color']); w.writeheader()
    for r in rows: w.writerow({k: v for k, v in r.items() if k != 'color'})

fig, ax = plt.subplots(figsize=(7.2, 7.0))
gap, step, y, yt = 1.0, 0.8, 0.0, []
for gi, (ds, dlab) in enumerate(DS):
    ys = []
    for r in [r for r in rows if r['dataset'] == ds]:
        m = 'o' if r['provenance'] == 'ours' else 'D'
        ax.plot([r['ci95_lo_pct'], r['ci95_hi_pct']], [y, y], color=r['color'], lw=1.4)
        ax.plot(r['rel_pct'], y, m, ms=6, color=r['color'],
                mfc=r['color'] if r['excludes_zero'] else 'white', mew=1.4)
        ys.append(y); y -= step
    yt.append((sum(ys) / len(ys), dlab))
    if ds.endswith('_c') and ds == 'tlio_c':
        ax.axhspan(ys[0] + step / 2 + gap / 4, -99, color='#ececec', zorder=0)
    y -= gap
ax.set_ylim(y + gap - step / 2, step)
ax.axvline(0, color='k', lw=0.8)
ax.set_yticks([t for t, _ in yt]); ax.set_yticklabels([l for _, l in yt], fontsize=8.5)
ax.set_xlabel('ATE change of Plug-in HN vs. Unmodified (% of Unmodified mean, 95% CI)', fontsize=8.5)
ax.tick_params(axis='x', labelsize=8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
h = [plt.Line2D([], [], color=c, marker='o' if s == 'ours' else 'D', lw=1.4, label=l) for _, _, l, s, c in NETS]
h += [plt.Line2D([], [], color='gray', marker='o', mfc='white', lw=0, label='open: CI includes 0')]
h += [plt.Rectangle((0, 0), 1, 1, color='#e6e6e6', label='shaded: confirmation splits (never evaluated before)')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.42, -0.09), ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(OUT / f'v10fig_plugin_by_provenance.{ext}', dpi=300)
print(OUT / 'v10fig_plugin_by_provenance.png')
for r in rows: print(r['dataset'], r['network'], r['rel_pct'], r['ci95_lo_pct'], r['ci95_hi_pct'], r['excludes_zero'])
