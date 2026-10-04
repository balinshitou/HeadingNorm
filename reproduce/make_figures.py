"""Redraw all ten figures of the paper from the archived results and compare them with the published versions.

    python reproduce/make_figures.py

The plotting code is tools/make_figures_terms_20261001.py, the script that drew the figures in the manuscript. It writes to
reproduce/output/figures/ (the archived figures in figures/v15/ are not touched). Each figure is compared with the published
one in two ways:
  - source data: the *_source_data.csv written next to every figure must be byte-identical to the archived one;
  - image: the PNG is compared pixel by pixel. Pixel identity is expected with matplotlib 3.11.0 and the Helvetica Neue
    font of macOS; with other fonts or versions the anti-aliasing and text metrics change while the data do not, so a
    pixel difference alone is reported but not counted as a failure (set HN_STRICT_PIXELS=1 to count it).
Output: reproduce/output/figures_summary.txt. Exit status 1 if a figure is missing or its source data differ.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'figures/v15'
OUT = ROOT / 'reproduce/output/figures'
FIGS = [('Figure 1', 'v4fig_problem_overview'), ('Figure 2', 'fig02_window_axis_and_sign'), ('Figure 3', 'fig3_hn_inference'),
        ('Figure 4', 'fig4_psi_experiment'), ('Figure 5', 'v4fig_repeatability'), ('Figure 6', 'v10fig_plugin_by_provenance'),
        ('Figure 7', 'fig8_shape_example'), ('Figure S1', 'v2fig_unified_paired'), ('Figure S2', 'fig10_sign_rule_frame_failure'),
        ('Figure S3', 'figS03_trajectories')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob('*'):
        f.unlink()
    env = {**os.environ, 'HN_FIG_OUT': str(OUT), 'MPLBACKEND': 'Agg'}
    r = subprocess.run([sys.executable, 'tools/make_figures_terms_20261001.py'], cwd=ROOT, env=env, capture_output=True, text=True)
    (OUT.parent / 'figures_log.txt').write_text(r.stdout + r.stderr)
    lines = [] if r.returncode == 0 else [f'plotting script FAILED (exit {r.returncode}): ' + ' | '.join((r.stdout + r.stderr).strip().splitlines()[-3:])]
    strict = os.environ.get('HN_STRICT_PIXELS') == '1'
    bad = n_pix = 0
    for label, stem in FIGS:
        png, src = OUT / f'{stem}.png', OUT / f'{stem}_source_data.csv'
        if not png.exists() or not src.exists():
            lines.append(f'{label:10s} MISSING  {stem}'); bad += 1
            continue
        same_src = src.read_bytes() == (ARCHIVE / src.name).read_bytes()
        a, b = mpimg.imread(ARCHIVE / png.name), mpimg.imread(png)
        if a.shape == b.shape and np.array_equal(a, b):
            pix = 'pixel-identical'; n_pix += 1
        elif a.shape != b.shape:
            pix = f'image size {b.shape[1]}x{b.shape[0]} vs {a.shape[1]}x{a.shape[0]}'
        else:
            d = np.abs(a.astype(float) - b.astype(float)).max(-1)
            pix = f'pixels differ ({100 * (d > 0).mean():.2f}% of pixels)'
        ok = same_src and (pix == 'pixel-identical' or not strict)
        bad += not ok
        lines.append(f"{label:10s} {'ok      ' if ok else 'DIFFERS '} source data {'identical' if same_src else 'DIFFER'}; image {pix}  ({stem})")
    lines.append(f'TOTAL: {len(FIGS)} figures; source data identical: {sum("source data identical" in l for l in lines)}; '
                 f'pixel-identical: {n_pix}; failures: {bad + (r.returncode != 0)}')
    if n_pix < len(FIGS) and not strict:
        lines.append('(pixel differences with identical source data come from fonts or the matplotlib version; see the docstring)')
    text = '\n'.join(lines)
    print(text)
    (OUT.parent / 'figures_summary.txt').write_text(text + '\n')
    sys.exit(1 if bad or r.returncode else 0)


if __name__ == '__main__':
    main()
