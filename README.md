# HeadingNorm: code, archived results and checkpoints

Code and data for the article

> Yanlin Li, Xichen Cui, Yu Quan. **HeadingNorm: Closed-Form Yaw Canonicalization for Inertial Velocity Regression without Retraining.** Submitted to *Sensors*.

HeadingNorm (HN) rotates each 1 s window of gravity-aligned IMU data onto the principal axis of its horizontal acceleration, fixes the sign of that axis with the projected third moment, and rotates the regressed velocity back. It has no learnable parameters and makes any deterministic per-window network exactly yaw-equivariant on non-degenerate windows, so it can be plugged into networks that are already trained.

This repository regenerates **every table and figure** of the main text and the Supplementary Materials from the archived results with one command, and contains the code that produced those results from the public datasets.

## Contents

- [Quick start: one command](#quick-start-one-command)
- [What is regenerated and how it is compared](#what-is-regenerated-and-how-it-is-compared)
- [Using HeadingNorm with a trained network](#using-headingnorm-with-a-trained-network)
- [Starting from the raw datasets](#starting-from-the-raw-datasets)
- [Repository layout](#repository-layout)
- [Data in this repository](#data-in-this-repository)
- [Notes](#notes)

## Quick start: one command

Requirements: Python 3.11 and the packages in `requirements.txt` (PyTorch included). No GPU and no dataset are needed.

```bash
git clone <repository URL> HeadingNorm-code && cd HeadingNorm-code
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce/run_all.py
```

The command runs three steps and takes about 15 s on a laptop CPU:

| Step | Script | What it does |
|---|---|---|
| 1 | `reproduce/verify_manifest.py` | checks that every file listed in `MANIFEST.tsv` is present with its recorded SHA-256 |
| 2 | `reproduce/make_tables.py` | regenerates Tables 1–9, Tables S1–S28 and the nine `supp_data/*.csv` files and compares them with the manuscript cell by cell |
| 3 | `reproduce/make_figures.py` | redraws Figures 1–7 and S1–S3 and compares them with the published figures |

The report is printed and written to `reproduce/output/REPORT.txt`. With the pinned package versions the expected result is:

```text
## Repository integrity (MANIFEST.tsv) (reproduce/verify_manifest.py): exit 0
files in manifest: 593; missing: 0; changed: 0
## Tables (reproduce/make_tables.py): exit 0
TOTAL: 46 tables, 4773 cells compared (4773 identical text, 0 equal at the displayed precision), 0 disagreements, 0 tables failed
## Figures (reproduce/make_figures.py): exit 0
Figure 1   ok       source data identical; image pixel-identical  (v4fig_problem_overview)
...
TOTAL: 10 figures; source data identical: 10; pixel-identical: 10; failures: 0
RESULT: every table and figure agrees with the manuscript
```

The generated tables are in `reproduce/output/tables/` (Markdown and CSV, one file per table) and the generated figures in `reproduce/output/figures/` (PNG, PDF and source data). Single tables can be regenerated with, e.g., `python reproduce/make_tables.py 8 S21`.

## What is regenerated and how it is compared

**Tables.** `reproduce/tables_main.py` (Tables 1–9), `reproduce/tables_supp.py` (Tables S1–S28) and `reproduce/tables_suppdata.py` (`supp_data/*.csv`) contain one function per table. Row and column labels are written in the code; every number is computed from the archived per-sequence records in `results/` with the statistics of the paper (subject means with equal subject weights, 10,000-sample percentile bootstrap with seed 20260906; the interval levels and decision rules are those stated in the table captions). `reproduce/tables_lib.py` holds the shared loaders and the bootstrap. A generated cell agrees with the manuscript (`paper/*.md`) when its text is identical, or when every number in it agrees at the precision shown in the manuscript; with the pinned versions all 4773 cells are textually identical.

**Figures.** `tools/make_figures_terms_20261001.py` is the script that drew the published figures; `reproduce/make_figures.py` runs it with the output directed to `reproduce/output/figures/`, so the archived figures in `figures/v15/` are never overwritten. Every figure is written together with a `*_source_data.csv` file holding the plotted values. A figure passes when its source data are byte-identical to the archived ones; the PNG is also compared pixel by pixel. Pixel identity requires matplotlib 3.11.0 and the Helvetica Neue font of macOS; on other systems another font is substituted, the images differ slightly in their text and anti-aliasing, and the report says so without counting it as a failure (`HN_STRICT_PIXELS=1` counts it).

| Item | Archived inputs | Code |
|---|---|---|
| Figure 1 | `results/e0_trajectory_overlay/` | `tools/make_v4_figures_20260919.py` |
| Figure 2 | `results/figure_inputs_20261004/fig02_window.npz` (one 1 s window) | `tools/mst_figures.py` |
| Figure 3 | schematic | `tools/make_hn_pipeline_figures_20260929.py` |
| Figure 4 | `results/e0_plugin_overlay_20260929/` | `tools/make_hn_pipeline_figures_20260929.py` |
| Figure 5 | `results/e1_heading_repeatability/`, `results/e6_sign_ablation/`, `results/hn_plugin_p0_20260912/` | `tools/make_v4_figures_20260919.py` |
| Figure 6 | `results/unified_eval_20260913/`, `results/confirm_20260913/`, `results/plugin_arch_20260923/` | `tools/make_plugin_provenance_figure_20260925.py` |
| Figure 7 | `results/fig8_shape_example_20261001/`, `results/figure_inputs_20261004/fig07_tlio_confirm_sequence.npz` | `tools/make_fig8_shape_example_20261001.py` |
| Figures S1–S3 | `results/unified_eval_20260913/`, `results/e6_sign_ablation/`, `figures/v11/figS03_trajectories_source_data.csv` | `tools/make_v2_figures_20260913.py`, `tools/mst_figures.py`, `tools/make_supp_figures_S1_S3_20260926.py` |
| Tables 1–9, S1–S28, `supp_data/` | per-sequence records in `results/` (table in “Starting from the raw datasets”) | `reproduce/tables_*.py` |

The figure scripts are called through `tools/make_figures_terms_20261001.py`, which applies the final wording of axis labels and legends and routes inputs and outputs; the plotting code itself is unchanged.

## Using HeadingNorm with a trained network

HeadingNorm is `models.HeadingNorm` in [`src/models.py`](src/models.py). Plugging it into a trained per-window network takes two lines:

```python
import sys; sys.path[:0] = ['src', 'tools']
from models import HeadingNorm, _unrotate

hn = HeadingNorm()

def plug_in(net, x):                    # x: [B, 6, 200] gravity-aligned gyroscope (0-2) and accelerometer (3-5), 200 Hz
    x_canonical, phi = hn(x)            # rotate each window into its canonical heading (closed form)
    return _unrotate(net(x_canonical), phi)   # predict and rotate the velocity back
```

`python examples/plug_in_demo.py` loads the ResNet18-YawAug checkpoint of this repository and prints, for 8 rotations of synthetic windows, the violation |F(Rx) − R F(x)| of yaw equivariance: about 0.1–0.25 m/s for the unmodified network and about 2×10⁻⁷ m/s (float32 rounding) with plug-in HN. Training with HN (training-time HN) uses the same module inside the network (`models.build(..., hnorm=True)`).

## Starting from the raw datasets

All per-sequence results in `results/` were produced by the scripts in `tools/` from the public datasets. To re-run them, obtain the datasets and third-party code and place them as described in [`data/README.md`](data/README.md) and [`external/README.md`](external/README.md):

| Input | Source |
|---|---|
| RoNIN dataset | https://ronin.cs.sfu.ca/ |
| RoNIN code (commit `805b7f0f28bb164ce89ada9ac05a9470dbe3d715`) and released ResNet/LSTM/TCN weights | https://github.com/Sachini/ronin, RoNIN project page |
| RIDI dataset | https://yanhangpublic.github.io/ridi/ |
| TLIO dataset | https://github.com/CathIAS/TLIO |
| IMUNet phone data | public data of the IMUNet authors (reference [39] of the paper) |

Build the caches:

```bash
python tools/rebuild_caches.py --ronin <RoNIN>/ --ronin-lists <RoNIN code>/lists --ridi <RIDI>/ --oxiod <empty dir> --out data
python tools/rebuild_tlio_cache.py --tlio external/tlio_golden          # TLIO-test (36)
python tools/build_imunet_cache_20260913.py                               # Phone-test (36); needs numpy-quaternion
python tools/build_confirm_caches_20260913.py                             # TLIO-confirm (318), Phone-confirm (87)
```

Then `python reproduce/check_raw_route.py` re-runs the pipeline on a few sequences (our checkpoints with plug-in HN on all five test sets, the released RoNIN ResNet under 8 headings and 8 conventions, three networks on TLIO-confirm, the dataset statistics of Table 1 and the inputs of Table S4 and Figures 2 and 7) and compares every value with the archive (identical files; per-sequence values within 10⁻⁴ m). On the authors' Apple M5 Pro all five steps pass, with a largest per-sequence difference of 1.0×10⁻⁶ m, in about 1.5 min; on a CPU steps 3–4 take about 30 min. `tools/confirm_anglefair_eval_20260927.py` accepts only Apple MPS, as in the archived runs, so step 5 is skipped on other machines.

A complete re-run of an experiment overwrites its folder in `results/`; afterwards `python reproduce/run_all.py` regenerates the tables and figures from the new records (`verify_manifest.py` will then list the re-run files as changed). The experiments, in the order in which they depend on each other:

| Archived results | Used in | Scripts |
|---|---|---|
| `results/sensors_v4/` | Tables 6, S1, S5, S7, S13, S24–S28 | `tools/sensors_evaluate.py`, `tools/sensors_analyze.py`, `tools/sensors_diagnostics.py` |
| `results/e1_heading_repeatability/`, `results/e1b_yaw_aligned/` | Tables 3, S10, S11, S17; Figure 5 | `tools/e1_heading_repeatability.py`, `tools/e1_analyze.py`, `tools/e1b_yaw_aligned.py` |
| `results/e3_backbone_frontend/` | Tables S1, S10 | `tools/e3_evaluate.py`, `tools/e3_analyze.py` |
| `results/e6_sign_ablation/` | Tables 3, 5, 6, S12, S24, S25; Figure S2 | `tools/e6_evaluate.py`, `tools/e6_analytic_check.py` |
| `results/hn_plugin_p0_20260912/`, `results/angle_fair_20260926/` | Tables 4, S17 | `tools/hn_plugin_p0_20260912.py`, `tools/angle_fair_plugin_20260926.py` |
| `results/unified_eval_20260913/` | Tables 7, S8, S9, S14, S15, S22; Figures 6, S1; `supp_data/` | `tools/unified_eval_20260913.py --group {window,seq,yawhn,yawfa2}`, `tools/unified_analyze_20260913.py`, `tools/unified_significance_20260913.py`, `tools/complex_gain_oracle_20260913.py` |
| `results/tta_repeat_20260913/`, `results/fa2_repeat_20260913/` | Table S15; `supp_data/testtime_fa2_*` | `tools/tta_repeat_20260913.py`, `tools/fa2_repeat_20260913.py` |
| `results/confirm_20260913/` | Tables 7, S18–S20, S23 | `tools/confirm_eval_20260913.py --group {yaw,rest}`, `tools/confirm_verdict_20260913.py` |
| `results/plugin_arch_20260923/`, `results/shape_verdict_20260923/` | Tables 7, S16, S23; Figure 6 | `tools/plugin_arch_eval_20260923.py`, `tools/plugin_arch_verdict_20260923.py`, `tools/shape_verdict_20260923.py` |
| `results/confirm_anglefair_20260927/` | Tables 8, 9, S21; `supp_data/confirm_anglefair_per_angle.csv` | `HN_DEVICE=mps tools/confirm_anglefair_eval_20260927.py --group {resnet,arch}` (Apple MPS only), `tools/confirm_anglefair_verdict_20260927.py` |
| `results/e0_trajectory_overlay/`, `results/e0_plugin_overlay_20260929/`, `results/fig8_shape_example_20261001/` | Figures 1, 4, 7 | `tools/e0_trajectory_overlay.py`, `tools/e0_plugin_overlay_20260929.py`, `tools/make_fig8_shape_example_20261001.py` (re-runs inference when its cache is removed) |
| `results/data_stats_20260929/`, `results/data_stats_20261004/`, `results/figure_inputs_20261004/` | Tables 1, S4; Figures 2, 7 | `tools/data_stats_20260929.py`, `tools/archive_small_inputs_20261004.py` |
| `verification/mst_revision_20260909/illustration/` | Figure S3 | `tools/replay_mst_illustration_20260909.py`, `tools/make_supp_figures_S1_S3_20260926.py extract` |

`tools/unified_eval_20260913.py`, `tools/hn_plugin_p0_20260912.py` and `tools/plugin_arch_eval_20260923.py` accept `--limit N` (and `tools/e1_heading_repeatability.py` `--limit-per-group N`) for a short run that writes to `verification/generated/` instead of `results/`. The scripts run on the CPU by default; `HN_DEVICE=auto` uses CUDA or Apple MPS (the archived runs used MPS). Per-sequence values agree across devices to within 10⁻⁴ m. The decision criteria of the confirmatory analyses were written before the runs and are in `docs/` (`53_*`, `89_*`, `91_*`, `100_*`; in Chinese).

**Retraining.** The checkpoints in `models/` can be retrained with `tools/run_sensors_revision.py` (`models/sensors_v4/`), `tools/train_rq_matrix_all_20260927.py --only <tag>` (`models/submission_frozen/`), `tools/run_e3_train.py` (`models/e3_backbone_frontend/`) and `tools/run_e6_train.py` (`models/e6_sign_ablation/`); each run took 12–20 min on an Apple M5 Pro (`HN_DEVICE=mps`). GPU kernels are not deterministic, so a retrained checkpoint matches the archived one statistically, not bit for bit; compare validation losses with `models/*/*.json` and evaluation results with `results/`.

## Repository layout

| Path | Content |
|---|---|
| `reproduce/` | One-command reproduction (`run_all.py`), table generators, figure comparison, raw-data check |
| `examples/` | `plug_in_demo.py`: plug-in HN on a trained checkpoint |
| `src/` | HeadingNorm and backbones (`models.py`), front ends, training and evaluation code, dataset readers |
| `tools/` | Evaluation, analysis and plotting scripts (dates in the names are the dates of the experiments) |
| `config/` | Subject-disjoint split, sequence lists, experiment protocols |
| `models/` | Checkpoints used in the paper (`*.pt`) with training metadata (`*.json`: settings, GN constants, validation history, training-file hashes) |
| `results/` | Per-sequence evaluation records and decision files of every experiment |
| `figures/v15/` | The published figures with their source data |
| `paper/` | The manuscript and Supplementary Materials (Markdown) used for the comparison |
| `supp_data/` | The numerical tables that accompany the Supplementary Materials |
| `verification/` | Re-evaluation of the released RoNIN checkpoint (Sections S3–S4) and the trajectories of Figure S3 |
| `provenance/` | Experiment registry and hashes of the training caches |
| `MANIFEST.tsv` | Size and SHA-256 of every file |

## Data in this repository

The repository contains the per-sequence results, decision files and figure source data of every experiment in the paper, the checkpoints of all networks trained for the paper (640 MB), and three small inputs exported from the evaluation caches by `tools/archive_small_inputs_20261004.py` (the phone-data statistics of Table S4, the window of Figure 2 and the ground truth of the sequence of Figure 7). The datasets themselves and the caches built from them (about 1.3 GB) are not included; they are rebuilt from the public datasets as described above. Apart from these three inputs, the only ground-truth excerpts are the trajectories drawn in Figures 1, 4, 7 and S3.

## Notes

- `PATCHES.md` lists the only difference from the authors' working copy: one line of `src/hn_paths.py` that searched a directory of the authors' machine for third-party data.
- Provenance fields inside some archived result files record paths of the authors' machine at the time of the run; they are kept unchanged because the files are archives.
- Comments in some scripts are in Chinese, the working language of the project.
- License: to be added by the authors before publication.
