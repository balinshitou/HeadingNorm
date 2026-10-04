#!/usr/bin/env python3
"""从已审计汇总自动生成 RQ1--RQ4 的论文表格和投稿图。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "generated"
FIG_DIR = ROOT / "figures" / "submission" / "rq_v3"
PAPER_OUT = ROOT / "paper" / "generated_rq1_rq4_results.md"
COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "black": "#000000", "gray": "#7A7A7A",
}
GROUPS = (("ronin", "seen", "RoNIN seen"),
          ("ronin", "unseen", "RoNIN unseen"), ("ridi", "", "RIDI"))
LABELS = {
    "a6_full": "HN-Transformer (ours)",
    "ronin_resnet18_complete_pipeline_control": "ResNet18",
    "imunet_complete_pipeline_control": "IMUNet",
    "a6_without_headingnorm": "−HeadingNorm",
    "a6_without_globalnorm": "−GlobalNorm",
    "a6_without_prefilter": "−PreFilter",
    "a6_mse_instead_of_huber": "Huber→MSE",
    "a6_without_headingnorm_and_globalnorm": "−HN−GN",
    "resnet18_shared_frontend": "ResNet18",
    "imunet_shared_frontend": "IMUNet",
}
BACKBONE_LABELS = {
    "a6_without_prefilter": "Transformer",
    "resnet18_shared_frontend": "ResNet18",
    "imunet_shared_frontend": "IMUNet",
}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
        "savefig.dpi": 600,
    })


def save_both(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fmt(mean, sd, digits=3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main() -> None:
    global FIG_DIR, PAPER_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="?",
                        default="eval_rq1_rq4_v3_registered_summary.json")
    parser.add_argument("--figure-dir", type=Path,
                        help="图输出目录；相对路径按项目根目录解释")
    parser.add_argument("--paper-out", type=Path,
                        help="自动表格输出文件；相对路径按项目根目录解释")
    args = parser.parse_args()
    if args.figure_dir is not None:
        FIG_DIR = args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir
    if args.paper_out is not None:
        PAPER_OUT = args.paper_out if args.paper_out.is_absolute() else ROOT / args.paper_out
    data = json.loads((RESULT_DIR / args.summary).read_text(encoding="utf-8"))
    if not data.get("matrix_complete"):
        raise RuntimeError("refuse to plot an incomplete RQ matrix")
    configure_plotting()
    aggregate = {(row["family"], row["dataset"], row["split"]): row
                 for row in data["aggregate"]}

    # Component ablation: all four seeds visible.
    ablations = ("a6_full", "a6_without_headingnorm", "a6_without_globalnorm",
                 "a6_without_prefilter", "a6_mse_instead_of_huber")
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 6.7 / 2.54),
                             constrained_layout=True)
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"],
               COLORS["purple"], COLORS["red"]]
    for ax, (dataset, split, title) in zip(axes, GROUPS):
        means, sds = [], []
        for family in ablations:
            row = aggregate[(family, dataset, split)]
            means.append(row["mean_ate"]); sds.append(row["sd_ate"])
        x = np.arange(len(ablations))
        ax.bar(x, means, yerr=sds, capsize=2, color=palette, alpha=.82)
        for index, family in enumerate(ablations):
            values = aggregate[(family, dataset, split)]["run_values_ate"]
            ax.scatter(np.full(len(values), index), values, s=10, color="black", zorder=3)
        ax.set_title(title); ax.set_xticks(x, [LABELS[f] for f in ablations], rotation=35,
                                           ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ATE (m; primary protocol)")
    save_both(fig, "fig_rq2_component_ablation")

    # Backbone-only comparison under the shared six-channel HN+GN front-end.
    backbones = ("a6_without_prefilter", "resnet18_shared_frontend",
                 "imunet_shared_frontend")
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 6.2 / 2.54),
                             constrained_layout=True)
    markers = ("o", "s", "^")
    for ax, (dataset, split, title) in zip(axes, GROUPS):
        for family, color, marker in zip(backbones, palette, markers):
            row = aggregate[(family, dataset, split)]
            ax.errorbar([BACKBONE_LABELS[family]], [row["mean_ate"]], yerr=[row["sd_ate"]],
                        marker=marker, color=color, capsize=3, linestyle="none")
            ax.scatter(np.full(4, BACKBONE_LABELS[family]), row["run_values_ate"],
                       color="black", s=10, zorder=3)
        ax.set_title(title); ax.tick_params(axis="x", labelrotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ATE (m; primary protocol)")
    save_both(fig, "fig_rq2_backbone_control")

    # HeadingNorm x GlobalNorm factorial plot.
    cells = {("off", "off"): "a6_without_headingnorm_and_globalnorm",
             ("off", "on"): "a6_without_headingnorm",
             ("on", "off"): "a6_without_globalnorm", ("on", "on"): "a6_full"}
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 5.7 / 2.54),
                             constrained_layout=True)
    for ax, (dataset, split, title) in zip(axes, GROUPS):
        for hn, color, marker, linestyle in (("off", COLORS["orange"], "s", "--"),
                                               ("on", COLORS["blue"], "o", "-")):
            means = [aggregate[(cells[(hn, gn)], dataset, split)]["mean_ate"]
                     for gn in ("off", "on")]
            sds = [aggregate[(cells[(hn, gn)], dataset, split)]["sd_ate"]
                   for gn in ("off", "on")]
            ax.errorbar([0, 1], means, yerr=sds, marker=marker, linestyle=linestyle,
                        color=color, capsize=3, label=f"HeadingNorm {hn}")
        ax.set_title(title); ax.set_xticks([0, 1], ["GlobalNorm off", "GlobalNorm on"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ATE (m; primary protocol)")
    axes[-1].legend(frameon=False)
    save_both(fig, "fig_rq2_heading_global_interaction")

    # Per-sequence ECDF uses the mean over four seeds to avoid treating seeds as
    # additional independent test sequences.
    seq = defaultdict(list)
    for row in data["per_sequence_seed_mean"]:
        seq[(row["family"], row["dataset"], row["split"])].append(row["mean_ate"])
    ecdf_families = ("a6_full", "ronin_resnet18_complete_pipeline_control",
                     "imunet_complete_pipeline_control")
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 5.8 / 2.54),
                             constrained_layout=True)
    for ax, (dataset, split, title) in zip(axes, GROUPS):
        for family, color, linestyle in zip(ecdf_families,
                                             (COLORS["blue"], COLORS["red"], COLORS["green"]),
                                             ("-", "--", "-.")):
            values = np.sort(np.asarray(seq[(family, dataset, split)], dtype=float))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, y, where="post", color=color, linestyle=linestyle,
                    linewidth=1.2, label=LABELS[family])
        ax.set_title(title); ax.set_xlabel("Per-sequence ATE (m)")
        ax.set_ylim(0, 1.01); ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[-1].legend(frameon=False, loc="lower right")
    save_both(fig, "fig_rq1_sequence_ate_ecdf")

    # RQ4: scale and direction channels for the three complete pipelines.
    mechanism_families = ecdf_families
    fig, axes = plt.subplots(2, 3, figsize=(17.8 / 2.54, 9.0 / 2.54),
                             constrained_layout=True)
    for col, (dataset, split, title) in enumerate(GROUPS):
        for row_index, (metric, ylabel, ideal) in enumerate(
                (("tlr", "Trajectory-length ratio", 1.0),
                 ("mcs", "Mean direction cosine", 1.0))):
            ax = axes[row_index, col]
            for family, color, marker in zip(mechanism_families,
                                              (COLORS["blue"], COLORS["red"], COLORS["green"]),
                                              ("o", "s", "^")):
                row = aggregate[(family, dataset, split)]
                ax.errorbar([LABELS[family]], [row[f"mean_{metric}"]],
                            yerr=[row[f"sd_{metric}"]], marker=marker, color=color,
                            capsize=3, linestyle="none")
            ax.axhline(ideal, color=COLORS["gray"], linewidth=.7, linestyle=":")
            ax.tick_params(axis="x", labelrotation=28)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabel)
    save_both(fig, "fig_rq4_scale_direction_channels")

    # Markdown tables generated from the same machine-readable summary.
    lines = [
        "# RQ1–RQ4 results (auto-generated; do not edit numbers)", "",
        f"Source: `{data['source']}`", f"Matrix: `{data['matrix']}`", "",
        "All cells report four-seed mean ± sample SD. Public-test exposure predates the v3 addendum; "
        "the matrix is frozen prospectively with respect to these new component results but is not an untouched confirmation study.", "",
        "## RQ1: complete-pipeline comparison", "",
        "| Pipeline | RoNIN seen ATE (m) | RoNIN unseen ATE (m) | RIDI ATE (m) |",
        "|---|---:|---:|---:|",
    ]
    for family in ecdf_families:
        lines.append("| " + LABELS[family] + " | " + " | ".join(
            fmt(aggregate[(family, ds, split)]["mean_ate"],
                aggregate[(family, ds, split)]["sd_ate"])
            for ds, split, _ in GROUPS) + " |")
    lines += ["", "## RQ2: HN-Transformer component ablation", "",
              "| Configuration | RoNIN seen ATE (m) | RoNIN unseen ATE (m) | RIDI ATE (m) |",
              "|---|---:|---:|---:|"]
    for family in ablations:
        lines.append("| " + LABELS[family] + " | " + " | ".join(
            fmt(aggregate[(family, ds, split)]["mean_ate"],
                aggregate[(family, ds, split)]["sd_ate"])
            for ds, split, _ in GROUPS) + " |")
    lines += ["", "## RQ2: backbone-only comparison", "",
              "All arms use HeadingNorm + GlobalNorm, raw six-channel input, Huber loss, AdamW, "
              "OneCycleLR, 20,000 updates, batch 256, and seeds 0–3. PreFilter is off in all arms.", "",
              "| Backbone | Parameters | RoNIN seen ATE (m) | RoNIN unseen ATE (m) | RIDI ATE (m) |",
              "|---|---:|---:|---:|---:|"]
    training = defaultdict(list)
    for row in data["training_runs"]:
        training[row["family"]].append(row)
    for family in backbones:
        params = {row["parameters"] for row in training[family]}
        if len(params) != 1:
            raise RuntimeError(f"parameter count differs within family {family}")
        lines.append(f"| {BACKBONE_LABELS[family]} | {next(iter(params)):,} | " + " | ".join(
            fmt(aggregate[(family, ds, split)]["mean_ate"],
                aggregate[(family, ds, split)]["sd_ate"])
            for ds, split, _ in GROUPS) + " |")
    lines += ["", "## RQ2: HeadingNorm × GlobalNorm interaction", "",
              "Interaction is `ATE(on,on) − ATE(off,on) − ATE(on,off) + ATE(off,off)`; "
              "zero denotes additivity on the ATE scale.", "",
              "| Test group | Interaction mean ± SD (m) | Four seed values (m) |",
              "|---|---:|---|"]
    interaction_grouped = defaultdict(list)
    for row in data["headingnorm_globalnorm_interaction"]:
        interaction_grouped[(row["dataset"], row["split"])].append(row["interaction_ate"])
    for dataset, split, label in GROUPS:
        values = np.asarray(interaction_grouped[(dataset, split)], dtype=float)
        lines.append(f"| {label} | {values.mean():.3f} ± {values.std(ddof=1):.3f} | "
                     + ", ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "## RQ1/RQ3: per-sequence failure-tail summary", "",
              "ATE is first averaged over four seeds for each sequence; quantiles are then computed across sequences.", "",
              "| Pipeline | Test group | n | Median (m) | P90 (m) | P95 (m) |",
              "|---|---|---:|---:|---:|---:|"]
    qlookup = {(row["family"], row["dataset"], row["split"]): row
               for row in data["sequence_quantiles"]}
    for family in ecdf_families:
        for dataset, split, label in GROUPS:
            row = qlookup[(family, dataset, split)]
            lines.append(f"| {LABELS[family]} | {label} | {row['n_sequences']} | "
                         f"{row['median_ate']:.3f} | {row['p90_ate']:.3f} | "
                         f"{row['p95_ate']:.3f} |")
    lines += ["", "## RQ4: speed-scale and direction channels", "",
              "| Pipeline | Test group | TLR | MCS | Speed residual RMS (m/s) | Direction residual RMS (m/s) | Median angular residual (deg) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for family in mechanism_families:
        for dataset, split, label in GROUPS:
            row = aggregate[(family, dataset, split)]
            lines.append(f"| {LABELS[family]} | {label} | "
                         f"{fmt(row['mean_tlr'], row['sd_tlr'])} | "
                         f"{fmt(row['mean_mcs'], row['sd_mcs'])} | "
                         f"{fmt(row['mean_rc_rms_speed'], row['sd_rc_rms_speed'])} | "
                         f"{fmt(row['mean_rc_rms_dir'], row['sd_rc_rms_dir'])} | "
                         f"{fmt(row['mean_rc_dtheta_med'], row['sd_rc_dtheta_med'], 2)} |")
    lines += ["", "## Selected validation checkpoints", "",
              "| Family | Validation vector error (m/s) | Best steps by seed |",
              "|---|---:|---|"]
    ordered = list(ablations) + ["a6_without_headingnorm_and_globalnorm",
                                 "resnet18_shared_frontend", "imunet_shared_frontend"]
    for family in ordered:
        rows_family = sorted(training[family], key=lambda row: row["seed"])
        values = np.asarray([row["selected_val_verr"] for row in rows_family], dtype=float)
        steps = ", ".join(str(row["best_step"]) for row in rows_family)
        lines.append(f"| {LABELS[family]} | {values.mean():.4f} ± {values.std(ddof=1):.4f} | {steps} |")
    lines.append("")
    PAPER_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[写出] {PAPER_OUT.relative_to(ROOT)}")
    for path in sorted(FIG_DIR.glob("fig_rq*")):
        print(f"[写出] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
