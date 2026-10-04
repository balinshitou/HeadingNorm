"""生成 RTE、前端移植和逐序列显著性表及配套图。

所有数字均来自冻结的逐序列记录，不手工录入。逐序列配对检验用于补充四种子的描述区间；
由于同一受试者的多条序列不严格独立，该检验只支持“效果覆盖多数已测序列”，不能直接外推
为总体人群结论。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "submission" / "rq_v3"
# Palette and rcParams mirror tools/make_rq_paper_assets.py so the new panel matches
# the other submission figures.
COLORS = {"blue": "#0072B2", "orange": "#E69F00"}

GROUPS = (("ronin", "seen", "RoNIN seen"),
          ("ronin", "unseen", "RoNIN unseen"),
          ("ridi", "", "RIDI"))

# Full-pipeline arms used in RQ1.
MAIN = (("HN-Transformer (ours)", "A6_v2_f100_s"),
        ("ResNet18", "ResNet18_v2_matched_s"),
        ("IMUNet", "IMUNet2024_v21_matched_s"))

# Front-end transplant: identical backbone and budget, HeadingNorm+GlobalNorm off vs on.
# PreFilter is off in both arms, so the difference is exactly the HN+GN pair.
TRANSPLANT = (("ResNet18", "ResNet18_v2_matched_s", "ResNet18_v3_hn_gn_s"),
              ("IMUNet", "IMUNet2024_v21_matched_s", "IMUNet2024_v3_hn_gn_s"))

SEEDS = range(4)


def load(path: Path):
    return json.load(open(path, encoding="utf-8"))


def seed_means(rows, prefix, ds, split, key):
    """每个种子返回一个值：该种子全部序列的均值。"""
    out = []
    for s in SEEDS:
        vals = [r[key] for r in rows
                if r["model"] == f"{prefix}{s}" and r["dataset"] == ds and r["split"] == split]
        if not vals:
            raise RuntimeError(f"missing run {prefix}{s} on {ds}/{split}")
        out.append(float(np.mean(vals)))
    return np.asarray(out)


def per_sequence(rows, prefix, ds, split, key):
    """每条序列先跨四个种子求均值。"""
    acc = defaultdict(list)
    for s in SEEDS:
        for r in rows:
            if r["model"] == f"{prefix}{s}" and r["dataset"] == ds and r["split"] == split:
                acc[r["seq"]].append(r[key])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def p_bound(p: float) -> str:
    """把 p 值报告为界限，避免没有意义的过多小数位。"""
    for thr in (0.001, 0.01, 0.05):
        if p < thr:
            return f"p < {thr:g}"
    return f"p = {p:.2f} (n.s.)"


def plot_transplant(rows) -> None:
    """绘制把同一结构化前端接到两种卷积主干后的成组柱状图。

    图中只放 ResNet18 和 IMUNet；两者开/关前端时均关闭 PreFilter，因此柱对隔离的是
    HeadingNorm+GlobalNorm 联合作用。Transformer 缺少完全对应的配对臂，强行加入会混入
    PreFilter 差异。
    """
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
        "savefig.dpi": 600,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 5.8 / 2.54),
                             constrained_layout=True)
    width = 0.34
    for ax, (ds, split, title) in zip(axes, GROUPS):
        for xi, (name, off, on) in enumerate(TRANSPLANT):
            a = seed_means(rows, off, ds, split, "ate")
            b = seed_means(rows, on, ds, split, "ate")
            for dx, vals, color, label in ((-width / 2, a, COLORS["orange"], "backbone only"),
                                           (+width / 2, b, COLORS["blue"], "+ HeadingNorm + GlobalNorm")):
                ax.bar(xi + dx, vals.mean(), width, yerr=vals.std(ddof=1), capsize=2.5,
                       color=color, edgecolor="black", linewidth=.5,
                       label=label if (xi == 0 and ax is axes[0]) else None)  # legend once
                ax.plot(np.full(len(vals), xi + dx), vals, ".", color="black",
                        markersize=2.5, zorder=3)
            ax.annotate(f"{100 * (b.mean() / a.mean() - 1):+.1f}%",
                        (xi, max(a.mean(), b.mean())), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7)
        ax.set_title(title)
        ax.set_xticks(range(len(TRANSPLANT)), [t[0] for t in TRANSPLANT])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    axes[0].set_ylabel("ATE (m; primary protocol)")
    # Figure-level legend keeps the per-panel percentage annotations unobstructed.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.09))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_rq2_frontend_transplant.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIG_DIR / 'fig_rq2_frontend_transplant.png').relative_to(ROOT)}")


def main():
    global FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="eval_rq1_rq4_v3_registered.json")
    ap.add_argument("--out", default="paper/generated_rte_significance.md")
    ap.add_argument("--figure-dir", type=Path,
                    help="图输出目录；相对路径按项目根目录解释")
    args = ap.parse_args()
    if args.figure_dir is not None:
        FIG_DIR = args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir

    src = ROOT / "results" / "generated" / args.records
    rows = load(src)

    L = ["# RTE, front-end transplant, and significance tables (auto-generated; do not edit numbers)",
         "", f"Source: `{src.relative_to(ROOT).as_posix()}` ({len(rows)} model-sequence records)",
         "", "Four-seed mean ± sample SD under the pre-specified main alignment protocol.", ""]

    # ---- RTE, main pipelines -------------------------------------------------
    L += ["## RQ1: relative trajectory error (60 s)", "",
          "| Pipeline | " + " | ".join(g[2] + " RTE (m)" for g in GROUPS) + " |",
          "|---|---:|---:|---:|"]
    rte = {}
    for name, pre in MAIN:
        cells = []
        for ds, split, _ in GROUPS:
            v = seed_means(rows, pre, ds, split, "rte")
            rte[(name, ds, split)] = v
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    L += ["", "### Seed-paired RTE differences (HN-Transformer minus comparator)", "",
          "| Comparator | Test group | Mean difference (m) | Relative reduction | HN better seeds |",
          "|---|---|---:|---:|---:|"]
    for name, _ in MAIN[1:]:
        for ds, split, label in GROUPS:
            a, b = rte[("HN-Transformer (ours)", ds, split)], rte[(name, ds, split)]
            d = a - b
            L.append(f"| {name} | {label} | {d.mean():+.3f} | "
                     f"{100 * (1 - a.mean() / b.mean()):.1f}% | {int((d < 0).sum())}/4 |")

    # ---- Front-end transplant ------------------------------------------------
    L += ["", "## RQ2: structured front end transplanted onto other backbones", "",
          "Both arms share backbone, data, budget, and seeds; PreFilter is off in both, so the",
          "difference isolates the HeadingNorm+GlobalNorm pair acting jointly.", "",
          "| Backbone | Test group | Without HN+GN (m) | With HN+GN (m) | Change | Improved seeds |",
          "|---|---|---:|---:|---:|---:|"]
    for name, off, on in TRANSPLANT:
        for ds, split, label in GROUPS:
            a = seed_means(rows, off, ds, split, "ate")
            b = seed_means(rows, on, ds, split, "ate")
            d = b - a
            L.append(f"| {name} | {label} | {a.mean():.3f} | {b.mean():.3f} | "
                     f"{100 * (b.mean() / a.mean() - 1):+.1f}% | {int((d < 0).sum())}/4 |")

    # ---- Per-sequence paired tests ------------------------------------------
    L += ["", "## Per-sequence paired Wilcoxon signed-rank tests", "",
          "ATE/RTE are first averaged over the four seeds for each sequence; the paired test then",
          "runs across sequences. p-values are reported as bounds.", ""]
    for key, title in (("ate", "ATE"), ("rte", "RTE")):
        L += [f"### {title}", "",
              "| Comparator | Test group | n | Wilcoxon | HN better sequences |",
              "|---|---|---:|---|---:|"]
        for name, pre in MAIN[1:]:
            for ds, split, label in GROUPS:
                A = per_sequence(rows, "A6_v2_f100_s", ds, split, key)
                B = per_sequence(rows, pre, ds, split, key)
                seqs = sorted(set(A) & set(B))
                x = np.array([A[s] for s in seqs])
                y = np.array([B[s] for s in seqs])
                _, p = wilcoxon(x, y)
                L.append(f"| {name} | {label} | {len(seqs)} | {p_bound(p)} | "
                         f"{int((x < y).sum())}/{len(seqs)} |")
        L.append("")

    plot_transplant(rows)

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
