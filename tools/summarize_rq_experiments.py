#!/usr/bin/env python3
"""审计并汇总冻结的 RQ1--RQ4 实验矩阵。"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "rq_experiment_matrix_v3.json"
MODEL_DIR = ROOT / "models" / "submission_frozen"   # 2026-10-01：models/generated 是其逐字节副本，已删
RESULT_DIR = ROOT / "results" / "generated"
METRICS = ("ate", "rte", "tlr", "mcs", "r4", "rc_rms_speed", "rc_rms_dir",
           "rc_dtheta_med")


def load_matrix() -> tuple[dict, dict[str, dict]]:
    matrix = json.loads(CONFIG.read_text(encoding="utf-8"))
    specs = {}
    for source in ("reused_frozen_references", "new_families"):
        for spec in matrix[source]:
            for seed in spec["seeds"]:
                tag = spec["tag_template"].format(seed=seed)
                if tag in specs:
                    raise RuntimeError(f"duplicate expected tag: {tag}")
                specs[tag] = {"family": spec["id"], "seed": seed, "source": source,
                              "role": spec.get("role", []), "args": spec["args"]}
    return matrix, specs


def finite_mean(rows: list[dict], key: str) -> float | None:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    source = RESULT_DIR / args.file
    rows = json.loads(source.read_text(encoding="utf-8"))
    matrix, specs = load_matrix()

    evaluated = {row["model"] for row in rows}
    expected = set(specs)
    if evaluated != expected:
        raise RuntimeError(
            f"incomplete RQ matrix: missing={sorted(expected-evaluated)}, "
            f"extra={sorted(evaluated-expected)}")
    if any(row.get("align") != "registered" for row in rows):
        raise RuntimeError("RQ primary summary accepts registered alignment only")

    expected_counts = {("ronin", "seen"): 32, ("ronin", "unseen"): 32,
                       ("ridi", ""): 94}
    buckets = defaultdict(list)
    for row in rows:
        key = (row["model"], row["dataset"], row.get("split", ""))
        buckets[key].append(row)
    for tag in expected:
        for (dataset, split), count in expected_counts.items():
            actual = len(buckets[(tag, dataset, split)])
            if actual != count:
                raise RuntimeError(f"{tag}/{dataset}/{split}: {actual} != {count}")

    # Validate training metadata and checkpoint selection for every arm.
    training_runs = []
    train_file_reference = None
    val_file_reference = None
    split_digest_reference = None
    for tag, spec in sorted(specs.items(), key=lambda item: (item[1]["family"], item[1]["seed"])):
        meta_path = MODEL_DIR / f"{tag}.json"
        checkpoint_path = MODEL_DIR / f"{tag}.pt"
        if not meta_path.exists() or not checkpoint_path.exists():
            raise RuntimeError(f"missing model artifact for {tag}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("seed") != spec["seed"] or meta.get("steps") != 20000:
            raise RuntimeError(f"seed/update mismatch for {tag}")
        if meta.get("batch_size") != 256 or meta.get("sampling") != "uniform_window":
            raise RuntimeError(f"batch/sampling mismatch for {tag}")
        if meta.get("subject_overlap"):
            raise RuntimeError(f"subject leakage for {tag}")
        train_files = tuple(row["path"] for row in meta["train_files"])
        val_files = tuple(row["path"] for row in meta["val_files"])
        split_digest = meta["split_config_sha256"]
        if train_file_reference is None:
            train_file_reference = train_files
            val_file_reference = val_files
            split_digest_reference = split_digest
        if train_files != train_file_reference or val_files != val_file_reference:
            raise RuntimeError(f"training/validation file mismatch for {tag}")
        if split_digest != split_digest_reference:
            raise RuntimeError(f"split config mismatch for {tag}")
        selected = min(meta["history"], key=lambda item: item["val_loss"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("best_step") != selected["step"]:
            raise RuntimeError(f"checkpoint selection mismatch for {tag}")
        if not math.isclose(float(checkpoint["val_loss"]), float(selected["val_loss"]),
                            abs_tol=1e-12):
            raise RuntimeError(f"checkpoint validation loss mismatch for {tag}")
        training_runs.append({
            "family": spec["family"], "model": tag, "seed": spec["seed"],
            "source": spec["source"], "roles": spec["role"],
            "parameters": meta["parameters"], "train_hours": meta["train_hours"],
            "n_train_seq": meta["n_train_seq"],
            "n_train_subjects": len(meta["train_subjects"]),
            "n_val_seq": meta["n_val_seq"],
            "n_val_subjects": len(meta["val_subjects"]),
            "best_step": selected["step"], "best_val_loss": selected["val_loss"],
            "selected_val_verr": selected["val_verr"], "minutes": meta["minutes"],
            "arch": meta["arch"], "pre": bool(checkpoint.get("pre", False)),
            "hnorm": bool(checkpoint.get("hnorm", False)),
            "gnorm": bool(checkpoint.get("gnorm", False)),
            "loss": checkpoint.get("loss"),
        })

    # Sequence-level metrics, run-level means, and across-seed summaries.
    per_run = []
    for (tag, dataset, split), selected in sorted(buckets.items()):
        record = {"family": specs[tag]["family"], "model": tag,
                  "seed": specs[tag]["seed"], "dataset": dataset, "split": split,
                  "n_sequences": len(selected)}
        for metric in METRICS:
            record[f"mean_{metric}"] = finite_mean(selected, metric)
        per_run.append(record)

    aggregate = []
    aggregate_buckets = defaultdict(list)
    for row in per_run:
        aggregate_buckets[(row["family"], row["dataset"], row["split"])].append(row)
    for (family, dataset, split), selected in sorted(aggregate_buckets.items()):
        record = {"family": family, "dataset": dataset, "split": split,
                  "n_runs": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[f"mean_{metric}"] for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            record[f"mean_{metric}"] = float(values.mean()) if len(values) else None
            record[f"sd_{metric}"] = (float(values.std(ddof=1)) if len(values) > 1 else None)
            record[f"run_values_{metric}"] = values.tolist()
        aggregate.append(record)

    sequence_buckets = defaultdict(list)
    for row in rows:
        sequence_buckets[(specs[row["model"]]["family"], row["dataset"],
                          row.get("split", ""), row["seq"])].append(row)
    per_sequence_seed_mean = []
    for (family, dataset, split, seq), selected in sorted(sequence_buckets.items()):
        if len(selected) != 4:
            raise RuntimeError(f"sequence does not have four seeds: {family}/{seq}")
        record = {"family": family, "dataset": dataset, "split": split, "seq": seq,
                  "n_seeds": len(selected)}
        for metric in METRICS:
            record[f"mean_{metric}"] = finite_mean(selected, metric)
        per_sequence_seed_mean.append(record)

    sequence_quantiles = []
    sequence_grouped = defaultdict(list)
    for row in per_sequence_seed_mean:
        sequence_grouped[(row["family"], row["dataset"], row["split"])].append(row)
    for (family, dataset, split), selected in sorted(sequence_grouped.items()):
        values = np.asarray([row["mean_ate"] for row in selected], dtype=float)
        sequence_quantiles.append({
            "family": family, "dataset": dataset, "split": split,
            "n_sequences": len(values), "median_ate": float(np.median(values)),
            "p90_ate": float(np.quantile(values, .90)),
            "p95_ate": float(np.quantile(values, .95)),
        })

    # Same-seed differences against full A6; negative error differences favor comparator.
    per_run_lookup = {(row["family"], row["seed"], row["dataset"], row["split"]): row
                      for row in per_run}
    paired = []
    full_family = "a6_full"
    other_families = sorted({row["family"] for row in per_run} - {full_family})
    for family in other_families:
        for seed in range(4):
            for dataset, split in expected_counts:
                full = per_run_lookup[(full_family, seed, dataset, split)]
                other = per_run_lookup[(family, seed, dataset, split)]
                record = {"comparison": f"{family} minus {full_family}",
                          "family": family, "seed": seed,
                          "dataset": dataset, "split": split}
                for metric in METRICS:
                    a, b = other[f"mean_{metric}"], full[f"mean_{metric}"]
                    record[f"difference_{metric}"] = (
                        float(a - b) if a is not None and b is not None else None)
                paired.append(record)

    cells = matrix["interaction"]["cells"]
    interaction = []
    for seed in range(4):
        families = {}
        for cell, template in cells.items():
            tag = template.format(seed=seed)
            families[cell] = specs[tag]["family"]
        for dataset, split in expected_counts:
            record = {"seed": seed, "dataset": dataset, "split": split,
                      "formula": "on_on - off_on - on_off + off_off"}
            for metric in METRICS:
                values = {cell: per_run_lookup[(family, seed, dataset, split)][f"mean_{metric}"]
                          for cell, family in families.items()}
                record[f"interaction_{metric}"] = (
                    float(values["on_on"] - values["off_on"] - values["on_off"] +
                          values["off_off"])
                    if all(value is not None for value in values.values()) else None)
            interaction.append(record)

    output = {
        "source": source.relative_to(ROOT).as_posix(),
        "matrix": CONFIG.relative_to(ROOT).as_posix(),
        "matrix_id": matrix["matrix_id"], "matrix_complete": True,
        "alignment": "registered", "failed_runs_retained": True,
        "prior_public_test_exposure_disclosed": True,
        "training_runs": training_runs, "per_run": per_run,
        "aggregate": aggregate,
        "per_sequence_seed_mean": per_sequence_seed_mean,
        "sequence_quantiles": sequence_quantiles,
        "paired_other_minus_full_a6": paired,
        "headingnorm_globalnorm_interaction": interaction,
    }
    target = RESULT_DIR / f"{source.stem}_summary.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] models={len(expected)} rows={len(rows)}")
    print(f"[OK] train/val files and checkpoint selection agree across {len(training_runs)} runs")
    print(f"[写出] {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
