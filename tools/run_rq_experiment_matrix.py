#!/usr/bin/env python3
"""执行冻结的 RQ1--RQ4 组件/主干矩阵，并保存可审计日志。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "rq_experiment_matrix_v3.json"
LOG_DIR = ROOT / "logs" / "rq_v3"
MODEL_DIR = ROOT / "models" / "submission_frozen"   # 2026-10-01：models/generated 是其逐字节副本，已删


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def load_matrix() -> dict:
    matrix = json.loads(CONFIG.read_text(encoding="utf-8"))
    if matrix.get("matrix_id") != "a6-rq1-rq4-v3":
        raise RuntimeError("unexpected RQ matrix id")
    if matrix.get("frozen_on") != "2026-09-01":
        raise RuntimeError("unexpected RQ matrix freeze date")
    return matrix


def expand_new(matrix: dict) -> list[tuple[str, str, list[str]]]:
    runs = []
    tags = set()
    for family in matrix["new_families"]:
        for seed in family["seeds"]:
            tag = family["tag_template"].format(seed=seed)
            if tag in tags:
                raise RuntimeError(f"duplicate matrix tag: {tag}")
            tags.add(tag)
            values = dict(matrix["common"])
            values.update(family["args"])
            values.update(tag=tag, seed=seed)
            cmd = [sys.executable, str(ROOT / "src" / "pkg_train.py")]
            for key, value in values.items():
                cmd.extend([flag(key), str(value)])
            runs.append((family["id"], tag, cmd))
    return runs


def expected_reference_tags(matrix: dict) -> list[str]:
    return [
        spec["tag_template"].format(seed=seed)
        for spec in matrix["reused_frozen_references"]
        for seed in spec["seeds"]
    ]


def validate_reference_artifacts(matrix: dict) -> None:
    missing = []
    for tag in expected_reference_tags(matrix):
        for suffix in (".pt", ".json"):
            path = MODEL_DIR / f"{tag}{suffix}"
            if not path.exists():
                missing.append(path.relative_to(ROOT).as_posix())
    if missing:
        raise RuntimeError(f"missing frozen reference artifacts: {missing}")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        raise SystemExit("--train-only and --eval-only are mutually exclusive")

    matrix = load_matrix()
    validate_reference_artifacts(matrix)
    runs = expand_new(matrix)
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    print(f"[RQ矩阵] id={matrix['matrix_id']} new_runs={len(runs)} "
          f"config_sha256={digest}", flush=True)
    for index, (family, tag, cmd) in enumerate(runs, 1):
        print(f"[{index:02d}/{len(runs):02d}] {family}: {tag}", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
        elif not args.eval_only:
            run_logged(cmd, LOG_DIR / f"{tag}.log")

    if args.dry_run or args.train_only:
        return

    all_tags = expected_reference_tags(matrix) + [tag for _, tag, _ in runs]
    out_name = "eval_rq1_rq4_v3_registered.json"
    eval_cmd = [
        sys.executable, str(ROOT / "src" / "pkg_eval.py"), *all_tags,
        "--datasets", "ronin,ridi", "--alignment", "registered",
        "--out", out_name,
    ]
    started = time.time()
    run_logged(eval_cmd, LOG_DIR / "evaluation.log")
    run_logged([
        sys.executable, str(ROOT / "tools" / "summarize_rq_experiments.py"),
        out_name,
    ], LOG_DIR / "summary.log")
    print(f"[完成] RQ矩阵训练/评测/汇总，用时 {(time.time()-started)/60:.1f} min（不含训练）")


if __name__ == "__main__":
    main()
