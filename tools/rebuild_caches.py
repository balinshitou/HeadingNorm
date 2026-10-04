"""从合法取得的官方数据目录重建论文缓存，并可逐字段对拍。

示例：把全部论文输入重建到独立目录（绝不覆盖包内冻结证据）::

    python tools/rebuild_caches.py --ronin /path/RoNIN_Dataset \
      --ronin-lists /path/ronin/lists --ridi /path/data_publish_v2 \
      --oxiod '/path/Oxford Inertial Odometry Dataset' --out /tmp/a6-cache

使用 ``--compare`` 可把每个重建数组与冻结缓存逐字段比较。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import raw_data as R  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ronin", type=Path, required=True)
    parser.add_argument("--ronin-lists", type=Path,
                        help="official list directory; optional when using bundled manifest")
    parser.add_argument("--ridi", type=Path, required=True)
    parser.add_argument("--oxiod", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path,
                        default=ROOT / "config" / "dataset_sequences.json")
    parser.add_argument("--compare", type=Path,
                        help="package root whose data/ arrays must match")
    parser.add_argument("--limit", type=int, default=0,
                        help="per group, for a smoke test; zero rebuilds all")
    return parser.parse_args()


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def assert_match(created, expected):
    if not expected.exists():
        raise FileNotFoundError(f"missing frozen comparison file: {expected}")
    with np.load(created, allow_pickle=False) as left, np.load(expected, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise RuntimeError(f"field mismatch {created.name}: {left.files} != {right.files}")
        differences = {}
        for key in left.files:
            a, b = left[key], right[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError(f"schema mismatch {created.name}:{key}: "
                                   f"{a.shape}/{a.dtype} != {b.shape}/{b.dtype}")
            if a.dtype.kind in "f":
                delta = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
                atol, rtol = ((2e-6, 1e-7) if a.dtype.itemsize <= 4 else (1e-10, 1e-12))
                if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                    raise RuntimeError(f"value mismatch {created.name}:{key}, max_abs={delta}")
                differences[key] = delta
            elif not np.array_equal(a, b):
                raise RuntimeError(f"value mismatch {created.name}:{key}")
        return differences


def limited(rows, n):
    return rows[:n] if n else rows


def main():
    args = parse_args()
    roots = R.RawRoots(args.ronin.resolve(), args.ronin_lists.resolve() if args.ronin_lists else None,
                       args.ridi.resolve(), args.oxiod.resolve())
    for path in (roots.ronin, roots.ridi, roots.oxiod):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.load(open(args.sequence_manifest, encoding="utf-8"))
    ronin_names = dict(manifest["ronin_train_validation"])
    ronin_names["test_seen"] = manifest["evaluation"]["seen"]
    ronin_names["test_unseen"] = manifest["evaluation"]["unseen"]
    counts = {"train": 0, "eval": 0, "compared": 0}
    comparison_max_abs = {}

    for record in limited(R.ronin_records(
            roots, ("train", "val"), names_by_split=ronin_names), args.limit):
        data = R.load_ronin_official(record["dir"])
        feat = np.asarray(data["features"], np.float32)
        gt64 = np.asarray(data["gt"], np.float64)
        gt = gt64.astype(np.float32)
        ts = np.asarray(data["ts_model"], np.float64)
        dt = (ts[R.WINDOW:] - ts[:-R.WINDOW])[:, None]
        targ = ((gt64[R.WINDOW:] - gt64[:-R.WINDOW]) / dt).astype(np.float32)
        out = args.out / "train" / "ronin" / f"{record['split']}__{record['seq']}.npz"
        save(out, feat=feat, targ=targ, gt=gt, ts=ts)
        counts["train"] += 1
        if args.compare:
            diffs = assert_match(out, args.compare / "data" / "train" / "ronin" / out.name)
            for key, value in diffs.items():
                comparison_max_abs[key] = max(comparison_max_abs.get(key, 0.0), value)
            counts["compared"] += 1

    ridi_wanted = set(manifest["evaluation"]["ridi"])
    oxiod_wanted = set(manifest["evaluation"]["oxiod_large"])
    eval_records = (R.ronin_records(
                        roots, ("test_seen", "test_unseen"), names_by_split=ronin_names) +
                    [r for r in R.ridi_records(roots) if r["seq"] in ridi_wanted] +
                    [r for r in R.oxiod_records(roots, only_large=True)
                     if r["seq"] in oxiod_wanted])
    groups = {}
    for record in eval_records:
        groups.setdefault(record["dataset"], []).append(record)
    for dataset in ("ronin", "ridi", "oxiod"):
        for record in limited(groups.get(dataset, []), args.limit):
            data = (R.load_ronin_official(record["dir"]) if dataset == "ronin"
                    else R.load_cross(record))
            seq = record["seq"]
            safe = f"{dataset}__{seq}".replace("/", "_").replace(" ", "-") + ".npz"
            split = ("seen" if record.get("split") == "test_seen" else
                     "unseen" if record.get("split") == "test_unseen" else "")
            out = args.out / "eval" / "benchmark" / safe
            save(out, feat=np.asarray(data["features"], np.float32),
                 tm=np.asarray(data["ts_model"], np.float64),
                 te=np.asarray(data["ts_eval"], np.float64),
                 gt=np.asarray(data["gt"], np.float32), rate=float(data["rate"]),
                 dataset=dataset, seq=seq, split=split)
            counts["eval"] += 1
            if args.compare:
                diffs = assert_match(out, args.compare / "data" / "eval" / "benchmark" / out.name)
                for key, value in diffs.items():
                    comparison_max_abs[key] = max(comparison_max_abs.get(key, 0.0), value)
                counts["compared"] += 1

    report = {"schema": "a6-cache-rebuild-v1", "counts": counts,
              "comparison_max_abs_by_field": comparison_max_abs,
              "roots": {k: str(v) if v is not None else None for k, v in vars(roots).items()},
              "comparison_root": str(args.compare.resolve()) if args.compare else None}
    (args.out / "rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
