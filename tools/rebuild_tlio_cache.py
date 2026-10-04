"""从 TLIO golden 发布数据重建事后官方测试集缓存。

输出数组是 VIO 派生、已偏置补偿的世界系 IMU；它与 186 条手机主评价物理分开，不能描述
为只使用原始 IMU 的基准实验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import raw_data as R  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tlio", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/tlio_posthoc")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/tlio_extension_v2_2.json")
    parser.add_argument("--sequence-manifest", type=Path,
                        default=ROOT / "config/dataset_sequences.json")
    parser.add_argument("--record", type=Path,
                        default=ROOT / "provenance/tlio_cache_rebuild_validation.json")
    args = parser.parse_args()
    tlio = args.tlio.resolve()
    test_list = tlio / "test_list.txt"
    if not test_list.exists():
        raise FileNotFoundError(test_list)
    config = json.load(open(args.config, encoding="utf-8"))
    manifest = json.load(open(args.sequence_manifest, encoding="utf-8"))
    wanted = manifest["evaluation"]["tlio_official_test"]
    official = test_list.read_text(encoding="utf-8").split()
    expected_hash = config["dataset"]["official_test_list_sha256"]
    observed_hash = sha256(test_list)
    if observed_hash != expected_hash:
        raise RuntimeError(f"TLIO test_list.txt SHA-256 mismatch: {observed_hash}")
    if official != wanted or len(wanted) != 36 or len(set(wanted)) != 36:
        raise RuntimeError("bundled TLIO sequence manifest does not match the official test list")

    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for record in R.tlio_records(tlio, wanted):
        data = R.load_tlio_processed(record)
        out = args.out / f"tlio__{record['seq']}.npz"
        np.savez_compressed(
            out,
            feat=np.asarray(data["features"], np.float32),
            tm=np.asarray(data["ts_model"], np.float64),
            te=np.asarray(data["ts_eval"], np.float64),
            gt=np.asarray(data["gt"], np.float32),
            rate=float(data["rate"]),
            dataset="tlio",
            seq=record["seq"],
            split="official_test",
            analysis_status="post_hoc_exploratory_not_confirmatory",
            input_attitude_source=data["input_attitude_source"],
        )
        source = Path(record["dir"]) / "imu0_resampled.npy"
        files.append({"seq": record["seq"], "source_sha256": sha256(source),
                      "cache_sha256": sha256(out), "samples": len(data["features"])})
    existing = sorted(args.out.glob("tlio__*.npz"))
    if len(existing) != 36 or {p.stem.split("__", 1)[1] for p in existing} != set(wanted):
        raise RuntimeError("TLIO cache directory contains a non-frozen sequence set")
    report = {
        "schema": "tlio-posthoc-cache-rebuild-v2.2",
        "result": "pass",
        "analysis_status": config["status"],
        "input_attitude_source": config["input"]["attitude_source"],
        "official_test_list_sha256": observed_hash,
        "n_sequences": len(files),
        "files": files,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TLIO缓存] official test={len(files)}; out={args.out}")
    print(f"[TLIO记录] {args.record}")


if __name__ == "__main__":
    main()
