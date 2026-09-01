#!/usr/bin/env python3
"""Aggregate the three frozen STTrack selector OOF seeds into Gate B."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


SEEDS = (2026, 2027, 2028)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.spec, args.root, args.output = (
        args.spec.resolve(), args.root.resolve(), args.output.resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if (spec.get("complete") is not True or
            args.root != Path(spec["output_root"]).resolve() or
            args.output != args.root / "final"):
        raise ValueError("selector aggregate binding mismatch")
    seed_results, seed_records = [], []
    selected_rows = []
    for seed in SEEDS:
        root = args.root / ("seed%d" % seed)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True or int(manifest["seed"]) != seed:
            raise ValueError("seed manifest mismatch")
        result_path = root / "result.json"
        if (sha256_file(result_path) != manifest["result"]["sha256"] or
                result_path.stat().st_size != manifest["result"]["bytes"]):
            raise ValueError("seed result record mismatch")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("complete") is not True or int(result["seed"]) != seed:
            raise ValueError("seed result mismatch")
        seed_results.append(result)
        seed_records.append({"seed": seed, "manifest": record(manifest_path),
                             "result": record(result_path)})
        for fold in result["fold_results"]:
            for row in fold.get("outer_selected", []):
                selected_rows.append({"seed": seed,
                                      "outer_fold": fold["outer_fold"], **row})
    per_seed = []
    for result in seed_results:
        overall = result["overall"]
        conditions = {
            "evaluated_outer_folds_ge_4":
                int(overall["evaluated_outer_folds"]) >= 4,
            "actions_ge_20": int(overall["actions"]) >= 20,
            "selected_sequences_ge_8":
                int(overall["selected_sequences"]) >= 8,
            "precision_ge_0_95": float(overall["precision"]) >= 0.95,
            "catastrophic_eq_0": int(overall["catastrophic"]) == 0,
            "zero_actions_is_failure": int(overall["actions"]) > 0,
        }
        per_seed.append({
            "seed": result["seed"], "overall": overall,
            "conditions": conditions, "passed": all(conditions.values()),
        })
    gate_passed = all(row["passed"] for row in per_seed)
    result = {
        "schema": "sttrack-lachtt-selector-oof-gateb-result/v1",
        "complete": True, "accepted": True,
        "per_seed": per_seed,
        "gate_b_passed": gate_passed,
        "decision": ("train_final_selector_train_only"
                     if gate_passed else
                     "stop_sttrack_selector_no_final_no_online_replay"),
        "selected_rows": selected_rows,
        "automatic_next_stage": False,
        "final_selector_training_authorized": gate_passed,
        "online_replay_authorized": False,
        "depthtrack_test_authorized": False,
        "cdtb_authorized": False,
        "vot_low22_authorized": False,
        "vot_full127_authorized": False,
        "claim_ceiling": "DepthTrack Train-152 nested sequence OOF only",
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "gate_b_result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-selector-oof-gateb-manifest/v1",
        "complete": True, "accepted": True,
        "spec": record(args.spec), "seed_records": seed_records,
        "aggregator": record(Path(__file__).resolve()),
        "result": record(result_path),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
