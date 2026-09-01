#!/usr/bin/env python3
"""Read-only closure audit for frozen M8b cached features and strict H10 labels.

This runner never opens dataset RGB, depth, or ground-truth files and never loads
STTrack, CLIP, or any tracking checkpoint.  It verifies that every cached tensor
is byte-identical to its frozen ledger entry, that its candidate axis follows the
six-branch trajectory order, and that every branch joins exactly one strict H10
label.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path, value):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        Path(temporary).write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl_gz(path, rows):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8",
                       compresslevel=6) as stream:
            for row in rows:
                stream.write(json.dumps(
                    row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_output(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments], text=True).strip()


def validate_binding(args, spec):
    binding = json_file(args.binding)
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain")
    expected = {
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "output": str(args.output),
    }
    if binding.get("complete") is not True:
        raise ValueError("binding is incomplete")
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError("binding mismatch for %s" % key)
    if not clean:
        raise ValueError("repository is not clean")
    if branch != spec["repository"]["branch"]:
        raise ValueError("repository branch mismatch")
    base = spec["repository"]["base_commit"]
    ancestor = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
         base, commit], check=False).returncode == 0
    if not ancestor:
        raise ValueError("runner commit does not descend from frozen base")
    authorizations = binding.get("authorizations", {})
    if authorizations.get("readonly_cached_closure") is not True:
        raise ValueError("binding does not authorize cached closure audit")
    forbidden = (
        "model_load", "m8b_implementation", "m8b_training",
        "tracking_checkpoint", "depthtrack_test", "cdtb", "vot_low22",
        "vot_full127", "qwen", "automatic_next_stage")
    if any(authorizations.get(name) is not False for name in forbidden):
        raise ValueError("binding contains an unsafe authorization")
    if args.output.exists():
        raise FileExistsError("output already exists")
    return binding, commit, branch


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def action_key(row):
    return (*event_key(row), str(row["branch_id"]))


def finite_h10(values):
    return (isinstance(values, list) and len(values) == 10 and
            all(value is not None and math.isfinite(float(value))
                for value in values))


def strict_label(row):
    branch = row["branch_ious"]
    public = row["public_ious"]
    if not finite_h10(branch) or not finite_h10(public):
        return "unavailable", None
    branch = [float(value) for value in branch]
    public = [float(value) for value in public]
    branch_mean = statistics.fmean(branch)
    public_mean = statistics.fmean(public)
    gain = branch_mean - public_mean
    early_hits = sum(value >= 0.5 for value in branch[:5])
    beneficial = branch_mean >= 0.5 and gain >= 0.2 and early_hits >= 2
    catastrophic = (
        (public_mean >= 0.5 and branch_mean <= 0.2) or
        gain <= -0.3 or
        (all(value <= 0.1 for value in branch) and
         not all(value <= 0.1 for value in public)))
    label = ("beneficial" if beneficial else
             "catastrophic" if catastrophic else "neutral")
    return label, {
        "branch_mean_iou": branch_mean,
        "public_mean_iou": public_mean,
        "mean_iou_gain": gain,
        "early_iou_hits": early_hits,
    }


def event_class(labels):
    return (
        "beneficial" if "beneficial" in labels else
        "catastrophic" if "catastrophic" in labels else
        "neutral" if "neutral" in labels else "unavailable")


def check_input_contract(spec):
    if sha256_file(Path(spec["plan"]["path"])) != spec["plan"]["sha256"]:
        raise ValueError("frozen plan hash mismatch")
    for shard in spec["inputs"]["shards"]:
        for name in ("event_ledger", "manifest"):
            item = shard[name]
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("frozen %s hash mismatch" % name)
        manifest = json_file(Path(shard["manifest"]["path"]))
        if manifest.get("accepted") is not True or manifest.get("complete") is not True:
            raise ValueError("collection manifest is not accepted and complete")
        forbidden_true = (
            "future_frame_text_used", "future_ground_truth_opened",
            "ground_truth_used_after_initialization", "metric_computed",
            "candidate_committed_to_public_tracker", "qwen_used",
            "depthtrack_test_run", "cdtb_run", "vot_low22_run",
            "vot_full127_run")
        if any(manifest.get(name) is not False for name in forbidden_true):
            raise ValueError("collection manifest violates read-only contract")
    labels = spec["inputs"]["labeled_actions"]
    if sha256_file(Path(labels["path"])) != labels["sha256"]:
        raise ValueError("frozen strict label hash mismatch")


def load_labels(spec):
    labels = {}
    duplicate_action_joins = 0
    label_conflicts = 0
    source_counts = Counter()
    strict_counts = Counter()
    rows = 0
    path = Path(spec["inputs"]["labeled_actions"]["path"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            key = action_key(row)
            if key in labels:
                duplicate_action_joins += 1
                continue
            recomputed, utility = strict_label(row)
            if recomputed != row.get("label"):
                label_conflicts += 1
            row["recomputed_label"] = recomputed
            row["strict_utility"] = utility
            labels[key] = row
            source_counts[str(row.get("label"))] += 1
            strict_counts[recomputed] += 1
    return {
        "by_key": labels,
        "rows": rows,
        "duplicate_action_joins": duplicate_action_joins,
        "label_conflicts": label_conflicts,
        "source_counts": source_counts,
        "strict_counts": strict_counts,
    }


def tensor_is_finite(value):
    return isinstance(value, torch.Tensor) and bool(
        torch.isfinite(value.float()).all().item())


def main():
    args = parse_args()
    spec = json_file(args.spec)
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    if spec["authorization"].get("readonly_cached_closure") is not True:
        raise ValueError("spec does not authorize read-only cached closure")
    if any(spec["authorization"].get(name) is not False for name in (
            "m8b_engineering_spec", "m8b_implementation", "m8b_training",
            "tracking_checkpoint", "depthtrack_test", "cdtb", "vot_low22",
            "vot_full127", "qwen", "automatic_next_stage")):
        raise ValueError("spec authorization drifted")
    binding, commit, branch = validate_binding(args, spec)
    check_input_contract(spec)
    label_state = load_labels(spec)
    labels = label_state["by_key"]

    expected = spec["expected"]
    expected_branches = list(expected["branches"])
    expected_feature_shapes = {
        name: tuple(shape) for name, shape in expected["feature_shapes"].items()}
    expected_anchor_shapes = {
        name: tuple(shape) for name, shape in expected["anchor_shapes"].items()}
    depth_min, depth_max = [float(value) for value in
                            spec["checks"]["verify_raw_depth_validity_range"]]

    counters = Counter()
    event_keys = set()
    joined_action_keys = set()
    sequences = set()
    feature_paths = set()
    feature_bytes_sum = 0
    anchor_by_sequence = {}
    anchor_summaries = {}
    strict_event_counts = Counter()
    closure_rows = []

    for shard_index, shard in enumerate(spec["inputs"]["shards"]):
        root = Path(shard["root"])
        ledger = Path(shard["event_ledger"]["path"])
        shard_rows = 0
        with ledger.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                shard_rows += 1
                counters["event_rows"] += 1
                row = json.loads(line)
                key = event_key(row)
                if key in event_keys:
                    counters["duplicate_event_keys"] += 1
                event_keys.add(key)
                sequence = key[0]
                sequences.add(sequence)

                trajectory = row.get("trajectory")
                branch_order_ok = isinstance(trajectory, list) and len(trajectory) >= 5
                if branch_order_ok:
                    for age in range(5):
                        age_row = trajectory[age]
                        names = [str(item.get("name"))
                                 for item in age_row.get("branches", [])]
                        if int(age_row.get("age", -1)) != age or names != expected_branches:
                            branch_order_ok = False
                            break
                if not branch_order_ok:
                    counters["candidate_axis_order_mismatches"] += 1

                feature_path = root / str(row["feature_path"])
                feature_paths.add(str(feature_path))
                expected_size = int(row["feature_bytes"])
                feature_bytes_sum += expected_size
                if not feature_path.is_file():
                    counters["missing_feature_files"] += 1
                    continue
                actual_size = feature_path.stat().st_size
                if actual_size != expected_size:
                    counters["feature_size_mismatches"] += 1
                actual_hash = sha256_file(feature_path)
                if actual_hash != row["feature_sha256"]:
                    counters["feature_hash_mismatches"] += 1

                try:
                    payload = torch.load(
                        feature_path, map_location="cpu", weights_only=True)
                except Exception:
                    counters["feature_load_errors"] += 1
                    continue
                if set(payload) != set(expected_feature_shapes):
                    counters["feature_shape_mismatches"] += 1
                for name, shape in expected_feature_shapes.items():
                    value = payload.get(name)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                        counters["feature_shape_mismatches"] += 1
                        continue
                    if not tensor_is_finite(value):
                        counters["feature_nonfinite_tensors"] += 1

                raw_depth = payload.get("raw_depth")
                if (isinstance(raw_depth, torch.Tensor) and
                        tuple(raw_depth.shape) == expected_feature_shapes["raw_depth"]):
                    validity = raw_depth[:, :, 1].float()
                    if (not tensor_is_finite(validity) or
                            float(validity.min()) < depth_min or
                            float(validity.max()) > depth_max):
                        counters["depth_validity_range_violations"] += 1

                scalars = payload.get("scalars")
                if (branch_order_ok and isinstance(scalars, torch.Tensor) and
                        tuple(scalars.shape) == expected_feature_shapes["scalars"]):
                    for age in range(5):
                        for index, branch_row in enumerate(
                                trajectory[age]["branches"]):
                            expected_triplet = (
                                float(branch_row["score"]),
                                float(branch_row["margin"]),
                                float(branch_row["entropy"]))
                            observed_triplet = tuple(
                                float(value) for value in scalars[age, index, :3])
                            if any(abs(left - right) > 1e-6 for left, right in
                                   zip(expected_triplet, observed_triplet)):
                                counters["candidate_axis_scalar_mismatches"] += 1

                anchor_path = root / str(row["anchor_path"])
                previous_anchor = anchor_by_sequence.setdefault(sequence, str(anchor_path))
                if previous_anchor != str(anchor_path):
                    counters["anchor_path_conflicts"] += 1
                if str(anchor_path) not in anchor_summaries:
                    if not anchor_path.is_file():
                        counters["missing_anchor_files"] += 1
                    else:
                        try:
                            anchor = torch.load(
                                anchor_path, map_location="cpu", weights_only=True)
                        except Exception:
                            counters["anchor_load_errors"] += 1
                            anchor = None
                        if anchor is not None:
                            if set(anchor) != set(expected_anchor_shapes):
                                counters["anchor_shape_mismatches"] += 1
                            for name, shape in expected_anchor_shapes.items():
                                value = anchor.get(name)
                                if (not isinstance(value, torch.Tensor) or
                                        tuple(value.shape) != shape):
                                    counters["anchor_shape_mismatches"] += 1
                                elif not tensor_is_finite(value):
                                    counters["anchor_nonfinite_tensors"] += 1
                            anchor_summaries[str(anchor_path)] = {
                                "path": str(anchor_path),
                                "sha256": sha256_file(anchor_path),
                                "bytes": anchor_path.stat().st_size,
                            }

                event_labels = []
                actions = []
                for branch_id in expected_branches:
                    action = labels.get((*key, branch_id))
                    if action is None:
                        counters["missing_action_joins"] += 1
                        continue
                    joined_action_keys.add((*key, branch_id))
                    label = action["recomputed_label"]
                    event_labels.append(label)
                    actions.append({
                        "branch_id": branch_id,
                        "strict_label": label,
                        "strict_utility": action["strict_utility"],
                    })
                classification = event_class(event_labels)
                strict_event_counts[classification] += 1
                closure_rows.append({
                    "sequence": sequence,
                    "event_id": key[1],
                    "trigger_frame": key[2],
                    "shard": shard_index,
                    "feature_path": str(feature_path),
                    "feature_sha256": actual_hash,
                    "feature_bytes": actual_size,
                    "anchor_path": str(anchor_path),
                    "branch_order": expected_branches,
                    "strict_event_class": classification,
                    "actions": actions,
                })
        if shard_rows != int(shard["event_ledger"]["rows"]):
            counters["ledger_row_count_mismatches"] += 1

    extra_action_keys = set(labels) - joined_action_keys
    counters["extra_action_joins"] = len(extra_action_keys)
    counters["feature_files"] = len(feature_paths)
    counters["anchor_files"] = len(anchor_summaries)
    counters["sequences_with_events"] = len(sequences)
    counters["feature_bytes_from_ledger"] = feature_bytes_sum

    expected_action_counts = Counter(expected["strict_action_counts"])
    expected_event_counts = Counter(expected["strict_event_counts"])
    gates = spec["gates"]
    conditions = {
        "event_rows_exact": counters["event_rows"] == int(expected["events"]),
        "duplicate_event_keys_zero": counters["duplicate_event_keys"] == 0,
        "sequences_exact": counters["sequences_with_events"] == int(
            expected["sequences_with_events"]),
        "feature_files_exact": counters["feature_files"] == int(
            expected["feature_files"]),
        "feature_bytes_exact": counters["feature_bytes_from_ledger"] == int(
            expected["feature_bytes_from_ledger"]),
        "anchor_files_exact": counters["anchor_files"] == int(
            expected["anchor_files"]),
        "label_rows_exact": label_state["rows"] == int(
            spec["inputs"]["labeled_actions"]["rows"]),
        "strict_action_counts_exact": label_state["strict_counts"] ==
            expected_action_counts,
        "strict_source_counts_exact": label_state["source_counts"] ==
            expected_action_counts,
        "strict_event_counts_exact": strict_event_counts == expected_event_counts,
        "missing_feature_files_max": counters["missing_feature_files"] <= int(
            gates["missing_feature_files_max"]),
        "feature_hash_mismatches_max": counters["feature_hash_mismatches"] <= int(
            gates["feature_hash_mismatches_max"]),
        "feature_size_mismatches_max": counters["feature_size_mismatches"] <= int(
            gates["feature_size_mismatches_max"]),
        "feature_shape_mismatches_max": counters["feature_shape_mismatches"] <= int(
            gates["feature_shape_mismatches_max"]),
        "feature_nonfinite_tensors_max": counters["feature_nonfinite_tensors"] <= int(
            gates["feature_nonfinite_tensors_max"]),
        "depth_validity_range_violations_max":
            counters["depth_validity_range_violations"] <= int(
                gates["depth_validity_range_violations_max"]),
        "missing_anchor_files_max": counters["missing_anchor_files"] <= int(
            gates["missing_anchor_files_max"]),
        "anchor_shape_mismatches_max": counters["anchor_shape_mismatches"] <= int(
            gates["anchor_shape_mismatches_max"]),
        "anchor_nonfinite_tensors_max": counters["anchor_nonfinite_tensors"] <= int(
            gates["anchor_nonfinite_tensors_max"]),
        "missing_action_joins_max": counters["missing_action_joins"] <= int(
            gates["missing_action_joins_max"]),
        "duplicate_action_joins_max": label_state["duplicate_action_joins"] <= int(
            gates["duplicate_action_joins_max"]),
        "label_conflicts_max": label_state["label_conflicts"] <= int(
            gates["label_conflicts_max"]),
        "extra_action_joins_zero": counters["extra_action_joins"] == 0,
        "candidate_axis_order_exact":
            counters["candidate_axis_order_mismatches"] == 0,
        "candidate_axis_scalar_exact":
            counters["candidate_axis_scalar_mismatches"] == 0,
        "feature_load_errors_zero": counters["feature_load_errors"] == 0,
        "anchor_load_errors_zero": counters["anchor_load_errors"] == 0,
        "anchor_path_conflicts_zero": counters["anchor_path_conflicts"] == 0,
        "ledger_row_count_mismatches_zero":
            counters["ledger_row_count_mismatches"] == 0,
        "original_rgb_depth_opened_false": True,
        "ground_truth_opened_false": True,
        "sttrack_checkpoint_loaded_false": True,
        "clip_checkpoint_loaded_false": True,
        "qwen_used_false": True,
        "public_state_mutations_zero": True,
    }
    accepted = all(conditions.values())
    decision = (
        "m8b_0_pass_freeze_engineering_spec_only" if accepted else
        "stop_m8b_0_cached_closure_failed")

    result = {
        "schema": "sttrack-lachtt-m8b-cached-closure-audit-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": decision,
        "claim_ceiling": spec["claim_ceiling"],
        "repository": {"path": str(REPOSITORY_ROOT), "branch": branch,
                       "commit": commit, "clean": True},
        "counts": dict(sorted(counters.items())),
        "strict_action_counts": dict(sorted(label_state["strict_counts"].items())),
        "strict_event_counts": dict(sorted(strict_event_counts.items())),
        "duplicate_action_joins": label_state["duplicate_action_joins"],
        "label_conflicts": label_state["label_conflicts"],
        "conditions": conditions,
        "unauthorized_actions": {
            "original_rgb_depth_opened": False,
            "ground_truth_opened": False,
            "sttrack_checkpoint_loaded": False,
            "clip_checkpoint_loaded": False,
            "model_forward": False,
            "training": False,
            "tracking_checkpoint_written": False,
            "depthtrack_test_run": False,
            "cdtb_run": False,
            "vot_low22_run": False,
            "vot_full127_run": False,
            "qwen_used": False,
            "automatic_next_stage": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=args.output.name + ".", dir=str(args.output.parent)))
    try:
        closure_path = temporary_root / "closure.jsonl.gz"
        result_path = temporary_root / "result.json"
        manifest_path = temporary_root / "manifest.json"
        atomic_jsonl_gz(closure_path, closure_rows)
        atomic_json(result_path, result)
        manifest = {
            "schema": "sttrack-lachtt-m8b-cached-closure-audit-manifest/v1",
            "complete": True,
            "accepted": accepted,
            "decision": decision,
            "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
            "binding": {"path": str(args.binding),
                        "sha256": sha256_file(args.binding)},
            "runner": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256_file(Path(__file__).resolve())},
            "repository_commit": commit,
            "inputs": {
                "plan": spec["plan"],
                "shards": spec["inputs"]["shards"],
                "labeled_actions": spec["inputs"]["labeled_actions"],
            },
            "outputs": {
                "result.json": {"sha256": sha256_file(result_path),
                                "bytes": result_path.stat().st_size},
                "closure.jsonl.gz": {"sha256": sha256_file(closure_path),
                                    "bytes": closure_path.stat().st_size,
                                    "rows": len(closure_rows)},
            },
            "scientific_scope": spec["claim_ceiling"],
        }
        atomic_json(manifest_path, manifest)
        for path in (result_path, closure_path, manifest_path):
            path.chmod(0o444)
        os.replace(temporary_root, args.output)
        args.output.chmod(0o555)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    print(json.dumps({
        "accepted": accepted,
        "decision": decision,
        "events": counters["event_rows"],
        "features": counters["feature_files"],
        "anchors": counters["anchor_files"],
        "missing_action_joins": counters["missing_action_joins"],
        "label_conflicts": label_state["label_conflicts"],
        "failed_conditions": sorted(
            name for name, passed in conditions.items() if not passed),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
