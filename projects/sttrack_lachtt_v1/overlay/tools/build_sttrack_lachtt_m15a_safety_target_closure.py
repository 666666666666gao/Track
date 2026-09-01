#!/usr/bin/env python3
"""Build an audited read-only multi-horizon safety-target closure."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_record_as(path, reported_path):
    record = file_record(path)
    record["path"] = str(Path(reported_path).resolve())
    return record


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl_gz(path, rows):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(
                    row, sort_keys=True, ensure_ascii=False,
                    allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_output(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True).strip()


def action_key(sequence, event_id, trigger_frame, branch_id):
    return (str(sequence), int(event_id), int(trigger_frame), str(branch_id))


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m15a-safety-target-closure-binding/v1",
        "complete": True,
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "output": str(args.output),
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    audit = binding.get("pre_execution_plan_audit", {})
    if (audit.get("path") !=
            "/home/SUTrack_RGBD_L/refine-logs/"
            "EXPERIMENT_AUDIT_M15A_SAFETY_TARGET_CLOSURE_"
            "PREEXECUTION_20260901.json" or
            audit.get("sha256") !=
            "8b5e99e5c0528cd99bffe2d2f368a21b1cf5458be1ce937fcc4cf4804bb698ec" or
            sha256_file(Path(audit["path"])) != audit["sha256"]):
        raise ValueError("pre-execution audit binding drifted")
    if (git_output("branch", "--show-current") !=
            spec["repository"]["branch"] or
            git_output("status", "--porcelain")):
        raise ValueError("repository state drifted")
    if subprocess.run([
            "git", "-C", str(REPOSITORY_ROOT), "merge-base",
            "--is-ancestor", spec["repository"]["base_commit"], commit,
    ], check=False).returncode != 0:
        raise ValueError("implementation is outside frozen ancestry")
    authorizations = binding.get("authorizations", {})
    if (authorizations.get("pre_execution_plan_audit_passed") is not True or
            authorizations.get("one_readonly_target_closure_run") is not True or
            authorizations.get("independent_result_audit_after_run") is not True or
            authorizations.get("m15b_engineering_plan_if_pass") is not True):
        raise ValueError("binding does not authorize M15a")
    for name in (
            "m15b_engineering_execution", "m15c_capacity",
            "sequence_disjoint_pilot", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, commit


def frozen_records(spec, binding):
    records = []
    for name in ("plan", "source_batch_spec", "labeled_actions"):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"]))
    for group in ("sealed_closure", "m14b_r1_negative_boundary"):
        for name, item in spec[group].items():
            if isinstance(item, dict) and "path" in item:
                records.append((group + ":" + name,
                                Path(item["path"]), item["sha256"]))
    audit = binding["pre_execution_plan_audit"]
    records.append(("pre_execution_plan_audit",
                    Path(audit["path"]), audit["sha256"]))
    return records


def verify_frozen(records):
    observed = []
    mismatches = []
    for name, path, expected_sha in records:
        if not path.is_file():
            mismatches.append({"name": name, "reason": "missing"})
            continue
        actual = sha256_file(path)
        observed.append({
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        })
        if actual != expected_sha:
            mismatches.append({
                "name": name,
                "reason": "sha256",
                "expected": expected_sha,
                "actual": actual,
            })
    return mismatches, observed


def load_selected_closure(spec, source_spec):
    selected_events = {
        (str(row["sequence"]), int(row["event_id"]),
         int(row["trigger_frame"]))
        for row in source_spec["selection"]["events"]
    }
    closure_rows = {}
    path = Path(spec["sealed_closure"]["closure"]["path"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            event = (str(row["sequence"]), int(row["event_id"]),
                     int(row["trigger_frame"]))
            if event not in selected_events:
                continue
            if event in closure_rows:
                raise ValueError("duplicate selected closure event")
            closure_rows[event] = row
    return selected_events, closure_rows


def load_selected_actions(spec, selected_keys):
    rows = {}
    duplicate_joins = 0
    total_rows = 0
    path = Path(spec["labeled_actions"]["path"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            total_rows += 1
            row = json.loads(line)
            key = action_key(
                row["sequence"], row["event_id"], row["trigger_frame"],
                row["branch_id"])
            if key not in selected_keys:
                continue
            if key in rows:
                duplicate_joins += 1
            else:
                rows[key] = row
    return rows, duplicate_joins, total_rows


def trailing_low_count(values, threshold):
    count = 0
    for value in reversed(values):
        if float(value) <= float(threshold):
            count += 1
        else:
            break
    return count


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    if (authorization.get(
            "pre_execution_independent_plan_audit_required") is not True or
            authorization.get("implementation_after_plan_audit_pass") is not True or
            authorization.get(
                "one_readonly_target_closure_run_after_binding") is not True or
            authorization.get("independent_result_audit_after_run") is not True or
            authorization.get("m15b_engineering_plan_if_pass") is not True):
        raise ValueError("spec does not authorize M15a")
    for name in (
            "m15b_engineering_execution", "m15c_capacity",
            "sequence_disjoint_pilot", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)
    binding, runner_path, commit = validate_binding(args, spec)
    audit = json_file(binding["pre_execution_plan_audit"]["path"])
    if (str(audit.get("protocol_gate", "")).lower() != "pass" or
            str(audit.get("integrity_verdict", "")).lower() != "pass" or
            audit.get("authorized") is not True or
            "one_readonly_target_closure_run_after_binding" not in
            audit.get("authorized_next_actions", [])):
        raise ValueError("M15a pre-execution audit drifted")
    boundary = json_file(
        spec["m14b_r1_negative_boundary"]["result"]["path"])
    if (boundary.get("accepted") is not False or
            boundary.get("decision") !=
            "m14b_r1_fail_stop_without_rescan"):
        raise ValueError("M14b-R1 stop boundary drifted")

    records = frozen_records(spec, binding)
    before_mismatches, observed = verify_frozen(records)
    source_spec = json_file(spec["source_batch_spec"]["path"])
    selected_events, closure_by_event = load_selected_closure(
        spec, source_spec)
    branch_order = tuple(spec["join"]["branch_order"])
    selected_keys = {
        action_key(sequence, event_id, trigger_frame, branch)
        for sequence, event_id, trigger_frame in selected_events
        for branch in branch_order
    }
    action_rows, duplicate_joins, labeled_rows = load_selected_actions(
        spec, selected_keys)

    counters = Counter()
    target_rows = []
    label_counts = Counter()
    horizon_counts = Counter()
    maximum_h10_parity_error = 0.0
    available_actions = 0
    threshold = float(spec["target_definition"]["low_overlap_threshold"])
    horizons = tuple(int(value) for value in
                     spec["target_definition"]["horizons"])
    for selected in source_spec["selection"]["events"]:
        event = (str(selected["sequence"]), int(selected["event_id"]),
                 int(selected["trigger_frame"]))
        closure = closure_by_event.get(event)
        if closure is None:
            counters["missing_closure_events"] += 1
            continue
        if tuple(closure.get("branch_order", [])) != branch_order:
            counters["branch_order_mismatches"] += 1
        actions = closure.get("actions", [])
        closure_actions = {
            str(row["branch_id"]): row for row in actions
        }
        if tuple(row.get("branch_id") for row in actions) != branch_order:
            counters["branch_order_mismatches"] += 1
        for candidate_index, branch_id in enumerate(branch_order):
            key = action_key(*event, branch_id)
            strict = closure_actions.get(branch_id)
            source = action_rows.get(key)
            if strict is None or source is None:
                counters["missing_joins"] += 1
                continue
            strict_label = str(strict["strict_label"])
            source_label = str(source["label"])
            if strict_label != source_label:
                counters["strict_label_mismatches"] += 1
            label_counts[strict_label] += 1
            if strict_label == "unavailable":
                counters["unexpected_unavailable_actions"] += 1
                continue
            available_actions += 1
            branch_ious = source.get("branch_ious", [])
            public_ious = source.get("public_ious", [])
            if len(branch_ious) != 10 or len(public_ious) != 10:
                counters["trajectory_length_mismatches"] += 1
                continue
            values = [float(value) for value in
                      tuple(branch_ious) + tuple(public_ious)]
            if not all(math.isfinite(value) for value in values):
                counters["trajectory_nonfinite_values"] += 1
                continue
            if any(value < 0.0 or value > 1.0 for value in values):
                counters["derived_range_violations"] += 1
            per_action = []
            for horizon in horizons:
                branch_prefix = values[:horizon]
                public_prefix = values[10:10 + horizon]
                branch_mean = sum(branch_prefix) / horizon
                public_mean = sum(public_prefix) / horizon
                gain = branch_mean - public_mean
                low_fraction = sum(
                    value <= threshold for value in branch_prefix) / horizon
                trailing_fraction = trailing_low_count(
                    branch_prefix, threshold) / horizon
                derived = (
                    branch_mean, public_mean, gain, low_fraction,
                    trailing_fraction)
                if not all(math.isfinite(value) for value in derived):
                    counters["derived_nonfinite_values"] += 1
                if (branch_mean < 0.0 or branch_mean > 1.0 or
                        public_mean < 0.0 or public_mean > 1.0 or
                        gain < -1.0 or gain > 1.0 or
                        low_fraction < 0.0 or low_fraction > 1.0 or
                        trailing_fraction < 0.0 or trailing_fraction > 1.0):
                    counters["derived_range_violations"] += 1
                row = {
                    "sequence": event[0],
                    "event_id": event[1],
                    "trigger_frame": event[2],
                    "candidate_index": candidate_index,
                    "branch_id": branch_id,
                    "strict_label": strict_label,
                    "available": True,
                    "horizon": horizon,
                    "branch_mean_iou": branch_mean,
                    "public_mean_iou": public_mean,
                    "gain": gain,
                    "low_overlap_fraction": low_fraction,
                    "trailing_low_run_fraction": trailing_fraction,
                }
                target_rows.append(row)
                per_action.append(row)
                horizon_counts[horizon] += 1
            h10 = next(row for row in per_action if row["horizon"] == 10)
            closure_utility = strict.get("strict_utility") or {}
            parity_pairs = (
                (h10["branch_mean_iou"], source["branch_mean_iou"]),
                (h10["public_mean_iou"], source["public_mean_iou"]),
                (h10["gain"], source["mean_iou_gain"]),
                (h10["branch_mean_iou"],
                 closure_utility.get("branch_mean_iou")),
                (h10["public_mean_iou"],
                 closure_utility.get("public_mean_iou")),
                (h10["gain"], closure_utility.get("mean_iou_gain")),
            )
            for derived, reference in parity_pairs:
                if reference is None:
                    counters["h10_missing_references"] += 1
                else:
                    maximum_h10_parity_error = max(
                        maximum_h10_parity_error,
                        abs(float(derived) - float(reference)))

    after_mismatches, _ = verify_frozen(records)
    gates = spec["gates"]
    conditions = {
        "selected_events_exact": len(selected_events) == int(
            gates["selected_events_exact"]),
        "candidate_axes_exact": len(selected_keys) == int(
            gates["candidate_axes_exact"]),
        "available_actions_exact": available_actions == int(
            gates["available_actions_exact"]),
        "horizon_records_exact": len(target_rows) == int(
            gates["horizon_records_exact"]),
        "horizon_distribution_exact": all(
            horizon_counts[horizon] == available_actions
            for horizon in horizons),
        "labeled_action_rows_exact": labeled_rows == int(
            spec["labeled_actions"]["rows"]),
        "missing_joins_max": counters["missing_joins"] <= int(
            gates["missing_joins_max"]),
        "duplicate_joins_max": duplicate_joins <= int(
            gates["duplicate_joins_max"]),
        "missing_closure_events_zero":
            counters["missing_closure_events"] == 0,
        "branch_order_mismatches_max":
            counters["branch_order_mismatches"] <= int(
                gates["branch_order_mismatches_max"]),
        "strict_label_mismatches_max":
            counters["strict_label_mismatches"] <= int(
                gates["strict_label_mismatches_max"]),
        "unexpected_unavailable_actions_zero":
            counters["unexpected_unavailable_actions"] == 0,
        "trajectory_length_mismatches_max":
            counters["trajectory_length_mismatches"] <= int(
                gates["trajectory_length_mismatches_max"]),
        "trajectory_nonfinite_values_max":
            counters["trajectory_nonfinite_values"] <= int(
                gates["trajectory_nonfinite_values_max"]),
        "derived_nonfinite_values_max":
            counters["derived_nonfinite_values"] <= int(
                gates["derived_nonfinite_values_max"]),
        "derived_range_violations_max":
            counters["derived_range_violations"] <= int(
                gates["derived_range_violations_max"]),
        "h10_missing_references_zero":
            counters["h10_missing_references"] == 0,
        "h10_parity_max_absolute_error":
            maximum_h10_parity_error <= float(
                gates["h10_parity_max_absolute_error"]),
        "source_hash_mismatches_before_max":
            len(before_mismatches) <= int(
                gates["source_hash_mismatches_max"]),
        "source_hash_mismatches_after_max":
            len(after_mismatches) <= int(
                gates["source_hash_mismatches_max"]),
        "original_rgb_opened_false": gates["original_rgb_opened"] is False,
        "original_depth_opened_false":
            gates["original_depth_opened"] is False,
        "original_ground_truth_opened_false":
            gates["original_ground_truth_opened"] is False,
        "model_checkpoint_loaded_false":
            gates["model_checkpoint_loaded"] is False,
        "optimizer_created_false": gates["optimizer_created"] is False,
    }
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m15a-safety-target-closure-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": (
            "m15a_pass_freeze_m15b_engineering_plan_only" if accepted else
            "m15a_fail_stop_without_model"),
        "claim_ceiling": spec["claim_ceiling"],
        "conditions": conditions,
        "counts": {
            "selected_events": len(selected_events),
            "candidate_axes": len(selected_keys),
            "available_actions": available_actions,
            "horizon_records": len(target_rows),
            "horizon_counts": dict(horizon_counts),
            "strict_label_counts": dict(label_counts),
            "labeled_action_rows": labeled_rows,
            "counters": dict(counters),
        },
        "target_definition": spec["target_definition"],
        "maximum_h10_parity_error": maximum_h10_parity_error,
        "frozen": {
            "before_mismatches": before_mismatches,
            "after_mismatches": after_mismatches,
            "observed": observed,
        },
        "repository": {
            "path": str(REPOSITORY_ROOT),
            "branch": spec["repository"]["branch"],
            "commit": commit,
            "clean": True,
        },
        "inputs": {
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "runner": file_record(runner_path),
        },
        "unauthorized_actions": {
            "original_rgb_opened": False,
            "original_depth_opened": False,
            "original_ground_truth_opened": False,
            "model_checkpoint_loaded": False,
            "optimizer_created": False,
            "m15b_engineering_execution": False,
            "m15c_capacity": False,
            "sequence_disjoint_pilot": False,
            "tracking_checkpoint": False,
            "online_replay": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
    }

    temporary_root = Path(tempfile.mkdtemp(
        prefix=args.output.name + ".", dir=str(args.output.parent)))
    try:
        closure_path = temporary_root / "target_closure.jsonl.gz"
        result_path = temporary_root / "result.json"
        manifest_path = temporary_root / "manifest.json"
        atomic_jsonl_gz(closure_path, target_rows)
        atomic_json(result_path, result)
        manifest = {
            "schema": "sttrack-lachtt-m15a-safety-target-closure-manifest/v1",
            "complete": True,
            "accepted": accepted,
            "payload": {
                "result": file_record_as(
                    result_path, args.output / "result.json"),
                "target_closure": {
                    **file_record_as(
                        closure_path, args.output / "target_closure.jsonl.gz"),
                    "rows": len(target_rows),
                },
            },
            "inputs": result["inputs"],
            "unauthorized_actions": result["unauthorized_actions"],
        }
        atomic_json(manifest_path, manifest)
        for path in (closure_path, result_path, manifest_path):
            path.chmod(0o444)
        os.replace(temporary_root, args.output)
        args.output.chmod(0o555)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "counts": result["counts"],
        "maximum_h10_parity_error": maximum_h10_parity_error,
        "result": file_record(args.output / "result.json"),
        "manifest": file_record(args.output / "manifest.json"),
        "target_closure": file_record(args.output / "target_closure.jsonl.gz"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
