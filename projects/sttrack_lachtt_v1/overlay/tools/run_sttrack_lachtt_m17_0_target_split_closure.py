#!/usr/bin/env python3
"""Build the frozen M17-0 strict target and sequence-split closure.

This runner is deliberately standard-library only.  It never imports a model,
opens RGB/Depth/GT, performs a forward pass, or writes a checkpoint.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


BRANCH_ORDER = (
    "current_peak0",
    "current_peak1",
    "last_reliable_peak0",
    "last_reliable_peak1",
    "velocity_peak0",
    "velocity_peak1",
)
HORIZONS = (3, 5, 10)
METRICS = (
    "branch_mean_iou",
    "public_mean_iou",
    "gain",
    "low_overlap_fraction",
    "trailing_low_run_fraction",
)
PARTITIONS = ("training", "heldout")


class ContractError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def stable_fold(sequence, salt, folds):
    payload = (salt + "\0" + sequence).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def finite_number(value):
    return (not isinstance(value, bool) and
            isinstance(value, (int, float)) and math.isfinite(float(value)))


def finite_iou(value):
    return finite_number(value) and 0.0 <= float(value) <= 1.0


def mean(values):
    return sum(values) / float(len(values))


def trailing_low_fraction(values, threshold):
    count = 0
    for value in reversed(values):
        if value <= threshold:
            count += 1
        else:
            break
    return count / float(len(values))


def trajectory_metrics(branch_ious, public_ious, horizon, threshold):
    branch = branch_ious[:horizon]
    public = public_ious[:horizon]
    branch_mean = mean(branch)
    public_mean = mean(public)
    result = {
        "branch_mean_iou": branch_mean,
        "public_mean_iou": public_mean,
        "gain": branch_mean - public_mean,
        "low_overlap_fraction": (
            sum(value <= threshold for value in branch) / float(horizon)),
        "trailing_low_run_fraction": trailing_low_fraction(
            branch, threshold),
    }
    if tuple(result) != METRICS or not all(
            finite_number(value) for value in result.values()):
        raise ContractError("trajectory metric contract drifted")
    return result


def recompute_label(branch_ious, public_ious):
    branch_mean = mean(branch_ious)
    public_mean = mean(public_ious)
    gain = branch_mean - public_mean
    early_hits = sum(value >= 0.5 for value in branch_ious[:5])
    beneficial = (
        branch_mean >= 0.5 and gain >= 0.2 and early_hits >= 2)
    catastrophic = (
        (public_mean >= 0.5 and branch_mean <= 0.2) or
        gain <= -0.3 or
        (all(value <= 0.1 for value in branch_ious) and
         not all(value <= 0.1 for value in public_ious)))
    label = "beneficial" if beneficial else (
        "catastrophic" if catastrophic else "neutral")
    return label, branch_mean, public_mean, gain, early_hits


def recompute_event_class(labels):
    if "beneficial" in labels:
        return "beneficial"
    if "catastrophic" in labels:
        return "catastrophic"
    if "neutral" in labels:
        return "neutral"
    return "unavailable"


def event_key(row):
    return (
        str(row["sequence"]),
        int(row["event_id"]),
        int(row["trigger_frame"]),
    )


def action_key(row):
    return event_key(row) + (str(row["branch_id"]),)


def jsonable_equal(first, second):
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)


def canonical_rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(
            row, sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def collect_hashed_records(value, records=None):
    if records is None:
        records = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
                value.get("sha256"), str):
            records.append({"path": value["path"], "sha256": value["sha256"]})
        for nested in value.values():
            collect_hashed_records(nested, records)
    elif isinstance(value, list):
        for nested in value:
            collect_hashed_records(nested, records)
    return records


def verify_hashed_records(spec):
    checked = []
    mismatches = []
    seen = set()
    for expected in collect_hashed_records(spec):
        key = (expected["path"], expected["sha256"])
        if key in seen:
            continue
        seen.add(key)
        path = Path(expected["path"])
        actual = sha256_file(path) if path.is_file() else None
        checked.append({
            "path": str(path),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
            "match": actual == expected["sha256"],
        })
        if actual != expected["sha256"]:
            mismatches.append(str(path))
    return checked, mismatches


def validate_binding(args, spec, binding):
    if binding.get("schema") != (
            "sttrack-lachtt-m17-0-target-split-closure-binding/v1"):
        raise ContractError("binding schema drifted")
    if binding.get("complete") is not True:
        raise ContractError("binding is incomplete")
    if binding.get("authorization", {}).get(
            "m17_0_readonly_target_split_closure") is not True:
        raise ContractError("binding does not authorize M17-0 closure")
    spec_record = binding["spec"]
    if (Path(spec_record["path"]).resolve() != args.spec or
            spec_record["sha256"] != sha256_file(args.spec)):
        raise ContractError("spec binding mismatch")
    runner_record = binding["runner"]
    runner_path = Path(__file__).resolve()
    if (Path(runner_record["path"]).resolve() != runner_path or
            runner_record["sha256"] != sha256_file(runner_path)):
        raise ContractError("runner binding mismatch")
    audit_record = binding["preexecution_audit"]
    audit_path = Path(audit_record["path"]).resolve()
    if audit_record["sha256"] != sha256_file(audit_path):
        raise ContractError("preexecution audit binding mismatch")
    audit = load_json(audit_path)
    allowed = audit.get("authorization_boundary", {}).get(
        "authorized_next_actions_after_pass", [])
    exact_authorization = (
        "run exactly one M17-0 read-only target/split closure if the "
        "binding passes its own preflight")
    if (audit.get("overall_verdict") != "PASS" or
            exact_authorization not in allowed):
        raise ContractError("audit does not authorize the bounded run")
    repo = binding["repository"]
    repo_path = Path(repo["path"]).resolve()
    spec_repo_path = Path(spec["repository"]["path"]).resolve()
    runner_path = Path(__file__).resolve()
    try:
        runner_relative = runner_path.relative_to(repo_path)
    except ValueError as error:
        raise ContractError("runner is outside the bound repository") from error
    if (repo_path != spec_repo_path or
            runner_relative != Path(
                "tools/run_sttrack_lachtt_m17_0_target_split_closure.py")):
        raise ContractError("repository path or runner location drifted")
    head = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True).strip()
    status_text = subprocess.check_output(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        text=True)
    if (head != repo["commit"] or
            branch != repo["branch"] or
            branch != spec["repository"]["branch"] or status_text):
        raise ContractError("repository binding or cleanliness mismatch")
    if args.output != Path(binding["output"]["path"]).resolve():
        raise ContractError("output binding mismatch")
    if args.output.exists() or binding["output"].get(
            "absent_before_execution") is not True:
        raise ContractError("output root precondition failed")
    if args.output != Path(spec["outputs"]["root"]).resolve():
        raise ContractError("spec output root mismatch")
    return audit, repo_path, head, branch


def revalidate_before_publication(
        args, spec, binding, binding_sha256, repo_path, head, branch):
    if (sha256_file(args.spec) != binding["spec"]["sha256"] or
            sha256_file(args.binding) != binding_sha256 or
            sha256_file(Path(__file__).resolve()) != binding["runner"]["sha256"] or
            sha256_file(binding["preexecution_audit"]["path"]) !=
            binding["preexecution_audit"]["sha256"]):
        raise ContractError("bound control artifact changed during execution")
    checked, mismatches = verify_hashed_records(spec)
    current_head = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True).strip()
    current_branch = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True).strip()
    status_text = subprocess.check_output(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        text=True)
    if (mismatches or current_head != head or current_branch != branch or
            status_text or args.output.exists()):
        raise ContractError("prepublication source/repository/output recheck failed")
    return checked


def validate_event_ledgers(spec, closure_by_event):
    ledgers = {}
    duplicate_events = 0
    axis_mismatches = 0
    record_mismatches = 0
    for shard in spec["frozen_inputs"]["collection_shards"]:
        root = Path(shard["root"]).resolve()
        manifest_path = root / "manifest.json"
        if (not manifest_path.is_file() or
                sha256_file(manifest_path) != shard["manifest_sha256"]):
            record_mismatches += 1
        rows = []
        with Path(shard["event_ledger"]["path"]).open(
                "r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != int(shard["event_ledger"]["rows"]):
            raise ContractError("event ledger row count drifted")
        for row in rows:
            key = event_key(row)
            if key in ledgers:
                duplicate_events += 1
                continue
            ledgers[key] = row
            closure = closure_by_event.get(key)
            if closure is None:
                record_mismatches += 1
                continue
            feature_path = (root / row["feature_path"]).resolve()
            anchor_path = (root / row["anchor_path"]).resolve()
            try:
                feature_path.relative_to(root)
                anchor_path.relative_to(root)
            except ValueError:
                record_mismatches += 1
                continue
            if (str(feature_path) != closure["feature_path"] or
                    int(row["feature_bytes"]) != int(closure["feature_bytes"]) or
                    row["feature_sha256"] != closure["feature_sha256"] or
                    str(anchor_path) != closure["anchor_path"] or
                    not feature_path.is_file() or
                    feature_path.stat().st_size != int(row["feature_bytes"]) or
                    not anchor_path.is_file()):
                record_mismatches += 1
            trajectory = row.get("trajectory")
            if (not isinstance(trajectory, list) or len(trajectory) != 10 or
                    [int(age.get("age", -1)) for age in trajectory] !=
                    list(range(10))):
                axis_mismatches += 1
                continue
            for age, point in enumerate(trajectory):
                names = [branch.get("name") for branch in point.get("branches", [])]
                if (tuple(names) != BRANCH_ORDER or
                        int(point.get("frame_index", -1)) != key[2] + age):
                    axis_mismatches += 1
                    break
    return ledgers, {
        "duplicate_events": duplicate_events,
        "candidate_axis_mismatches": axis_mismatches,
        "event_record_mismatches": record_mismatches,
    }


def build_closure(spec):
    expected = spec["frozen_inputs"]["counts"]
    tolerance = float(spec["stage_m17_0_target_and_split_closure"][
        "h10_recalculation_tolerance"])
    threshold = float(spec["stage_m17_0_target_and_split_closure"][
        "low_overlap_threshold"])
    closure_rows = load_gzip_jsonl(
        spec["frozen_inputs"]["m8b_closure"]["path"])
    action_rows = load_gzip_jsonl(
        spec["frozen_inputs"]["labeled_actions"]["path"])

    duplicate_events = len(closure_rows) - len({event_key(row) for row in closure_rows})
    duplicate_actions = len(action_rows) - len({action_key(row) for row in action_rows})
    closure_by_event = {event_key(row): row for row in closure_rows}
    actions_by_key = {action_key(row): row for row in action_rows}
    ledgers, ledger_audit = validate_event_ledgers(spec, closure_by_event)

    action_counts = Counter()
    event_counts = Counter()
    partition_event_counts = {name: Counter() for name in PARTITIONS}
    partition_action_counts = {name: Counter() for name in PARTITIONS}
    targets = {name: [] for name in PARTITIONS}
    split_entries = []
    missing_joins = 0
    extra_action_keys = set(actions_by_key)
    label_conflicts = 0
    axis_mismatches = 0
    nonfinite_targets = 0
    h10_metric_mismatches = 0
    partial_availability_events = 0
    folds = int(spec["sequence_split"]["fold_count"])
    salt = spec["sequence_split"]["salt"]
    heldout_fold = int(spec["sequence_split"]["evaluation_fold"])

    for closure in sorted(closure_rows, key=event_key):
        key = event_key(closure)
        if key not in ledgers:
            missing_joins += 1
        if tuple(closure.get("branch_order", ())) != BRANCH_ORDER:
            axis_mismatches += 1
        actions = closure.get("actions")
        if (not isinstance(actions, list) or len(actions) != len(BRANCH_ORDER) or
                tuple(row.get("branch_id") for row in actions) != BRANCH_ORDER):
            axis_mismatches += 1
            continue
        fold = stable_fold(key[0], salt, folds)
        partition = "heldout" if fold == heldout_fold else "training"
        labels = []
        available_flags = []
        event_targets = []
        for role_id, action_stub in enumerate(actions):
            branch_id = BRANCH_ORDER[role_id]
            joined_key = key + (branch_id,)
            action = actions_by_key.get(joined_key)
            if action is None:
                missing_joins += 1
                continue
            extra_action_keys.discard(joined_key)
            label = str(action["label"])
            labels.append(label)
            action_counts[label] += 1
            available = label != "unavailable"
            available_flags.append(available)
            stub_label = str(action_stub.get("strict_label"))
            if stub_label != label:
                label_conflicts += 1
            if not available:
                if action_stub.get("strict_utility") is not None:
                    label_conflicts += 1
                continue
            branch_ious = action.get("branch_ious")
            public_ious = action.get("public_ious")
            if (not isinstance(branch_ious, list) or len(branch_ious) != 10 or
                    not isinstance(public_ious, list) or len(public_ious) != 10 or
                    not all(finite_iou(value) for value in
                            branch_ious + public_ious)):
                nonfinite_targets += 1
                continue
            branch_ious = [float(value) for value in branch_ious]
            public_ious = [float(value) for value in public_ious]
            computed_label, branch_mean, public_mean, gain, early_hits = (
                recompute_label(branch_ious, public_ious))
            if computed_label != label:
                label_conflicts += 1
            for field, actual in (
                    ("branch_mean_iou", branch_mean),
                    ("public_mean_iou", public_mean),
                    ("mean_iou_gain", gain)):
                if (not finite_number(action.get(field)) or
                        abs(float(action[field]) - actual) > tolerance):
                    h10_metric_mismatches += 1
            if int(action.get("early_hits", -1)) != early_hits:
                h10_metric_mismatches += 1
            utility = action_stub.get("strict_utility")
            if (not isinstance(utility, dict) or
                    abs(float(utility["branch_mean_iou"]) - branch_mean) > tolerance or
                    abs(float(utility["public_mean_iou"]) - public_mean) > tolerance or
                    abs(float(utility["mean_iou_gain"]) - gain) > tolerance or
                    int(utility["early_iou_hits"]) != early_hits):
                h10_metric_mismatches += 1
            horizon_targets = {
                str(horizon): trajectory_metrics(
                    branch_ious, public_ious, horizon, threshold)
                for horizon in HORIZONS
            }
            if not all(finite_number(value) for horizon in horizon_targets.values()
                       for value in horizon.values()):
                nonfinite_targets += 1
            event_targets.append({
                "record_type": "action_target",
                "partition": partition,
                "fold": fold,
                "sequence": key[0],
                "event_id": key[1],
                "trigger_frame": key[2],
                "branch_id": branch_id,
                "candidate_role_id": role_id,
                "strict_event_class": str(closure["strict_event_class"]),
                "strict_label": label,
                "early_hits_h5": early_hits,
                "targets": horizon_targets,
            })
            partition_action_counts[partition][label] += 1
        if len(available_flags) != len(BRANCH_ORDER):
            missing_joins += len(BRANCH_ORDER) - len(available_flags)
        if available_flags and any(available_flags) != all(available_flags):
            partial_availability_events += 1
        event_class = recompute_event_class(labels)
        event_counts[event_class] += 1
        if event_class != str(closure.get("strict_event_class")):
            label_conflicts += 1
        available_event = event_class != "unavailable"
        if available_event:
            partition_event_counts[partition][event_class] += 1
            targets[partition].extend(event_targets)
        elif event_targets:
            partial_availability_events += 1
        split_entries.append({
            "sequence": key[0],
            "event_id": key[1],
            "trigger_frame": key[2],
            "fold": fold,
            "partition": partition,
            "strict_h10_available": available_event,
            "candidate_count": len(BRANCH_ORDER),
        })

    split_sequences = {
        name: sorted({row["sequence"] for row in split_entries
                      if row["partition"] == name}) for name in PARTITIONS
    }
    available_sequences = {
        name: sorted({row["sequence"] for row in split_entries
                      if row["partition"] == name and
                      row["strict_h10_available"]}) for name in PARTITIONS
    }
    overlap = sorted(set(split_sequences["training"]) &
                     set(split_sequences["heldout"]))
    summary = {
        "events": len(closure_rows),
        "actions": len(action_rows),
        "sequences": len({row["sequence"] for row in split_entries}),
        "strict_action_counts": dict(sorted(action_counts.items())),
        "strict_event_counts": dict(sorted(event_counts.items())),
        "partition_event_counts": {
            name: dict(sorted(partition_event_counts[name].items()))
            for name in PARTITIONS
        },
        "partition_action_counts": {
            name: dict(sorted(partition_action_counts[name].items()))
            for name in PARTITIONS
        },
        "partition_all_sequence_counts": {
            name: len(split_sequences[name]) for name in PARTITIONS
        },
        "partition_available_sequence_counts": {
            name: len(available_sequences[name]) for name in PARTITIONS
        },
        "target_action_rows": sum(len(targets[name]) for name in PARTITIONS),
        "serialized_training_target_rows": len(targets["training"]),
        "committed_heldout_target_rows": len(targets["heldout"]),
    }
    target_commitments = {
        name: {
            "action_rows": len(targets[name]),
            "canonical_jsonl_sha256": canonical_rows_sha256(targets[name]),
        } for name in PARTITIONS
    }
    expected_action_counts = expected["strict_action_counts"]
    expected_event_counts = expected["strict_event_counts"]
    split_spec = spec["sequence_split"]
    conditions = {
        "event_rows_exact": len(closure_rows) == int(expected["events"]),
        "action_rows_exact": len(action_rows) == int(expected["actions"]),
        "sequence_count_exact": summary["sequences"] == int(expected["sequences"]),
        "strict_action_counts_exact": all(
            action_counts[name] == int(count)
            for name, count in expected_action_counts.items()),
        "strict_event_counts_exact": all(
            event_counts[name] == int(count)
            for name, count in expected_event_counts.items()),
        "duplicate_event_keys_zero": duplicate_events == 0,
        "duplicate_action_keys_zero": duplicate_actions == 0,
        "missing_joins_zero": missing_joins == 0 and not extra_action_keys,
        "label_conflicts_zero": label_conflicts == 0,
        "candidate_axis_mismatches_zero": (
            axis_mismatches == 0 and
            ledger_audit["candidate_axis_mismatches"] == 0),
        "event_ledger_duplicates_zero": ledger_audit["duplicate_events"] == 0,
        "event_ledger_records_exact": ledger_audit["event_record_mismatches"] == 0,
        "partial_availability_events_zero": partial_availability_events == 0,
        "nonfinite_targets_zero": nonfinite_targets == 0,
        "h10_metric_mismatches_zero": h10_metric_mismatches == 0,
        "target_action_rows_exact": summary["target_action_rows"] == 4692,
        "training_split_exact": (
            sum(partition_event_counts["training"].values()) ==
            int(split_spec["training"]["available_events"]) and
            sum(partition_action_counts["training"].values()) ==
            int(split_spec["training"]["available_actions"]) and
            len(available_sequences["training"]) ==
            int(split_spec["training"]["available_sequences"]) and
            len(split_sequences["training"]) ==
            int(split_spec["training"]["all_event_sequences"])),
        "heldout_split_exact": (
            sum(partition_event_counts["heldout"].values()) ==
            int(split_spec["heldout"]["available_events"]) and
            sum(partition_action_counts["heldout"].values()) ==
            int(split_spec["heldout"]["available_actions"]) and
            len(available_sequences["heldout"]) ==
            int(split_spec["heldout"]["available_sequences"]) and
            len(split_sequences["heldout"]) ==
            int(split_spec["heldout"]["all_event_sequences"])),
        "training_event_classes_exact": jsonable_equal(
            dict(partition_event_counts["training"]),
            split_spec["training"]["event_classes"]),
        "heldout_event_classes_exact": jsonable_equal(
            dict(partition_event_counts["heldout"]),
            split_spec["heldout"]["event_classes"]),
        "training_action_labels_exact": jsonable_equal(
            dict(partition_action_counts["training"]),
            split_spec["training"]["action_labels"]),
        "heldout_action_labels_exact": jsonable_equal(
            dict(partition_action_counts["heldout"]),
            split_spec["heldout"]["action_labels"]),
        "sequence_overlap_zero": not overlap,
    }
    diagnostics = {
        "duplicate_events": duplicate_events,
        "duplicate_actions": duplicate_actions,
        "missing_joins": missing_joins,
        "extra_action_keys": len(extra_action_keys),
        "label_conflicts": label_conflicts,
        "candidate_axis_mismatches": axis_mismatches,
        "partial_availability_events": partial_availability_events,
        "nonfinite_targets": nonfinite_targets,
        "h10_metric_mismatches": h10_metric_mismatches,
        "ledger_audit": ledger_audit,
        "sequence_overlap": overlap,
    }
    split_ledger = {
        "schema": "sttrack-lachtt-m17-0-sequence-split-ledger/v1",
        "fold_function": spec["sequence_split"]["fold_function"],
        "salt": salt,
        "fold_count": folds,
        "heldout_fold": heldout_fold,
        "target_storage_contract": (
            "trajectory_targets serializes training numeric targets only; "
            "heldout numeric targets are omitted and represented solely by "
            "a canonical SHA256 commitment. A future M17-1 evaluator may "
            "rederive and verify heldout targets from the sealed source only "
            "after optimization is complete"),
        "target_commitments": target_commitments,
        "summary": summary,
        "all_sequences": split_sequences,
        "available_sequences": available_sequences,
        "events": split_entries,
    }
    return (targets, target_commitments, split_ledger, summary,
            conditions, diagnostics)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_training_targets(path, targets, target_commitments):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({
                    "record_type": "partition_start",
                    "partition": "training",
                    "action_rows": len(targets["training"]),
                    "canonical_jsonl_sha256": target_commitments[
                        "training"]["canonical_jsonl_sha256"],
                }, sort_keys=True, allow_nan=False) + "\n")
                for row in targets["training"]:
                    stream.write(json.dumps(
                        row, sort_keys=True, allow_nan=False) + "\n")
                stream.write(json.dumps({
                    "record_type": "partition_end",
                    "partition": "training",
                    "action_rows": len(targets["training"]),
                }, sort_keys=True, allow_nan=False) + "\n")
                stream.write(json.dumps({
                    "record_type": "heldout_target_commitment",
                    "partition": "heldout",
                    "action_rows": target_commitments["heldout"]["action_rows"],
                    "canonical_jsonl_sha256": target_commitments[
                        "heldout"]["canonical_jsonl_sha256"],
                    "numeric_targets_serialized": False,
                }, sort_keys=True, allow_nan=False) + "\n")
        raw.flush()
        os.fsync(raw.fileno())


def chmod_readonly_files(root):
    for path in root.iterdir():
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR |
               stat.S_IRGRP | stat.S_IXGRP |
               stat.S_IROTH | stat.S_IXOTH)


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = load_json(args.spec)
    binding = load_json(args.binding)
    if spec.get("schema") != (
            "sttrack-lachtt-m17-r1-sequence-disjoint-conservative-"
            "survival-pilot-spec/v1"):
        raise ContractError("R1 spec schema drifted")
    if spec.get("complete") is not True:
        raise ContractError("R1 spec is incomplete")
    binding_sha256 = sha256_file(args.binding)
    audit, repo_path, head, branch = validate_binding(args, spec, binding)
    checked_sources, source_mismatches = verify_hashed_records(spec)
    if source_mismatches:
        raise ContractError("frozen source hash mismatch")
    (targets, target_commitments, split_ledger, summary,
     conditions, diagnostics) = build_closure(spec)
    conditions["source_hash_mismatches_zero"] = not source_mismatches
    conditions["source_records_checked_exact"] = len(checked_sources) == 19
    conditions["preexecution_audit_pass"] = audit["overall_verdict"] == "PASS"
    conditions["repository_clean_and_bound"] = True
    conditions["output_root_absent_before_execution"] = True
    conditions["heldout_numeric_targets_not_serialized"] = True
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m17-0-target-split-closure-result/v1",
        "complete": True,
        "accepted": accepted,
        "evaluation_type": "real_gt_train_only_readonly_target_split_closure",
        "claim_ceiling": (
            "Sealed DepthTrack Train target/split closure only; no model, "
            "training, checkpoint, Test, CDTB, VOT, Qwen or benchmark claim."),
        "conditions": conditions,
        "summary": summary,
        "diagnostics": diagnostics,
        "source_records_checked": len(checked_sources),
        "decision": (
            "m17_0_pass_authorize_independent_result_audit_only" if accepted
            else "m17_0_fail_stop_no_training"),
        "authorization": {
            "independent_m17_0_result_audit": accepted,
            "m17_1_implementation": False,
            "m17_1_execution": False,
            "full_six_fold_oof": False,
            "online_replay": False,
            "tracking_checkpoint": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
            "automatic_next_stage": False,
        },
    }
    parent = args.output.parent
    if not parent.is_dir():
        raise ContractError("output parent does not exist")
    temporary = Path(tempfile.mkdtemp(
        prefix=args.output.name + ".tmp.", dir=str(parent))).resolve()
    if temporary.parent != parent:
        raise ContractError("temporary output escaped intended parent")
    try:
        target_path = temporary / "trajectory_targets.jsonl.gz"
        split_path = temporary / "split_ledger.json"
        result_path = temporary / "result.json"
        manifest_path = temporary / "manifest.json"
        write_training_targets(target_path, targets, target_commitments)
        write_json(split_path, split_ledger)
        write_json(result_path, result)
        final_checked_sources = revalidate_before_publication(
            args, spec, binding, binding_sha256,
            repo_path, head, branch)
        if not jsonable_equal(final_checked_sources, checked_sources):
            raise ContractError("frozen source records changed during execution")
        manifest = {
            "schema": "sttrack-lachtt-m17-0-target-split-closure-manifest/v1",
            "complete": True,
            "accepted": accepted,
            "repository": {
                "path": str(repo_path),
                "commit": head,
                "branch": branch,
                "clean": True,
            },
            "inputs": {
                "plan": file_record(spec["recovery"]["original_plan"]["path"]),
                "spec": file_record(args.spec),
                "binding": file_record(args.binding),
                "preexecution_audit": file_record(
                    binding["preexecution_audit"]["path"]),
                "runner": file_record(Path(__file__).resolve()),
                "frozen_source_records": checked_sources,
            },
            "outputs": {
                "result.json": file_record_as(
                    result_path, args.output / "result.json"),
                "trajectory_targets.jsonl.gz": file_record_as(
                    target_path, args.output / "trajectory_targets.jsonl.gz"),
                "split_ledger.json": file_record_as(
                    split_path, args.output / "split_ledger.json"),
            },
            "unauthorized_actions": {
                "original_rgb_opened": False,
                "original_depth_opened": False,
                "ground_truth_opened": False,
                "model_imported": False,
                "model_forward": False,
                "optimizer": False,
                "checkpoint_written": False,
                "qwen": False,
                "depthtrack_test": False,
                "cdtb": False,
                "vot_low22": False,
                "vot_full127": False,
                "public_tracker_mutation": False,
            },
        }
        write_json(manifest_path, manifest)
        expected_files = set(spec["outputs"]["m17_0_files"])
        actual_files = {path.name for path in temporary.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise ContractError("output file set drifted")
        chmod_readonly_files(temporary)
        os.replace(str(temporary), str(args.output))
    except Exception:
        if temporary.exists() and temporary.parent == parent:
            shutil.rmtree(temporary)
        raise
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "output": str(args.output),
        "summary": summary,
    }, sort_keys=True, allow_nan=False))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
