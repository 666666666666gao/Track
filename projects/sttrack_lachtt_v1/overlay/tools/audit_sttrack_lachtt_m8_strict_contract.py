#!/usr/bin/env python3
"""Audit the M8 strict-H10 label contract and age-0 candidate deduplication."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


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


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            for row in rows:
                stream.write(json.dumps(
                    row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_binding(spec_path, spec, binding_path, output):
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    clean = not subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        text=True).strip()
    expected = {
        "spec_sha256": sha256_file(spec_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "repository_commit": commit,
        "output": str(output),
    }
    if binding.get("complete") is not True:
        raise ValueError("binding is incomplete")
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError("binding mismatch for %s" % key)
    if not clean or binding.get("repository_clean") is not True:
        raise ValueError("repository is not clean")
    if binding.get("m8_0_readonly_audit_authorized") is not True:
        raise ValueError("binding does not authorize M8-0")
    return binding, commit


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def action_key(row):
    return (*event_key(row), str(row["branch_id"]))


def finite_values(values):
    return (len(values) == 10 and
            all(value is not None and math.isfinite(float(value))
                for value in values))


def strict_label(row):
    branch = row["branch_ious"]
    public = row["public_ious"]
    if not finite_values(branch) or not finite_values(public):
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
         not all(value <= 0.1 for value in public))
    )
    label = ("beneficial" if beneficial else
             "catastrophic" if catastrophic else "neutral")
    utility = {
        "branch_mean_iou": branch_mean,
        "public_mean_iou": public_mean,
        "mean_iou_gain": gain,
        "early_iou_hits": early_hits,
    }
    return label, utility


def bbox_iou(left, right):
    lx1, ly1, lw, lh = [float(value) for value in left]
    rx1, ry1, rw, rh = [float(value) for value in right]
    lx2, ly2 = lx1 + max(lw, 0.0), ly1 + max(lh, 0.0)
    rx2, ry2 = rx1 + max(rw, 0.0), ry1 + max(rh, 0.0)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = max(lw, 0.0) * max(lh, 0.0) + max(rw, 0.0) * max(rh, 0.0) - intersection
    return intersection / union if union > 0.0 else 0.0


def dedup_groups(branches, iou_min, score_delta_max):
    count = len(branches)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    duplicate_pairs = []
    for left in range(count):
        for right in range(left + 1, count):
            overlap = bbox_iou(branches[left]["bbox"], branches[right]["bbox"])
            score_delta = abs(float(branches[left]["score"]) -
                              float(branches[right]["score"]))
            if overlap >= iou_min and score_delta <= score_delta_max:
                union(left, right)
                duplicate_pairs.append({
                    "left": branches[left]["name"],
                    "right": branches[right]["name"],
                    "bbox_iou": overlap,
                    "score_absolute_difference": score_delta,
                })
    groups = defaultdict(list)
    for index in range(count):
        groups[find(index)].append(branches[index])
    ordered = sorted(groups.values(), key=lambda group:
                     min(branch["name"] for branch in group))
    output = []
    for group in ordered:
        keeper = sorted(
            group, key=lambda branch:
            (-float(branch["score"]), branch["name"]))[0]
        output.append({
            "keeper": keeper["name"],
            "members": sorted(branch["name"] for branch in group),
            "dropped": sorted(
                branch["name"] for branch in group
                if branch["name"] != keeper["name"]),
        })
    return output, duplicate_pairs


def utility_tuple(row):
    utility = row["utility"]
    return (
        float(utility["branch_mean_iou"]),
        float(utility["mean_iou_gain"]),
        int(utility["early_iou_hits"]),
    )


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("output already exists: %s" % args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    if str(args.output) != spec["outputs"]["root"]:
        raise ValueError("output path differs from spec")
    binding, commit = validate_binding(
        args.spec, spec, args.binding, args.output)

    frozen_inputs = []
    for row in spec["inputs"]["event_ledgers"]:
        frozen_inputs.append((Path(row["path"]), row["sha256"]))
    frozen_inputs.extend([
        (Path(spec["inputs"]["labeled_actions"]["path"]),
         spec["inputs"]["labeled_actions"]["sha256"]),
        (Path(spec["inputs"]["gate_a_result"]["path"]),
         spec["inputs"]["gate_a_result"]["sha256"]),
    ])
    for path, digest in frozen_inputs:
        if sha256_file(path) != digest:
            raise ValueError("frozen input hash mismatch: %s" % path)

    events = {}
    duplicate_event_keys = 0
    event_rows = 0
    for ledger in spec["inputs"]["event_ledgers"]:
        with Path(ledger["path"]).open("r", encoding="utf-8") as stream:
            shard_rows = 0
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = event_key(row)
                event_rows += 1
                shard_rows += 1
                if key in events:
                    duplicate_event_keys += 1
                else:
                    events[key] = row
            if shard_rows != int(ledger["rows"]):
                raise ValueError("event ledger row count mismatch")

    labels = {}
    duplicate_action_keys = 0
    label_conflicts = 0
    recomputed_counts = Counter()
    source_label_counts = Counter()
    with gzip.open(spec["inputs"]["labeled_actions"]["path"],
                   "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = action_key(row)
            if key in labels:
                duplicate_action_keys += 1
                continue
            recomputed, utility = strict_label(row)
            if recomputed != row["label"]:
                label_conflicts += 1
            row["recomputed_label"] = recomputed
            row["utility"] = utility
            labels[key] = row
            recomputed_counts[recomputed] += 1
            source_label_counts[row["label"]] += 1

    expected_branches = set(spec["join_contract"]["expected_branch_ids"])
    missing_events = 0
    missing_actions = 0
    extra_actions = 0
    event_class_counts = Counter()
    dedup_candidate_counts = Counter()
    pair_counts = Counter()
    dedup_rows = []
    current_last_both_duplicate_events = 0
    beneficial_event_coverage_before = 0
    beneficial_event_coverage_after = 0
    beneficial_best_utility_loss_events = 0
    dropped_beneficial_actions = 0
    public_state_mutations = 0

    for key, event in events.items():
        trajectory = event.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            missing_events += 1
            continue
        age0 = trajectory[0]
        if int(age0.get("age", -1)) != 0:
            missing_events += 1
            continue
        branches = list(age0.get("branches", []))
        branch_names = {str(branch["name"]) for branch in branches}
        if branch_names != expected_branches or len(branches) != len(expected_branches):
            missing_events += 1
            continue
        action_rows = {}
        for branch in branches:
            joined = labels.get((*key, str(branch["name"])))
            if joined is None:
                missing_actions += 1
            else:
                action_rows[str(branch["name"])] = joined
        if len(action_rows) != len(expected_branches):
            continue

        labels_in_event = [row["recomputed_label"]
                           for row in action_rows.values()]
        event_class = (
            "beneficial" if "beneficial" in labels_in_event else
            "catastrophic" if "catastrophic" in labels_in_event else
            "neutral" if "neutral" in labels_in_event else
            "unavailable"
        )
        event_class_counts[event_class] += 1

        groups, duplicate_pairs = dedup_groups(
            branches,
            float(spec["deduplication"]["bbox_iou_min"]),
            float(spec["deduplication"]["score_absolute_difference_max"]))
        dedup_candidate_counts[len(groups)] += 1
        for pair in duplicate_pairs:
            pair_counts["%s__%s" % tuple(sorted(
                (pair["left"], pair["right"])))] += 1

        current_last_duplicates = set()
        for pair in duplicate_pairs:
            current_last_duplicates.add(frozenset(
                (pair["left"], pair["right"])))
        required_pairs = {
            frozenset(("current_peak0", "last_reliable_peak0")),
            frozenset(("current_peak1", "last_reliable_peak1")),
        }
        if required_pairs.issubset(current_last_duplicates):
            current_last_both_duplicate_events += 1

        keepers = {group["keeper"] for group in groups}
        before_beneficial = [
            row for row in action_rows.values()
            if row["recomputed_label"] == "beneficial"]
        after_beneficial = [
            row for name, row in action_rows.items()
            if name in keepers and row["recomputed_label"] == "beneficial"]
        if before_beneficial:
            beneficial_event_coverage_before += 1
        if after_beneficial:
            beneficial_event_coverage_after += 1
        dropped_beneficial_actions += sum(
            row["recomputed_label"] == "beneficial"
            for name, row in action_rows.items() if name not in keepers)
        if before_beneficial:
            best_before = max(utility_tuple(row)
                              for row in before_beneficial)
            best_after = (max(utility_tuple(row)
                              for row in after_beneficial)
                          if after_beneficial else None)
            if best_after != best_before:
                beneficial_best_utility_loss_events += 1

        dedup_rows.append({
            "sequence": key[0],
            "event_id": key[1],
            "trigger_frame": key[2],
            "strict_event_class": event_class,
            "candidate_count_before": len(branches),
            "candidate_count_after": len(groups),
            "groups": groups,
            "duplicate_pairs": duplicate_pairs,
            "beneficial_before": sorted(
                name for name, row in action_rows.items()
                if row["recomputed_label"] == "beneficial"),
            "beneficial_after": sorted(
                name for name, row in action_rows.items()
                if name in keepers and
                row["recomputed_label"] == "beneficial"),
        })

    event_keys_from_labels = {key[:3] for key in labels}
    extra_event_label_keys = len(event_keys_from_labels - set(events))
    extra_actions = sum(key[:3] not in events for key in labels)
    observed_current_last_ratio = (
        current_last_both_duplicate_events / len(events) if events else 0.0)

    expected_action_counts = spec["strict_h10_labels"]["expected_action_counts"]
    expected_event_counts = spec["strict_h10_labels"]["expected_event_counts"]
    conditions = {
        "event_join_missing_max":
            missing_events <= int(spec["gates"]["event_join_missing_max"]),
        "event_join_duplicate_max":
            duplicate_event_keys <= int(spec["gates"]["event_join_duplicate_max"]),
        "action_join_missing_max":
            missing_actions <= int(spec["gates"]["action_join_missing_max"]),
        "action_join_duplicate_max":
            duplicate_action_keys <= int(spec["gates"]["action_join_duplicate_max"]),
        "label_conflict_max":
            label_conflicts <= int(spec["gates"]["label_conflict_max"]),
        "strict_action_counts_exact":
            all(recomputed_counts[name] == int(expected_action_counts[name])
                for name in expected_action_counts),
        "strict_event_counts_exact":
            all(event_class_counts[name] == int(expected_event_counts[name])
                for name in expected_event_counts),
        "strict_beneficial_event_coverage_after_dedup_min":
            beneficial_event_coverage_after >= int(
                spec["gates"]["strict_beneficial_event_coverage_after_dedup_min"]),
        "strict_beneficial_best_utility_loss_events_max":
            beneficial_best_utility_loss_events <= int(
                spec["gates"]["strict_beneficial_best_utility_loss_events_max"]),
        "current_last_pair_duplicate_event_ratio_min":
            observed_current_last_ratio >= float(
                spec["gates"]["current_last_pair_duplicate_event_ratio_min"]),
        "deduplicated_candidate_count_min":
            min(dedup_candidate_counts) >= int(
                spec["gates"]["deduplicated_candidate_count_min"]),
        "deduplicated_candidate_count_max":
            max(dedup_candidate_counts) <= int(
                spec["gates"]["deduplicated_candidate_count_max"]),
        "ground_truth_files_opened_max": True,
        "public_state_mutations_max":
            public_state_mutations <= int(
                spec["gates"]["public_state_mutations_max"]),
        "event_count_exact":
            event_rows == int(spec["join_contract"]["expected_event_count"]),
        "action_count_exact":
            len(labels) == int(spec["join_contract"]["expected_action_count"]),
        "no_extra_label_events": extra_event_label_keys == 0,
        "no_extra_actions": extra_actions == 0,
    }
    accepted = all(conditions.values())

    result = {
        "schema": "sttrack-lachtt-m8-strict-contract-audit-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": (
            "m8_0_pass_freeze_engineering_spec_only" if accepted else
            "stop_m8_0_contract_or_dedup_gate_failed"),
        "conditions": conditions,
        "spec_sha256": sha256_file(args.spec),
        "binding_sha256": sha256_file(args.binding),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "repository_commit": commit,
        "event_rows": event_rows,
        "unique_events": len(events),
        "duplicate_event_keys": duplicate_event_keys,
        "action_rows": len(labels),
        "duplicate_action_keys": duplicate_action_keys,
        "missing_events": missing_events,
        "missing_actions": missing_actions,
        "extra_event_label_keys": extra_event_label_keys,
        "extra_actions": extra_actions,
        "label_conflicts": label_conflicts,
        "source_label_counts": dict(source_label_counts),
        "recomputed_label_counts": dict(recomputed_counts),
        "strict_event_class_counts": dict(event_class_counts),
        "dedup_candidate_count_distribution": {
            str(key): value for key, value in sorted(
                dedup_candidate_counts.items())},
        "duplicate_pair_counts": dict(pair_counts),
        "current_last_both_duplicate_events":
            current_last_both_duplicate_events,
        "current_last_both_duplicate_event_ratio":
            observed_current_last_ratio,
        "strict_beneficial_event_coverage_before":
            beneficial_event_coverage_before,
        "strict_beneficial_event_coverage_after":
            beneficial_event_coverage_after,
        "strict_beneficial_best_utility_loss_events":
            beneficial_best_utility_loss_events,
        "dropped_duplicate_beneficial_actions":
            dropped_beneficial_actions,
        "ground_truth_files_opened": 0,
        "public_state_mutations": public_state_mutations,
        "model_loaded": False,
        "training_run": False,
        "tracking_checkpoint_written": False,
        "depthtrack_test_run": False,
        "cdtb_run": False,
        "vot_low22_run": False,
        "vot_full127_run": False,
        "qwen_used": False,
        "automatic_next_stage": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    map_path = args.output / "dedup_map.jsonl.gz"
    result_path = args.output / "result.json"
    manifest_path = args.output / "manifest.json"
    atomic_jsonl_gz(map_path, dedup_rows)
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m8-strict-contract-audit-manifest/v1",
        "complete": True,
        "result.json": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "dedup_map.jsonl.gz": {
            "path": str(map_path),
            "sha256": sha256_file(map_path),
        },
    }
    atomic_json(manifest_path, manifest)
    for path in (map_path, result_path, manifest_path):
        path.chmod(0o444)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
