#!/usr/bin/env python3
"""Freeze STTrack LACH-TT Train-152 collection and Gate-A contract."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--language-manifest", required=True, type=Path)
    parser.add_argument("--clip-model", required=True, type=Path)
    parser.add_argument("--protected-plan", required=True, type=Path)
    parser.add_argument("--protected-trace", required=True, action="append",
                        type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--gate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


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
    args = parse_args()
    for key, value in vars(args).items():
        if key == "protected_trace":
            setattr(args, key, [path.resolve() for path in value])
        elif isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.collection_root.exists() or args.gate_root.exists():
        raise FileExistsError("formal output root already exists")
    status = subprocess.check_output(
        ["git", "-C", str(args.repository), "status", "--porcelain"],
        text=True)
    if status.strip():
        raise RuntimeError("repository must be clean before freezing")
    commit = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "HEAD"],
        text=True).strip()
    if len(args.protected_trace) != 2:
        raise ValueError("exactly two protected traces are required")
    trace_records = {}
    event_counts = {}
    for path in args.protected_trace:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard = int(payload.get("shard", -1))
        if (shard not in (0, 1) or payload.get("complete") is not True or
                payload.get("ground_truth_used_after_initialization") is not False or
                payload.get("metric_computed") is not False):
            raise ValueError("protected trace contract mismatch")
        count = sum(1 for row in payload["rows"]
                    if isinstance(row.get("risk_recovery_shadow"), dict) and
                    row["risk_recovery_shadow"].get("event_started") and
                    int(row["frame_index"]) + 10 <=
                    int(payload["sequence_frame_counts"][row["sequence"]]))
        trace_records["protected_trace_shard_%d" % shard] = record(path)
        event_counts[str(shard)] = count
    paths = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "language_manifest": args.language_manifest,
        "clip_model": args.clip_model,
        "protected_plan": args.protected_plan,
        "runner": (args.repository /
                   "tools/run_sttrack_lachtt_train152_collection.py"),
        "observation": (args.repository /
                        "lib/test/tracker/sttrack_lachtt_observation.py"),
        "model": args.repository / "lib/models/sttrack/sttrack.py",
    }
    bindings = {name: record(path) for name, path in paths.items()}
    bindings.update(trace_records)
    expected_checkpoint = (
        "cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98")
    if bindings["checkpoint"]["sha256"] != expected_checkpoint:
        raise ValueError("official checkpoint hash mismatch")
    plan = json.loads(args.protected_plan.read_text(encoding="utf-8"))
    if sorted(int(item["shard"]) for item in plan["shards"]) != [0, 1]:
        raise ValueError("protected plan shard mismatch")
    value = {
        "schema": "sttrack-lachtt-train152-spec/v1",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_before_collection": True,
        "future_ground_truth_opened": False,
        "repository_commit": commit,
        "repository_clean": True,
        "dataset_root": str(args.dataset_root),
        "collection_root": str(args.collection_root),
        "gate_root": str(args.gate_root),
        "bindings": bindings,
        "expected_event_counts": event_counts,
        "claim_ceiling": (
            "DepthTrack Train-152 candidate action-space capacity only; no "
            "deployable selector, public benchmark, or VOT claim."),
        "public_schedule": {
            "source": "sealed GT-free STTrack risk-recovery full152 trace",
            "trigger": "score<0.30 OR center_jump>0.75 OR abs_log_area>0.70",
            "cooldown_frames": 10,
            "public_replay": False,
            "public_state_mutated": False,
        },
        "action_protocol": {
            "sources": ["current", "last_reliable", "velocity"],
            "current": "public bbox at trigger_frame-1",
            "last_reliable": (
                "most recent pre-trigger public bbox with score>=0.30, "
                "center_jump<=0.75 and abs_log_area<=0.70; otherwise init"),
            "velocity": (
                "one-step linear xywh extrapolation from the two most recent "
                "reliable public boxes, time-gap normalized; otherwise last"),
            "age0_search_factor": 6.0,
            "age0_peaks_per_source": 2,
            "age0_nms_kernel": 3,
            "branch_count": 6,
            "age1_to_9_search_factor": 4.0,
            "rollout_ages": 10,
            "feature_ages": [0, 1, 2, 3, 4],
            "label_only_ages": [5, 6, 7, 8, 9],
            "template": "immutable first-frame template duplicated twice",
            "query": "reset at age0 then independent RGB/depth state per branch",
            "commit": False,
        },
        "candidate_observation": {
            "native_rgb_roi": 768,
            "native_depth_roi": 768,
            "native_fused_roi": 768,
            "clip_candidate_image": 768,
            "immutable_clip_initial_image": 768,
            "immutable_clip_identity_text": 768,
            "query_rgb_mean": 768,
            "query_depth_mean": 768,
            "raw_depth_roi": [2, 16, 16],
            "raw_depth_channels": ["robust_normalized_log_depth", "validity_mask"],
            "response_and_relative_geometry": True,
            "distractor_bank": "causal top-3 other branches reconstructed within event",
            "forbidden_model_inputs": [
                "sequence ID", "frame ID", "source ID", "peak rank",
                "dataset/subgroup label", "GT", "future IoU", "future text"],
        },
        "label_rules": {
            "future_visible_frames": 10,
            "early_visible_frames": 5,
            "beneficial_definition": (
                "branch_mean_iou>=0.5 AND gain>=0.2 AND at least 2 of first "
                "5 visible branch IoUs>=0.5"),
            "catastrophic_definition": (
                "(public_mean_iou>=0.5 AND branch_mean_iou<=0.2) OR "
                "gain<=-0.3 OR (all 10 branch IoUs<=0.1 AND not all 10 "
                "public IoUs<=0.1)"),
            "unavailable": "fewer than 10 aligned visible frames",
            "confirmed_failure": (
                "first frame of each contiguous run of 10 available public "
                "IoUs<=0.1; reset on gap or unavailable GT"),
            "rescue": (
                "branch trigger precedes failure; complete H10; at least 2/5 "
                "early IoUs>=0.5; mean gain>=0.2; not all H10 IoUs<=0.1"),
            "toy07_indoor_320_exception": (
                "only first 1367 of 1406 GT rows align to frames; exact GT "
                "hash must be frozen again before Gate A"),
        },
        "gate_a": {
            "minimum_beneficial_actions": 30,
            "minimum_positive_sequences": 10,
            "minimum_rescued_confirmed_failures": 10,
            "minimum_rescued_failure_sequences": 5,
            "require_current_source_capacity": True,
            "require_last_or_velocity_source_capacity": True,
            "all_conditions_required": True,
            "passing_authorizes": [
                "selector training", "DepthTrack Train complete-sequence OOF"],
            "passing_does_not_authorize": [
                "online commit", "DepthTrack Test", "CDTB", "VOT low22",
                "VOT full127"],
        },
        "output_contract": {
            "atomic_new_root": True,
            "maximum_collection_bytes": 2147483648,
            "minimum_free_disk_before_formal_launch_bytes": 7516192768,
            "required_per_shard": ["events.jsonl", "manifest.json"],
            "formal_shards": 2,
            "qwen_used": False,
            "future_frame_text_used": False,
        },
        "automatic_next_stage": False,
    }
    atomic_json(args.output, value)
    os.chmod(args.output, 0o444)
    print(json.dumps({
        "output": str(args.output), "sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "repository_commit": commit,
        "expected_event_counts": event_counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
