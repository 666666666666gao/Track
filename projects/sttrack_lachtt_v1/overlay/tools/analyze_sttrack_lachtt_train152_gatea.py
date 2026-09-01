#!/usr/bin/env python3
"""Post-seal Gate-A analysis for STTrack LACH-TT Train-152 collection."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
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


def atomic_gzip_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_bbox(values):
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(result) != 4 or not all(math.isfinite(value) for value in result)
            or result[2] <= 0.0 or result[3] <= 0.0):
        return None
    return result


def iou(first, second):
    first, second = finite_bbox(first), finite_bbox(second)
    if first is None or second is None:
        return None
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else None


def parse_gt_line(line):
    text = line.strip().replace("\t", ",").replace(" ", ",")
    values = [part for part in text.split(",") if part]
    try:
        return finite_bbox([float(value) for value in values])
    except ValueError:
        return None


def load_ground_truth(dataset_root, sequence, expected_frames):
    path = dataset_root / sequence / "groundtruth.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    if sequence == "toy07_indoor_320":
        if (len(lines) != 1406 or expected_frames != 1367 or
                sha256_file(path) !=
                "683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2"):
            raise ValueError("toy07 GT exception contract mismatch")
        lines = lines[:expected_frames]
    elif len(lines) != expected_frames:
        raise ValueError("GT/frame mismatch: %s" % sequence)
    return [parse_gt_line(line) for line in lines], path


def label_action(public_boxes, branch_boxes, gt_boxes):
    if len(public_boxes) != 10 or len(branch_boxes) != 10 or len(gt_boxes) != 10:
        raise ValueError("H10 alignment mismatch")
    public_ious = [iou(box, gt) for box, gt in zip(public_boxes, gt_boxes)]
    branch_ious = [iou(box, gt) for box, gt in zip(branch_boxes, gt_boxes)]
    if any(value is None for value in public_ious + branch_ious):
        return {
            "label": "unavailable", "public_ious": public_ious,
            "branch_ious": branch_ious,
        }
    public_mean = sum(public_ious) / 10.0
    branch_mean = sum(branch_ious) / 10.0
    gain = branch_mean - public_mean
    early_hits = sum(value >= 0.5 for value in branch_ious[:5])
    beneficial = branch_mean >= 0.5 and gain >= 0.2 and early_hits >= 2
    catastrophic = (
        (public_mean >= 0.5 and branch_mean <= 0.2) or gain <= -0.3 or
        (all(value <= 0.1 for value in branch_ious) and
         not all(value <= 0.1 for value in public_ious)))
    label = "beneficial" if beneficial else (
        "catastrophic" if catastrophic else "neutral")
    return {
        "label": label,
        "public_mean_iou": public_mean,
        "branch_mean_iou": branch_mean,
        "mean_iou_gain": gain,
        "early_hits": early_hits,
        "public_ious": public_ious,
        "branch_ious": branch_ious,
    }


def confirmed_failure_starts(public_boxes, gt_boxes):
    overlaps = [iou(box, gt) for box, gt in zip(public_boxes, gt_boxes)]
    starts = []
    run_start = None
    for index, value in enumerate(overlaps + [None]):
        low = value is not None and value <= 0.1
        if low and run_start is None:
            run_start = index
        if not low and run_start is not None:
            if index - run_start >= 10:
                starts.append(run_start)
            run_start = None
    return starts, overlaps


def validate_feature_file(path):
    values = torch.load(path, map_location="cpu")
    expected = {
        "clip_image": [5, 6, 768], "native_depth": [5, 6, 768],
        "native_fused": [5, 6, 768], "native_rgb": [5, 6, 768],
        "query_depth": [5, 6, 768], "query_rgb": [5, 6, 768],
        "raw_depth": [5, 6, 2, 16, 16], "scalars": [5, 6, 15],
    }
    if sorted(values) != sorted(expected):
        raise ValueError("feature schema mismatch")
    for name, shape in expected.items():
        if list(values[name].shape) != shape:
            raise ValueError("feature shape mismatch: %s" % name)
        if not torch.isfinite(values[name].float()).all().item():
            raise ValueError("non-finite feature: %s" % name)


def load_collection(collection_root, spec):
    events = []
    manifests = []
    for shard in (0, 1):
        root = collection_root / ("shard%d" % shard)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("complete") is not True or
                manifest.get("accepted") is not True or
                manifest.get("future_ground_truth_opened") is not False or
                manifest.get("ground_truth_used_after_initialization") is not False or
                manifest.get("candidate_committed_to_public_tracker") is not False or
                manifest.get("metric_computed") is not False or
                int(manifest["event_count"]) !=
                int(spec["expected_event_counts"][str(shard)])):
            raise ValueError("collection manifest contract mismatch")
        metadata_path = Path(manifest["metadata"]["path"])
        if (record(metadata_path)["sha256"] != manifest["metadata"]["sha256"] or
                metadata_path.stat().st_size != manifest["metadata"]["bytes"]):
            raise ValueError("metadata record mismatch")
        with metadata_path.open("r", encoding="utf-8") as stream:
            shard_events = [json.loads(line) for line in stream]
        if len(shard_events) != manifest["event_count"]:
            raise ValueError("event row count mismatch")
        for event in shard_events:
            feature_path = root / event["feature_path"]
            if (feature_path.stat().st_size != event["feature_bytes"] or
                    sha256_file(feature_path) != event["feature_sha256"]):
                raise ValueError("feature record mismatch")
            validate_feature_file(feature_path)
        events.extend(shard_events)
        manifests.append(record(manifest_path))
    keys = [(row["sequence"], int(row["trigger_frame"])) for row in events]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate collection event key")
    return events, manifests


def main():
    args = parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if (spec.get("complete") is not True or amendment.get("complete") is not True or
            amendment.get("future_ground_truth_opened") is not False or
            amendment["base_spec"]["sha256"] != sha256_file(args.spec) or
            args.collection_root != Path(spec["collection_root"]).resolve() or
            args.dataset_root != Path(spec["dataset_root"]).resolve() or
            args.output != Path(spec["gate_root"]).resolve()):
        raise ValueError("Gate-A input binding mismatch")
    events, manifests = load_collection(args.collection_root, spec)

    protected_rows = {}
    gt_by_sequence = {}
    gt_records = []
    failure_starts = {}
    public_overlap_by_sequence = {}
    for shard in (0, 1):
        trace_path = Path(spec["bindings"][
            "protected_trace_shard_%d" % shard]["path"])
        if sha256_file(trace_path) != spec["bindings"][
                "protected_trace_shard_%d" % shard]["sha256"]:
            raise ValueError("protected trace hash mismatch")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        grouped = {}
        for row in trace["rows"]:
            grouped.setdefault(row["sequence"], []).append(row)
        for sequence, rows in grouped.items():
            rows.sort(key=lambda row: int(row["frame_index"]))
            if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
                raise ValueError("public trace frame gap")
            public_boxes = [row["public_bbox"] for row in rows]
            gt, gt_path = load_ground_truth(args.dataset_root, sequence, len(rows))
            starts, overlaps = confirmed_failure_starts(public_boxes, gt)
            protected_rows[sequence] = rows
            gt_by_sequence[sequence] = gt
            failure_starts[sequence] = starts
            public_overlap_by_sequence[sequence] = overlaps
            gt_records.append(record(gt_path))

    labeled = []
    event_by_key = {}
    for event in events:
        sequence = event["sequence"]
        trigger = int(event["trigger_frame"])
        gt_window = gt_by_sequence[sequence][trigger:trigger + 10]
        public_boxes = [row["bbox"] for row in event["public"]]
        trajectories = event["trajectory"]
        names = [row["name"] for row in trajectories[0]["branches"]]
        event_by_key[(sequence, trigger)] = event
        for branch_index, name in enumerate(names):
            branch_boxes = [age["branches"][branch_index]["bbox"]
                            for age in trajectories]
            result = label_action(public_boxes, branch_boxes, gt_window)
            first = trajectories[0]["branches"][branch_index]
            labeled.append({
                "sequence": sequence, "event_id": event["event_id"],
                "trigger_frame": trigger, "branch_id": name,
                "source": first["source"], "peak_rank": first["peak_rank"],
                **result,
            })

    rescues = []
    for sequence, starts in failure_starts.items():
        for start in starts:
            event = event_by_key.get((sequence, start))
            if event is None:
                continue
            candidates = [row for row in labeled
                          if row["sequence"] == sequence and
                          int(row["trigger_frame"]) == start and
                          row["label"] != "unavailable" and
                          int(row["early_hits"]) >= 2 and
                          float(row["mean_iou_gain"]) >= 0.2 and
                          not all(value <= 0.1 for value in row["branch_ious"])]
            if not candidates:
                continue
            selected = sorted(
                candidates,
                key=lambda row: (float(row["branch_mean_iou"]),
                                 float(row["mean_iou_gain"]),
                                 str(row["branch_id"])), reverse=True)[0]
            rescues.append({
                "sequence": sequence, "confirmed_failure_start": start,
                "trigger_frame": start, "branch_id": selected["branch_id"],
                "source": selected["source"],
                "branch_mean_iou": selected["branch_mean_iou"],
                "public_mean_iou": selected["public_mean_iou"],
                "mean_iou_gain": selected["mean_iou_gain"],
                "early_hits": selected["early_hits"],
            })

    beneficial = [row for row in labeled if row["label"] == "beneficial"]
    catastrophic = [row for row in labeled if row["label"] == "catastrophic"]
    positive_sequences = sorted({row["sequence"] for row in beneficial})
    rescue_sequences = sorted({row["sequence"] for row in rescues})
    capacity_sources = {row["source"] for row in beneficial}
    capacity_sources.update(row["source"] for row in rescues)
    conditions = {
        "beneficial_actions_ge_30": len(beneficial) >= 30,
        "positive_sequences_ge_10": len(positive_sequences) >= 10,
        "rescued_confirmed_failures_ge_10": len(rescues) >= 10,
        "rescued_failure_sequences_ge_5": len(rescue_sequences) >= 5,
        "current_source_capacity": "current" in capacity_sources,
        "last_or_velocity_source_capacity": bool(
            {"last_reliable", "velocity"} & capacity_sources),
    }
    passed = all(conditions.values())
    label_counts = {name: sum(row["label"] == name for row in labeled)
                    for name in ("beneficial", "neutral", "catastrophic",
                                 "unavailable")}
    total_confirmed = sum(len(values) for values in failure_starts.values())

    args.output.mkdir(parents=True)
    labeled_path = args.output / "labeled_actions.jsonl.gz"
    rescues_path = args.output / "failure_rescues.jsonl.gz"
    atomic_gzip_jsonl(labeled_path, labeled)
    atomic_gzip_jsonl(rescues_path, rescues)
    result = {
        "schema": "sttrack-lachtt-train152-gatea-result/v1",
        "complete": True,
        "accepted": True,
        "future_ground_truth_opened": True,
        "public_evaluation": False,
        "vot_run": False,
        "event_count": len(events),
        "action_count": len(labeled),
        "label_counts": label_counts,
        "positive_sequence_count": len(positive_sequences),
        "positive_sequences": positive_sequences,
        "confirmed_failure_count": total_confirmed,
        "rescued_confirmed_failure_count": len(rescues),
        "rescued_failure_sequence_count": len(rescue_sequences),
        "rescued_failure_sequences": rescue_sequences,
        "capacity_sources": sorted(capacity_sources),
        "conditions": conditions,
        "gate_a_passed": passed,
        "decision": ("selector_training_authorized_train_only"
                     if passed else "stop_sttrack_actionspace_no_selector"),
        "automatic_next_stage": False,
        "depthtrack_test_authorized": False,
        "cdtb_authorized": False,
        "vot_low22_authorized": False,
        "vot_full127_authorized": False,
    }
    result_path = args.output / "gate_a_result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-train152-gatea-manifest/v1",
        "complete": True, "accepted": True,
        "spec": record(args.spec), "amendment": record(args.amendment),
        "collection_manifests": manifests,
        "ground_truth_files": gt_records,
        "labeled_actions": record(labeled_path),
        "failure_rescues": record(rescues_path),
        "result": record(result_path),
        "analyzer": record(Path(__file__).resolve()),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
