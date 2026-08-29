#!/usr/bin/env python3
"""Compare two frozen VOT trajectories frame by frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vot.region import RegionType
from vot.tracker import Trajectory
from vot.workspace import Workspace


def load_trajectory(
    workspace_path: Path,
    tracker_id: str,
    sequence_name: str,
    anchor: int,
) -> Trajectory:
    workspace = Workspace.load(str(workspace_path))
    trackers = workspace.registry.resolve(
        tracker_id,
        storage=workspace.storage.substorage("results"),
        skip_unknown=False,
    )
    if len(trackers) != 1:
        raise RuntimeError("expected exactly one tracker")
    experiment = workspace.stack.experiments["baseline"]
    sequence = next(
        (
            item
            for item in experiment.transform(workspace.dataset)
            if item.name == sequence_name
        ),
        None,
    )
    if sequence is None:
        raise ValueError(f"sequence not found: {sequence_name}")
    results = experiment.results(trackers[0], sequence)
    return Trajectory.read(results, f"{sequence_name}_{anchor:08d}")


def rectangle(region) -> Optional[Tuple[float, float, float, float]]:
    if region.type == RegionType.SPECIAL:
        return None
    value = region.convert(RegionType.RECTANGLE)
    return value.x, value.y, value.width, value.height


def region_value(region) -> Dict[str, Any]:
    if region.type == RegionType.SPECIAL:
        return {"type": "special", "code": int(region.code)}
    bbox = rectangle(region)
    return {"type": "rectangle", "bbox": list(bbox)}


def first_or_none(current: Optional[Dict[str, Any]], value: Dict[str, Any]):
    return value if current is None else current


def trace_event_at(path: Path, frame: int) -> Optional[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if (
                isinstance(value, dict)
                and value.get("type") == "transaction_frame"
                and int(value.get("frame_id", -1)) == frame
            ):
                return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-workspace", type=Path, required=True)
    parser.add_argument("--baseline-tracker", required=True)
    parser.add_argument("--candidate-workspace", type=Path, required=True)
    parser.add_argument("--candidate-tracker", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--max-frame", type=int, default=-1)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--bbox-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--confidence-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()

    baseline = load_trajectory(
        args.baseline_workspace,
        args.baseline_tracker,
        args.sequence,
        args.anchor,
    )
    candidate = load_trajectory(
        args.candidate_workspace,
        args.candidate_tracker,
        args.sequence,
        args.anchor,
    )
    if len(baseline) != len(candidate):
        raise RuntimeError("trajectory lengths differ")
    frame_count = len(baseline)
    if args.max_frame >= 0:
        frame_count = min(frame_count, args.max_frame + 1)

    first_type_difference = None
    first_nonzero_bbox_difference = None
    first_bbox_violation = None
    first_nonzero_confidence_difference = None
    first_confidence_violation = None
    max_bbox_difference = 0.0
    max_confidence_difference = 0.0

    for frame in range(frame_count):
        baseline_region = baseline.region(frame)
        candidate_region = candidate.region(frame)
        if baseline_region.type != candidate_region.type:
            first_type_difference = first_or_none(
                first_type_difference,
                {
                    "frame": frame,
                    "baseline": region_value(baseline_region),
                    "candidate": region_value(candidate_region),
                },
            )
            continue
        baseline_bbox = rectangle(baseline_region)
        candidate_bbox = rectangle(candidate_region)
        if baseline_bbox is not None and candidate_bbox is not None:
            error = max(
                abs(left - right)
                for left, right in zip(baseline_bbox, candidate_bbox)
            )
            max_bbox_difference = max(max_bbox_difference, error)
            record = {
                "frame": frame,
                "max_absolute_error": error,
                "baseline_bbox": list(baseline_bbox),
                "candidate_bbox": list(candidate_bbox),
            }
            if error > 0:
                first_nonzero_bbox_difference = first_or_none(
                    first_nonzero_bbox_difference, record
                )
            if error > args.bbox_tolerance:
                first_bbox_violation = first_or_none(first_bbox_violation, record)

        baseline_confidence = baseline.properties(frame).get("confidence")
        candidate_confidence = candidate.properties(frame).get("confidence")
        if isinstance(baseline_confidence, (int, float)) and isinstance(
            candidate_confidence, (int, float)
        ):
            error = abs(float(baseline_confidence) - float(candidate_confidence))
            max_confidence_difference = max(max_confidence_difference, error)
            record = {
                "frame": frame,
                "absolute_error": error,
                "baseline_confidence": float(baseline_confidence),
                "candidate_confidence": float(candidate_confidence),
            }
            if error > 0:
                first_nonzero_confidence_difference = first_or_none(
                    first_nonzero_confidence_difference, record
                )
            if error > args.confidence_tolerance:
                first_confidence_violation = first_or_none(
                    first_confidence_violation, record
                )

    parity = not any(
        (
            first_type_difference,
            first_bbox_violation,
            first_confidence_violation,
        )
    )
    branch_alignment = None
    if args.trace is not None and first_bbox_violation is not None:
        frame = int(first_bbox_violation["frame"])
        event = trace_event_at(args.trace, frame)
        if event is not None:
            baseline_bbox = first_bbox_violation["baseline_bbox"]
            candidate_bbox = first_bbox_violation["candidate_bbox"]
            protected_bbox = event.get("protected_bbox")
            tentative_bbox = event.get("tentative_bbox")

            def error(left, right):
                if not isinstance(left, list) or not isinstance(right, list):
                    return None
                if len(left) != len(right):
                    return None
                return max(abs(float(a) - float(b)) for a, b in zip(left, right))

            baseline_to_protected = error(baseline_bbox, protected_bbox)
            baseline_to_tentative = error(baseline_bbox, tentative_bbox)
            candidate_to_protected = error(candidate_bbox, protected_bbox)
            candidate_to_tentative = error(candidate_bbox, tentative_bbox)
            closest_to_baseline = None
            if baseline_to_protected is not None and baseline_to_tentative is not None:
                closest_to_baseline = (
                    "protected"
                    if baseline_to_protected <= baseline_to_tentative
                    else "tentative"
                )
            closest_to_candidate = None
            if candidate_to_protected is not None and candidate_to_tentative is not None:
                closest_to_candidate = (
                    "protected"
                    if candidate_to_protected <= candidate_to_tentative
                    else "tentative"
                )
            branch_alignment = {
                "frame": frame,
                "decision": event.get("decision"),
                "baseline_to_protected_max_error": baseline_to_protected,
                "baseline_to_tentative_max_error": baseline_to_tentative,
                "candidate_to_protected_max_error": candidate_to_protected,
                "candidate_to_tentative_max_error": candidate_to_tentative,
                "closest_branch_to_formal_baseline": closest_to_baseline,
                "closest_branch_to_formal_candidate": closest_to_candidate,
                "protected_bbox": protected_bbox,
                "tentative_bbox": tentative_bbox,
            }
    output = {
        "schema": "vot_anchor_trajectory_parity/v1",
        "verdict": "PARITY_EXACT" if parity else "PARITY_VIOLATION",
        "sequence": args.sequence,
        "anchor": args.anchor,
        "compared_frames": frame_count,
        "bbox_tolerance": args.bbox_tolerance,
        "confidence_tolerance": args.confidence_tolerance,
        "max_bbox_absolute_error": max_bbox_difference,
        "max_confidence_absolute_error": max_confidence_difference,
        "first_type_difference": first_type_difference,
        "first_nonzero_bbox_difference": first_nonzero_bbox_difference,
        "first_bbox_violation": first_bbox_violation,
        "first_nonzero_confidence_difference": first_nonzero_confidence_difference,
        "first_confidence_violation": first_confidence_violation,
        "branch_alignment_at_first_bbox_violation": branch_alignment,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
