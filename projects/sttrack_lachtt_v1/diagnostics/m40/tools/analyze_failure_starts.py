#!/usr/bin/env python3
"""Read-only census of M39 STTrack failure-start search geometry."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from vot.region import RegionType
from vot.region.io import read_trajectory
from vot.region.raster import calculate_overlap
from vot.region.shapes import Rectangle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def region_box(region):
    rectangle = region.convert(RegionType.RECTANGLE)
    return rectangle.x, rectangle.y, rectangle.width, rectangle.height


def gt_regions(path):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split(",")]
        if len(values) != 4:
            raise ValueError(f"non-rectangle groundtruth in {path}")
        result.append(Rectangle(*values))
    return result


def confidence_values(path):
    return [
        None if value.strip() == "" else float(value)
        for value in path.read_text(encoding="utf-8").splitlines()
    ]


def crop_geometry(state_box, target_box, factor):
    state_x, state_y, state_w, state_h = state_box
    target_x, target_y, target_w, target_h = target_box
    side = math.ceil(math.sqrt(state_w * state_h) * factor)
    state_cx = state_x + 0.5 * state_w
    state_cy = state_y + 0.5 * state_h
    x1 = round(state_cx - 0.5 * side)
    y1 = round(state_cy - 0.5 * side)
    x2 = x1 + side
    y2 = y1 + side
    target_cx = target_x + 0.5 * target_w
    target_cy = target_y + 0.5 * target_h
    intersection_width = max(0.0, min(x2, target_x + target_w) - max(x1, target_x))
    intersection_height = max(0.0, min(y2, target_y + target_h) - max(y1, target_y))
    target_area = target_w * target_h
    fraction = intersection_width * intersection_height / target_area
    return {
        "side": side,
        "center_inside": x1 <= target_cx < x2 and y1 <= target_cy < y2,
        "box_fraction_inside": fraction,
        "box_fully_inside": fraction >= 1.0 - 1e-12,
    }


def minimum_factors(state_box, target_box):
    state_x, state_y, state_w, state_h = state_box
    target_x, target_y, target_w, target_h = target_box
    state_cx = state_x + 0.5 * state_w
    state_cy = state_y + 0.5 * state_h
    target_cx = target_x + 0.5 * target_w
    target_cy = target_y + 0.5 * target_h
    scale = math.sqrt(state_w * state_h)
    center_factor = 2.0 * max(
        abs(target_cx - state_cx), abs(target_cy - state_cy)) / scale
    full_box_factor = 2.0 * max(
        abs(target_cx - state_cx) + 0.5 * target_w,
        abs(target_cy - state_cy) + 0.5 * target_h,
    ) / scale
    return center_factor, full_box_factor


def search_class(coverage):
    if coverage["4.0"]["center_inside"]:
        return "inside_factor4"
    if coverage["6.0"]["center_inside"]:
        return "factor6_only"
    if coverage["7.0"]["center_inside"]:
        return "factor7_only"
    return "outside_factor7"


def summarize_rows(rows, all_sequences, per_sequence_failures):
    classes = [
        "inside_factor4", "factor6_only", "factor7_only", "outside_factor7"]
    class_counts = {
        label: sum(row["onset_search_class"] == label for row in rows)
        for label in classes}
    confidences = [row["failure_confidence"] for row in rows]
    center_factors = [row["minimum_factor_for_target_center"] for row in rows]
    summary = {
        "failed_anchor_count": len(rows),
        "onset_search_class_counts": class_counts,
        "onset_search_class_percent": {
            key: 100.0 * value / len(rows) for key, value in class_counts.items()},
        "factor4_full_target_box_count": sum(
            row["factor4_box_fully_inside"] for row in rows),
        "factor4_at_least_half_target_box_count": sum(
            row["factor4_box_fraction_inside"] >= 0.5 for row in rows),
        "high_confidence_ge_0_75_count": sum(value >= 0.75 for value in confidences),
        "high_confidence_and_inside_factor4_count": sum(
            row["failure_confidence"] >= 0.75
            and row["onset_search_class"] == "inside_factor4" for row in rows),
        "failure_confidence_quantiles": {
            "q10": quantile(confidences, 0.10),
            "q25": quantile(confidences, 0.25),
            "q50": quantile(confidences, 0.50),
            "q75": quantile(confidences, 0.75),
            "q90": quantile(confidences, 0.90),
        },
        "minimum_factor_for_target_center_quantiles": {
            "q50": quantile(center_factors, 0.50),
            "q75": quantile(center_factors, 0.75),
            "q90": quantile(center_factors, 0.90),
            "q95": quantile(center_factors, 0.95),
            "max": max(center_factors),
        },
        "h10_frame_count": 10 * len(rows),
        "h10_target_center_inside_counts": {
            str(factor): sum(row[f"h10_factor{int(factor)}_center_inside_count"] for row in rows)
            for factor in (4, 6, 7)},
        "per_sequence": [],
    }
    for sequence in all_sequences:
        sequence_rows = [row for row in rows if row["sequence"] == sequence]
        sequence_counts = {
            label: sum(row["onset_search_class"] == label for row in sequence_rows)
            for label in classes}
        failure_count = len(sequence_rows)
        summary["per_sequence"].append({
            "sequence": sequence,
            "total_anchors": per_sequence_failures[sequence]["anchors"],
            "confirmed_failures": failure_count,
            "inside_factor4": sequence_counts["inside_factor4"],
            "factor6_only": sequence_counts["factor6_only"],
            "factor7_only": sequence_counts["factor7_only"],
            "outside_factor7": sequence_counts["outside_factor7"],
            "inside_factor4_percent": (
                100.0 * sequence_counts["inside_factor4"] / failure_count
                if failure_count else None),
            "median_failure_confidence": (
                median(row["failure_confidence"] for row in sequence_rows)
                if failure_count else None),
            "high_confidence_ge_0_75": sum(
                row["failure_confidence"] >= 0.75 for row in sequence_rows),
            "median_minimum_center_factor": (
                median(row["minimum_factor_for_target_center"] for row in sequence_rows)
                if failure_count else None),
        })
    return summary


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in spec["paths"].items()}
    for key, expected in spec["input_sha256"].items():
        actual = sha256_file(paths[key])
        if actual != expected:
            raise ValueError(f"input hash mismatch for {key}: {actual}")

    m39 = json.loads(paths["m39_result"].read_text(encoding="utf-8"))
    default = m39["arms"]["default"]
    analysis = json.loads(paths["default_analysis"].read_text(encoding="utf-8"))
    dimensions = analysis["sequences"]
    result_root = paths["trajectory_root"]
    sequence_root = paths["sequence_root"]
    factors = spec["search_factors"]
    rows = []

    for anchor_key, outcome in sorted(default["failure_outcomes"].items()):
        if not outcome["failed"]:
            continue
        sequence = outcome["sequence"]
        anchor = outcome["anchor"]
        direction = outcome["direction"]
        progress = outcome["progress"]
        trajectory_name = f"{sequence}_{anchor:08d}"
        trajectory_path = result_root / sequence / f"{trajectory_name}.bin"
        confidence_path = result_root / sequence / f"{trajectory_name}_confidence.value"
        with trajectory_path.open("rb") as handle:
            regions = read_trajectory(handle)
        confidences = confidence_values(confidence_path)
        groundtruth = gt_regions(sequence_root / sequence / "groundtruth.txt")
        if direction == "forward":
            source_indices = [anchor + index for index in range(len(regions))]
        else:
            source_indices = [anchor - index for index in range(len(regions))]
        if len(regions) != outcome["run_length"] or len(confidences) != len(regions):
            raise ValueError(f"trajectory length mismatch for {anchor_key}")
        if progress < 1 or progress + 10 > len(regions):
            raise ValueError(f"invalid failure progress for {anchor_key}")

        previous_trajectory_index = progress - 1
        previous_source_index = source_indices[previous_trajectory_index]
        failure_source_index = source_indices[progress]
        if previous_trajectory_index == 0:
            previous_state = region_box(groundtruth[previous_source_index])
        else:
            previous_state = region_box(regions[previous_trajectory_index])
        target_box = region_box(groundtruth[failure_source_index])
        prediction_box = region_box(regions[progress])
        coverage = {
            str(factor): crop_geometry(previous_state, target_box, factor)
            for factor in factors}
        center_factor, full_box_factor = minimum_factors(previous_state, target_box)
        width = dimensions[sequence]["width"]
        height = dimensions[sequence]["height"]
        bounds = (width, height)
        failure_iou = calculate_overlap(
            regions[progress], groundtruth[failure_source_index], bounds)
        previous_iou = calculate_overlap(
            (groundtruth[previous_source_index]
             if previous_trajectory_index == 0
             else regions[previous_trajectory_index]),
            groundtruth[previous_source_index], bounds)

        h10_ious = []
        h10_confidences = []
        h10_coverage_counts = {factor: 0 for factor in factors}
        for trajectory_index in range(progress, progress + 10):
            source_index = source_indices[trajectory_index]
            previous_index = trajectory_index - 1
            previous_source = source_indices[previous_index]
            state = (
                region_box(groundtruth[previous_source])
                if previous_index == 0
                else region_box(regions[previous_index]))
            target = region_box(groundtruth[source_index])
            for factor in factors:
                h10_coverage_counts[factor] += crop_geometry(
                    state, target, factor)["center_inside"]
            h10_ious.append(calculate_overlap(
                regions[trajectory_index], groundtruth[source_index], bounds))
            h10_confidences.append(confidences[trajectory_index])
        if not all(value <= 0.1 for value in h10_ious):
            raise ValueError(f"failure H10 mismatch for {anchor_key}: {h10_ious}")
        if any(value is None or not math.isfinite(value) for value in h10_confidences):
            raise ValueError(f"invalid failure confidence for {anchor_key}")

        pred_cx = prediction_box[0] + 0.5 * prediction_box[2]
        pred_cy = prediction_box[1] + 0.5 * prediction_box[3]
        gt_cx = target_box[0] + 0.5 * target_box[2]
        gt_cy = target_box[1] + 0.5 * target_box[3]
        gt_scale = math.sqrt(target_box[2] * target_box[3])
        failure_confidence = confidences[progress]
        row = {
            "anchor_key": anchor_key,
            "sequence": sequence,
            "anchor": anchor,
            "direction": direction,
            "progress": progress,
            "run_length": outcome["run_length"],
            "source_frame_zero_based": failure_source_index,
            "source_frame_one_based": failure_source_index + 1,
            "previous_iou": previous_iou,
            "failure_iou": failure_iou,
            "failure_confidence": failure_confidence,
            "h10_mean_confidence": mean(h10_confidences),
            "h10_mean_iou": mean(h10_ious),
            "prediction_center_error_gt_scale": math.hypot(
                pred_cx - gt_cx, pred_cy - gt_cy) / gt_scale,
            "prediction_to_gt_area_ratio": (
                prediction_box[2] * prediction_box[3]
                / (target_box[2] * target_box[3])),
            "minimum_factor_for_target_center": center_factor,
            "minimum_factor_for_full_target_box": full_box_factor,
            "onset_search_class": search_class(coverage),
            "factor4_crop_side": coverage["4.0"]["side"],
            "factor4_center_inside": coverage["4.0"]["center_inside"],
            "factor4_box_fraction_inside": coverage["4.0"]["box_fraction_inside"],
            "factor4_box_fully_inside": coverage["4.0"]["box_fully_inside"],
            "factor6_center_inside": coverage["6.0"]["center_inside"],
            "factor7_center_inside": coverage["7.0"]["center_inside"],
            "h10_factor4_center_inside_count": h10_coverage_counts[4.0],
            "h10_factor6_center_inside_count": h10_coverage_counts[6.0],
            "h10_factor7_center_inside_count": h10_coverage_counts[7.0],
        }
        if failure_confidence is None or not math.isfinite(failure_confidence):
            raise ValueError(f"invalid onset confidence for {anchor_key}")
        rows.append(row)

    if len(rows) != default["confirmed_failures"]:
        raise ValueError("failed-anchor count mismatch")
    all_sequences = sorted(default["per_sequence_failures"])
    summary = summarize_rows(rows, all_sequences, default["per_sequence_failures"])
    hardest = sorted(
        rows, key=lambda row: row["minimum_factor_for_target_center"], reverse=True)[:20]
    high_confidence_inside = sorted(
        [row for row in rows
         if row["onset_search_class"] == "inside_factor4"
         and row["failure_confidence"] >= 0.75],
        key=lambda row: row["failure_confidence"], reverse=True)[:20]

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "m40_failure_start_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": "sttrack_m40_failure_start_census_v1",
        "status": "complete",
        "scope": {
            "tracker": default["tracker"],
            "sequence_count": len(all_sequences),
            "anchor_count": len(default["failure_outcomes"]),
            "confirmed_failure_count": len(rows),
            "optimizer_steps": 0,
            "gpu_inference_frames": 0,
            "new_checkpoint": False,
            "public_metric": False,
        },
        "definitions": spec["definitions"],
        "input_sha256": spec["input_sha256"],
        "summary": summary,
        "hardest_search_domain_examples": hardest,
        "high_confidence_inside_factor4_examples": high_confidence_inside,
        "rows_csv": str(csv_path),
        "rows_csv_sha256": sha256_file(csv_path),
    }
    result_path = output_dir / "m40_failure_start_census.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "summary": summary,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "csv_sha256": result["rows_csv_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
