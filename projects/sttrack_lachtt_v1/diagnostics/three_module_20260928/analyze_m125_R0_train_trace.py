"""CPU-only description of sparse R0 training traces; not final-model evaluation."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(gt, box, crop):
    gx, gy, gw, gh = map(float, gt)
    bx, by, bw, bh = map(float, box)
    x, y, size = map(float, crop)
    assert all(math.isfinite(v) for v in (*gt, *box, *crop))
    assert gw > 0 and gh > 0 and bw > 0 and bh > 0 and size > 0
    intersection = max(0., min(gx + gw, bx + bw) - max(gx, bx)) * max(
        0., min(gy + gh, by + bh) - max(gy, by))
    crop_intersection = max(0., min(gx + gw, x + size) - max(gx, x)) * max(
        0., min(gy + gh, y + size) - max(gy, y))
    return dict(
        selected_iou=intersection / (gw * gh + bw * bh - intersection),
        gt_area_fraction_in_crop=crop_intersection / (gw * gh),
        gt_center_inside_crop=(x <= gx + gw / 2 < x + size
                               and y <= gy + gh / 2 < y + size))


def describe(rows):
    assert rows
    inside = [row for row in rows if row["gt_center_inside_crop"]]
    outside = [row for row in rows if not row["gt_center_inside_crop"]]
    return dict(
        valid_samples=len(rows), center_inside=len(inside), center_outside=len(outside),
        zero_crop_intersection=sum(row["gt_area_fraction_in_crop"] == 0 for row in rows),
        selected_zero_iou=sum(row["selected_iou"] == 0 for row in rows),
        selected_iou_at_least_half=sum(row["selected_iou"] >= .5 for row in rows),
        inside_selected_zero_iou=sum(row["selected_iou"] == 0 for row in inside),
        outside_selected_zero_iou=sum(row["selected_iou"] == 0 for row in outside),
        mean_selected_iou=sum(row["selected_iou"] for row in rows) / len(rows),
        mean_gt_area_fraction_in_crop=sum(row["gt_area_fraction_in_crop"] for row in rows) / len(rows))


def main(root, output):
    assert not output.exists()
    gate = read(root / "gate.json")
    assert gate["status"] == "reviewed_M125_R0_fixed6_source_gate"
    assert gate["fixed_budget_passes"] == gate["fixed_final_pass"] == 6
    assert read(root / "result.json")["status"] == "complete_M125_R0_fixed6_and_three_official_datasets"
    assert (root / "controller.exit").read_text().strip() == "0"
    spec_path, code = Path(gate["spec"]), Path(gate["code_root"])
    assert sha(spec_path) == "3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425"
    spec = read(spec_path)
    assert spec["seed"] == 2027 and len(spec["sequence_order"]) == 152
    assert Path(spec["dataset_root"]) == Path("/root/autodl-tmp/depthtrack/train/sequences")
    for path, digest in gate["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    sys.path.insert(0, str(code))
    from train_m122_full_causal import load_truth
    truths = {case["sequence"]: load_truth(Path(spec["dataset_root"]) / case["sequence"], case)
              for case in spec["sequence_order"]}
    expected = [(case["sequence"], frame) for case in spec["sequence_order"]
                for frame in range(1, case["rgb_frames"])
                if frame == 1 or frame % 500 == 0 or frame == case["rgb_frames"] - 1]
    source_paths = [Path(__file__), root / "gate.json", root / "result.json", spec_path,
                    code / "train_m125_R0_fixed_pass.py", code / "m123_joint_full_tracker.py",
                    code / "train_m122_full_causal.py"]
    summaries, samples = [], []
    for epoch in range(1, 7):
        folder = root / ("passes/pass-%02d/fit" % epoch)
        result_path, trace_path = folder / "result.json", folder / "sampled_state_trace.jsonl"
        result = read(result_path)
        assert result["epochs"] == epoch and result["sequence_runs"] == epoch * 152
        assert result["track_calls"] == epoch * 219802
        assert result["spec_sha256"] == sha(spec_path)
        assert result["source_sha256"] == sha(code / "train_m125_R0_fixed_pass.py")
        assert result["tracker_source_sha256"] == sha(code / "m123_joint_full_tracker.py")
        assert result["frozen_before_after_exact"]
        assert result["GT_reinitializations_after_first_frame"] == 0
        assert result["GT_used_after_action_for_loss_only"]
        assert not result["external_metric_checkpoint_selection"]
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert [(row["sequence"], row["frame"]) for row in rows] == expected
        local = []
        for row in rows:
            assert row["epoch"] == epoch
            gt = truths[row["sequence"]][row["frame"]]
            valid = bool(np.isfinite(gt).all() and (gt[2:] > 0).all())
            assert valid == row["GT_valid_for_loss"]
            sample = dict(epoch=epoch, sequence=row["sequence"], frame=row["frame"], GT_valid=valid,
                          gt_x=gt[0], gt_y=gt[1], gt_w=gt[2], gt_h=gt[3],
                          box_x=row["bbox"][0], box_y=row["bbox"][1],
                          box_w=row["bbox"][2], box_h=row["bbox"][3],
                          crop_x=row["crop_origin"][0], crop_y=row["crop_origin"][1],
                          crop_size=row["crop_origin"][2], selected_iou=None,
                          gt_area_fraction_in_crop=None, gt_center_inside_crop=None)
            if valid:
                sample.update(geometry(gt, row["bbox"], row["crop_origin"]))
            local.append(sample)
        valid = [row for row in local if row["GT_valid"]]
        summaries.append(dict(epoch=epoch, samples=len(local), invalid_GT_samples=len(local) - len(valid),
                              all_valid=describe(valid),
                              after_frame_one=describe([row for row in valid if row["frame"] > 1])))
        samples.extend(local)
        source_paths.extend([result_path, trace_path])
    inputs = {str(path): sha(path) for path in source_paths}
    inputs.update({str(Path(spec["dataset_root"]) / case["sequence"] / "groundtruth.txt"):
                   case["groundtruth_sha256"] for case in spec["sequence_order"]})
    report = dict(status="complete_sparse_R0_training_trace_description",
                  observed_at_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                  neural_executions=0, optimizer_steps=0, rows=len(samples), sequences_per_pass=152,
                  passes=summaries, input_sha256=inputs,
                  limitations=[
                      "States were sampled at frame 1, every 500 frames and each sequence's final frame.",
                      "Uniform sample counts are not uniform frame weights; this is not an all-frame rate.",
                      "Model parameters change during each training pass; these are not final-checkpoint trajectories.",
                      "Sparse states cannot locate failure onsets or recover full candidate-set coverage.",
                      "Frame 500 sampling aligns with cadence50; sampled writes cannot estimate all-frame write frequency.",
                      "Center inside the search crop does not imply a usable or correctly ranked candidate.",
                      "GT is used retrospectively for this description only; no Test50/CDTB/VOT metric is computed.",
                      "Present source/result/trace hashes provide retrospective binding, not original trace execution attestation."
                  ])
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(dict(rows=report["rows"], passes=summaries)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
