import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


METRIC_SHA256 = "05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgbd(color_path, depth_path):
    rgb = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError("Missing RGB or depth frame")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
    depth = depth.astype(np.float32)
    median_depth = np.median(depth)
    if median_depth > 0:
        limit = min(median_depth * 3.0, 10000.0)
        depth[depth > limit] = limit
    normalized = cv2.normalize(
        depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    return np.concatenate(
        (rgb, cv2.applyColorMap(normalized, cv2.COLORMAP_JET)), axis=2
    )


def tracker_for(source, checkpoint):
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.test.tracker.cadtrack import CADTrack

    update_config_from_file(str(source / "experiments/cadtrack/cadtrack.yaml"))
    params = SimpleNamespace(
        cfg=cfg,
        checkpoint=str(checkpoint),
        template_factor=cfg.TEST.TEMPLATE_FACTOR,
        template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE,
        debug=0,
        save_all_boxes=False,
    )
    return CADTrack(params)


def track(args):
    cases_path = args.cases.resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    expected = {"depthtrack_test": (50, 76373), "cdtb": (80, 101956)}
    if args.dataset not in expected:
        raise ValueError("Unknown dataset")
    if (len(cases), sum(row["frames"] for row in cases)) != expected[args.dataset]:
        raise ValueError("Dataset case list does not match the locked full-set count")
    checkpoint_sha256 = sha256(args.checkpoint)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tracker = tracker_for(args.source.resolve(), args.checkpoint.resolve())
    started = time.time()
    rows = []
    dataset_root = args.dataset_root.resolve()
    for case_index, case in enumerate(cases, 1):
        folder = dataset_root / case["sequence"]
        boxes = []
        scores = []
        for frame_index in range(case["frames"]):
            stem = "%08d" % (frame_index + 1)
            image = load_rgbd(
                folder / "color" / (stem + ".jpg"),
                folder / "depth" / (stem + ".png"),
            )
            if image.shape[2] != 6:
                raise ValueError("Expected RGB plus three-channel depth input")
            if frame_index == 0:
                bbox = [float(value) for value in case["init_bbox"]]
                tracker.initialize(image, {"init_bbox": bbox})
                prediction, confidence = bbox, 1.0
            else:
                result = tracker.track(image)
                prediction = result["target_bbox"]
                confidence = result["best_score"]
            prediction = np.asarray(prediction, dtype=np.float64).reshape(4)
            confidence = float(confidence)
            if (not np.isfinite(prediction).all() or
                    not np.isfinite(confidence) or
                    (prediction[2:] <= 0).any()):
                raise ValueError("Non-finite or invalid tracker output")
            boxes.append(prediction)
            scores.append(confidence)

        box_path = output / (case["sequence"] + ".txt")
        score_path = output / (case["sequence"] + "_all_scores.txt")
        np.savetxt(box_path, np.asarray(boxes), fmt="%.6f", delimiter=",")
        np.savetxt(score_path, np.asarray(scores), fmt="%.6f")
        rows.append({
            "sequence": case["sequence"],
            "frames": case["frames"],
            "groundtruth_sha256": case["gt_sha256"],
            "boxes_sha256": sha256(box_path),
            "scores_sha256": sha256(score_path),
            "elapsed_seconds": time.time() - started,
        })
        print(json.dumps({
            "dataset": args.dataset,
            "sequence_index": case_index,
            "sequence_count": len(cases),
            **rows[-1],
        }, sort_keys=True), flush=True)

    receipt = {
        "status": "complete_CADTrack_RGBD_OPE_predictions",
        "dataset": args.dataset,
        "checkpoint_sha256": checkpoint_sha256,
        "cases_sha256": sha256(cases_path),
        "runner_sha256": sha256(__file__),
        "sequences": rows,
        "frames": sum(row["frames"] for row in rows),
        "elapsed_seconds": time.time() - started,
        "groundtruth_opened_during_tracking": False,
        "optimizer_steps": 0,
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


def score(args):
    output = args.output.resolve()
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "complete_CADTrack_RGBD_OPE_predictions":
        raise ValueError("Predictions are not sealed")
    if sha256(args.checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the prediction receipt")
    if receipt["dataset"] != args.dataset:
        raise ValueError("Prediction dataset mismatch")
    if sha256(args.metric_source) != METRIC_SHA256:
        raise ValueError("Metric source digest differs from the locked implementation")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if sha256(args.cases) != receipt["cases_sha256"]:
        raise ValueError("Case list differs from prediction receipt")
    if len(receipt["sequences"]) != len(cases):
        raise ValueError("Incomplete prediction sequence set")
    for case, row in zip(cases, receipt["sequences"]):
        if case["sequence"] != row["sequence"] or case["frames"] != row["frames"]:
            raise ValueError("Prediction sequence order/count mismatch")
        if sha256(args.dataset_root / case["sequence"] / "groundtruth.txt") != case["gt_sha256"]:
            raise ValueError("Ground-truth digest mismatch")
        if sha256(output / (case["sequence"] + ".txt")) != row["boxes_sha256"]:
            raise ValueError("Prediction file digest mismatch")
        if sha256(output / (case["sequence"] + "_all_scores.txt")) != row["scores_sha256"]:
            raise ValueError("Confidence file digest mismatch")

    spec = importlib.util.spec_from_file_location(
        "locked_depthtrack_pr_metric", args.metric_source.resolve()
    )
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    values = metric.evaluate_depthtrack_results(
        args.dataset_root.resolve(), output, resolution=100,
        sequence_names=[case["sequence"] for case in cases],
    )
    result = {
        "status": "complete_CADTrack_RGBD_OPE_metrics",
        "dataset": args.dataset,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "cases_sha256": receipt["cases_sha256"],
        "runner_sha256": receipt["runner_sha256"],
        "metric_source_sha256": METRIC_SHA256,
        "receipt_sha256": sha256(receipt_path),
        "metrics": values,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["track", "score"])
    parser.add_argument("--dataset", choices=["depthtrack_test", "cdtb"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "track":
        track(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
