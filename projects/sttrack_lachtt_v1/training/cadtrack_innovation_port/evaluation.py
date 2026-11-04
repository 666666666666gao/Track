"""Official full-set OPE and VOT entrypoints for one trained CADTrack port final."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import struct


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def initialization_key(rgb_sha, bbox):
    return hashlib.sha256(
        bytes.fromhex(rgb_sha) + struct.pack(">4d", *bbox)
    ).hexdigest()


def checked_plan(path):
    plan = read(path)
    assert plan["schema"] == "CADTrack_M125_port_official_eval_v1"
    assert sha(plan["bundle_path"]) == plan["bundle_sha256"]
    bundle = read(plan["bundle_path"])
    assert bundle["schema"] == "CADTrack_port_eval_binding_v1"
    assert all(plan[key] == value for key, value in bundle["bindings"].items())
    for name, digest in plan["source_sha256"].items():
        assert sha(name) == digest, name
    assert sha(plan["training_result_path"]) == plan["training_result_sha256"]
    trained = read(plan["training_result_path"])
    assert trained["status"] == "complete_train"
    assert trained["calls"] == 1318812 and trained["final_sha256"] == plan["final_sha256"]
    assert trained["frozen_before_after_exact"] and trained["module_roundtrip_exact"]
    assert sha(plan["final_path"]) == plan["final_sha256"]
    assert sha(plan["bank_path"]) == plan["bank_sha256"]
    assert sha(plan["binding_path"]) == plan["binding_sha256"]
    assert sha(plan["checkpoint_path"]) == plan["checkpoint_sha256"]
    assert sha(plan["config_path"]) == plan["config_sha256"]
    assert sha(plan["clip_weight_path"]) == plan["clip_weight_sha256"]
    assert sha(plan["evaluation_source_path"]) == plan["evaluation_source_sha256"]
    assert sha(plan["evaluation_preparer_path"]) == plan["evaluation_preparer_sha256"]
    assert sha(plan["evaluation_runner_path"]) == plan["evaluation_runner_sha256"]
    for name in ("launch", "terminal", "training_rows"):
        assert sha(plan[name + "_path"]) == plan[name + "_sha256"]
    for name in ("failure_source", "failure_common"):
        assert sha(plan[name + "_path"]) == plan[name + "_sha256"]
    assert sha(plan["vot_identity_tool_path"]) == plan["vot_identity_tool_sha256"]
    assert sha(plan["vot_reference_path"]) == plan["vot_reference_sha256"]
    if plan["dataset"] == "vot":
        assert sha(plan["vot_bridge_path"]) == plan["vot_bridge_sha256"]
    if plan["dataset"] != "vot":
        assert sha(plan["cases_path"]) == plan["cases_sha256"]
        assert sha(plan["metric_source"]) == plan["metric_source_sha256"]
    return plan, trained


def load_tracker(plan):
    sys.path[:0] = [plan["code_root"], plan["legacy_code_root"]]
    import torch
    torch.set_num_threads(1)
    from runtime import load_final_tracker
    bank = torch.load(plan["bank_path"], map_location="cpu")
    assert bank["format"] == "M122_human_official_initialization_v1"
    assert bank["human_confirmed"]
    assert bank["binding_sha256"] == plan["binding_sha256"]
    assert bank["encoder_sha256"] == plan["clip_weight_sha256"]
    assert len(bank["keys"]) == len(set(bank["keys"])) == 1895
    assert tuple(bank["tokens"].shape) == (1895, 5, 768)
    assert tuple(bank["mask"].shape) == (1895, 5)
    assert bank["mask"].dtype == torch.bool and bool(bank["mask"][:, 0].all())
    assert bool(torch.isfinite(bank["tokens"]).all()) and bool(torch.isfinite(bank["empty"]).all())
    binding = read(plan["binding_path"])
    records = binding["rows"]
    assert len(records) == 1895 and len({row["key"] for row in records}) == 1895
    assert set(bank["keys"]) == {row["key"] for row in records}
    indices = {key: index for index, key in enumerate(bank["keys"])}
    tracker = load_final_tracker(
        Path(plan["repository"]), Path(plan["checkpoint_path"]),
        Path(plan["config_path"]), Path(plan["clip_weight_path"]),
        bank, Path(plan["final_path"]))
    from full_dense_tracker import state_digest
    initial = dict(
        native_clip=tracker.actor.frozen_digest(),
        A_B=state_digest(tracker.actor.decoder),
        C=state_digest(tracker.writer))
    return torch, tracker, bank, indices, initial, state_digest


def ope_track(plan_path):
    plan, trained = checked_plan(plan_path)
    assert plan["dataset"] in ("depthtrack_test", "cdtb")
    cases = read(plan["cases_path"])
    expected = {"depthtrack_test": (50, 76373), "cdtb": (80, 101956)}[plan["dataset"]]
    assert (len(cases), sum(case["frames"] for case in cases)) == expected
    torch, tracker, bank, indices, initial, state_digest = load_tracker(plan)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    records = []
    started = time.time()
    dataset = plan["dataset"]
    for case in cases:
        folder = Path(plan["dataset_root"]) / case["sequence"]
        rgb = folder / "color" / "00000001.jpg"
        key = dataset + ":" + initialization_key(sha(rgb), case["init_bbox"])
        assert key in indices
        boxes = [case["init_bbox"]]
        scores = [1.0]
        for frame in range(case["frames"]):
            stem = "%08d" % (frame + 1)
            image = get_rgbd_frame(
                str(folder / "color" / (stem + ".jpg")),
                str(folder / "depth" / (stem + ".png")),
                dtype="rgbcolormap", depth_clip=True)
            if frame == 0:
                tracker.initialize(image, case["init_bbox"], indices[key])
            else:
                with torch.no_grad():
                    prediction = tracker.track(image)
                boxes.append(prediction["target_bbox"])
                scores.append(prediction["best_score"])
        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        assert boxes.shape == (case["frames"], 4)
        assert scores.shape == (case["frames"],)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert (boxes[:, 2:] > 0).all()
        box_file = output / (case["sequence"] + ".txt")
        score_file = output / (case["sequence"] + "_all_scores.txt")
        np.savetxt(box_file, boxes, fmt="%.6f", delimiter=",")
        np.savetxt(score_file, scores, fmt="%.6f")
        assert np.abs(
            np.loadtxt(box_file, delimiter=",").reshape(-1, 4) - boxes).max() <= 5.01e-7
        assert np.abs(np.loadtxt(score_file).reshape(-1) - scores).max() <= 5.01e-7
        record = dict(
            sequence=case["sequence"], frames=case["frames"], bank_key=key,
            bbox_sha256=sha(box_file), confidence_sha256=sha(score_file),
            elapsed_seconds=time.time() - started)
        records.append(record)
        print(json.dumps(record), flush=True)
    assert tracker.actor.frozen_digest() == initial["native_clip"]
    assert state_digest(tracker.actor.decoder) == initial["A_B"]
    assert state_digest(tracker.writer) == initial["C"]
    receipt = dict(
        status="complete_CADTrack_port_OPE_predictions",
        plan_sha256=sha(plan_path), training_result_sha256=plan["training_result_sha256"],
        final_sha256=plan["final_sha256"], bank_sha256=plan["bank_sha256"],
        binding_sha256=plan["binding_sha256"], sequences=records,
        frames=sum(row["frames"] for row in records), subsequent_GT_opened=False,
        optimizer_steps=0, frozen_before_after_exact=True,
        elapsed_seconds=time.time() - started)
    write(output / "receipt.json", receipt)
    assert sha(plan["final_path"]) == trained["final_sha256"]


def ope_analyze(plan_path):
    plan, _ = checked_plan(plan_path)
    assert plan["dataset"] in ("depthtrack_test", "cdtb")
    output = Path(plan["output"])
    receipt_path = output / "receipt.json"
    receipt = read(receipt_path)
    assert receipt["status"] == "complete_CADTrack_port_OPE_predictions"
    assert receipt["plan_sha256"] == sha(plan_path)
    cases = read(plan["cases_path"])
    assert [row["sequence"] for row in receipt["sequences"]] == [
        case["sequence"] for case in cases]
    for case, row in zip(cases, receipt["sequences"]):
        assert case["frames"] == row["frames"]
        assert sha(output / (case["sequence"] + ".txt")) == row["bbox_sha256"]
        assert sha(output / (case["sequence"] + "_all_scores.txt")) == row["confidence_sha256"]
        gt = Path(plan["dataset_root"]) / case["sequence"] / "groundtruth.txt"
        assert sha(gt) == case["gt_sha256"]
    spec = importlib.util.spec_from_file_location(
        "locked_depthtrack_pr", plan["metric_source"])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    values = metric.evaluate_depthtrack_results(
        plan["dataset_root"], output, resolution=100,
        sequence_names=[case["sequence"] for case in cases])
    assert values["sequences"] == len(cases)
    assert values["frames"] == receipt["frames"]
    write(output / "metrics.json", dict(
        status="complete", metrics=values, dataset=plan["dataset"],
        final_sha256=plan["final_sha256"],
        training_result_sha256=plan["training_result_sha256"],
        plan_sha256=sha(plan_path), receipt_sha256=sha(receipt_path),
        metric_source_sha256=plan["metric_source_sha256"]))
    print(json.dumps(values, indent=2), flush=True)


def vot_track(plan_path):
    plan, _ = checked_plan(plan_path)
    assert plan["dataset"] == "vot"
    torch, tracker, bank, indices, initial, state_digest = load_tracker(plan)
    spec = importlib.util.spec_from_file_location("locked_vot_bridge", plan["vot_bridge_path"])
    vot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vot)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    handle = vot.VOT("rectangle", channels="rgbd")
    bbox = list(handle.region())
    files = handle.frame()
    assert isinstance(files, list) and len(files) == 2
    key = "vot:" + initialization_key(sha(files[0]), bbox)
    assert key in indices
    image = get_rgbd_frame(files[0], files[1], dtype="rgbcolormap", depth_clip=True)
    tracker.initialize(image, bbox, indices[key])
    while True:
        files = handle.frame()
        if files is None:
            break
        assert isinstance(files, list) and len(files) == 2
        image = get_rgbd_frame(files[0], files[1], dtype="rgbcolormap", depth_clip=True)
        with torch.no_grad():
            prediction = tracker.track(image)
        handle.report(vot.Rectangle(*prediction["target_bbox"]), prediction["best_score"])
    assert tracker.actor.frozen_digest() == initial["native_clip"]
    assert state_digest(tracker.actor.decoder) == initial["A_B"]
    assert state_digest(tracker.writer) == initial["C"]


def vot_protocol(plan_path):
    plan, _ = checked_plan(plan_path)
    assert plan["dataset"] == "vot"
    import vot
    from vot.workspace import Workspace
    root = Path(plan_path).resolve().parent
    reference = read(plan["vot_reference_path"])
    assert vot.__version__ == reference["toolkit"] == "0.7.1"
    manifest = read(root / "run/shard_manifest.json")
    for shard in manifest["shards"]:
        workspace = Workspace.load(shard["root"])
        experiments = list(workspace.stack)
        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment.identifier == "baseline"
        assert type(experiment).__module__ + "." + type(experiment).__name__ == (
            "vot.experiment.multistart.MultiStartExperiment")
        assert experiment.dump() == reference["results"]["baseline"]["parameters"]
    print(json.dumps(dict(status="VOT_protocol_reference_exact",
        toolkit=vot.__version__, shards=len(manifest["shards"]),
        reference_sha256=sha(plan["vot_reference_path"]))), flush=True)


def vot_seal(plan_path):
    plan, _ = checked_plan(plan_path)
    assert plan["dataset"] == "vot"
    vot_root = Path(plan_path).resolve().parent
    run_root = vot_root / "run"
    manifest = read(run_root / "shard_manifest.json")
    merge = read(run_root / "merge_result.json")
    assert merge["status"] == "complete_CADTrack_port_full127_merge"
    assert merge["source_manifest_sha256"] == sha(run_root / "shard_manifest.json")
    assert merge["anchor_count"] == 1765 and merge["result_file_count"] == 5295
    assert len(merge["result_sha256"]) == 5295
    for name, digest in merge["result_sha256"].items():
        assert sha(run_root / "master" / name) == digest, name
    assert [row["completed_anchors"] for row in merge["launches"]] == [441, 441, 441, 442]
    assert all(row["exit_code"] == 0 for row in merge["launches"])
    analysis_path = run_root / "master/analysis" / (manifest["tracker"] + "_full127.json")
    analysis = read(analysis_path)
    assert set(analysis["trackers"]) == {manifest["tracker"]}
    assert set(analysis["sequences"]) == set(manifest["sequences"])
    baseline = analysis["results"]["baseline"]
    assert [row["type"] for row in baseline["parameters"]["analyses"]] == [
        "vot.analysis.multistart.EAOScore",
        "vot.analysis.multistart.EAOCurve",
        "vot.analysis.multistart.AverageAccuracyRobustness"]
    values = baseline["results"]
    assert len(values) == 3 and len(values[0]) == len(values[2]) == 1
    metrics = dict(EAO=float(values[0][0][0]) * 100,
        ACC=float(values[2][0][0]) * 100, ROB=float(values[2][0][1]) * 100)
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in metrics.values())
    sys.path.insert(0, "/home/SUTrack_RGBD_L")
    spec = importlib.util.spec_from_file_location(
        "locked_failure_audit", plan["failure_source_path"])
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    from run_evaluation import checked_trajectory_coverage
    failures, per_sequence, outcomes, settings = checked_trajectory_coverage(
        run_root, manifest["tracker"], manifest["sequences"],
        audit.collect_confirmed_failure_outcomes)
    result = dict(status="complete_full127", model=manifest["tracker"],
        final_sha256=plan["final_sha256"], bundle_sha256=plan["bundle_sha256"],
        training_result_sha256=plan["training_result_sha256"],
        bank_sha256=plan["bank_sha256"], binding_sha256=plan["binding_sha256"],
        metrics_percent=metrics, anchors=1765, sequences=127,
        merge_sha256=sha(run_root / "merge_result.json"),
        analysis_sha256=sha(analysis_path), confirmed_failures=failures,
        per_sequence_failures=per_sequence, failure_outcomes=outcomes,
        failure_settings=settings, external_optimizer_steps=0,
        independent_completed_audit=False)
    write(vot_root / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ope_track", "ope_analyze", "vot_track", "vot_protocol", "vot_seal"),
                        required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    {"ope_track": ope_track, "ope_analyze": ope_analyze,
     "vot_track": vot_track, "vot_protocol": vot_protocol,
     "vot_seal": vot_seal}[args.mode](args.plan)


if __name__ == "__main__":
    main()
