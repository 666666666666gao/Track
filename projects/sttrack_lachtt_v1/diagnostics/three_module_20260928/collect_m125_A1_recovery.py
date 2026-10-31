"""Recover the completed A1 run with CPU scoring and paired failure accounting."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from analyze_m122_rob_contributions import account


TRACKER = "sttrack_m125_anchored_U2"
FINAL_SHA = "cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f"
SOURCES = {
    "native_anchored_selection.py": "ccdd8bc4ce8ecaf483260a20043ad09dbcb05bfed9a1c280d3fb515460a94685",
    "anchored_profile_tracker.py": "a93ec16c9fa99347a07001dc12753f5ea020fc7d347d790fe6676df0ea51265d",
    "run_anchored.py": "e5bc5ed80dfdaaff12609627f40163da6816adaef46b6b09e5f24972398ddac0",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(args):
    root, cad = args.root, args.cad
    assert not args.output.exists()
    bundle_path = cad / "profiles/U2/bundle.json"
    bundle = read(bundle_path)
    assert bundle["final_sha256"] == FINAL_SHA == sha(Path(bundle["final_path"]))
    for name, digest in SOURCES.items():
        assert sha(root / "code" / name) == digest, name
    for name, digest in bundle["source_sha256"].items():
        assert sha(Path(name)) == digest, name

    inputs = {str(bundle_path): sha(bundle_path)}
    for path in sorted((root / "code").glob("*.py")):
        inputs[str(path)] = sha(path)
    metrics, ope_files = {}, {}
    for dataset, count, frames in [("depthtrack_test", 50, 76373), ("cdtb", 80, 101956)]:
        plan_path = cad / "profiles/U2" / dataset / "plan.json"
        plan = read(plan_path)
        assert plan["bundle_sha256"] == sha(bundle_path)
        for field in ("cases", "bank", "binding", "metric_source"):
            path = Path(plan[field] if field == "metric_source" else plan[field + "_path"])
            assert sha(path) == plan[field + "_sha256"], str(path)
            inputs[str(path)] = sha(path)
        cases = read(Path(plan["cases_path"]))
        assert len(cases) == count and sum(case["frames"] for case in cases) == frames
        folder = root / dataset
        expected = set()
        for case in cases:
            name = case["sequence"]
            boxes, scores = folder / (name + ".txt"), folder / (name + "_all_scores.txt")
            expected.update([boxes.name, scores.name])
            bbox = np.loadtxt(str(boxes), delimiter=",")
            confidence = np.loadtxt(str(scores))
            assert bbox.shape == (case["frames"], 4)
            assert confidence.shape == (case["frames"],)
            assert np.isfinite(bbox).all() and np.isfinite(confidence).all()
            assert np.allclose(bbox[0], case["init_bbox"], rtol=0, atol=1e-6)
            gt = Path(plan["dataset_root"]) / name / "groundtruth.txt"
            assert sha(gt) == case["gt_sha256"], str(gt)
            for path in (boxes, scores, gt):
                ope_files[str(path)] = sha(path)
        assert {p.name for p in folder.glob("*.txt")} == expected
        scorer = load_module("A1_recovery_metric_" + dataset, Path(plan["metric_source"]))
        recomputed = scorer.evaluate_depthtrack_results(
            plan["dataset_root"], folder, resolution=100,
            sequence_names=[case["sequence"] for case in cases])
        saved = read(folder / "metrics.json")
        assert saved["profile"] == "U2" and saved["margin"] == .25 and saved["radius"] == 1
        assert recomputed == saved["metrics"], dataset
        metrics[dataset] = {key: recomputed[key + "_percent"]
                            for key in ("precision", "recall", "f_score")}
        inputs[str(plan_path)] = sha(plan_path)
        inputs[str(folder / "metrics.json")] = sha(folder / "metrics.json")
        print("OPE_RECOMPUTED", dataset, metrics[dataset], flush=True)

    manifest_path = cad / "profiles/U2/vot/run/shard_manifest.json"
    manifest = read(manifest_path)
    assert manifest["total_anchor_count"] == 1765 and len(manifest["sequences"]) == 127
    expected_files, trajectories = {}, []
    for shard in manifest["shards"]:
        names = shard["expected_trajectories"]
        assert len(names) == shard["anchor_count"]
        for name in names:
            trajectories.append(name)
            sequence = name.rsplit("_", 1)[0]
            for suffix in (".bin", "_confidence.value", "_time.value"):
                relative = Path("results") / TRACKER / "baseline" / sequence / (name + suffix)
                source = root / "vot" / ("shard-%02d" % shard["index"]) / relative
                merged = root / "vot/master" / relative
                assert source.stat().st_size > 0
                assert sha(source) == sha(merged), str(relative)
                expected_files[str(relative)] = sha(merged)
    assert len(trajectories) == len(set(trajectories)) == 1765
    assert len(expected_files) == 5295
    actual_files = {str(path.relative_to(root / "vot/master"))
                    for path in (root / "vot/master/results" / TRACKER / "baseline").rglob("*")
                    if path.is_file()}
    assert actual_files == set(expected_files)

    plan = read(cad / "profiles/U2/vot/plan.json")
    binding_path = Path(plan["binding_path"])
    assert sha(binding_path) == plan["binding_sha256"]
    binding = read(binding_path)
    keys = {row["key"] for row in binding["rows"] if row["dataset"] == "vot"}
    assert len(keys) == 1765
    events, counts, seen = {}, {}, set()
    for path in sorted((root / "vot/events").glob("*.json")):
        event = read(path)
        key = event["initialization_key"]
        assert key in keys and key not in seen
        assert path.stem == hashlib.sha256(key.encode()).hexdigest()
        seen.add(key)
        events[path.name] = sha(path)
        for name, value in event["counts"].items():
            assert isinstance(value, int) and value >= 0
            counts[name] = counts.get(name, 0) + value
    assert seen == keys

    analysis_path = root / "vot/master/analysis/A1_anchored_U2_full127.json"
    values = read(analysis_path)["results"]["baseline"]["results"]
    metrics["vot"] = dict(EAO=100 * values[0][0][0], ACC=100 * values[2][0][0],
                          ROB=100 * values[2][0][1])
    assert all(math.isfinite(v) and 0 <= v <= 100 for group in metrics.values() for v in group.values())
    failure_source, failure_common = Path(bundle["failure_source_path"]), Path(bundle["failure_common_path"])
    assert sha(failure_source) == "e96a375a3792a80ab12e5671db468f559d514558a05d1181fe7afec48bcb514a"
    assert sha(failure_common) == "cbbe55132a4e64157011c0cbfa9162bd583f09588e3e497dcf741a8d5174a6bb"
    sys.path.insert(0, str(failure_source.parent.parent))
    audit = load_module("A1_recovery_failures", failure_source)
    outcomes, failures, per_sequence, settings = audit.collect_confirmed_failure_outcomes(
        root / "vot/master", TRACKER, expected_anchors=1765)
    assert len(per_sequence) == 127
    candidate = dict(failure_outcomes=outcomes, confirmed_failures=failures,
                     per_sequence_failures=per_sequence, failure_settings=settings,
                     metrics_percent=metrics["vot"])
    native = read(args.native)
    assert sha(args.native) == "51213ef55085d270a56ae2952b8871c06557891d418ae572ec3fb932102f1d8b"
    u2_path = cad / "profiles/U2/vot/result.json"
    u2 = read(u2_path)
    assert u2["final_sha256"] == FINAL_SHA
    comparison = account(native, dict(A1=candidate, U2=u2))
    u2_as_reference = dict(u2, metrics_percent=dict(rob=u2["metrics_percent"]["ROB"]))
    paired = account(u2_as_reference, dict(A1=candidate))
    dt, cd, vot = metrics["depthtrack_test"], metrics["cdtb"], metrics["vot"]
    passed = dict(depthtrack_P=dt["precision"] >= 65.2, depthtrack_R=dt["recall"] >= 64.9,
                  depthtrack_F=dt["f_score"] >= 65.1, cdtb_P=cd["precision"] >= 72.9,
                  cdtb_R=cd["recall"] >= 75.6, cdtb_F=cd["f_score"] >= 74.2,
                  vot_EAO=vot["EAO"] > 77.9, vot_ACC=vot["ACC"] > 82.1, vot_ROB=vot["ROB"] > 93.7)
    for path in (manifest_path, binding_path, analysis_path, args.native, u2_path, failure_source, failure_common):
        inputs[str(path)] = sha(path)
    report = dict(status="completed_A1_recovery_CPU_checks", observed_at_utc=datetime.now(timezone.utc).isoformat(),
                  root=str(root), final_sha256=FINAL_SHA, profile="U2", margin=.25, radius=1,
                  metrics_percent=metrics, passed=passed, joint_pass=all(passed.values()),
                  confirmed_failures=failures, anchors=1765, sequences=127, result_files=5295,
                  template_counts=counts, source_and_input_sha256=inputs,
                  ope_prediction_and_GT_sha256=ope_files, vot_master_files_sha256=expected_files,
                  event_files_sha256=events, failure_outcomes=outcomes, per_sequence_failures=per_sequence,
                  failure_settings=settings, neural_executions=0, optimizer_steps=0,
                  limitations=[
                      "Retrospective checks of present sources, checkpoint and plans; no new historical execution attestation.",
                      "A1 OPE metrics lack original final/bundle hashes; run source and inherited CAD plans supply the retrospective binding.",
                      "EAO and ACC are read from the original official analysis; OPE and failure/ROB accounting are recomputed here.",
                      "No per-frame native/learned selection trace; paired failures do not prove a non-local jump mechanism.",
                      "The U2 base was chosen with test-metric knowledge; this is an engineering comparison."
                  ])
    args.output.mkdir(parents=True)
    write(args.output / "recovery_result.json", report)
    write(args.output / "ROB_vs_native.json", comparison)
    write(args.output / "ROB_vs_U2.json", paired)
    rows = paired["comparisons_vs_native"]["A1"]["anchors"]
    with (args.output / "A1_vs_U2_anchor_ROB.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(metrics=metrics, passed=passed, confirmed_failures=failures,
                          counts=counts, paired_vs_U2={key: value for key, value in paired["comparisons_vs_native"]["A1"].items()
                                                      if key not in ("anchors", "sequences", "worst_ten", "best_ten")})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cad", type=Path, default=Path("/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016"))
    parser.add_argument("--native", type=Path, default=Path("/root/autodl-tmp/sttrack_default_full127_v1_20260905/result.json"))
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
