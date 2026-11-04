"""Fixed-checkpoint OPE and VOT entrypoints for pure fine-tuned CADTrack."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def checked_plan(path, verify_artifacts=True):
    plan = read(path)
    assert plan["schema"] == "CADTrack_pure_finetune_evaluation_v1"
    if verify_artifacts:
        assert sha(plan["checkpoint_path"]) == plan["checkpoint_sha256"]
        assert sha(plan["selection_path"]) == plan["selection_file_sha256"]
        assert sha(plan["evaluation_source_path"]) == plan["evaluation_source_sha256"]
        assert sha(plan["source_manifest_path"]) == plan["source_manifest_sha256"]
        assert sha(plan["evaluator_path"]) == plan["evaluator_sha256"]
        source_manifest = read(plan["source_manifest_path"])
        assert source_manifest["evaluator_path"] == plan["evaluator_path"]
        assert source_manifest["evaluator_sha256"] == plan["evaluator_sha256"]
        for source, digest in source_manifest["source_sha256"].items():
            assert sha(source) == digest, source
    if plan["dataset"] == "vot":
        if verify_artifacts:
            assert sha(plan["vot_identity_tool_path"]) == plan["vot_identity_tool_sha256"]
            assert sha(plan["vot_coverage_tool_path"]) == plan["vot_coverage_tool_sha256"]
            assert sha(plan["vot_reference_path"]) == plan["vot_reference_sha256"]
        execution = read(plan["vot_execution_path"])
        if verify_artifacts:
            assert sha(plan["vot_execution_path"]) == plan["vot_execution_sha256"]
            for source, digest in execution["source_sha256"].items():
                assert sha(source) == digest, source
    profile = plan["inference_profile"]
    if verify_artifacts:
        assert sha(profile["config_path"]) == profile["config_sha256"]
    assert plan["selection"] == "test-set-selected"
    assert plan["selection_metric"] == "DepthTrack Test50 F-score"
    assert len(plan["retained_checkpoints"]) == 5
    assert [row["epoch"] for row in plan["retained_checkpoints"]] == [
        31, 32, 33, 34, 35
    ]
    for row in plan["retained_checkpoints"]:
        if verify_artifacts:
            assert sha(row["path"]) == row["sha256"]
    if verify_artifacts:
        assert sha(plan["bundle_path"]) == plan["bundle_sha256"]
    bundle = read(plan["bundle_path"])
    assert bundle["schema"] == "CADTrack_pure_finetune_bundle_v1"
    assert bundle["checkpoint_sha256"] == plan["checkpoint_sha256"]
    assert bundle["selected_epoch"] == plan["selected_epoch"]
    assert bundle["inference_profile"] == plan["inference_profile"]
    assert bundle["selection_file_sha256"] == plan["selection_file_sha256"]
    assert bundle["evaluation_source_sha256"] == plan["evaluation_source_sha256"]
    assert bundle["source_manifest_sha256"] == plan["source_manifest_sha256"]
    assert bundle["evaluator_sha256"] == plan["evaluator_sha256"]
    if plan["dataset"] in ("depthtrack_test", "cdtb"):
        dataset = bundle["datasets"][plan["dataset"]]
        assert dataset["cases_sha256"] == plan["cases_sha256"]
        assert dataset["metric_source_sha256"] == plan["metric_source_sha256"]
    else:
        assert bundle["vot_execution_sha256"] == plan["vot_execution_sha256"]
        assert bundle["vot_reference_sha256"] == plan["vot_reference_sha256"]
        assert bundle["vot_identity_tool_sha256"] == plan["vot_identity_tool_sha256"]
        assert bundle["vot_coverage_tool_sha256"] == plan["vot_coverage_tool_sha256"]
    for path_key, sha_key in (
        ("cases_path", "cases_sha256"),
        ("metric_source", "metric_source_sha256"),
    ):
        if path_key in plan:
            if verify_artifacts:
                assert sha(plan[path_key]) == plan[sha_key]
    return plan


def gate_outcomes(metrics):
    targets = {
        "depthtrack_test": {"P": 65.2, "R": 64.9, "F": 65.1},
        "cdtb": {"P": 72.9, "R": 75.6, "F": 74.2},
        "vot": {"EAO": 77.9, "ACC": 82.1, "ROB": 93.7},
    }
    outcomes = {}
    for dataset, thresholds in targets.items():
        outcomes[dataset] = {}
        for name, threshold in thresholds.items():
            value = float(metrics[dataset][name])
            assert math.isfinite(value) and 0 <= value <= 100
            passed = value > threshold if dataset == "vot" else value >= threshold
            outcomes[dataset][name] = {
                "value": value,
                "threshold": threshold,
                "passed": passed,
            }
    return outcomes


def runtime_environment(plan, gpu=None):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(plan["evaluation_source_path"]).parent),
                str(plan["probe_root"]),
                str(plan["source_root"]),
            ]
        ),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONOPTIMIZE="0",
    )
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    return env


def stage(root, name, command, cwd, env, gpu=None):
    root = Path(root)
    launch_path = root / (name + ".launch.json")
    log_path = root / (name + ".log")
    assert not launch_path.exists() and not log_path.exists()
    launch = dict(
        stage=name,
        command=list(map(str, command)),
        cwd=str(cwd),
        gpu=gpu,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    write(launch_path, launch)
    with log_path.open("xb") as log:
        result = subprocess.run(
            list(map(str, command)),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    launch.update(
        exit_code=result.returncode,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    (root / (name + ".exit")).write_text(str(result.returncode) + "\n")
    write(root / (name + ".receipt.json"), launch)
    assert result.returncode == 0, "Original stage failed; no automatic retry."
    return launch


def vot_track(plan_path):
    plan = checked_plan(plan_path, verify_artifacts=False)
    assert plan["dataset"] == "vot"
    source_manifest = read(plan["source_manifest_path"])
    assert sha(plan["source_manifest_path"]) == plan["source_manifest_sha256"]
    assert sha(plan["vot_execution_path"]) == plan["vot_execution_sha256"]
    sys.path[:0] = [str(plan["probe_root"]), str(plan["source_root"])]
    import torch
    import vot
    import evaluate_ope

    assert sha(plan["evaluator_path"]) == source_manifest["evaluator_sha256"]
    assert Path(plan["evaluator_path"]).read_bytes() == Path(evaluate_ope.__file__).read_bytes()
    from evaluate_ope import load_rgbd, tracker_for

    torch.set_num_threads(1)
    tracker = tracker_for(Path(plan["source_root"]), Path(plan["checkpoint_path"]))
    handle = vot.VOT("rectangle", channels="rgbd")
    bbox = list(handle.region())
    files = handle.frame()
    assert isinstance(files, list) and len(files) == 2
    image = load_rgbd(files[0], files[1])
    tracker.initialize(image, {"init_bbox": bbox})
    while True:
        files = handle.frame()
        if files is None:
            break
        assert isinstance(files, list) and len(files) == 2
        image = load_rgbd(files[0], files[1])
        with torch.no_grad():
            prediction = tracker.track(image)
        tracker_box = np.asarray(
            prediction["target_bbox"], dtype=np.float64
        ).reshape(4)
        confidence = prediction["best_score"]
        confidence = 0.0 if confidence is None else float(confidence)
        assert np.isfinite(tracker_box).all() and np.isfinite(confidence)
        handle.report(vot.Rectangle(*tracker_box.tolist()), confidence)


def vot_protocol(plan_path):
    plan = checked_plan(plan_path)
    assert plan["dataset"] == "vot"
    import vot
    from vot.workspace import Workspace

    reference = read(plan["vot_reference_path"])
    assert sha(plan["vot_reference_path"]) == plan["vot_reference_sha256"]
    assert vot.__version__ == reference["toolkit"] == "0.7.1"
    expected = reference["results"]["baseline"]
    workspaces = []
    for path in plan["protocol_workspaces"]:
        workspace = Workspace.load(path)
        experiments = list(workspace.stack)
        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment.identifier == "baseline"
        assert (
            type(experiment).__module__ + "." + type(experiment).__name__
            == expected["type"]
        )
        assert experiment.dump() == expected["parameters"]
        workspaces.append(
            dict(
                workspace=str(path),
                config_sha256=sha(Path(path) / "config.yaml"),
                tracker_sha256=sha(Path(path) / "trackers.ini"),
            )
        )
    write(
        plan["protocol_receipt"],
        dict(
            status="VOT_protocol_reference_exact",
            toolkit=vot.__version__,
            reference_sha256=sha(plan["vot_reference_path"]),
            protocol=expected["parameters"],
            workspaces=workspaces,
        ),
    )


def vot_seal(plan_path):
    plan = checked_plan(plan_path)
    assert plan["dataset"] == "vot"
    root = Path(plan_path).resolve().parent
    run_root = root / "run"
    manifest = read(run_root / "shard_manifest.json")
    merge = read(run_root / "merge_result.json")
    assert merge["status"] == "complete_CADTrack_pure_finetune_full127_merge"
    assert merge["source_manifest_sha256"] == sha(run_root / "shard_manifest.json")
    assert merge["anchor_count"] == 1765 and merge["result_file_count"] == 5295
    assert len(merge["result_sha256"]) == 5295
    for name, digest in merge["result_sha256"].items():
        assert sha(run_root / "master" / name) == digest
    assert [row["completed_anchors"] for row in merge["launches"]] == [
        441, 441, 441, 442
    ]
    assert all(row["exit_code"] == 0 for row in merge["launches"])

    analysis_path = (
        run_root / "master/analysis" / (manifest["tracker"] + "_full127.json")
    )
    analysis = read(analysis_path)
    reference = read(plan["vot_reference_path"])
    assert analysis["toolkit"] == reference["toolkit"] == "0.7.1"
    assert set(analysis["trackers"]) == {manifest["tracker"]}
    assert analysis["sequences"] == reference["sequences"]
    baseline = analysis["results"]["baseline"]
    expected = reference["results"]["baseline"]
    assert baseline["type"] == expected["type"]
    assert baseline["parameters"] == expected["parameters"]
    descriptors = baseline["parameters"]["analyses"]
    assert [row["type"] for row in descriptors] == [
        "vot.analysis.multistart.EAOScore",
        "vot.analysis.multistart.EAOCurve",
        "vot.analysis.multistart.AverageAccuracyRobustness",
    ]
    values = baseline["results"]
    assert len(values) == 3 and len(values[0]) == len(values[2]) == 1
    metrics = dict(
        EAO=float(values[0][0][0]) * 100,
        ACC=float(values[2][0][0]) * 100,
        ROB=float(values[2][0][1]) * 100,
    )
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in metrics.values())
    write(
        root / "result.json",
        dict(
            status="complete_full127",
            tracker=manifest["tracker"],
            checkpoint_sha256=plan["checkpoint_sha256"],
            selected_epoch=plan["selected_epoch"],
            metrics_percent=metrics,
            anchors=1765,
            sequences=127,
            analysis_sha256=sha(analysis_path),
            merge_sha256=sha(run_root / "merge_result.json"),
            independent_completed_audit=False,
        ),
    )
    return metrics


def complete_anchors(shard, tracker):
    count = 0
    for name in shard["expected_trajectories"]:
        folder_name = name.rsplit("_", 1)[0]
        folder = Path(shard["root"]) / "results" / tracker / "baseline" / folder_name
        files = [
            folder / (name + suffix)
            for suffix in (".bin", "_confidence.value", "_time.value")
        ]
        count += int(all(path.is_file() and path.stat().st_size > 0 for path in files))
    return count


def run_vot(root, plan, metric_python, execution_sha256):
    root = Path(root)
    vot_root = root / "vot"
    run_root = vot_root / "run"
    assert sha(vot_root / "execution.json") == execution_sha256
    manifest = read(run_root / "shard_manifest.json")
    execution = read(vot_root / "execution.json")
    for source, digest in execution["source_sha256"].items():
        assert sha(source) == digest, source
    tracker = manifest["tracker"]
    logs_root = run_root / "controller_logs"
    logs_root.mkdir(exist_ok=False)
    launches = []
    for offset in (0, 2):
        active = []
        for shard in manifest["shards"][offset : offset + 2]:
            folder = Path(shard["root"])
            assert not (folder / "results").exists()
            command = [str(metric_python), "-m", "vot", "evaluate", "--workspace", ".", tracker]
            env = runtime_environment(plan, gpu=shard["gpu"])
            log_path = logs_root / ("shard-%02d.log" % shard["index"])
            with log_path.open("xb") as log:
                child = subprocess.Popen(
                    command, cwd=str(folder), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            launch = dict(
                index=shard["index"],
                gpu=shard["gpu"],
                pid=child.pid,
                command=command,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            launches.append(launch)
            active.append((child, shard, launch))
        write(run_root / "launch.json", launches)
        for child, shard, launch in active:
            code = child.wait()
            launch.update(
                exit_code=code,
                exit=code,
                completed_anchors=complete_anchors(shard, tracker),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            (run_root / ("shard-%02d.exit" % shard["index"])).write_text(
                str(code) + "\n"
            )
        write(run_root / "launch.json", launches)
        assert all(
            row["exit_code"] == 0
            and row["completed_anchors"]
            == manifest["shards"][row["index"]]["anchor_count"]
            for row in launches[offset : offset + 2]
        ), "VOT shard failed; no automatic retry."

    master = run_root / "master"
    master.mkdir(exist_ok=False)
    (master / "sequences").mkdir()
    (master / "config.yaml").write_text(
        "registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n",
        encoding="utf-8",
    )
    shutil.copy2(
        Path(manifest["shards"][0]["root"]) / "trackers.ini",
        master / "trackers.ini",
    )
    (master / "sequences" / "list.txt").write_text(
        "".join(name + "\n" for name in manifest["sequences"]),
        encoding="utf-8",
    )
    for name in manifest["sequences"]:
        (master / "sequences" / name).symlink_to(
            Path(manifest["source"][name]["root"]).resolve(),
            target_is_directory=True,
        )

    copied = {}
    for shard in manifest["shards"]:
        for name in shard["expected_trajectories"]:
            source_folder = Path(shard["root"]) / "results" / tracker / "baseline" / name.rsplit("_", 1)[0]
            destination = master / "results" / tracker / "baseline" / name.rsplit("_", 1)[0]
            destination.mkdir(parents=True, exist_ok=True)
            for suffix in (".bin", "_confidence.value", "_time.value"):
                source = source_folder / (name + suffix)
                target = destination / source.name
                assert not target.exists()
                shutil.copy2(source, target)
                assert sha(source) == sha(target)
                copied[str(target.relative_to(master))] = sha(target)
    assert len(copied) == 5295
    for relative, digest in copied.items():
        assert sha(master / relative) == digest
    write(
        run_root / "merge_result.json",
        dict(
            status="complete_CADTrack_pure_finetune_full127_merge",
            tracker=tracker,
            source_manifest_sha256=sha(run_root / "shard_manifest.json"),
            anchor_count=1765,
            result_file_count=5295,
            master_workspace=str(master),
            result_sha256=dict(sorted(copied.items())),
            launches=launches,
        ),
    )

    analysis_name = tracker + "_full127"
    analysis_path = master / "analysis" / (analysis_name + ".json")
    stage(
        vot_root,
        "official_analysis",
        [
            metric_python, "-m", "vot", "analysis", "--workspace", str(master),
            "--format", "json", "--name", analysis_name, tracker,
        ],
        root,
        runtime_environment(plan),
    )
    assert analysis_path.is_file()
    analysis = read(analysis_path)
    assert set(analysis["trackers"]) == {tracker}
    assert set(analysis["sequences"]) == set(manifest["sequences"])
    baseline = analysis["results"]["baseline"]
    types = [row["type"] for row in baseline["parameters"]["analyses"]]
    assert types == [
        "vot.analysis.multistart.EAOScore",
        "vot.analysis.multistart.EAOCurve",
        "vot.analysis.multistart.AverageAccuracyRobustness",
    ]
    values = baseline["results"]
    metrics = dict(
        EAO=float(values[0][0][0]) * 100,
        ACC=float(values[2][0][0]) * 100,
        ROB=float(values[2][0][1]) * 100,
    )
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in metrics.values())
    write(
        vot_root / "result.json",
        dict(
            status="complete_CADTrack_pure_finetune_full127",
            tracker=tracker,
            checkpoint_sha256=plan["checkpoint_sha256"],
            selected_epoch=plan["selected_epoch"],
            metrics_percent=metrics,
            anchors=1765,
            sequences=127,
            analysis_sha256=sha(analysis_path),
            merge_sha256=sha(run_root / "merge_result.json"),
            independent_completed_audit=False,
        ),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--tracker-python", type=Path, required=True)
    parser.add_argument("--metric-python", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    selection = read(root / "selection.json")
    assert selection["status"] == "CADTrack_pure_finetune_final_fixed"
    assert selection["optimizer_steps_after_training"] == 0
    assert selection["selection"] == "test-set-selected"
    assert sha(Path(__file__).resolve()) == selection["runner_sha256"]
    plans = {
        name: root / name / "plan.json"
        for name in ("depthtrack_test", "cdtb", "vot")
    }
    for name in ("depthtrack_test", "cdtb"):
        assert plans[name] == Path(selection["plans"][name]["path"])
        assert sha(plans[name]) == selection["plans"][name]["sha256"]
    assert plans["vot"] == Path(selection["vot_plan"])
    assert sha(plans["vot"]) == selection["vot_plan_sha256"]
    plan_data = {name: checked_plan(path) for name, path in plans.items()}
    assert {plan["checkpoint_sha256"] for plan in plan_data.values()} == {
        selection["checkpoint_sha256"]
    }
    assert len({plan["bundle_sha256"] for plan in plan_data.values()}) == 1
    stages = root / "stages"
    stages.mkdir(exist_ok=False)
    environment_plan = plan_data["depthtrack_test"]
    for dataset in ("depthtrack_test", "cdtb"):
        plan = plan_data[dataset]
        output = Path(plan["output"])
        track_command = [
            str(args.tracker_python.resolve()), "-B",
            str(plan["evaluator_path"]), "track",
            "--dataset", dataset,
            "--source", str(plan["source_root"]),
            "--checkpoint", str(plan["checkpoint_path"]),
            "--dataset-root", str(plan["dataset_root"]),
            "--cases", str(plan["cases_path"]),
            "--output", str(output),
            "--metric-source", str(plan["metric_source"]),
        ]
        stage(
            stages, dataset + "_track", track_command, root,
            runtime_environment(plan, gpu=0 if dataset == "depthtrack_test" else 1),
            gpu=0 if dataset == "depthtrack_test" else 1,
        )
        score_command = [
            str(args.metric_python.resolve()), "-B",
            str(plan["evaluator_path"]), "score",
            "--dataset", dataset,
            "--source", str(plan["source_root"]),
            "--checkpoint", str(plan["checkpoint_path"]),
            "--dataset-root", str(plan["dataset_root"]),
            "--cases", str(plan["cases_path"]),
            "--output", str(output),
            "--metric-source", str(plan["metric_source"]),
        ]
        stage(
            stages, dataset + "_score", score_command, root,
            runtime_environment(plan),
        )

    vot_plan = plan_data["vot"]
    protocol_receipt = Path(vot_plan["protocol_receipt"])
    stage(
        stages,
        "vot_protocol",
        [
            str(args.metric_python.resolve()), "-B",
            str(vot_plan["evaluation_source_path"]),
            "--mode", "vot_protocol", "--plan", str(plans["vot"]),
        ],
        root,
        runtime_environment(vot_plan),
    )
    protocol = read(protocol_receipt)
    assert protocol["status"] == "VOT_protocol_reference_exact"
    assert protocol["reference_sha256"] == vot_plan["vot_reference_sha256"]
    assert len(protocol["workspaces"]) == 4
    assert [row["config_sha256"] for row in protocol["workspaces"]] == [
        sha(Path(path) / "config.yaml") for path in vot_plan["protocol_workspaces"]
    ]
    assert [row["tracker_sha256"] for row in protocol["workspaces"]] == [
        sha(Path(path) / "trackers.ini") for path in vot_plan["protocol_workspaces"]
    ]
    run_vot(
        root,
        vot_plan,
        args.metric_python.resolve(),
        selection["vot_execution_sha256"],
    )
    stage(
        stages,
        "vot_seal",
        [
            str(args.metric_python.resolve()), "-B",
            str(vot_plan["evaluation_source_path"]),
            "--mode", "vot_seal", "--plan", str(plans["vot"]),
        ],
        root,
        runtime_environment(vot_plan),
    )
    vot_result = read(root / "vot/result.json")
    vot_metrics = vot_result["metrics_percent"]

    identity_command = [
        str(args.metric_python.resolve()), "-B", vot_plan["vot_identity_tool_path"],
        "--analysis", root / "vot/run/master/analysis" / (vot_plan["tracker"] + "_full127.json"),
        "--reference", vot_plan["vot_reference_path"],
        "--manifest", root / "vot/run/shard_manifest.json",
        "--sealed-result", root / "vot/result.json",
    ]
    stage(
        stages, "vot_identity_audit", identity_command, root,
        runtime_environment(vot_plan),
    )
    coverage_report = root / "vot" / "coverage.json"
    coverage_command = [
        str(args.metric_python.resolve()), "-B", vot_plan["vot_coverage_tool_path"],
        "--workspace", root / "vot/run/master",
        "--manifest", root / "vot/run/shard_manifest.json",
        "--merge", root / "vot/run/merge_result.json",
        "--execution", root / "vot/execution.json",
        "--execution-sha256", selection["vot_execution_sha256"],
        "--analysis",
        root / "vot/run/master/analysis" / (vot_plan["tracker"] + "_full127.json"),
        "--output", coverage_report,
    ]
    stage(
        stages, "vot_trajectory_coverage", coverage_command, root,
        runtime_environment(vot_plan),
    )
    coverage = read(coverage_report)
    assert coverage["trajectory_coverage_checked"] is True
    assert coverage["checkpoint_source_binding_checked"] is False
    assert coverage["independent_completed_audit"] is False
    metrics = {
        dataset: read(Path(plan_data[dataset]["output"]) / "metrics.json")["metrics"]
        for dataset in ("depthtrack_test", "cdtb")
    }
    metrics["vot"] = vot_metrics
    gates = gate_outcomes(metrics)
    all_pass = all(
        item["passed"]
        for dataset in gates.values()
        for item in dataset.values()
    )
    result = dict(
        status="complete_CADTrack_pure_finetune_three_datasets",
        datasets=metrics,
        gates=gates,
        any_joint_pass=all_pass,
        project_status="ACTIVE_MET" if all_pass else "ACTIVE_UNMET",
        checkpoint_sha256=selection["checkpoint_sha256"],
        selected_epoch=selection["selected_epoch"],
        selection="test-set-selected",
        selection_bias_disclosed=True,
        runner_sha256=sha(Path(__file__).resolve()),
        protocol_receipt_sha256=sha(protocol_receipt),
        vot_identity_report_sha256=sha(stages / "vot_identity_audit.receipt.json"),
        trajectory_coverage_sha256=sha(coverage_report),
        checkpoint_source_binding_checked=False,
        independent_completed_audit=False,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    write(root / "all_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("vot_track", "vot_protocol", "vot_seal"), required=True
    )
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    {
        "vot_track": vot_track,
        "vot_protocol": vot_protocol,
        "vot_seal": vot_seal,
    }[args.mode](args.plan)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mode":
        cli()
    else:
        main()
