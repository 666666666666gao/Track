"""Run the frozen CADTrack port final on full OPE and VOT-RGBD127."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluation import checked_plan, read, sha, write


def environment(plan, gpu=None, metric=False):
    env = dict(os.environ,
        PYTHONPATH=os.pathsep.join([
            str(Path(plan["evaluation_source_path"]).parent), plan["code_root"],
            plan["legacy_code_root"], plan["repository"], "/home/SUTrack_RGBD_L"]),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1")
    if gpu is None or metric:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def stage(root, name, command, cwd, env, gpu=None):
    launch = dict(stage=name, command=list(map(str, command)), cwd=str(cwd),
        gpu=gpu, started_at=datetime.now(timezone.utc).isoformat())
    launch_path = root / (name + ".launch.json")
    log_path = root / (name + ".log")
    assert not launch_path.exists() and not log_path.exists()
    write(launch_path, launch)
    with log_path.open("xb") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    launch.update(exit_code=result.returncode,
                  finished_at=datetime.now(timezone.utc).isoformat())
    (root / (name + ".exit")).write_text(str(result.returncode) + "\n")
    write(root / (name + ".receipt.json"), launch)
    assert result.returncode == 0, "Original stage failed; no automatic retry."
    return launch


def checked_trajectory_coverage(root, tracker, sequences, collect_outcomes):
    analysis = read(root / "master/analysis" / (tracker + "_full127.json"))
    assert set(analysis["sequences"]) == set(sequences)
    outcomes, failures, per_sequence, settings = collect_outcomes(
        root / "master", tracker, expected_anchors=1765)
    assert len(outcomes) == 1765 and len(per_sequence) == 127
    for sequence, row in per_sequence.items():
        assert row["anchors"] > 0
        assert sequence in analysis["sequences"]
    return failures, per_sequence, outcomes, settings


def tracker_files(root, tracker, name):
    folder = Path(root) / "results" / tracker / "baseline" / name.rsplit("_", 1)[0]
    return [folder / (name + suffix)
            for suffix in (".bin", "_confidence.value", "_time.value")]


def complete_anchors(shard, tracker):
    return sum(
        all(path.is_file() and path.stat().st_size > 0
            for path in tracker_files(shard["root"], tracker, name))
        for name in shard["expected_trajectories"])


def gate_outcomes(metrics):
    gates = {
        "depthtrack_test": dict(P=65.2, R=64.9, F=65.1),
        "cdtb": dict(P=72.9, R=75.6, F=74.2),
        "vot": dict(EAO=77.9, ACC=82.1, ROB=93.7)}
    outcomes = {}
    for dataset, targets in gates.items():
        observed = metrics[dataset]
        assert all(math.isfinite(float(observed[key])) and 0 <= observed[key] <= 100
                   for key in targets)
        outcomes[dataset] = {
            key: dict(value=float(observed[key]), threshold=threshold,
                      passed=float(observed[key]) > threshold
                      if dataset == "vot" else float(observed[key]) >= threshold)
            for key, threshold in targets.items()}
    return outcomes


def run_vot(evaluation_root, plan, metric_python, execution_sha256):
    vot_root = evaluation_root / "vot"
    run_root = vot_root / "run"
    assert sha(vot_root / "execution.json") == execution_sha256
    manifest = read(run_root / "shard_manifest.json")
    execution = read(vot_root / "execution.json")
    assert execution["status"] == "frozen_before_CADTrack_port_VOT_tracking"
    for path, digest in execution["source_sha256"].items():
        assert sha(path) == digest, path
    assert manifest["tracker"] == "cadtrack_abc_epoch20_innovation_fixed6"
    logs_root = run_root / "controller_logs"
    logs_root.mkdir(exist_ok=False)
    launches = []
    for offset in (0, 2):
        children = []
        for shard in manifest["shards"][offset:offset + 2]:
            folder = Path(shard["root"])
            assert not (folder / "results").exists()
            command = [metric_python, "-m", "vot", "evaluate", "--workspace", ".",
                       manifest["tracker"]]
            env = environment(plan, gpu=shard["gpu"])
            log_path = logs_root / ("shard-%02d.log" % shard["index"])
            with log_path.open("xb") as log:
                child = subprocess.Popen(command, cwd=folder, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            launch = dict(index=shard["index"], gpu=shard["gpu"], pid=child.pid,
                          started_at=datetime.now(timezone.utc).isoformat(),
                          command=command)
            children.append((child, shard, launch))
            launches.append(launch)
        write(run_root / "launch.json", launches)
        for child, shard, launch in children:
            code = child.wait()
            launch.update(exit_code=code,
                completed_anchors=complete_anchors(shard, manifest["tracker"]),
                finished_at=datetime.now(timezone.utc).isoformat())
            (run_root / ("shard-%02d.exit" % shard["index"])).write_text(str(code) + "\n")
        write(run_root / "launch.json", launches)
        assert all(row["exit_code"] == 0
            and row["completed_anchors"] == manifest["shards"][row["index"]]["anchor_count"]
            for row in launches[offset:offset + 2]), "VOT shard failed; no automatic retry."

    master = run_root / "master"
    master.mkdir(exist_ok=False)
    sequences = master / "sequences"
    sequences.mkdir()
    (master / "config.yaml").write_text(
        "registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n")
    shutil.copy2(Path(manifest["shards"][0]["root"]) / "trackers.ini",
                 master / "trackers.ini")
    (sequences / "list.txt").write_text(
        "".join(name + "\n" for name in manifest["sequences"]))
    for name in manifest["sequences"]:
        (sequences / name).symlink_to(
            Path(manifest["source"][name]["root"]).resolve(),
            target_is_directory=True)
    copied = {}
    for shard in manifest["shards"]:
        for name in shard["expected_trajectories"]:
            destination = master / "results" / manifest["tracker"] / "baseline" / name.rsplit("_", 1)[0]
            destination.mkdir(parents=True, exist_ok=True)
            for source in tracker_files(shard["root"], manifest["tracker"], name):
                target = destination / source.name
                assert not target.exists()
                shutil.copy2(source, target)
                assert sha(source) == sha(target)
                copied[str(target.relative_to(master))] = sha(target)
    assert sum(row["anchor_count"] for row in manifest["shards"]) == 1765
    assert len(copied) == 5295
    for relative_path, digest in copied.items():
        assert sha(master / relative_path) == digest
    merge = dict(status="complete_CADTrack_port_full127_merge",
        tracker=manifest["tracker"],
        source_manifest_sha256=sha(run_root / "shard_manifest.json"),
        anchor_count=1765, result_file_count=5295,
        master_workspace=str(master), result_sha256=dict(sorted(copied.items())),
        launches=launches)
    write(run_root / "merge_result.json", merge)
    analysis_name = manifest["tracker"] + "_full127"
    analysis_command = [metric_python, "-m", "vot", "analysis", "--workspace",
        str(master), "--format", "json", "--name", analysis_name, manifest["tracker"]]
    stage(vot_root, "official_analysis", analysis_command, evaluation_root,
          environment(plan, metric=True))
    seal_command = [metric_python, "-B", "-u", plan["evaluation_source_path"],
                    "--mode", "vot_seal", "--plan", vot_root / "plan.json"]
    stage(vot_root, "seal", seal_command, evaluation_root,
          environment(plan, metric=True))
    return read(vot_root / "result.json")["metrics_percent"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--tracker-python", type=Path, required=True)
    parser.add_argument("--metric-python", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    selection = read(root / "selection.json")
    assert sha(Path(__file__).resolve()) == selection["evaluation_runner_sha256"]
    assert selection["status"] == "CADTrack_port_final_fixed_before_full_metrics"
    assert selection["optimizer_steps_after_training"] == 0
    plans = {name: Path(selection["plans"][name]["path"])
             for name in ("depthtrack_test", "cdtb", "vot")}
    plan_data = {}
    for name, path in plans.items():
        assert sha(path) == selection["plans"][name]["sha256"]
        plan_data[name], _ = checked_plan(path)
        assert plan_data[name]["final_sha256"] == selection["final_sha256"]
        assert plan_data[name]["training_result_sha256"] == selection["training_result_sha256"]
    result_bundle_sha = plan_data["depthtrack_test"]["bundle_sha256"]
    assert all(plan["bundle_sha256"] == result_bundle_sha for plan in plan_data.values())
    vot_execution_sha256 = selection["plans"]["vot"]["execution_sha256"]
    assert sha(root / "vot/execution.json") == vot_execution_sha256
    stage_root = root / "stages"
    stage_root.mkdir(exist_ok=False)
    stage(stage_root, "vot_protocol",
        [args.metric_python.resolve(), "-B", plan_data["vot"]["evaluation_source_path"],
         "--mode", "vot_protocol", "--plan", plans["vot"]],
        root, environment(plan_data["vot"], metric=True))
    track_jobs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for gpu, dataset in enumerate(("depthtrack_test", "cdtb")):
            plan = plan_data[dataset]
            track_jobs.append(pool.submit(stage, stage_root, dataset + "_track",
                [args.tracker_python.resolve(), "-B", "-u",
                 plan["evaluation_source_path"], "--mode", "ope_track",
                 "--plan", plans[dataset]], root,
                environment(plan, gpu=gpu), gpu))
        [job.result() for job in track_jobs]
    metrics = {}
    for dataset in ("depthtrack_test", "cdtb"):
        plan = plan_data[dataset]
        stage(stage_root, dataset + "_analyze",
            [args.metric_python.resolve(), "-B", "-u",
             plan["evaluation_source_path"], "--mode", "ope_analyze",
             "--plan", plans[dataset]], root, environment(plan, metric=True))
        raw = read(Path(plan["output"]) / "metrics.json")["metrics"]
        metrics[dataset] = dict(
            P=raw["precision_percent"], R=raw["recall_percent"],
            F=raw["f_score_percent"])
    vot_metrics = run_vot(root, plan_data["vot"], str(args.metric_python.resolve()),
                          vot_execution_sha256)
    identity_command = [sys.executable, "-B", plan_data["vot"]["vot_identity_tool_path"],
        "--analysis", root / "vot/run/master/analysis/cadtrack_abc_epoch20_innovation_fixed6_full127.json",
        "--reference", plan_data["vot"]["vot_reference_path"],
        "--manifest", root / "vot/run/shard_manifest.json",
        "--sealed-result", root / "vot/result.json"]
    stage(stage_root, "vot_identity_audit", identity_command, root,
          environment(plan_data["vot"], metric=True))
    metrics["vot"] = vot_metrics
    outcomes = gate_outcomes(metrics)
    all_pass = all(row["passed"] for dataset in outcomes.values() for row in dataset.values())
    result = dict(status="complete_CADTrack_port_three_official_datasets",
        datasets=metrics, gates=outcomes, any_joint_pass=all_pass,
        project_status="ACTIVE_MET" if all_pass else "ACTIVE_UNMET",
        final_sha256=selection["final_sha256"],
        training_result_sha256=selection["training_result_sha256"],
        bundle_sha256=result_bundle_sha,
        optimizer_steps_after_training=0, automatic_retries=0,
        checkpoint_or_threshold_selected_by_test=False,
        finished_at=datetime.now(timezone.utc).isoformat())
    write(root / "all_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
