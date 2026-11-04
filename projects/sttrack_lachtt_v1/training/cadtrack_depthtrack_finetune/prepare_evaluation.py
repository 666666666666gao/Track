"""Bind one selected pure CADTrack fine-tune checkpoint to full RGB-D tests."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

LOCKED_METRIC_SHA256 = "05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc"
VOT_REFERENCE_SHA256 = "73e0b6c15713735240d5d50acd51a95593155f2e3471a9f3d13d38249f5df2e6"
PROBE_ROOT = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe")


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


def verify_final_five(training_root, checkpoint_path, selected_epoch):
    training = Path(training_root)
    checkpoint = Path(checkpoint_path).resolve()
    assert selected_epoch in range(31, 36)
    launch = read(training / "launch.json")
    assert launch["status"] == "complete"
    rows_path = training / "epoch_metrics.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [31, 32, 33, 34, 35]
    best = read(training / "best.json")
    assert best["selection"] == "DepthTrack Test50 F-score, epochs 31-35"
    assert best["epoch"] in range(31, 36)

    retained = []
    metric_rows = []
    for epoch, row in zip(range(31, 36), rows):
        path = training / "final_five" / ("epoch_%02d.pth.tar" % epoch)
        assert path.is_file()
        digest = sha(path)
        assert digest == row["checkpoint_sha256"]
        metric_path = (
            training / "evaluations" / ("epoch_%02d" % epoch) / "metrics.json"
        )
        metrics = read(metric_path)
        assert metrics["checkpoint_sha256"] == digest
        assert metrics["metrics"] == row["metrics"]
        retained.append(dict(epoch=epoch, path=str(path), sha256=digest))
        metric_rows.append(dict(epoch=epoch, metrics=metrics["metrics"]))

    best_row = rows[best["epoch"] - 31]
    best_path = training / "final_five" / ("epoch_%02d.pth.tar" % best["epoch"])
    assert best["checkpoint_sha256"] == best_row["checkpoint_sha256"]
    assert best["f_score"] == best_row["metrics"]["f_score"]
    assert best_path.resolve() == Path(best["checkpoint_path"]).resolve()
    assert best["epoch"] == max(
        metric_rows, key=lambda item: item["metrics"]["f_score"]
    )["epoch"]

    selected_row = rows[selected_epoch - 31]
    selected_path = training / "final_five" / ("epoch_%02d.pth.tar" % selected_epoch)
    assert checkpoint == selected_path.resolve()
    assert sha(checkpoint) == selected_row["checkpoint_sha256"]
    selected = dict(
        epoch=selected_epoch,
        checkpoint_path=str(selected_path),
        checkpoint_sha256=selected_row["checkpoint_sha256"],
        metrics=selected_row["metrics"],
    )
    return selected, retained, metric_rows


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "training-root",
        "base-evaluation",
        "checkpoint",
        "selected-epoch",
        "evaluation-source",
        "evaluation-runner",
        "vot-identity-tool",
        "vot-coverage-tool",
        "vot-reference",
        "output",
    ):
        parser.add_argument(
            "--" + name,
            type=int if name == "selected-epoch" else Path,
            required=True,
        )
    args = parser.parse_args()

    training = args.training_root.resolve()
    base = args.base_evaluation.resolve()
    checkpoint = args.checkpoint.resolve()
    source = args.evaluation_source.resolve()
    runner = args.evaluation_runner.resolve()
    identity_tool = args.vot_identity_tool.resolve()
    coverage_tool = args.vot_coverage_tool.resolve()
    reference = args.vot_reference.resolve()
    output = args.output.resolve()
    assert not output.exists()

    selected, retained, metric_rows = verify_final_five(
        training, checkpoint, args.selected_epoch
    )

    case_plans = {
        name: read(base / name / "plan.json")
        for name in ("depthtrack_test", "cdtb")
    }
    base_bundle_sha256 = sha(base / "bundle.json")
    for name, expected in (("depthtrack_test", (50, 76373)), ("cdtb", (80, 101956))):
        plan = case_plans[name]
        assert plan["bundle_sha256"] == base_bundle_sha256
        assert plan["dataset"] == name
        assert sha(plan["cases_path"]) == plan["cases_sha256"]
        assert plan["metric_source_sha256"] == LOCKED_METRIC_SHA256
        assert sha(plan["metric_source"]) == LOCKED_METRIC_SHA256
        cases = read(plan["cases_path"])
        assert (len(cases), sum(row["frames"] for row in cases)) == expected

    vot_manifest_path = base / "vot" / "run" / "shard_manifest.json"
    manifest = read(vot_manifest_path)
    assert manifest["schema"] == "M122_full127_frozen_shards_v1"
    assert len(manifest["sequences"]) == 127
    assert manifest["total_anchor_count"] == 1765
    assert [row["anchor_count"] for row in manifest["shards"]] == [441, 441, 441, 442]
    assert len(reference.read_bytes()) > 0
    assert sha(reference) == VOT_REFERENCE_SHA256

    probe_root = PROBE_ROOT
    source_root = probe_root / "source"
    config_path = source_root / "experiments/cadtrack/cadtrack.yaml"
    evaluator = probe_root / "evaluate_ope.py"
    assert (
        source.is_file()
        and runner.is_file()
        and identity_tool.is_file()
        and coverage_tool.is_file()
        and evaluator.is_file()
    )
    assert checkpoint.is_file() and config_path.is_file()
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inference_profile = dict(
        config_path=str(config_path),
        config_sha256=sha(config_path),
        template_factor=float(config["TEST"]["TEMPLATE_FACTOR"]),
        template_size=int(config["TEST"]["TEMPLATE_SIZE"]),
        search_factor=float(config["TEST"]["SEARCH_FACTOR"]),
        search_size=int(config["TEST"]["SEARCH_SIZE"]),
        update_intervals=int(config["TEST"]["UPDATE_INTERVALS"]),
        update_threshold=float(config["TEST"]["UPDATE_THRESHOLD"]),
        input="RGB plus JET-normalized depth using evaluate_ope.load_rgbd",
    )
    assert inference_profile == dict(
        config_path=str(config_path),
        config_sha256=sha(config_path),
        template_factor=2.0,
        template_size=128,
        search_factor=4.0,
        search_size=256,
        update_intervals=5,
        update_threshold=0.65,
        input="RGB plus JET-normalized depth using evaluate_ope.load_rgbd",
    )
    import torch
    checkpoint_obj = torch.load(str(checkpoint), map_location="cpu")
    assert checkpoint_obj["epoch"] == selected["epoch"]
    assert set(checkpoint_obj) == {"epoch", "net_type", "net"}

    output.mkdir(parents=True)
    checkpoint_selection_path = output / "checkpoint_selection.json"
    write(
        checkpoint_selection_path,
        dict(
            schema="CADTrack_pure_finetune_checkpoint_selection_v1",
            selected_epoch=selected["epoch"],
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            selected_by="user",
            selection="test-set-selected",
            selection_metric="DepthTrack Test50 F-score",
            selection_method=(
                "User-specified epoch after reviewing all Test50 scores "
                "for epochs 31-35."
            ),
            selection_bias_disclosed=True,
            selected_metrics=selected["metrics"],
            candidate_metrics=metric_rows,
            automatic_test50_max_epoch=read(training / "best.json")["epoch"],
        ),
    )
    source_copy = output / "evaluation.py"
    runner_copy = output / "run_evaluation.py"
    evaluator_copy = output / "evaluate_ope.py"
    identity_copy = output / "vot_analysis_identity.py"
    coverage_copy = output / "vot_trajectory_coverage.py"
    shutil.copy2(source, source_copy)
    shutil.copy2(runner, runner_copy)
    shutil.copy2(evaluator, evaluator_copy)
    shutil.copy2(identity_tool, identity_copy)
    shutil.copy2(coverage_tool, coverage_copy)
    reference_copy = output / "A1_anchored_U2_full127.json"
    shutil.copy2(reference, reference_copy)
    source_manifest_path = output / "source_manifest.json"
    source_manifest = {
        str(path): sha(path)
        for path in sorted(source_root.rglob("*"), key=str)
        if path.is_file()
    }
    write(
        source_manifest_path,
        dict(
            schema="CADTrack_pure_finetune_source_manifest_v1",
            source_sha256=source_manifest,
            evaluator_path=str(evaluator_copy),
            evaluator_sha256=sha(evaluator_copy),
        ),
    )

    tracker = "cadtrack_pure_finetune_epoch_%02d" % selected["epoch"]
    common = dict(
        schema="CADTrack_pure_finetune_evaluation_v1",
        dataset="",
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        selected_epoch=selected["epoch"],
        selection="test-set-selected",
        selection_metric="DepthTrack Test50 F-score",
        selection_rows=metric_rows,
        selection_path=str(checkpoint_selection_path),
        selection_file_sha256=sha(checkpoint_selection_path),
        retained_checkpoints=retained,
        inference_profile=inference_profile,
        probe_root=str(probe_root),
        source_root=str(source_root),
        source_manifest_path=str(source_manifest_path),
        source_manifest_sha256=sha(source_manifest_path),
        evaluator_path=str(evaluator_copy),
        evaluator_sha256=sha(evaluator_copy),
        evaluation_source_path=str(source_copy),
        evaluation_source_sha256=sha(source_copy),
        bundle_path=str(output / "bundle.json"),
    )
    plans = {}
    for name, old in case_plans.items():
        folder = output / name
        folder.mkdir()
        cases = read(old["cases_path"])
        plan = dict(
            common,
            dataset=name,
            bundle_path=str(output / "bundle.json"),
            cases_path=old["cases_path"],
            cases_sha256=old["cases_sha256"],
            dataset_root=old["dataset_root"],
            metric_source=old["metric_source"],
            metric_source_sha256=old["metric_source_sha256"],
            output=str(folder / "predictions"),
        )
        write(folder / "plan.json", plan)
        plans[name] = str(folder / "plan.json")

    vot_root = output / "vot"
    vot_root.mkdir()
    run_root = vot_root / "run"
    run_root.mkdir()
    vote_plan = dict(
        common,
        dataset="vot",
        bundle_path=str(output / "bundle.json"),
        tracker=tracker,
        vot_reference_path=str(reference_copy),
        vot_reference_sha256=sha(reference_copy),
        vot_identity_tool_path=str(identity_copy),
        vot_identity_tool_sha256=sha(identity_copy),
        vot_coverage_tool_path=str(coverage_copy),
        vot_coverage_tool_sha256=sha(coverage_copy),
        protocol_workspaces=[
            str(run_root / ("shard-%02d" % row["index"]))
            for row in manifest["shards"]
        ],
        protocol_receipt=str(vot_root / "protocol_receipt.json"),
        vot_execution_path=str(vot_root / "execution.json"),
        output=str(vot_root),
    )
    write(vot_root / "plan.json", vote_plan)

    shards = []
    execution_paths = [
        source_copy, runner_copy, identity_copy, coverage_copy, reference_copy,
        evaluator_copy, source_manifest_path,
        checkpoint, config_path,
        training / "epoch_metrics.jsonl", training / "best.json",
        checkpoint_selection_path,
    ]
    for old in manifest["shards"]:
        target = run_root / ("shard-%02d" % old["index"])
        target.mkdir()
        shutil.copytree(Path(old["root"]) / "sequences", target / "sequences", symlinks=True)
        shutil.copy2(Path(old["root"]) / "config.yaml", target / "config.yaml")
        gpu = old["index"] % 2
        (target / "trackers.ini").write_text(
            "[%s]\n"
            "label = Pure CADTrack DepthTrack fine-tune epoch %d\n"
            "protocol = traxpython\n"
            "command = pure_cadtrack_vot\n"
            "paths = %s\n"
            "python = /root/autodl-tmp/envs/sttrack/bin/python\n"
            "env_CUDA_VISIBLE_DEVICES = %d\n"
            "env_PYTHONPATH = %s\n"
            "env_PYTHONDONTWRITEBYTECODE = 1\n"
            "env_PYTHONOPTIMIZE = 0\n"
            "timeout = 600\n"
            "restart = false\n"
            % (
                tracker, selected["epoch"], vot_root, gpu,
                os.pathsep.join((str(vot_root), str(probe_root), str(source_root))),
            ),
            encoding="utf-8",
        )
        wrapper = vot_root / "pure_cadtrack_vot.py"
        wrapper.write_text(
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from evaluation import vot_track\n"
            "vot_track(%r)\n"
            % (str(output), str(vot_root / "plan.json")),
            encoding="utf-8",
        )
        row = dict(old, root=str(target), gpu=gpu)
        assert sha(target / "sequences" / "list.txt") == old["list_sha256"]
        row["trackers_sha256"] = sha(target / "trackers.ini")
        shards.append(row)
        execution_paths.extend(
            [
                target / "config.yaml",
                target / "trackers.ini",
                target / "sequences" / "list.txt",
            ]
        )
        for name in old["expected_trajectories"]:
            sequence_name = name.rsplit("_", 1)[0]
            execution_paths.append(
                Path(manifest["source"][sequence_name]["root"]) / "anchor.value"
            )

    vot_manifest = dict(
        manifest,
        tracker=tracker,
        shards=shards,
        native_source="CADTrack epoch20 pretrained baseline fine-tuned on DepthTrack Train152",
        source_manifest_path=str(vot_manifest_path),
        source_manifest_sha256=sha(vot_manifest_path),
    )
    write(run_root / "shard_manifest.json", vot_manifest)
    execution_paths.extend(
        [run_root / "shard_manifest.json", source_copy, runner_copy, wrapper]
    )
    execution = dict(
        status="frozen_before_CADTrack_pure_finetune_VOT_tracking",
        source_sha256={
            str(path): sha(path)
            for path in sorted(set(execution_paths), key=str)
        },
        anchors=1765,
        sequences=127,
        result_files=5295,
        workers=2,
        reused_prediction_anchors=0,
    )
    write(vot_root / "execution.json", execution)
    vote_plan["vot_execution_sha256"] = sha(vot_root / "execution.json")
    write(vot_root / "plan.json", vote_plan)

    bundle = dict(
        schema="CADTrack_pure_finetune_bundle_v1",
        checkpoint_sha256=sha(checkpoint),
        selected_epoch=selected["epoch"],
        inference_profile=inference_profile,
        selection="test-set-selected",
        retained_checkpoints=retained,
        selection_file_sha256=sha(checkpoint_selection_path),
        selection_metrics_file_sha256=sha(training / "epoch_metrics.jsonl"),
        evaluation_source_sha256=sha(source_copy),
        evaluation_runner_sha256=sha(runner_copy),
        evaluator_sha256=sha(evaluator_copy),
        source_manifest_sha256=sha(source_manifest_path),
        vot_identity_tool_sha256=sha(identity_copy),
        vot_coverage_tool_sha256=sha(coverage_copy),
        vot_reference_sha256=sha(reference_copy),
        vot_source_manifest_sha256=sha(vot_manifest_path),
        datasets={
            name: dict(
                cases_sha256=case_plans[name]["cases_sha256"],
                metric_source_sha256=case_plans[name]["metric_source_sha256"],
                sequence_count=len(read(case_plans[name]["cases_path"])),
                frame_count=sum(
                    row["frames"] for row in read(case_plans[name]["cases_path"])
                ),
            )
            for name in ("depthtrack_test", "cdtb")
        },
        plans=plans,
        vot_plan=str(vot_root / "plan.json"),
        vot_execution_sha256=vote_plan["vot_execution_sha256"],
    )
    write(output / "bundle.json", bundle)
    for name, path in plans.items():
        plan = read(path)
        plan["bundle_sha256"] = sha(output / "bundle.json")
        write(path, plan)
    vote_plan["bundle_sha256"] = sha(output / "bundle.json")
    write(vot_root / "plan.json", vote_plan)

    selection_output = dict(
        status="CADTrack_pure_finetune_final_fixed",
        checkpoint_sha256=sha(checkpoint),
        selected_epoch=selected["epoch"],
        selection="test-set-selected",
        selection_metric="DepthTrack Test50 F-score",
        selected_by="user",
        checkpoint_selection_sha256=sha(checkpoint_selection_path),
        selection_bias_disclosed=True,
        optimizer_steps_after_training=0,
        runner_sha256=sha(runner_copy),
        vot_execution_sha256=sha(vot_root / "execution.json"),
        plans={
            name: dict(path=str(path), sha256=sha(path))
            for name, path in plans.items()
        },
        vot_plan=str(vot_root / "plan.json"),
        vot_plan_sha256=sha(vot_root / "plan.json"),
    )
    write(output / "selection.json", selection_output)
    print(
        json.dumps(
            dict(
                status="prepared_CADTrack_pure_finetune_full_official_plans",
                selected_epoch=selected["epoch"],
                checkpoint_sha256=sha(checkpoint),
                selection="test-set-selected",
                selected_by="user",
                plans=selection_output["plans"],
                vot_plan=selection_output["vot_plan"],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
