"""Bind one completed CADTrack port final to the existing official RGB-D test plans."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

LOCKED_METRIC_SHA256 = "05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc"
REFERENCE_SHA256 = "73e0b6c15713735240d5d50acd51a95593155f2e3471a9f3d13d38249f5df2e6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initialization_key(rgb_sha, bbox):
    import struct
    return hashlib.sha256(
        bytes.fromhex(rgb_sha) + struct.pack(">4d", *bbox)
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("training-root", "base-evaluation", "checkpoint", "config",
                 "clip-weight", "evaluation-source", "evaluation-runner",
                 "vot-identity-tool",
                 "vot-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run_root = args.training_root.resolve()
    base = args.base_evaluation.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    clip_weight_path = args.clip_weight.resolve()
    fit = run_root / "fit"
    result_path = fit / "result.json"
    final_path = fit / "final.pt"
    terminal = read(run_root / "terminal.json")
    child = read(run_root / "child.json")
    assert (run_root / "train.exit").read_text().strip() == "0"
    assert terminal["exit_code"] == 0
    assert terminal["child_pid"] == child["child_pid"]
    assert terminal["supervisor_pid"] == child["supervisor_pid"]
    launch_path = run_root / "launch.json"
    launch = read(launch_path)
    assert terminal["command"] == child["command"] == launch["command"]
    command = child["command"]
    assert command[command.index("--mode") + 1] == "train"
    assert Path(command[command.index("--output") + 1]).resolve() == fit
    trained = read(result_path)
    assert trained["status"] == "complete_train"
    assert trained["calls"] == 1318812
    assert trained["final_sha256"] == sha(final_path)
    assert trained["frozen_before_after_exact"] and trained["module_roundtrip_exact"]
    assert not trained["formal_metrics"]
    assert not args.output.exists()

    checkpoint_sha = sha(checkpoint_path)
    config_sha = sha(config_path)
    clip_sha = sha(clip_weight_path)
    final = torch.load(final_path, map_location="cpu")
    protocol = final["protocol"]
    assert protocol["mode"] == "train" and protocol["seed"] == 2027
    assert protocol["baseline_sha256"] == checkpoint_sha
    assert protocol["inputs"]["config"] == config_sha
    assert protocol["inputs"]["clip_weight"] == clip_sha
    assert protocol["passes"] == 6 and protocol["final_selection"].startswith("fixed pass06")
    spec_path = Path(command[command.index("--spec") + 1])
    assert sha(spec_path) == protocol["inputs"]["spec"]
    spec = read(spec_path)
    assert len(spec["sequence_order"]) == 152 and spec["seed"] == 2027
    rows_path = fit / "sequences.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 912
    calls = 0
    for epoch in range(6):
        for case, row in zip(spec["sequence_order"], rows[epoch * 152:(epoch + 1) * 152]):
            assert row["epoch"] == epoch + 1 and row["sequence"] == case["sequence"]
            assert row["frames"] == case["rgb_frames"] - 1
            calls += row["frames"]
            assert row["calls"] == calls and math.isfinite(row["loss_sum"])
    assert calls == trained["calls"] == 1318812
    assert rows[-1]["steps"] == trained["steps"] and rows[-1]["C_steps"] == trained["C_steps"]

    base_plans = {
        name: read(base / name / "plan.json")
        for name in ("depthtrack_test", "cdtb", "vot")}
    base_bundle = read(base / "bundle.json")
    assert base_bundle["schema"] == "M125_region_fixed6_Train152_official_v1"
    for name, plan in base_plans.items():
        assert sha(plan["base_plan_path"]) == plan["base_plan_sha256"]
        assert plan["dataset"] == name
    bundle_sha = sha(base / "bundle.json")
    assert all(plan["bundle_sha256"] == bundle_sha for plan in base_plans.values())
    bank_path = Path(base_plans["depthtrack_test"]["bank_path"])
    binding_path = Path(base_plans["depthtrack_test"]["binding_path"])
    bank_sha, binding_sha = sha(bank_path), sha(binding_path)
    assert all(row["bank_sha256"] == bank_sha and row["binding_sha256"] == binding_sha
               for row in base_plans.values())
    bank = torch.load(bank_path, map_location="cpu")
    binding = read(binding_path)
    assert bank["format"] == "M122_human_official_initialization_v1"
    assert bank["human_confirmed"]
    assert bank["binding_sha256"] == binding_sha
    assert bank["encoder_sha256"] == sha(clip_weight_path)
    assert len(bank["keys"]) == len(set(bank["keys"])) == 1895
    assert tuple(bank["tokens"].shape) == (1895, 5, 768)
    assert tuple(bank["mask"].shape) == (1895, 5)
    assert set(bank["keys"]) == {row["key"] for row in binding["rows"]}
    assert len(binding["rows"]) == 1895
    assert {key for key in bank["keys"] if key.startswith("vot:")} == {
        row["key"] for row in binding["rows"] if row["dataset"] == "vot"}
    binding_by_key = {row["key"]: row for row in binding["rows"]}

    for dataset, expected in (("depthtrack_test", (50, 76373)), ("cdtb", (80, 101956))):
        plan = base_plans[dataset]
        assert sha(plan["cases_path"]) == plan["cases_sha256"]
        assert plan["metric_source_sha256"] == LOCKED_METRIC_SHA256
        assert sha(plan["metric_source"]) == LOCKED_METRIC_SHA256
        cases = read(plan["cases_path"])
        assert (len(cases), sum(case["frames"] for case in cases)) == expected
        for case in cases:
            image = Path(plan["dataset_root"]) / case["sequence"] / "color/00000001.jpg"
            key = dataset + ":" + initialization_key(sha(image), case["init_bbox"])
            row = binding_by_key[key]
            assert row["human_confirmed"] and row["image_sha256"] == sha(image)
            assert row["init_bbox"] == case["init_bbox"]
    assert all(row["human_confirmed"] for row in binding["rows"])
    vot_rows = [row for row in binding["rows"] if row["dataset"] == "vot"]
    assert len(vot_rows) == 1765
    for row in vot_rows:
        assert sha(row["image"]) == row["image_sha256"]
        assert row["key"] == "vot:" + initialization_key(row["image_sha256"], row["init_bbox"])

    evaluation_source = args.evaluation_source.resolve()
    evaluation_runner = args.evaluation_runner.resolve()
    assert evaluation_source.is_file()
    assert evaluation_runner.is_file()
    identity_tool = args.vot_identity_tool.resolve()
    reference = args.vot_reference.resolve()
    assert sha(reference) == REFERENCE_SHA256
    code_root, legacy_root, repository = (
        run_root / "code", run_root / "legacy-code", run_root / "native-source")
    sources = {
        str(path): sha(path)
        for folder in (code_root, legacy_root, repository)
        for path in folder.rglob("*") if path.is_file()
    }
    for name, digest in launch["source_sha256"].items():
        path = (run_root / name).resolve()
        assert sha(path) == digest, name
        assert sources[str(path)] == digest
    code_sources = {path.name: sha(path) for path in code_root.glob("*.py")}
    legacy_sources = {path.name: sha(path) for path in legacy_root.glob("*.py")}
    assert protocol["source"] == code_sources
    assert protocol["legacy_source"] == legacy_sources
    assert len([p for p in sources if p.startswith(str(code_root))]) >= 1
    output = args.output.resolve()
    output.mkdir(parents=True)
    eval_source_path = output / "evaluation.py"
    shutil.copy2(evaluation_source, eval_source_path)
    evaluation_source_sha = sha(eval_source_path)
    runner_path = output / "run_evaluation.py"
    shutil.copy2(evaluation_runner, runner_path)
    runner_sha = sha(runner_path)
    identity_tool_path = output / "vot_analysis_identity.py"
    shutil.copy2(identity_tool, identity_tool_path)
    identity_tool_sha = sha(identity_tool_path)
    reference_path = output / "A1_anchored_U2_full127.json"
    shutil.copy2(reference, reference_path)
    reference_sha = sha(reference_path)
    preparer_path = output / "prepare_evaluation.py"
    shutil.copy2(Path(__file__).resolve(), preparer_path)
    preparer_sha = sha(preparer_path)

    common = dict(
        schema="CADTrack_M125_port_official_eval_v1",
        training_result_path=str(result_path), training_result_sha256=sha(result_path),
        final_path=str(final_path), final_sha256=sha(final_path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        config_path=str(config_path), config_sha256=config_sha,
        clip_weight_path=str(clip_weight_path), clip_weight_sha256=clip_sha,
        bank_path=str(bank_path.resolve()), bank_sha256=bank_sha,
        binding_path=str(binding_path.resolve()), binding_sha256=binding_sha,
        repository=str(repository), code_root=str(code_root), legacy_code_root=str(legacy_root),
        evaluation_source_path=str(eval_source_path),
        evaluation_source_sha256=evaluation_source_sha,
        evaluation_preparer_path=str(preparer_path),
        evaluation_preparer_sha256=preparer_sha,
        evaluation_runner_path=str(runner_path),
        evaluation_runner_sha256=runner_sha, source_sha256=sources,
        launch_path=str(launch_path), launch_sha256=sha(launch_path),
        terminal_path=str(run_root / "terminal.json"),
        terminal_sha256=sha(run_root / "terminal.json"),
        training_rows_path=str(rows_path), training_rows_sha256=sha(rows_path),
        failure_source_path=base_bundle["failure_source_path"],
        failure_source_sha256=sha(base_bundle["failure_source_path"]),
        failure_common_path=base_bundle["failure_common_path"],
        failure_common_sha256=sha(base_bundle["failure_common_path"]),
        vot_identity_tool_path=str(identity_tool_path),
        vot_identity_tool_sha256=identity_tool_sha,
        vot_reference_path=str(reference_path),
        vot_reference_sha256=reference_sha)
    bundle_path = output / "bundle.json"
    write(bundle_path, dict(schema="CADTrack_port_eval_binding_v1", bindings=common))
    common = dict(common, bundle_path=str(bundle_path), bundle_sha256=sha(bundle_path))
    plans = {}
    for dataset in ("depthtrack_test", "cdtb"):
        old = base_plans[dataset]
        folder = output / dataset
        folder.mkdir()
        plan = dict(common, dataset=dataset,
            cases_path=old["cases_path"], cases_sha256=old["cases_sha256"],
            dataset_root=old["dataset_root"],
            metric_source=old["metric_source"],
            metric_source_sha256=old["metric_source_sha256"],
            output=str(folder / "predictions"))
        write(folder / "plan.json", plan)
        plans[dataset] = dict(path=str(folder / "plan.json"), sha256=sha(folder / "plan.json"))

    vot = output / "vot"
    vot.mkdir()
    run = vot / "run"
    run.mkdir()
    original = base / "vot/run"
    manifest = read(original / "shard_manifest.json")
    assert manifest["schema"] == "M122_full127_frozen_shards_v1"
    assert len(manifest["sequences"]) == 127 and manifest["total_anchor_count"] == 1765
    assert set(manifest["source"]) == set(manifest["sequences"])
    assert [row["index"] for row in manifest["shards"]] == [0, 1, 2, 3]
    assert [row["anchor_count"] for row in manifest["shards"]] == [441, 441, 441, 442]
    names = [name for row in manifest["shards"] for name in row["expected_trajectories"]]
    assert len(names) == len(set(names)) == 1765
    tracker_name = "cadtrack_abc_epoch20_innovation_fixed6"
    wrapper = vot / "selected_cadtrack_vot.py"
    wrapper.write_text(
        "import sys\n"
        "sys.path[:0]=" + repr([str(output), str(code_root), str(legacy_root), str(repository)]) + "\n"
        "from evaluation import vot_track\n"
        "vot_track(" + repr(str(vot / "plan.json")) + ")\n")
    bridge_path = Path(base_bundle["vot_bridge_path"])
    vot_plan = dict(common, dataset="vot", vot_bridge_path=str(bridge_path),
        vot_bridge_sha256=sha(bridge_path))
    write(vot / "plan.json", vot_plan)
    shards = []
    execution_files = [eval_source_path, runner_path, Path(vot / "plan.json"), wrapper,
                       final_path, result_path, identity_tool_path, reference_path]
    original_execution = read(base / "vot/execution.json")
    for row in manifest["shards"]:
        old_root = Path(row["root"])
        target = run / ("shard-%02d" % row["index"])
        target.mkdir()
        shutil.copytree(old_root / "sequences", target / "sequences", symlinks=True)
        shutil.copy2(old_root / "config.yaml", target / "config.yaml")
        gpu = row["index"] % 2
        pythonpath = ":".join(map(str, (vot, output, code_root, legacy_root, repository)))
        trackers = (
            f"[{tracker_name}]\n"
            "label = CADTrack epoch20 plus RGB-D port, Train152 six-pass fixed final\n"
            "protocol = traxpython\n"
            "command = selected_cadtrack_vot\n"
            f"paths = {vot}\n"
            "python = /root/autodl-tmp/envs/sttrack/bin/python\n"
            f"env_CUDA_VISIBLE_DEVICES = {gpu}\n"
            f"env_PYTHONPATH = {pythonpath}\n"
            "env_TOKENIZERS_PARALLELISM = false\n"
            "env_PYTHONDONTWRITEBYTECODE = 1\n"
            "timeout = 600\n"
            "restart = false\n")
        (target / "trackers.ini").write_text(trackers)
        shards.append(dict(row, root=str(target), gpu=gpu,
                           trackers_sha256=sha(target / "trackers.ini")))
        execution_files.extend(target / name for name in
            ("config.yaml", "trackers.ini", "sequences/list.txt"))
        for original_path, digest in original_execution["source_sha256"].items():
            original_path = Path(original_path)
            if old_root / "sequences" in original_path.parents:
                copied = target / original_path.relative_to(old_root)
                assert sha(copied) == digest
                execution_files.append(copied)
    for info in manifest["source"].values():
        anchor = Path(info["root"]) / "anchor.value"
        assert sha(anchor) == info["anchor_sha256"]
        execution_files.append(anchor)
    vot_manifest = dict(manifest, tracker=tracker_name, shards=shards,
        native_source="CADTrack epoch20 with frozen CLIP and Train152 A/B/C port; fixed pass06 final",
        source_manifest_path=str(original / "shard_manifest.json"),
        source_manifest_sha256=sha(original / "shard_manifest.json"))
    write(run / "shard_manifest.json", vot_manifest)
    execution_files.append(run / "shard_manifest.json")
    execution = dict(
        status="frozen_before_CADTrack_port_VOT_tracking",
        source_sha256={str(path): sha(path) for path in execution_files},
        anchors=1765, sequences=127, result_files=5295, workers=2,
        reused_prediction_anchors=0,
        original_manifest_sha256=sha(original / "shard_manifest.json"))
    write(vot / "execution.json", execution)
    plans["vot"] = dict(path=str(vot / "plan.json"), sha256=sha(vot / "plan.json"),
                        run=str(run), execution_sha256=sha(vot / "execution.json"))
    write(output / "selection.json", dict(
        status="CADTrack_port_final_fixed_before_full_metrics",
        training_result_sha256=sha(result_path), final_sha256=sha(final_path),
        bank_sha256=bank_sha, binding_sha256=binding_sha, plans=plans,
        evaluation_runner_sha256=runner_sha,
        bundle_sha256=sha(bundle_path),
        external_metric_checkpoint_selection=False, optimizer_steps_after_training=0,
        fixed_final_pass=6))
    print(json.dumps(dict(status="prepared_CADTrack_port_full_official_plans",
                          plans=plans), indent=2), flush=True)


if __name__ == "__main__":
    main()
