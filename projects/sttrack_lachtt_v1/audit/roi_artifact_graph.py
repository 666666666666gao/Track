"""Read the completed ROI artifact graph; never instantiate a tracking actor."""
import argparse
import ast
import configparser
import hashlib
import json
import os
from pathlib import Path
import sys


FINAL = "1a6f8e4429ce6182bb42e4b7d5b084ced56168faa9ebc7d27bb7771871ff39cf"
BUNDLE = "c928b115afdb494e8492bebfd4ba9447bba6e8e3f9e171e51847f004cc6cec0c"
EXECUTION = "51a3041dadbefb842622ccebd9253f79ea2bd507bc80796b0ac3b1f6244b06b1"
COVERAGE = "8a29c7757a6eaad5ed0201f954373f7825cbe5424f772a78c98d0cf135859c4f"


def require(condition, label):
    if not condition:
        raise ValueError(label)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sequence_history(trained):
    rows = trained["sequence_records"]
    epochs = trained["epochs"]
    require(len(rows) == epochs * 152, "sequence record count")
    order = [row["sequence"] for row in rows[:152]]
    require(len(set(order)) == 152, "152 distinct training sequences")
    calls = 0
    for index, row in enumerate(rows):
        require(row["epoch"] == index // 152 + 1, "epoch position")
        require(row["sequence_index"] == index % 152, "sequence position")
        require(row["sequence"] == order[index % 152], "sequence order")
        require(row["sequences_per_epoch"] == 152, "sequences per epoch")
        calls += row["track_calls"]
        require(row["total_track_calls"] == calls, "cumulative tracking calls")
    require(calls == epochs * 219802 == trained["track_calls"], "tracking budget")
    return rows


def check_shard_partition(shards):
    require([shard["index"] for shard in shards] == [0, 1, 2, 3], "shard indexes")
    require([shard["gpu"] for shard in shards] == [0, 1, 0, 1], "shard GPUs")
    require([shard["anchor_count"] for shard in shards] == [441, 441, 441, 442],
            "shard anchor partition")


def audit(root, coverage_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    require(sys.flags.optimize == 0 and __debug__, "original runtime requires assertions")
    evaluation = root / "evaluation"
    bundle_path = evaluation / "bundle.json"
    require(sha(bundle_path) == BUNDLE, "externally pinned ROI bundle")
    bundle = read(bundle_path)
    gate_path = Path(bundle["source_gate_path"])
    require(sha(gate_path) == bundle["source_gate_sha256"], "source gate binding")
    gate = read(gate_path)
    require(gate["source_sha256"] == bundle["source_sha256"], "gate/bundle source inventory")
    require(len(bundle["source_sha256"]) == 215, "215-source inventory")
    for path, digest in gate["input_sha256"].items():
        require(sha(path) == digest, "gate input: " + path)
    require(sha(root / "source_review.json") == gate["source_review_report_sha256"],
            "review report binding")
    require(sha(root / "candidate_gate.json") == gate["source_review_candidate_sha256"],
            "review candidate binding")

    # This original entry validates the common base plan, sources and frozen weights.
    sys.path[:0] = [str(root / "code"), gate["base_code_root"]]
    from m125_region_fixed_official_runtime import checked_plan
    vot_plan, checked_bundle = checked_plan(evaluation / "vot/plan.json")
    base = checked_bundle.pop("base_bundle")
    require(checked_bundle == bundle, "loaded runtime bundle")
    for key in ("repository", "native_checkpoint_path", "native_checkpoint_sha256",
                "clip_weight_path", "clip_weight_sha256"):
        require(bundle[key] == base[key], "shared frozen baseline: " + key)
    require(bundle["region_arm"] == "roi" and bundle["C_arm"] == "current"
            and bundle["C_threshold"] == 0.5, "fixed inference profile")
    selection = read(evaluation / "selection.json")
    require(selection["status"] == "M125_region_fixed6_final_before_metrics"
            and len(selection["models"]) == 1, "single fixed-final selection")
    require(selection["final_sha256"] == bundle["final_sha256"] == FINAL,
            "common final identity")
    require(selection["bundle_sha256"] == BUNDLE, "selection bundle")
    require(selection["models"][0]["final_sha256"] == FINAL
            and selection["models"][0]["bundle_sha256"] == BUNDLE
            and selection["models"][0]["plans"] == selection["plans"], "single model plans")

    history = []
    pass_receipts = []
    for epoch in range(1, 7):
        folder = root / "passes" / ("pass-%02d" % epoch)
        trained = read(folder / "fit/result.json")
        require(trained["epochs"] == epoch and trained["seed"] == 2027,
                "pass epoch/seed")
        require(trained["training_sequences"] == 152
                and trained["sequence_runs"] == epoch * 152, "pass budget")
        rows = sequence_history(trained)
        require(rows[:len(history)] == history, "cumulative history prefix")
        history = rows
        require(sha(folder / "fit/final.pt") == trained["final_sha256"], "pass final hash")
        receipt_path = folder / "train_fixed_pass.receipt.json"
        receipt = read(receipt_path)
        require(receipt["exit_code"] == 0
                and (folder / "train_fixed_pass.exit").read_text().strip() == "0",
                "original pass exit")
        command = receipt["command"]
        require(command[command.index("--epoch") + 1] == str(epoch), "pass argv epoch")
        require(Path(command[command.index("--output") + 1]) == folder / "fit",
                "pass argv output")
        if epoch == 1:
            require("--resume" not in command and trained["fresh_common_warm_start"],
                    "fresh first pass")
        else:
            previous = root / "passes" / ("pass-%02d" % (epoch - 1)) / "fit/latest.pt"
            require(Path(command[command.index("--resume") + 1]) == previous,
                    "original resume path")
            require(sha(previous) == trained["resumed_checkpoint_sha256"], "resume hash")
        pass_receipts.append(receipt)
    require(sha(bundle["training_result_path"]) == bundle["training_result_sha256"]
            == selection["training_result_sha256"], "final training receipt binding")
    require(Path(bundle["training_result_path"]) == root / "passes/pass-06/fit/result.json",
            "fixed pass06 receipt")
    require(Path(bundle["final_path"]) == root / "passes/pass-06/fit/final.pt",
            "fixed pass06 checkpoint")

    import torch
    from region_conditioned_target_decoder import RegionConditionedTargetDecoder
    from template_write_C import TemplateWriteC
    weights = torch.load(bundle["final_path"], map_location="cpu")
    require(weights["format"] == "M125_region_fixed_final_v1"
            and weights["region_arm"] == "roi", "checkpoint format")
    decoder = RegionConditionedTargetDecoder("roi")
    writer = TemplateWriteC()
    decoder.load_state_dict(weights["A_B"], strict=True)
    writer.load_state_dict(weights["C"], strict=True)
    for key in ("A_B", "C"):
        require(state_digest(weights[key]) == trained["final_state_sha256"][key],
                "checkpoint state digest: " + key)

    plan_digests = {}
    for dataset in ("depthtrack_test", "cdtb", "vot"):
        path = evaluation / dataset / "plan.json"
        plan = read(path)
        plan_digests[dataset] = sha(path)
        require(selection["plans"][dataset]["path"] == str(path)
                and selection["plans"][dataset]["sha256"] == plan_digests[dataset],
                "selected plan: " + dataset)
        require(plan["dataset"] == dataset, "plan dataset")
        for key in ("bundle_path", "bundle_sha256", "bank_path", "bank_sha256",
                    "binding_path", "binding_sha256"):
            require(plan[key] == vot_plan[key], "common initialization: " + key)
        base_plan = read(plan["base_plan_path"])
        require(sha(plan["base_plan_path"]) == plan["base_plan_sha256"],
                "inherited base plan")
        require(base_plan["bundle_path"] == read(vot_plan["base_plan_path"])["bundle_path"],
                "common inherited baseline")
        if dataset != "vot":
            output = Path(plan["output"])
            prediction = read(output / "receipt.json")
            metrics = read(output / "metrics.json")
            require(prediction["plan_sha256"] == sha(path), "prediction plan binding")
            require(prediction["status"] == "complete_M122_official_OPE_predictions"
                    and prediction["sequences"] == len(read(plan["cases_path"]))
                    and prediction["frames"] == metrics["metrics"]["frames"],
                    "OPE prediction scope")
            require(metrics["plan_sha256"] == sha(path)
                    and metrics["receipt_sha256"] == sha(output / "receipt.json")
                    and metrics["metric_source_sha256"] == plan["metric_source_sha256"],
                    "OPE metric receipt binding")
            require(prediction["final_sha256"] == metrics["final_sha256"] == FINAL,
                    "OPE final binding")
            require(prediction["bundle_sha256"] == metrics["bundle_sha256"] == BUNDLE,
                    "OPE bundle binding")
    execution_path = evaluation / "vot/execution.json"
    require(sha(execution_path) == EXECUTION, "externally pinned VOT execution")
    execution = read(execution_path)
    require(selection["bank_sha256"] == vot_plan["bank_sha256"]
            and selection["binding_sha256"] == vot_plan["binding_sha256"],
            "selection initialization binding")
    wrapper = evaluation / "vot/selected_m125_region_fixed_vot.py"
    wrapper_tree = ast.parse(wrapper.read_text())
    calls = [node.value for node in wrapper_tree.body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    require(len(calls) == 1 and isinstance(calls[0].func, ast.Name)
            and calls[0].func.id == "vot_track"
            and ast.literal_eval(calls[0].args[0]) == str(evaluation / "vot/plan.json"),
            "selected VOT plan entry")
    for path in (bundle_path, evaluation / "vot/plan.json", wrapper):
        require(execution["source_sha256"][str(path)] == sha(path), "execution graph edge")
    manifest = read(evaluation / "vot/run/shard_manifest.json")
    check_shard_partition(manifest["shards"])
    for shard in manifest["shards"]:
        ini = Path(shard["root"]) / "trackers.ini"
        require(sha(ini) == shard["trackers_sha256"]
                == execution["source_sha256"][str(ini)], "shard runtime binding")
        config = configparser.ConfigParser()
        config.read(str(ini))
        require(config.sections() == [manifest["tracker"]], "one shard tracker")
        tracker = config[manifest["tracker"]]
        require(tracker["command"] == "selected_m125_region_fixed_vot"
                and tracker["paths"] == str(evaluation / "vot")
                and tracker["env_PYTHONPATH"] == str(evaluation / "vot")
                and tracker["env_CUDA_VISIBLE_DEVICES"] == str(shard["gpu"]),
                "shard selected runtime/profile")
    sealed = read(evaluation / "vot/result.json")
    for key in ("final_sha256", "bundle_sha256", "bank_sha256", "binding_sha256"):
        require(sealed[key] == selection[key], "VOT sealed binding: " + key)
    complete = read(root / "result.json")
    require(complete["stages"][:6] == pass_receipts, "original training stage receipts")
    expected_stages = (["train_fixed_pass"] * 6 + ["prepare", "depthtrack_test_track",
                        "cdtb_track", "depthtrack_test_analyze", "cdtb_analyze",
                        "vot_track", "vot_official_analysis", "vot_seal", "collect"])
    require([row["stage"] for row in complete["stages"]] == expected_stages
            and all(row["exit_code"] == 0 for row in complete["stages"]), "all original stages")
    for row in complete["stages"][6:]:
        require(read(root / (row["stage"] + ".receipt.json")) == row, "original stage receipt")
        require((root / (row["stage"] + ".exit")).read_text().strip() == "0",
                "original stage exit")
    require(sha(coverage_path) == COVERAGE, "previous detailed coverage report")
    coverage = read(coverage_path)
    require(coverage["trajectory_coverage_checked"]
            and coverage["execution_sha256"] == EXECUTION, "coverage execution binding")
    return dict(status="current_ROI_artifact_graph_verified", final_sha256=FINAL,
                bundle_sha256=BUNDLE, plan_sha256=plan_digests, bound_sources=215,
                full_passes=6, sequence_runs=len(history), calls=trained["track_calls"],
                strict_checkpoint_load=True, checkpoint_state_digests_exact=True,
                original_stages=15, original_shard_runtime_bindings=4,
                coverage_report_sha256=COVERAGE, current_artifact_graph_checked=True,
                historic_execution_provenance_proven=False, independent_completed_audit=False,
                neural_inference=False, metrics_recomputed=False,
                auditor_source_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new report path required")
    result = audit(args.root.resolve(), args.coverage.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
