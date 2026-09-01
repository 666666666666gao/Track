#!/usr/bin/env python3
"""Bound task-decoupled fixed-batch target--distractor capacity test."""

import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_cached_strict_router import (  # noqa: E402
    cached_strict_router_loss,
)
from lib.models.sttrack.lachtt_target_distractor_memory import (  # noqa: E402
    build_target_distractor_relations,
)
from lib.models.sttrack.lachtt_target_distractor_fixed_pool import (  # noqa: E402
    TEMPORAL_STATISTICS,
    TEMPORAL_WIDTH,
    fixed_temporal_pool,
)
from lib.models.sttrack.lachtt_target_distractor_task_towers import (  # noqa: E402
    TASK_NAMES,
    TargetDistractorTaskTowersRouter,
)
from tools.smoke_sttrack_lachtt_m10a_memory import (  # noqa: E402
    file_record,
    load_native_batch,
)
from tools.smoke_sttrack_lachtt_m8b_cached import (  # noqa: E402
    atomic_json,
    batch_tensors,
    frozen_records,
    gradient_diagnostics,
    load_closure,
    scale_gradients,
    sha256_file,
    state_digest,
    validate_selection,
    verify_frozen,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_output(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True).strip()


def atomic_jsonl_gz(path, rows):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    model = REPOSITORY_ROOT / spec["model"]["path"]
    relation_extractor = (REPOSITORY_ROOT /
                          spec["m10_relation_extractor"]["path"])
    fixed_pool_source = (REPOSITORY_ROOT /
                         spec["m11_fixed_pool_source"]["path"])
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m12-task-towers-overfit-binding/v1",
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "relation_extractor_path": str(relation_extractor),
        "relation_extractor_sha256": sha256_file(relation_extractor),
        "fixed_pool_source_path": str(fixed_pool_source),
        "fixed_pool_source_sha256": sha256_file(fixed_pool_source),
        "output": str(args.output),
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    if (git_output("branch", "--show-current") !=
            spec["repository"]["branch"] or
            git_output("status", "--porcelain")):
        raise ValueError("repository state drifted")
    if subprocess.run([
            "git", "-C", str(REPOSITORY_ROOT), "merge-base",
            "--is-ancestor", spec["repository"]["base_commit"], commit,
    ], check=False).returncode != 0:
        raise ValueError("implementation is outside frozen ancestry")
    authorization = binding.get("authorizations", {})
    if authorization.get("one_fixed_batch_capacity_run") is not True:
        raise ValueError("binding does not authorize M12")
    for name in (
            "sequence_disjoint_pilot_execution", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    if binding.get("authorizations", {}).get(
            "independent_integrity_audit_after_run") is not True:
        raise ValueError("binding does not authorize the required audit")
    return (binding, runner, model, relation_extractor, fixed_pool_source,
            commit)


def bound_records(spec, source_spec, selected_native_rows):
    records = frozen_records(source_spec)
    for name in ("plan", "source_batch_spec", "native_anchor_index",
                 "native_anchor_manifest"):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"], None))
    for name, item in spec["m11"].items():
        records.append(("m11_" + name, Path(item["path"]),
                        item["sha256"], None))
    for key in ("m10_relation_extractor", "m11_fixed_pool_source"):
        item = spec[key]
        records.append((key, REPOSITORY_ROOT / item["path"],
                        item["sha256"], None))
    index_root = Path(spec["native_anchor_index"]["path"]).parent
    for sequence, row in selected_native_rows.items():
        records.append((
            "native_anchor:" + sequence,
            index_root / row["path"], row["sha256"], int(row["bytes"])))
    return records


def evaluate(outputs, batch):
    available = batch["label_available"] & batch["candidate_valid"]
    event_prediction = torch.sigmoid(outputs["event_commit_logit"]) >= 0.5
    event_correct = int((event_prediction == batch["event_target"]).sum().item())
    best_rank_correct = 0
    best_rank_total = 0
    for index in range(available.shape[0]):
        beneficial = torch.nonzero(
            available[index] & batch["beneficial_target"][index],
            as_tuple=False).flatten()
        if beneficial.numel() == 0:
            continue
        target = int(beneficial[
            batch["gain_target"][index, beneficial].argmax()].item())
        predicted = int(outputs["candidate_rank_logits"][index].masked_fill(
            ~available[index], -float("inf")).argmax().item())
        best_rank_total += 1
        best_rank_correct += int(predicted == target)
    benefit_prediction = torch.sigmoid(
        outputs["candidate_benefit_logits"]) >= 0.5
    catastrophe_prediction = torch.sigmoid(
        outputs["candidate_catastrophe_logits"]) >= 0.5
    denominator = int(available.sum().item())
    benefit_correct = int((
        benefit_prediction[available] ==
        batch["beneficial_target"][available]).sum().item())
    catastrophe_correct = int((
        catastrophe_prediction[available] ==
        batch["catastrophic_target"][available]).sum().item())
    gain_mae = float(torch.abs(
        outputs["candidate_h10_gain"][available] -
        batch["gain_target"][available]).mean().item())
    return {
        "event_commit_correct": event_correct,
        "event_commit_total": int(available.shape[0]),
        "beneficial_event_best_rank_correct": best_rank_correct,
        "beneficial_event_best_rank_total": best_rank_total,
        "candidate_benefit_accuracy": benefit_correct / denominator,
        "candidate_catastrophe_accuracy": catastrophe_correct / denominator,
        "candidate_gain_mae": gain_mae,
    }


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    if (authorization.get("implementation") is not True or
            authorization.get("one_fixed_batch_capacity_run") is not True):
        raise ValueError("spec does not authorize M12")
    for name in (
            "sequence_disjoint_pilot_execution", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)
    if (authorization.get("independent_integrity_audit_after_run") is not True or
            authorization.get("sequence_disjoint_pilot_spec_if_pass") is not True):
        raise ValueError("required post-run authorization contract drifted")
    (binding, runner_path, model_path, relation_extractor_path,
     fixed_pool_source_path, commit) = validate_binding(args, spec)
    source_spec = json_file(spec["source_batch_spec"]["path"])
    if sha256_file(Path(spec["source_batch_spec"]["path"])) != \
            spec["source_batch_spec"]["sha256"]:
        raise ValueError("source batch spec drifted")
    m11_result = json_file(spec["m11"]["result"]["path"])
    if (m11_result.get("accepted") is not False or
            m11_result.get("decision") != "m11_fail_stop_without_rescan" or
            m11_result.get("metrics", {}).get("final", {}).get(
                "event_commit_correct") != 8 or
            m11_result.get("metrics", {}).get("final", {}).get(
                "beneficial_event_best_rank_correct") != 2):
        raise ValueError("M11 negative/capacity boundary drifted")
    m11_audit = json_file(spec["m11"]["audit"]["path"])
    if (m11_audit.get("integrity_verdict") != "PASS" or
            m11_audit.get("m11", {}).get("accepted") is not False):
        raise ValueError("M11 independent audit boundary drifted")

    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, selected_native_rows = load_native_batch(
        spec, source_spec)
    records = bound_records(spec, source_spec, selected_native_rows)
    before_mismatches, frozen_observed = verify_frozen(records)
    model_spec = spec["model"]
    relation_spec = spec["m10_relation_extractor"]
    fixed_pool_spec = spec["m11_fixed_pool_source"]
    relations = build_target_distractor_relations(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        alpha=float(relation_spec["ema_alpha"]),
        epsilon=float(relation_spec["l2_epsilon"]),
        top_k=int(relation_spec["native_anchor_top_k"]),
        depth_missing_floor=float(relation_spec["depth_missing_floor"]),
    ).detach()
    if tuple(relations.shape) != (8, 5, 6, int(model_spec["relation_dim"])):
        raise ValueError("relation shape drifted")

    optimization = spec["optimization"]
    torch.manual_seed(int(optimization["seed"]))
    pooled = fixed_temporal_pool(relations)
    if tuple(pooled.shape) != (8, 6, int(model_spec["temporal_width"])):
        raise ValueError("fixed temporal pool shape drifted")
    if (tuple(TEMPORAL_STATISTICS) !=
            tuple(fixed_pool_spec["temporal_statistics"]) or
            TEMPORAL_WIDTH != int(fixed_pool_spec["temporal_width"])):
        raise ValueError("fixed temporal pool definition drifted")
    model = TargetDistractorTaskTowersRouter(
        hidden_dim=int(model_spec["task_hidden_dim"]),
        residual_scale=float(model_spec["residual_scale"]))
    if sum(parameter.numel() for parameter in model.parameters()) != int(
            model_spec["expected_parameters"]):
        raise ValueError("model parameter count drifted")
    forbidden_types = set(model_spec["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if type(module).__name__ in forbidden_types]
    if tuple(TASK_NAMES) != tuple(model_spec["task_names"]):
        raise ValueError("task name contract drifted")
    task_parameter_ids = {
        name: {id(parameter) for parameter in model.task_parameters(name)}
        for name in TASK_NAMES
    }
    all_task_ids = [parameter_id for ids in task_parameter_ids.values()
                    for parameter_id in ids]
    shared_parameter_ids = len(all_task_ids) - len(set(all_task_ids))
    if set(all_task_ids) != {id(parameter)
                             for parameter in model.parameters()}:
        raise ValueError("unassigned task parameter found")
    permutation = torch.tensor([2, 5, 1, 4, 0, 3], dtype=torch.long)
    inverse_permutation = torch.argsort(permutation)
    model.eval()
    with torch.no_grad():
        reference_outputs = model(relations, batch["candidate_valid"])
        permuted_outputs = model(
            relations[:, :, permutation],
            batch["candidate_valid"][:, permutation])
    permutation_errors = []
    for name, value in reference_outputs.items():
        if value.ndim == 2:
            restored = permuted_outputs[name][:, inverse_permutation]
        else:
            restored = permuted_outputs[name]
        permutation_errors.append(float(torch.max(torch.abs(
            value - restored)).item()))
    permutation_error = max(permutation_errors)
    relation_max_abs = float(relations.abs().max().item())
    isolation_loss_keys = {
        "commit": ("event_commit",),
        "rank": ("conditional_rank", "pairwise"),
        "benefit": ("benefit",),
        "catastrophe": ("catastrophe",),
        "gain": ("gain",),
    }
    gradient_isolation = {}
    for task_name in TASK_NAMES:
        model.zero_grad(set_to_none=True)
        task_outputs = model(relations, batch["candidate_valid"])
        task_losses = cached_strict_router_loss(
            task_outputs, batch["event_target"], batch["gain_target"],
            batch["beneficial_target"], batch["catastrophic_target"],
            batch["label_available"], batch["candidate_valid"],
            pairwise_margin=float(optimization["pairwise_margin"]))
        task_loss = sum(task_losses[name]
                        for name in isolation_loss_keys[task_name])
        task_loss.backward()
        counts = {}
        nonfinite = 0
        for parameter_task in TASK_NAMES:
            parameters = model.task_parameters(parameter_task)
            counts[parameter_task] = sum(
                parameter.grad is not None and
                torch.count_nonzero(parameter.grad).item() > 0
                for parameter in parameters)
            nonfinite += sum(
                parameter.grad is not None and
                not torch.isfinite(parameter.grad).all().item()
                for parameter in parameters)
        gradient_isolation[task_name] = {
            "loss": float(task_loss.detach()),
            "nonzero_gradient_tensors": counts,
            "own_nonzero_gradient_tensors": counts[task_name],
            "cross_task_nonzero_gradient_tensors": sum(
                count for name, count in counts.items()
                if name != task_name),
            "nonfinite_gradient_tensors": nonfinite,
        }
    model.zero_grad(set_to_none=True)
    maximum_cross_task_gradients = max(
        row["cross_task_nonzero_gradient_tensors"]
        for row in gradient_isolation.values())
    minimum_own_task_gradients = min(
        row["own_nonzero_gradient_tensors"]
        for row in gradient_isolation.values())
    isolation_nonfinite_gradients = sum(
        row["nonfinite_gradient_tensors"]
        for row in gradient_isolation.values())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]))
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    before_state_sha256 = state_digest(model)
    model.eval()
    with torch.no_grad():
        initial_outputs = model(relations, batch["candidate_valid"])
        initial_losses = cached_strict_router_loss(
            initial_outputs, batch["event_target"], batch["gain_target"],
            batch["beneficial_target"], batch["catastrophic_target"],
            batch["label_available"], batch["candidate_valid"],
            pairwise_margin=float(optimization["pairwise_margin"]))
        initial_metrics = evaluate(initial_outputs, batch)
    initial_total = float(initial_losses["total"])

    trace = []
    completed_steps = 0
    maximum_preclip = 0.0
    maximum_postclip = 0.0
    total_nonfinite_gradients = 0
    total_nonfinite_outputs = 0
    total_nonfinite_losses = 0
    stopped_reason = None
    for step in range(1, int(optimization["steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(relations, batch["candidate_valid"])
        losses = cached_strict_router_loss(
            outputs, batch["event_target"], batch["gain_target"],
            batch["beneficial_target"], batch["catastrophic_target"],
            batch["label_available"], batch["candidate_valid"],
            pairwise_margin=float(optimization["pairwise_margin"]))
        nonfinite_outputs = sum(
            not torch.isfinite(value.float()).all().item()
            for value in outputs.values())
        nonfinite_losses = sum(
            not math.isfinite(float(value.detach()))
            for value in losses.values())
        total_nonfinite_outputs += nonfinite_outputs
        total_nonfinite_losses += nonfinite_losses
        if nonfinite_outputs or nonfinite_losses:
            stopped_reason = "nonfinite_forward_or_loss"
            break
        losses["total"].backward()
        preclip, nonfinite_gradients, _ = gradient_diagnostics(model, 0)
        total_nonfinite_gradients += nonfinite_gradients
        maximum_preclip = max(maximum_preclip, preclip)
        if (nonfinite_gradients or not math.isfinite(preclip) or
                preclip <= 0.0 or
                preclip > float(optimization["preclip_total_l2_max"])):
            stopped_reason = "preclip_gradient_gate"
            break
        clip_max = float(optimization["global_clip_max_norm"])
        scale_gradients(model, min(1.0, clip_max / (preclip + 1e-12)))
        postclip, postclip_nonfinite, _ = gradient_diagnostics(model, 0)
        total_nonfinite_gradients += postclip_nonfinite
        maximum_postclip = max(maximum_postclip, postclip)
        if (postclip_nonfinite or not math.isfinite(postclip) or
                postclip > float(optimization["postclip_total_l2_max"])):
            stopped_reason = "postclip_gradient_gate"
            break
        optimizer.step()
        completed_steps += 1
        trace.append({
            "step": step,
            "losses": {name: float(value.detach())
                       for name, value in losses.items()},
            "preclip_total_l2": preclip,
            "postclip_total_l2": postclip,
        })

    model.eval()
    with torch.no_grad():
        final_outputs = model(relations, batch["candidate_valid"])
        final_losses = cached_strict_router_loss(
            final_outputs, batch["event_target"], batch["gain_target"],
            batch["beneficial_target"], batch["catastrophic_target"],
            batch["label_available"], batch["candidate_valid"],
            pairwise_margin=float(optimization["pairwise_margin"]))
        final_metrics = evaluate(final_outputs, batch)
    final_total = float(final_losses["total"])
    loss_ratio = final_total / initial_total
    changed_tensors = sum(
        not torch.equal(before_state[name], value)
        for name, value in model.state_dict().items())
    after_state_sha256 = state_digest(model)
    after_mismatches, _ = verify_frozen(records)

    gates = spec["gates"]
    conditions = {
        "frozen_hash_mismatches_before_max": len(before_mismatches) <= int(
            gates["frozen_hash_mismatches_max"]),
        "frozen_hash_mismatches_after_max": len(after_mismatches) <= int(
            gates["frozen_hash_mismatches_max"]),
        "batch_composition_exact": composition == Counter(
            source_spec["selection"]["composition"]),
        "distinct_sequences_exact": len(sequences) == 8,
        "task_towers_exact": len(model.towers) == int(
            gates["task_towers_exact"]),
        "shared_parameter_ids_max": shared_parameter_ids <= int(
            gates["shared_parameter_ids_max"]),
        "cross_task_nonzero_gradient_tensors_max": (
            maximum_cross_task_gradients <= int(
                gates["cross_task_nonzero_gradient_tensors_max"])),
        "own_task_nonzero_gradient_tensors_min": (
            minimum_own_task_gradients >= int(
                gates["own_task_nonzero_gradient_tensors_min"])),
        "gradient_isolation_nonfinite_max": isolation_nonfinite_gradients <= 0,
        "relation_max_abs": relation_max_abs <= float(
            gates["relation_max_abs"]),
        "permutation_error_max": permutation_error <= float(
            gates["permutation_error_max"]),
        "forbidden_module_count_max": len(forbidden_modules) <= int(
            gates["forbidden_module_count_max"]),
        "steps_exact": completed_steps == int(gates["steps_exact"]),
        "loss_ratio_max": loss_ratio <= float(gates["loss_ratio_max"]),
        "event_commit_correct": final_metrics["event_commit_correct"] == int(
            gates["event_commit_correct"]),
        "beneficial_event_best_rank_correct": final_metrics[
            "beneficial_event_best_rank_correct"] == int(
                gates["beneficial_event_best_rank_correct"]),
        "candidate_benefit_accuracy_min": final_metrics[
            "candidate_benefit_accuracy"] >= float(
                gates["candidate_benefit_accuracy_min"]),
        "candidate_catastrophe_accuracy_min": final_metrics[
            "candidate_catastrophe_accuracy"] >= float(
                gates["candidate_catastrophe_accuracy_min"]),
        "candidate_gain_mae_max": final_metrics[
            "candidate_gain_mae"] <= float(gates["candidate_gain_mae_max"]),
        "nonfinite_outputs_max": total_nonfinite_outputs <= int(
            gates["nonfinite_outputs_max"]),
        "nonfinite_losses_max": total_nonfinite_losses <= int(
            gates["nonfinite_losses_max"]),
        "nonfinite_gradients_max": total_nonfinite_gradients <= int(
            gates["nonfinite_gradients_max"]),
        "changed_trainable_tensors_min": changed_tensors >= int(
            gates["changed_trainable_tensors_min"]),
        "stopped_reason_none": stopped_reason is None,
    }
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m12-task-towers-overfit-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m12_pass_freeze_sequence_disjoint_pilot_spec_only"
                     if accepted else "m12_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "conditions": conditions,
        "batch": {
            "composition": dict(composition),
            "sequences": sorted(sequences),
            "events": len(selected_rows),
            "relations_shape": list(relations.shape),
        },
        "optimization": {
            "requested_steps": int(optimization["steps"]),
            "completed_steps": completed_steps,
            "stopped_reason": stopped_reason,
            "initial_losses": {name: float(value)
                               for name, value in initial_losses.items()},
            "final_losses": {name: float(value)
                             for name, value in final_losses.items()},
            "loss_ratio": loss_ratio,
            "maximum_preclip_total_l2": maximum_preclip,
            "maximum_postclip_total_l2": maximum_postclip,
            "nonfinite_outputs": total_nonfinite_outputs,
            "nonfinite_losses": total_nonfinite_losses,
            "nonfinite_gradients": total_nonfinite_gradients,
        },
        "metrics": {
            "initial": initial_metrics,
            "final": final_metrics,
        },
        "model": {
            "parameters": sum(parameter.numel()
                              for parameter in model.parameters()),
            "m11_parameters": int(model_spec["m11_parameters"]),
            "parameter_increase": int(model_spec["parameter_increase"]),
            "task_names": list(TASK_NAMES),
            "task_towers": len(model.towers),
            "task_hidden_dim": int(model_spec["task_hidden_dim"]),
            "shared_parameter_ids": shared_parameter_ids,
            "gradient_isolation": gradient_isolation,
            "temporal_statistics": list(TEMPORAL_STATISTICS),
            "temporal_width": TEMPORAL_WIDTH,
            "relation_max_abs": relation_max_abs,
            "permutation_error": permutation_error,
            "forbidden_modules": forbidden_modules,
            "state_sha256_before": before_state_sha256,
            "state_sha256_after": after_state_sha256,
            "changed_trainable_tensors": changed_tensors,
            "checkpoint_written": False,
        },
        "frozen": {
            "before_mismatches": before_mismatches,
            "after_mismatches": after_mismatches,
            "observed": frozen_observed,
        },
        "repository": {
            "path": str(REPOSITORY_ROOT),
            "branch": spec["repository"]["branch"],
            "commit": commit,
            "clean": True,
        },
        "inputs": {
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "runner": file_record(runner_path),
            "model": file_record(model_path),
            "relation_extractor": file_record(relation_extractor_path),
            "fixed_pool_source": file_record(fixed_pool_source_path),
        },
        "authorization": {
            "independent_integrity_audit": True,
            "sequence_disjoint_pilot_spec": accepted,
            "sequence_disjoint_pilot_execution": False,
            "formal_training": False,
            "tracking_checkpoint": False,
            "online_replay": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
    }
    args.output.mkdir(parents=True)
    trace_path = args.output / "training_trace.jsonl.gz"
    result_path = args.output / "result.json"
    manifest_path = args.output / "manifest.json"
    atomic_jsonl_gz(trace_path, trace)
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m12-task-towers-overfit-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "payload": {
            "result": file_record(result_path),
            "training_trace": file_record(trace_path),
        },
        "unauthorized_actions": {
            "checkpoint_written": False,
            "formal_training": False,
            "online_replay": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
    }
    atomic_json(manifest_path, manifest)
    for path in (trace_path, result_path, manifest_path):
        path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "optimization": result["optimization"],
        "metrics": result["metrics"],
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
