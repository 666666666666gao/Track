#!/usr/bin/env python3
"""Bound capacity test for learned candidate-specific RoI association."""

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
from lib.models.sttrack.lachtt_learned_bounded_roi_association import (  # noqa: E402
    BLOCK_FAMILY_INDICES,
    EMBEDDING_WIDTH,
    FAMILY_BLOCK_COUNTS,
    RAW_DIFFERENCE_BLOCKS,
    SCALAR_RELATION_DIM,
    LearnedBoundedRoIAssociationRouter,
    build_detached_roi_differences,
)
from lib.models.sttrack.lachtt_rich_roi_relation import (  # noqa: E402
    FAMILY_NAMES,
    PROJECTION_WIDTH,
    RICH_RELATION_DIM,
    TEMPORAL_STATISTICS,
    TEMPORAL_WIDTH,
    build_rich_roi_relations,
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
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m14b-r1-wiring-recovery-binding/v1",
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "output": str(args.output),
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    plan_audit = binding.get("pre_execution_plan_audit", {})
    if (plan_audit.get("path") !=
            "/home/SUTrack_RGBD_L/refine-logs/"
            "EXPERIMENT_AUDIT_M14B_R1_PREEXECUTION_PLAN_20260901.json" or
            plan_audit.get("sha256") !=
            "9bd43d05ec08ea43f442ea93ff3c5a1a7d601d290d7548d0c45ee64172f738ea" or
            sha256_file(Path(plan_audit["path"])) !=
            plan_audit["sha256"]):
        raise ValueError("pre-execution plan audit binding drifted")
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
    if (authorization.get("one_fixed_batch_capacity_run") is not True or
            authorization.get("pre_execution_plan_audit_passed") is not True or
            authorization.get("independent_result_audit_after_run") is not True):
        raise ValueError("binding does not authorize M14b")
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
            "independent_result_audit_after_run") is not True:
        raise ValueError("binding does not authorize the required audit")
    return binding, runner, model, commit


def bound_records(spec, source_spec, selected_native_rows, binding):
    records = frozen_records(source_spec)
    for name in ("plan", "source_batch_spec", "native_anchor_index",
                 "native_anchor_manifest"):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"], None))
    for boundary_name in ("m13b_negative_boundary",
                          "m14a_positive_boundary"):
        for name, item in spec[boundary_name].items():
            if isinstance(item, dict) and "path" in item:
                records.append((boundary_name + ":" + name,
                                Path(item["path"]), item["sha256"], None))
    for name, item in spec["m14b_invalid_attempt"].items():
        if isinstance(item, dict) and "path" in item:
            records.append(("m14b_invalid_attempt:" + name,
                            Path(item["path"]), item["sha256"], None))
    audit = binding["pre_execution_plan_audit"]
    records.append(("pre_execution_plan_audit", Path(audit["path"]),
                    audit["sha256"], None))
    for name in ("model_source", "m14a_runner_source"):
        item = spec[name]
        records.append((name, REPOSITORY_ROOT / item["path"],
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
    required_gate_keys = {
        "steps_exact", "loss_ratio_max", "event_commit_correct",
        "beneficial_event_best_rank_correct",
        "candidate_benefit_accuracy_min",
        "candidate_catastrophe_accuracy_min", "candidate_gain_mae_max",
        "raw_difference_shape", "raw_difference_max_abs",
        "raw_difference_nonfinite_max", "scalar_relation_shape",
        "scalar_relation_max_abs", "scalar_relation_nonfinite_max",
        "initial_fixed_relation_parity_error_max",
        "raw_builder_permutation_error_max",
        "initial_relation_permutation_error_max",
        "model_permutation_error_max", "projector_parameters_exact",
        "projector_bias_tensors_exact", "router_parameters_exact",
        "total_parameters_exact", "forbidden_module_count_max",
        "first_step_projector_nonzero_gradient_tensors_exact",
        "projector_changed_tensors_exact", "changed_trainable_tensors_min",
        "output_nonfinite_max", "loss_nonfinite_max",
        "gradient_nonfinite_max", "frozen_hash_mismatches_max",
    }
    missing_gate_keys = sorted(required_gate_keys - set(spec.get("gates", {})))
    if missing_gate_keys:
        raise ValueError("missing required gate keys before execution: %s" %
                         ",".join(missing_gate_keys))
    authorization = spec["authorization"]
    if (authorization.get("implementation_after_plan_audit_pass") is not True or
            authorization.get(
                "one_fixed_batch_capacity_run_after_plan_audit_pass") is not True or
            authorization.get("independent_result_audit_after_run") is not True):
        raise ValueError("spec does not authorize M14b")
    for name in (
            "sequence_disjoint_pilot_execution", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)
    if (authorization.get(
            "pre_execution_independent_plan_audit_required") is not True or
            authorization.get("sequence_disjoint_pilot_plan_if_pass") is not True):
        raise ValueError("required post-run authorization contract drifted")
    binding, runner_path, model_path, commit = validate_binding(args, spec)
    source_spec = json_file(spec["source_batch_spec"]["path"])
    if sha256_file(Path(spec["source_batch_spec"]["path"])) != \
            spec["source_batch_spec"]["sha256"]:
        raise ValueError("source batch spec drifted")
    m13b_result = json_file(
        spec["m13b_negative_boundary"]["result"]["path"])
    m13b_audit = json_file(
        spec["m13b_negative_boundary"]["audit"]["path"])
    if (m13b_result.get("accepted") is not False or
            m13b_result.get("decision") !=
            "m13b_fail_stop_without_rescan" or
            str(m13b_audit.get("integrity_verdict", "")).lower() != "pass"):
        raise ValueError("M13b negative boundary drifted")
    m14a_result = json_file(
        spec["m14a_positive_boundary"]["result"]["path"])
    m14a_audit = json_file(
        spec["m14a_positive_boundary"]["audit"]["path"])
    if (m14a_result.get("accepted") is not True or
            m14a_result.get("decision") !=
            "m14a_pass_freeze_capacity_plan_only" or
            str(m14a_audit.get("integrity_verdict", "")).lower() != "pass" or
            str(m14a_audit.get("scientific_gate", "")).lower() !=
            "pass_engineering_smoke_only"):
        raise ValueError("M14a positive boundary drifted")
    plan_audit = json_file(binding["pre_execution_plan_audit"]["path"])
    if (str(plan_audit.get("protocol_gate", "")).lower() != "pass" or
            str(plan_audit.get("integrity_verdict", "")).lower() != "pass" or
            "one_r1_fixed_batch_capacity_run_after_binding" not in
            plan_audit.get("authorization_boundary", {}).get(
                "authorized_after_protocol_pass", []) or
            plan_audit.get("original_m14b", {}).get(
                "valid_scientific_result") is not False or
            plan_audit.get("preexecution_state", {}).get(
                "r1_output_exists") is not False):
        raise ValueError("M14b-R1 pre-execution audit boundary drifted")
    invalid_attempt = spec["m14b_invalid_attempt"]
    if (invalid_attempt.get("valid_scientific_result") is not False or
            invalid_attempt.get("original_output_exists") is not False or
            Path(invalid_attempt["original_output"]).exists()):
        raise ValueError("original invalid M14b attempt boundary drifted")

    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, selected_native_rows = load_native_batch(
        spec, source_spec)
    records = bound_records(
        spec, source_spec, selected_native_rows, binding)
    before_mismatches, frozen_observed = verify_frozen(records)
    model_spec = spec["model"]
    raw_spec = spec["raw_difference"]
    differences, block_gates, scalar = build_detached_roi_differences(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(raw_spec["ema_alpha"]),
        epsilon=float(raw_spec["l2_epsilon"]),
        soft_distractor_scale=float(
            raw_spec["soft_distractor_cosine_scale"]),
    )
    fixed_relations = build_rich_roi_relations(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(raw_spec["ema_alpha"]),
        epsilon=float(raw_spec["l2_epsilon"]),
        soft_distractor_scale=float(
            raw_spec["soft_distractor_cosine_scale"]),
        base_projection_seed=int(spec["learned_projector"]["base_seed"]),
    ).detach()

    extractor_permutation = torch.tensor(
        [2, 5, 1, 4, 0, 3], dtype=torch.long)
    permuted_features = {
        name: value[:, :, extractor_permutation]
        for name, value in batch["features"].items()
    }
    permuted_differences, permuted_gates, permuted_scalar = \
        build_detached_roi_differences(
        permuted_features, batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(raw_spec["ema_alpha"]),
        epsilon=float(raw_spec["l2_epsilon"]),
        soft_distractor_scale=float(
            raw_spec["soft_distractor_cosine_scale"]),
    )
    raw_builder_permutation_error = max(
        float(torch.max(torch.abs(
            differences[:, :, extractor_permutation] -
            permuted_differences)).item()),
        float(torch.max(torch.abs(
            block_gates[:, :, extractor_permutation] -
            permuted_gates)).item()),
        float(torch.max(torch.abs(
            scalar[:, :, extractor_permutation] -
            permuted_scalar)).item()),
    )
    difference_nonfinite = int(not torch.isfinite(differences).all().item())
    scalar_nonfinite = int(not torch.isfinite(scalar).all().item())
    difference_max_abs = float(differences.abs().max().item())
    scalar_max_abs = float(scalar.abs().max().item())

    optimization = spec["optimization"]
    torch.manual_seed(int(optimization["seed"]))
    if (tuple(TEMPORAL_STATISTICS) !=
            tuple(model_spec["temporal_statistics"]) or
            TEMPORAL_WIDTH != int(model_spec["temporal_width"])):
        raise ValueError("fixed temporal pool definition drifted")
    if (tuple(raw_spec["families"]) != tuple(FAMILY_NAMES) or
            tuple(raw_spec["family_block_counts"]) != FAMILY_BLOCK_COUNTS or
            tuple(raw_spec["family_index_per_block"]) !=
            BLOCK_FAMILY_INDICES or
            int(spec["learned_projector"]["input_dim"]) != EMBEDDING_WIDTH or
            int(spec["learned_projector"]["output_dim"]) !=
            PROJECTION_WIDTH or RAW_DIFFERENCE_BLOCKS != 16 or
            SCALAR_RELATION_DIM != 49 or
            int(model_spec["relation_dim"]) != RICH_RELATION_DIM):
        raise ValueError("learned bounded relation contract drifted")
    model = LearnedBoundedRoIAssociationRouter(
        hidden_dim=int(model_spec["router_hidden_dim"]),
        residual_scale=float(model_spec["residual_scale"]),
        base_projection_seed=int(spec["learned_projector"]["base_seed"]))
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    router_parameters = sum(
        parameter.numel() for parameter in model.router.parameters()
        if parameter.requires_grad)
    projector_parameters = sum(
        parameter.numel() for projector in model.projectors
        for parameter in projector.parameters() if parameter.requires_grad)
    projector_bias_tensors = sum(
        projector.bias is not None for projector in model.projectors)
    if (parameter_count != int(model_spec["total_parameters"]) or
            router_parameters != int(model_spec["router_parameters"]) or
            projector_parameters != int(
                spec["learned_projector"]["parameters"])):
        raise ValueError("model parameter count drifted")
    forbidden_fragments = tuple(model_spec["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if any(fragment in type(module).__name__
               for fragment in forbidden_fragments)]
    initial_relations = model.project_relations(
        differences, block_gates, scalar)
    fixed_parity_error = float(torch.max(torch.abs(
        initial_relations.detach() - fixed_relations)).item())
    permuted_initial_relations = model.project_relations(
        permuted_differences, permuted_gates, permuted_scalar)
    initial_relation_permutation_error = float(torch.max(torch.abs(
        initial_relations[:, :, extractor_permutation].detach() -
        permuted_initial_relations.detach())).item())
    permutation = torch.tensor([2, 5, 1, 4, 0, 3], dtype=torch.long)
    inverse_permutation = torch.argsort(permutation)
    model.eval()
    with torch.no_grad():
        reference_outputs = model(
            differences, block_gates, scalar, batch["candidate_valid"])
        permuted_outputs = model(
            differences[:, :, permutation],
            block_gates[:, :, permutation], scalar[:, :, permutation],
            batch["candidate_valid"][:, permutation])
    permutation_errors = []
    for name, value in reference_outputs.items():
        if value.ndim == 2:
            restored = permuted_outputs[name][:, inverse_permutation]
        else:
            restored = permuted_outputs[name]
        permutation_errors.append(float(torch.max(torch.abs(
            value - restored)).item()))
    model_permutation_error = max(permutation_errors)
    initial_relation_max_abs = float(initial_relations.detach().abs().max().item())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]))
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    before_state_sha256 = state_digest(model)
    model.eval()
    with torch.no_grad():
        initial_outputs = model(
            differences, block_gates, scalar, batch["candidate_valid"])
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
    first_step_projector_nonzero_gradient_tensors = 0
    stopped_reason = None
    for step in range(1, int(optimization["steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            differences, block_gates, scalar, batch["candidate_valid"])
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
        if step == 1:
            first_step_projector_nonzero_gradient_tensors = sum(
                parameter.grad is not None and
                torch.isfinite(parameter.grad).all().item() and
                float(parameter.grad.detach().abs().max().item()) > 0.0
                for projector in model.projectors
                for parameter in projector.parameters())
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
        final_outputs = model(
            differences, block_gates, scalar, batch["candidate_valid"])
        final_relations = model.project_relations(
            differences, block_gates, scalar)
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
    changed_projector_tensors = sum(
        name.startswith("projectors.") and
        not torch.equal(before_state[name], value)
        for name, value in model.state_dict().items())
    final_relation_max_abs = float(final_relations.abs().max().item())
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
        "raw_difference_shape_exact": list(differences.shape) == gates[
            "raw_difference_shape"],
        "raw_difference_max_abs": difference_max_abs <= float(
            gates["raw_difference_max_abs"]),
        "raw_difference_nonfinite_max": difference_nonfinite <= int(
            gates["raw_difference_nonfinite_max"]),
        "scalar_relation_shape_exact": list(scalar.shape) == gates[
            "scalar_relation_shape"],
        "scalar_relation_max_abs": scalar_max_abs <= float(
            gates["scalar_relation_max_abs"]),
        "scalar_relation_nonfinite_max": scalar_nonfinite <= int(
            gates["scalar_relation_nonfinite_max"]),
        "initial_fixed_relation_parity_error_max":
            fixed_parity_error <= float(
                gates["initial_fixed_relation_parity_error_max"]),
        "raw_builder_permutation_error_max":
            raw_builder_permutation_error <= float(
                gates["raw_builder_permutation_error_max"]),
        "initial_relation_permutation_error_max":
            initial_relation_permutation_error <= float(
                gates["initial_relation_permutation_error_max"]),
        "model_permutation_error_max": model_permutation_error <= float(
            gates["model_permutation_error_max"]),
        "projector_parameters_exact": projector_parameters == int(
            gates["projector_parameters_exact"]),
        "projector_bias_tensors_exact": projector_bias_tensors == int(
            gates["projector_bias_tensors_exact"]),
        "router_parameters_exact": router_parameters == int(
            gates["router_parameters_exact"]),
        "total_parameters_exact": parameter_count == int(
            gates["total_parameters_exact"]),
        "forbidden_module_count_max": len(forbidden_modules) <= int(
            gates["forbidden_module_count_max"]),
        "first_step_projector_nonzero_gradient_tensors_exact":
            first_step_projector_nonzero_gradient_tensors == int(
                gates[
                    "first_step_projector_nonzero_gradient_tensors_exact"]),
        "projector_changed_tensors_exact":
            changed_projector_tensors == int(
                gates["projector_changed_tensors_exact"]),
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
        "output_nonfinite_max": total_nonfinite_outputs <= int(
            gates["output_nonfinite_max"]),
        "loss_nonfinite_max": total_nonfinite_losses <= int(
            gates["loss_nonfinite_max"]),
        "gradient_nonfinite_max": total_nonfinite_gradients <= int(
            gates["gradient_nonfinite_max"]),
        "changed_trainable_tensors_min": changed_tensors >= int(
            gates["changed_trainable_tensors_min"]),
        "stopped_reason_none": stopped_reason is None,
    }
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m14b-r1-learned-bounded-roi-capacity-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m14b_r1_pass_freeze_sequence_disjoint_pilot_plan_only"
                     if accepted else "m14b_r1_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "conditions": conditions,
        "batch": {
            "composition": dict(composition),
            "sequences": sorted(sequences),
            "events": len(selected_rows),
            "raw_differences_shape": list(differences.shape),
            "block_gates_shape": list(block_gates.shape),
            "scalar_relation_shape": list(scalar.shape),
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
            "parameters": parameter_count,
            "router_parameters": router_parameters,
            "projector_parameters": projector_parameters,
            "projector_bias_tensors": projector_bias_tensors,
            "temporal_statistics": list(TEMPORAL_STATISTICS),
            "temporal_width": TEMPORAL_WIDTH,
            "relation_dim": RICH_RELATION_DIM,
            "initial_relation_max_abs": initial_relation_max_abs,
            "final_relation_max_abs": final_relation_max_abs,
            "raw_difference_max_abs": difference_max_abs,
            "raw_difference_nonfinite": difference_nonfinite,
            "scalar_relation_max_abs": scalar_max_abs,
            "scalar_relation_nonfinite": scalar_nonfinite,
            "initial_fixed_relation_parity_error": fixed_parity_error,
            "raw_builder_permutation_error":
                raw_builder_permutation_error,
            "initial_relation_permutation_error":
                initial_relation_permutation_error,
            "model_permutation_error": model_permutation_error,
            "families": list(FAMILY_NAMES),
            "family_block_counts": list(FAMILY_BLOCK_COUNTS),
            "block_family_indices": list(BLOCK_FAMILY_INDICES),
            "difference_blocks": RAW_DIFFERENCE_BLOCKS,
            "projection_width": PROJECTION_WIDTH,
            "forbidden_modules": forbidden_modules,
            "first_step_projector_nonzero_gradient_tensors":
                first_step_projector_nonzero_gradient_tensors,
            "state_sha256_before": before_state_sha256,
            "state_sha256_after": after_state_sha256,
            "changed_trainable_tensors": changed_tensors,
            "changed_projector_tensors": changed_projector_tensors,
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
        },
        "authorization": {
            "independent_result_audit": True,
            "sequence_disjoint_pilot_plan": accepted,
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
        "schema": "sttrack-lachtt-m14b-r1-learned-bounded-roi-capacity-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "payload": {
            "result": file_record(result_path),
            "training_trace": file_record(trace_path),
        },
        "unauthorized_actions": {
            "checkpoint_written": False,
            "sequence_disjoint_pilot_execution": False,
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
