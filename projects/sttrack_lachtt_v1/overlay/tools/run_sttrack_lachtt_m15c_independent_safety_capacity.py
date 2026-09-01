#!/usr/bin/env python3
"""Fixed eight-event capacity test for independent utility/safety paths."""

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
from lib.models.sttrack.lachtt_independent_utility_safety import (  # noqa: E402
    IndependentUtilitySafetyRouter,
)
from lib.models.sttrack.lachtt_learned_bounded_roi_association import (  # noqa: E402
    build_detached_roi_differences,
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
from tools.smoke_sttrack_lachtt_m15b_r2_independent_utility_safety import (  # noqa: E402
    changed_named,
    count_nonzero_finite_gradients,
    dependency_records,
    file_record,
    git_output,
    json_file,
    load_native_batch,
    load_trajectory_targets,
    permutation_errors,
)


PREAUDIT_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_AUDIT_M15C_R1_PREAUDIT_SCHEMA_WIRING_RECOVERY_PREEXEC_20260901.json")
PREAUDIT_SHA256 = \
    "b94f5b09eebb046003724896364e6b52534a46ea1ebc827ef1c8fed760019d58"
HELPER_PATH = REPOSITORY_ROOT / \
    "tools/smoke_sttrack_lachtt_m15b_r2_independent_utility_safety.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def atomic_jsonl_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(
                    row, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    model = REPOSITORY_ROOT / \
        "lib/models/sttrack/lachtt_independent_utility_safety.py"
    commit = git_output("rev-parse", "HEAD")
    m15b_result = json_file(spec["m15b_r3_boundary"]["result"]["path"])
    expected_helper = m15b_result["runner"]
    expected = {
        "schema": "sttrack-lachtt-m15c-r1-preaudit-schema-wiring-recovery-binding/v1",
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "plan_path": spec["plan"]["path"],
        "plan_sha256": spec["plan"]["sha256"],
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "model_path": str(model.resolve()),
        "model_sha256": sha256_file(model),
        "engineering_helper": file_record(HELPER_PATH),
        "labeled_actions": file_record(spec["labeled_actions"]["path"]),
        "m15a_target_closure": file_record(
            spec["m15a_target_closure"]["path"]),
        "native_anchor_index": file_record(
            spec["relation_evidence"]["native_anchor_index"]["path"]),
        "native_anchor_manifest": file_record(
            spec["relation_evidence"]["native_anchor_manifest"]["path"]),
        "dependency_records": dependency_records(spec),
        "pre_execution_plan_audit": {
            "path": str(PREAUDIT_PATH), "sha256": PREAUDIT_SHA256,
        },
        "output": str(args.output),
        "output_root_absent_at_binding": True,
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    if (expected_helper["path"] != str(HELPER_PATH) or
            expected_helper["sha256"] != sha256_file(HELPER_PATH)):
        raise ValueError("M15b helper identity drifted")
    if sha256_file(PREAUDIT_PATH) != PREAUDIT_SHA256:
        raise ValueError("pre-execution audit hash drifted")
    audit = json_file(PREAUDIT_PATH)
    expected_authorization = [
        "apply exact wiring-only runner patch: compare authorization_boundary.authorized_after_pass to the four frozen values",
        "update R1 preaudit/schema/decision/output binding strings and args.output == R1 spec.output.root binding only",
        "create a new read-only R1 binding after implementation",
        "run exactly one R1 200-step fixed-batch capacity execution after binding",
        "perform independent post-result audit",
    ]
    if (str(audit.get("overall_verdict", "")).lower() != "pass" or
            str(audit.get("integrity_verdict", "")).lower() != "pass" or
            audit.get("authorization_boundary", {}).get(
                "authorized_after_pass") != expected_authorization):
        raise ValueError("pre-execution audit authorization drifted")
    if (git_output("branch", "--show-current") !=
            spec["repository"]["branch"] or
            git_output("status", "--porcelain")):
        raise ValueError("repository state drifted")
    if subprocess.run([
            "git", "-C", str(REPOSITORY_ROOT), "merge-base",
            "--is-ancestor", spec["repository"]["base_commit"], commit,
    ], check=False).returncode != 0:
        raise ValueError("implementation is outside frozen ancestry")
    authorizations = binding.get("authorizations", {})
    for name in (
            "pre_execution_plan_audit_passed", "one_capacity_execution",
            "independent_result_audit_after_run",
            "sequence_disjoint_plan_if_pass"):
        if authorizations.get(name) is not True:
            raise ValueError("binding authorization missing: %s" % name)
    for name in (
            "second_m15c_capacity_execution", "sequence_disjoint_pilot",
            "formal_training", "tracking_checkpoint", "online_replay",
            "depthtrack_test", "cdtb", "vot_low22", "vot_full127",
            "qwen", "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, model.resolve(), commit


def capacity_frozen_records(spec, source_spec, native_records, binding):
    records = frozen_records(source_spec)
    for name, item in (
            ("plan", spec["plan"]),
            ("source_batch_spec", spec["source_batch_spec"]),
            ("labeled_actions", spec["labeled_actions"]),
            ("m15a_target_closure", spec["m15a_target_closure"]),
            ("m15a_result", spec["m15a_target_closure"]["result"]),
            ("m15a_manifest", spec["m15a_target_closure"]["manifest"]),
            ("m15a_result_audit",
             spec["m15a_target_closure"]["result_audit"]),
            ("m14a_contract_spec",
             spec["relation_evidence"]["m14a_contract_spec"]),
            ("native_anchor_index",
             spec["relation_evidence"]["native_anchor_index"]),
            ("native_anchor_manifest",
             spec["relation_evidence"]["native_anchor_manifest"]),
            ("m15b_r3_result", spec["m15b_r3_boundary"]["result"]),
            ("m15b_r3_manifest", spec["m15b_r3_boundary"]["manifest"]),
            ("m15b_r3_result_audit",
             spec["m15b_r3_boundary"]["result_audit"])):
        records.append((name, Path(item["path"]), item["sha256"], None))
    for item in spec["relation_evidence"]["dependencies"]:
        records.append((
            "dependency:" + item["path"], REPOSITORY_ROOT / item["path"],
            item["sha256"], None))
    helper = file_record(HELPER_PATH)
    records.append(("engineering_helper", HELPER_PATH,
                    helper["sha256"], helper["bytes"]))
    records.append(("pre_execution_plan_audit", PREAUDIT_PATH,
                    PREAUDIT_SHA256, None))
    for item in native_records:
        records.append(("native_anchor:" + item["sequence"],
                        Path(item["path"]), item["sha256"], item["bytes"]))
    if binding.get("selected_native_payloads") != native_records:
        raise ValueError("binding selected native payloads drifted")
    return records


def forward_losses(model, differences, block_gates, scalar, batch,
                   trajectory_target, trajectory_available, spec):
    outputs = model(
        differences, block_gates, scalar, batch["candidate_valid"])
    strict_outputs = {
        "event_commit_logit": outputs["event_commit_logit"],
        "candidate_rank_logits": outputs["candidate_rank_logits"],
        "candidate_benefit_logits": outputs["candidate_benefit_logits"],
        "candidate_catastrophe_logits":
            outputs["candidate_catastrophe_logits"],
        "candidate_h10_gain": outputs["candidate_trajectory"][:, :, 2, 2],
    }
    strict = cached_strict_router_loss(
        strict_outputs, batch["event_target"], batch["gain_target"],
        batch["beneficial_target"], batch["catastrophic_target"],
        batch["label_available"], batch["candidate_valid"],
        pairwise_margin=float(spec["loss"]["pairwise_margin"]),
    )
    mask = trajectory_available.unsqueeze(-1).expand_as(trajectory_target)
    trajectory_l1 = (
        torch.abs(outputs["candidate_trajectory"] - trajectory_target) *
        mask.float()).sum() / mask.float().sum()
    total = strict["total"] + \
        float(spec["loss"]["trajectory_weight"]) * trajectory_l1
    losses = {name: value for name, value in strict.items()}
    losses["trajectory_l1"] = trajectory_l1
    losses["total_with_trajectory"] = total
    return outputs, losses


def evaluate(outputs, losses, batch, trajectory_target,
             trajectory_available):
    available = batch["label_available"] & batch["candidate_valid"]
    event_prediction = outputs["event_commit_logit"] >= 0.0
    event_correct = int((event_prediction == batch["event_target"]).sum().item())
    rank_correct = 0
    rank_events = 0
    for event_index in range(available.shape[0]):
        if not bool(batch["event_target"][event_index]):
            continue
        beneficial = torch.nonzero(
            available[event_index] & batch["beneficial_target"][event_index],
            as_tuple=False).flatten()
        if beneficial.numel() == 0:
            continue
        rank_events += 1
        best = beneficial[
            batch["gain_target"][event_index, beneficial].argmax()]
        predicted = outputs["candidate_rank_logits"][event_index].masked_fill(
            ~available[event_index], -float("inf")).argmax()
        rank_correct += int(int(predicted) == int(best))
    benefit_prediction = outputs["candidate_benefit_logits"] >= 0.0
    benefit_correct = int((
        (benefit_prediction == batch["beneficial_target"]) & available
    ).sum().item())
    catastrophe_prediction = outputs["candidate_catastrophe_logits"] >= 0.0
    catastrophic = batch["catastrophic_target"] & available
    noncatastrophic = ~batch["catastrophic_target"] & available
    catastrophe_tp = int((catastrophe_prediction & catastrophic).sum().item())
    catastrophe_tn = int((~catastrophe_prediction & noncatastrophic).sum().item())
    trajectory_error = torch.abs(
        outputs["candidate_trajectory"] - trajectory_target)
    trajectory_mask = trajectory_available.unsqueeze(-1).expand_as(
        trajectory_error)
    overall = float(trajectory_error[trajectory_mask].mean().item())
    by_horizon = [float(trajectory_error[:, :, index][
        trajectory_mask[:, :, index]].mean().item()) for index in range(3)]
    by_metric = [float(trajectory_error[:, :, :, index][
        trajectory_mask[:, :, :, index]].mean().item()) for index in range(5)]
    h10_gain = float(trajectory_error[:, :, 2, 2][
        trajectory_available[:, :, 2]].mean().item())
    return {
        "losses": {name: float(value.detach())
                   for name, value in losses.items()},
        "event_commit_correct": event_correct,
        "event_commit_total": int(batch["event_target"].numel()),
        "conditional_rank_correct": rank_correct,
        "conditional_rank_total": rank_events,
        "benefit_correct": benefit_correct,
        "benefit_total": int(available.sum().item()),
        "benefit_accuracy": benefit_correct / float(available.sum().item()),
        "catastrophe_true_positive": catastrophe_tp,
        "catastrophe_positive_total": int(catastrophic.sum().item()),
        "catastrophe_true_negative": catastrophe_tn,
        "catastrophe_negative_total": int(noncatastrophic.sum().item()),
        "catastrophe_accuracy": (catastrophe_tp + catastrophe_tn) /
            float(available.sum().item()),
        "trajectory_overall_mae": overall,
        "trajectory_horizon_mae": dict(zip(("3", "5", "10"), by_horizon)),
        "trajectory_metric_mae": dict(zip((
            "branch_mean_iou", "public_mean_iou", "gain",
            "low_overlap_fraction", "trailing_low_run_fraction"),
            by_metric)),
        "h10_gain_mae": h10_gain,
    }


def trace_row(step, metrics, preclip, postclip, nonfinite, executed):
    return {
        "step": int(step),
        "total_loss": metrics["losses"]["total_with_trajectory"],
        "strict_total_loss": metrics["losses"]["total"],
        "trajectory_l1": metrics["losses"]["trajectory_l1"],
        "preclip_total_l2": float(preclip),
        "postclip_total_l2": float(postclip),
        "nonfinite_gradients": int(nonfinite),
        "optimizer_step_executed": bool(executed),
    }


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    if args.output != Path(spec["output"]["root"]).resolve():
        raise ValueError("output root drifted from M15c spec")
    if (spec.get("complete") is not True or
            spec.get("created_before_implementation_and_execution") is not True):
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    for name in (
            "repeat_independent_preexecution_audit_required",
            "implementation_after_preexecution_audit_pass",
            "one_capacity_execution_after_binding",
            "independent_result_audit_after_run",
            "sequence_disjoint_plan_if_pass"):
        if authorization.get(name) is not True:
            raise ValueError("spec authorization missing: %s" % name)
    for name in (
            "second_m15c_capacity_execution", "sequence_disjoint_pilot",
            "formal_training", "tracking_checkpoint", "online_replay",
            "depthtrack_test", "cdtb", "vot_low22", "vot_full127",
            "qwen", "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)

    binding, runner_path, model_path, commit = validate_binding(args, spec)
    source_spec = json_file(spec["source_batch_spec"]["path"])
    if sha256_file(Path(spec["source_batch_spec"]["path"])) != \
            spec["source_batch_spec"]["sha256"]:
        raise ValueError("source batch spec drifted")
    m15b_result = json_file(spec["m15b_r3_boundary"]["result"]["path"])
    m15b_audit = json_file(
        spec["m15b_r3_boundary"]["result_audit"]["path"])
    if (m15b_result.get("accepted") is not True or
            m15b_result.get("decision") !=
            "m15b_r3_pass_freeze_m15c_plan_only" or
            str(m15b_audit.get("overall_verdict", "")).lower() != "pass" or
            str(m15b_audit.get("integrity_verdict", "")).lower() != "pass"):
        raise ValueError("M15b-R3 accepted boundary drifted")

    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, native_records = load_native_batch(
        spec, source_spec)
    trajectory_target, trajectory_available, target_rows = \
        load_trajectory_targets(spec, source_spec)
    records = capacity_frozen_records(
        spec, source_spec, native_records, binding)
    before_mismatches, frozen_observed = verify_frozen(records)

    builder = spec["relation_evidence"]["builder_parameters"]
    differences, block_gates, scalar = build_detached_roi_differences(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(builder["ema_alpha"]),
        epsilon=float(builder["l2_epsilon"]),
        soft_distractor_scale=float(builder["soft_distractor_scale"]),
        native_anchor_top_k=int(builder["native_anchor_top_k"]),
        depth_missing_floor=float(builder["depth_missing_floor"]),
    )
    seed = int(spec["optimization"]["seed"])
    torch.manual_seed(seed)
    architecture = spec["architecture"]
    model = IndependentUtilitySafetyRouter(
        hidden_dim=int(architecture["hidden_dim"]), residual_scale=0.1,
        base_projection_seed=int(builder["base_projection_seed"]),
    )
    utility_parameters = list(model.utility_parameters())
    safety_parameters = list(model.safety_parameters())
    parameter_id_intersection = len(
        {id(value) for value in utility_parameters} &
        {id(value) for value in safety_parameters})
    parameter_counts = {
        "utility_projectors": sum(
            value.numel() for value in model.utility_projectors.parameters()),
        "utility_router": sum(
            value.numel() for value in model.utility_router.parameters()),
        "safety_projectors": sum(
            value.numel() for value in model.safety_projectors.parameters()),
        "safety_critic": sum(
            value.numel() for value in model.safety_critic.parameters()),
        "total": sum(value.numel() for value in model.parameters()
                     if value.requires_grad),
    }
    forbidden_fragments = tuple(architecture["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if any(fragment in type(module).__name__
               for fragment in forbidden_fragments)
    ]
    initial_state = {name: value.detach().clone()
                     for name, value in model.state_dict().items()}
    initial_state_sha256 = state_digest(model)
    predecessor_initial_sha256 = m15b_result["state_sha256"]["before"]
    model.eval()
    with torch.no_grad():
        initial_outputs, initial_losses = forward_losses(
            model, differences, block_gates, scalar, batch,
            trajectory_target, trajectory_available, spec)
        initial_metrics = evaluate(
            initial_outputs, initial_losses, batch,
            trajectory_target, trajectory_available)
    trace = [trace_row(0, initial_metrics, 0.0, 0.0, 0, False)]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["optimization"]["learning_rate"]),
        weight_decay=float(spec["optimization"]["weight_decay"]))
    utility_projector_coverage = set()
    safety_projector_coverage = set()
    utility_nonprojector_gradient_seen = False
    safety_nonprojector_gradient_seen = False
    steps_completed = 0
    gradient_failure = None
    preclip_values, postclip_values = [], []
    model.train()
    for step in range(1, int(spec["optimization"]["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, losses = forward_losses(
            model, differences, block_gates, scalar, batch,
            trajectory_target, trajectory_available, spec)
        losses["total_with_trajectory"].backward()
        for index, projector in enumerate(model.utility_projectors):
            if count_nonzero_finite_gradients(projector.parameters()) > 0:
                utility_projector_coverage.add(index)
        for index, projector in enumerate(model.safety_projectors):
            if count_nonzero_finite_gradients(projector.parameters()) > 0:
                safety_projector_coverage.add(index)
        utility_nonprojector_gradient_seen |= (
            count_nonzero_finite_gradients(
                model.utility_router.parameters()) > 0)
        safety_nonprojector_gradient_seen |= (
            count_nonzero_finite_gradients(
                model.safety_critic.parameters()) > 0)
        preclip, nonfinite, _ = gradient_diagnostics(model, 0)
        preclip_safe = (
            nonfinite == 0 and math.isfinite(preclip) and preclip > 0.0 and
            preclip <= float(spec["optimization"]["preclip_total_l2_max"]))
        if not preclip_safe:
            gradient_failure = {
                "step": step, "phase": "preclip", "norm": preclip,
                "nonfinite": nonfinite,
            }
            break
        maximum = float(spec["optimization"]["global_gradient_clip"])
        scale_gradients(model, min(1.0, maximum / (preclip + 1e-12)))
        postclip, post_nonfinite, _ = gradient_diagnostics(model, 0)
        postclip_safe = (
            post_nonfinite == 0 and math.isfinite(postclip) and
            postclip <= float(spec["optimization"]["postclip_total_l2_max"]))
        if not postclip_safe:
            gradient_failure = {
                "step": step, "phase": "postclip", "norm": postclip,
                "nonfinite": post_nonfinite,
            }
            break
        optimizer.step()
        steps_completed = step
        preclip_values.append(preclip)
        postclip_values.append(postclip)
        model.eval()
        with torch.no_grad():
            step_outputs, step_losses = forward_losses(
                model, differences, block_gates, scalar, batch,
                trajectory_target, trajectory_available, spec)
            step_metrics = evaluate(
                step_outputs, step_losses, batch,
                trajectory_target, trajectory_available)
        trace.append(trace_row(
            step, step_metrics, preclip, postclip, nonfinite, True))
        model.train()

    model.eval()
    with torch.no_grad():
        final_outputs, final_losses = forward_losses(
            model, differences, block_gates, scalar, batch,
            trajectory_target, trajectory_available, spec)
        final_metrics = evaluate(
            final_outputs, final_losses, batch,
            trajectory_target, trajectory_available)
    loss_ratio = (
        final_metrics["losses"]["total_with_trajectory"] /
        initial_metrics["losses"]["total_with_trajectory"])
    event_permutation_error, candidate_permutation_error, \
        permutation_details, permutation = permutation_errors(
            model, differences, block_gates, scalar,
            batch["candidate_valid"], seed)
    current_state = model.state_dict()
    changes = {
        "utility_projectors": changed_named(
            initial_state, current_state, "utility_projectors."),
        "utility_router": changed_named(
            initial_state, current_state, "utility_router."),
        "safety_projectors": changed_named(
            initial_state, current_state, "safety_projectors."),
        "safety_critic": changed_named(
            initial_state, current_state, "safety_critic."),
    }
    final_state_sha256 = state_digest(model)
    after_mismatches, _ = verify_frozen(records)
    gates = spec["gates"]
    horizon_mae = list(final_metrics["trajectory_horizon_mae"].values())
    metric_mae = list(final_metrics["trajectory_metric_mae"].values())
    conditions = {
        "source_hashes_before": len(before_mismatches) == 0,
        "source_hashes_after": len(after_mismatches) == 0,
        "batch_composition": composition == Counter(
            source_spec["selection"]["composition"]),
        "distinct_sequences": len(sequences) == 8,
        "target_rows": target_rows == 144,
        "target_availability": trajectory_available.all().item(),
        "fresh_initial_state": initial_state_sha256 ==
            predecessor_initial_sha256,
        "parameter_counts": parameter_counts == {
            "utility_projectors": 36864, "utility_router": 44679,
            "safety_projectors": 36864, "safety_critic": 40346,
            "total": int(gates["total_parameters_exact"]),
        },
        "parameter_id_intersection": parameter_id_intersection == 0,
        "forbidden_modules": len(forbidden_modules) == 0,
        "steps_completed": steps_completed == int(
            gates["steps_completed_exact"]),
        "gradient_safety": gradient_failure is None,
        "trace_rows": len(trace) == int(spec["training_trace"]["rows_exact"]),
        "utility_projector_gradient_coverage":
            len(utility_projector_coverage) == 6,
        "safety_projector_gradient_coverage":
            len(safety_projector_coverage) == 6,
        "utility_nonprojector_gradient": utility_nonprojector_gradient_seen,
        "safety_nonprojector_gradient": safety_nonprojector_gradient_seen,
        "utility_projectors_changed": changes["utility_projectors"] == 6,
        "safety_projectors_changed": changes["safety_projectors"] == 6,
        "utility_nonprojectors_changed": changes["utility_router"] >= 1,
        "safety_nonprojectors_changed": changes["safety_critic"] >= 1,
        "loss_ratio": loss_ratio <= float(gates["loss_ratio_max"]),
        "event_commit": final_metrics["event_commit_correct"] == int(
            gates["event_commit_correct_exact"]),
        "conditional_rank": final_metrics["conditional_rank_correct"] == int(
            gates["conditional_rank_correct_exact"]),
        "benefit_accuracy": final_metrics["benefit_accuracy"] >= float(
            gates["benefit_accuracy_min"]),
        "catastrophe_true_positive":
            final_metrics["catastrophe_true_positive"] == int(
                gates["catastrophe_true_positive_exact"]),
        "catastrophe_true_negative":
            final_metrics["catastrophe_true_negative"] == int(
                gates["catastrophe_true_negative_exact"]),
        "catastrophe_accuracy": final_metrics["catastrophe_accuracy"] >= float(
            gates["catastrophe_accuracy_min"]),
        "trajectory_overall_mae":
            final_metrics["trajectory_overall_mae"] <= float(
                gates["trajectory_overall_mae_max"]),
        "trajectory_horizon_mae": max(horizon_mae) <= float(
            gates["trajectory_each_horizon_mae_max"]),
        "trajectory_metric_mae": max(metric_mae) <= float(
            gates["trajectory_each_metric_mae_max"]),
        "h10_gain_mae": final_metrics["h10_gain_mae"] <= float(
            gates["h10_gain_mae_max"]),
        "event_permutation": event_permutation_error <= float(
            gates["event_permutation_error_max"]),
        "candidate_permutation": candidate_permutation_error <= float(
            gates["candidate_permutation_error_max"]),
        "checkpoint_count": int(gates["checkpoint_count_exact"]) == 0,
        "output_file_set_preregistered": sorted(spec["output"]["files"]) ==
            sorted(gates["output_file_set_exact"]),
    }
    accepted = all(conditions.values())
    failed_conditions = sorted(
        name for name, passed in conditions.items() if not passed)
    result = {
        "schema": "sttrack-lachtt-m15c-r1-preaudit-schema-wiring-recovery-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m15c_r1_pass_freeze_sequence_disjoint_plan_only"
                     if accepted else "m15c_r1_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "source_hashes": {
            "labeled_actions": spec["labeled_actions"]["sha256"],
            "m15a_target_closure": spec["m15a_target_closure"]["sha256"],
            "m15b_r3_result": spec["m15b_r3_boundary"]["result"]["sha256"],
            "native_anchor_index": spec["relation_evidence"]
                ["native_anchor_index"]["sha256"],
            "native_anchor_manifest": spec["relation_evidence"]
                ["native_anchor_manifest"]["sha256"],
        },
        "repository": {
            "path": str(REPOSITORY_ROOT), "commit": commit,
            "branch": spec["repository"]["branch"], "clean": True,
        },
        "runner": file_record(runner_path),
        "model": file_record(model_path),
        "engineering_helper": file_record(HELPER_PATH),
        "input_counts": {
            "events": len(selected_rows), "candidates": 48,
            "target_rows": target_rows,
            "available_horizon_records": int(
                trajectory_available.sum().item()),
            "composition": dict(composition),
            "sequences": sorted(sequences),
        },
        "input_shapes": {
            "raw_difference": list(differences.shape),
            "block_gate": list(block_gates.shape),
            "scalar": list(scalar.shape),
            "trajectory_target": list(trajectory_target.shape),
        },
        "parameter_counts": parameter_counts,
        "parameter_id_intersection": parameter_id_intersection,
        "training": {
            "seed": seed,
            "steps_requested": int(spec["optimization"]["steps"]),
            "steps_completed": steps_completed,
            "gradient_failure": gradient_failure,
            "trace_rows": len(trace),
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_ratio": loss_ratio,
        "gradient_safety": {
            "preclip_max": max(preclip_values) if preclip_values else None,
            "postclip_max": max(postclip_values) if postclip_values else None,
            "all_steps_safe": gradient_failure is None,
        },
        "projector_gradient_coverage": {
            "utility": sorted(utility_projector_coverage),
            "safety": sorted(safety_projector_coverage),
        },
        "projector_changes": {
            "utility": changes["utility_projectors"],
            "safety": changes["safety_projectors"],
        },
        "nonprojector_changes": {
            "utility": changes["utility_router"],
            "safety": changes["safety_critic"],
        },
        "state_sha256": {
            "initial": initial_state_sha256,
            "m15b_predecessor_initial": predecessor_initial_sha256,
            "final": final_state_sha256,
        },
        "permutation_errors": {
            "permutation": permutation.tolist(),
            "event": event_permutation_error,
            "candidate": candidate_permutation_error,
            "details": permutation_details,
        },
        "forbidden_module_count": len(forbidden_modules),
        "forbidden_modules": forbidden_modules,
        "checkpoint_count": 0,
        "conditions": conditions,
        "failed_conditions": failed_conditions,
        "frozen": {
            "before_mismatches": before_mismatches,
            "after_mismatches": after_mismatches,
            "observed": frozen_observed,
        },
        "authorization": {
            "independent_result_audit": True,
            "sequence_disjoint_plan": accepted,
            "sequence_disjoint_execution": False,
            "second_m15c_capacity_execution": False,
            "tracking_checkpoint": False,
            "depthtrack_test": False, "cdtb": False,
            "vot_low22": False, "vot_full127": False, "qwen": False,
        },
    }

    args.output.mkdir(parents=True)
    trace_path = args.output / "training_trace.jsonl.gz"
    atomic_jsonl_gz(trace_path, trace)
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m15c-r1-preaudit-schema-wiring-recovery-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "identity": {
            "plan": file_record(spec["plan"]["path"]),
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "pre_execution_audit": file_record(PREAUDIT_PATH),
            "repository_commit": commit,
            "runner": file_record(runner_path),
            "model": file_record(model_path),
            "engineering_helper": file_record(HELPER_PATH),
            "dependencies": dependency_records(spec),
            "labeled_actions": file_record(spec["labeled_actions"]["path"]),
            "m15a_target_closure": file_record(
                spec["m15a_target_closure"]["path"]),
            "selected_native_payloads": native_records,
        },
        "payload": {
            "result": file_record(result_path),
            "training_trace": file_record(trace_path),
        },
        "unauthorized_actions": {
            "checkpoint_written": False,
            "second_capacity_run": False,
            "sequence_disjoint_execution": False,
            "formal_training": False,
            "depthtrack_test": False, "cdtb": False,
            "vot_low22": False, "vot_full127": False, "qwen": False,
        },
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)
    actual_files = sorted(path.name for path in args.output.iterdir())
    if actual_files != sorted(spec["output"]["files"]):
        raise RuntimeError("output file set drifted")
    for path in (trace_path, result_path, manifest_path):
        path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "failed_conditions": failed_conditions,
        "steps_completed": steps_completed,
        "loss_ratio": loss_ratio,
        "final_metrics": final_metrics,
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
        "training_trace": file_record(trace_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
