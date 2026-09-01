#!/usr/bin/env python3
"""Exact-capacity replay for canonical candidate-role ordering.

This runner changes only the parameter-free M16a role canonicalization.  The
data, relation builder, model parameters, loss, optimizer, and capacity gates
are loaded from the frozen M15c-R1 scientific identity.
"""

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_cached_strict_router import (  # noqa: E402
    cached_strict_router_loss,
)
from lib.models.sttrack.lachtt_canonical_role_router import (  # noqa: E402
    CANDIDATE_ROLE_COUNT,
    CanonicalRoleIndependentUtilitySafetyRouter,
)
from lib.models.sttrack.lachtt_learned_bounded_roi_association import (  # noqa: E402
    build_detached_roi_differences,
)
from tools.run_sttrack_lachtt_m15c_independent_safety_capacity import (  # noqa: E402
    atomic_jsonl_gz,
    capacity_frozen_records,
    evaluate,
    trace_row,
)
from tools.smoke_sttrack_lachtt_m8b_cached import (  # noqa: E402
    atomic_json,
    batch_tensors,
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
)
from tools.smoke_sttrack_lachtt_m16a_canonical_role_ordering import (  # noqa: E402
    permutation_checks,
)


PREAUDIT_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_AUDIT_M16B_R1_TRACE_SCHEMA_AND_DEPENDENCY_RECOVERY_"
    "PREEXEC_20260901.json")
PREAUDIT_SHA256 = \
    "9ea9e536afa9eda4df3d4eca74e4d5466f3065f7aedcef9ab5954e312bce507f"
MODEL_PATH = REPOSITORY_ROOT / \
    "lib/models/sttrack/lachtt_canonical_role_router.py"
PARENT_MODEL_PATH = REPOSITORY_ROOT / \
    "lib/models/sttrack/lachtt_independent_utility_safety.py"
M15C_RUNNER_PATH = REPOSITORY_ROOT / \
    "tools/run_sttrack_lachtt_m15c_independent_safety_capacity.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def resolve_json_path(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def compare_final_result(reference, current, required_paths):
    mismatches = []
    for dotted_path in required_paths:
        left = resolve_json_path(reference, dotted_path)
        right = resolve_json_path(current, dotted_path)
        if left != right:
            mismatches.append({
                "path": dotted_path,
                "reference": left,
                "current": right,
            })
    return mismatches


def forward_losses(model, differences, block_gates, scalar, batch,
                   role_ids, trajectory_target, trajectory_available,
                   m15c_spec):
    outputs = model(
        differences, block_gates, scalar, batch["candidate_valid"], role_ids)
    strict_outputs = {
        "event_commit_logit": outputs["event_commit_logit"],
        "candidate_rank_logits": outputs["candidate_rank_logits"],
        "candidate_benefit_logits": outputs["candidate_benefit_logits"],
        "candidate_catastrophe_logits":
            outputs["candidate_catastrophe_logits"],
        "candidate_h10_gain": outputs["candidate_trajectory"][:, :, 2, 2],
    }
    loss_spec = m15c_spec["loss"]
    strict = cached_strict_router_loss(
        strict_outputs, batch["event_target"], batch["gain_target"],
        batch["beneficial_target"], batch["catastrophic_target"],
        batch["label_available"], batch["candidate_valid"],
        pairwise_margin=float(loss_spec["pairwise_margin"]),
    )
    mask = trajectory_available.unsqueeze(-1).expand_as(trajectory_target)
    trajectory_l1 = (
        torch.abs(outputs["candidate_trajectory"] - trajectory_target) *
        mask.float()).sum() / mask.float().sum()
    losses = {name: value for name, value in strict.items()}
    losses["trajectory_l1"] = trajectory_l1
    losses["total_with_trajectory"] = (
        strict["total"] +
        float(loss_spec["trajectory_weight"]) * trajectory_l1)
    return outputs, losses


def validate_authorization(spec):
    if (spec.get("complete") is not True or
            spec.get("created_before_runner_binding_and_execution") is not
            True):
        raise ValueError("M16b-R1 spec is incomplete")
    authorization = spec["authorization"]
    for name in (
            "independent_preexecution_audit_required",
            "runner_implementation_only_after_preaudit_pass",
            "one_capacity_run_only_after_binding",
            "independent_result_audit_after_run",
            "sequence_disjoint_plan_only_after_result_audit_pass"):
        if authorization.get(name) is not True:
            raise ValueError("spec authorization missing: %s" % name)
    for name in (
            "sequence_disjoint_execution", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)


def validate_identity_item(item, expected_path=None):
    path = Path(item["path"])
    if expected_path is not None and path.resolve() != Path(expected_path).resolve():
        raise ValueError("identity path drifted: %s" % path)
    if sha256_file(path) != item["sha256"]:
        raise ValueError("identity SHA drifted: %s" % path)
    return path


def validate_binding(args, spec, m15c_spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m16b-r1-trace-schema-and-dependency-recovery-binding/v1",
        "spec": file_record(args.spec),
        "plan": file_record(spec["plan"]["path"]),
        "incident": file_record(spec["incident"]["path"]),
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner": file_record(runner),
        "canonical_model": file_record(MODEL_PATH),
        "parent_model": file_record(PARENT_MODEL_PATH),
        "m15c_spec": file_record(
            spec["m15c_scientific_identity"]["spec"]["path"]),
        "m15c_runner": file_record(M15C_RUNNER_PATH),
        "m15c_result": file_record(
            spec["m15c_scientific_identity"]["result"]["path"]),
        "m15c_manifest": file_record(
            spec["m15c_scientific_identity"]["manifest"]["path"]),
        "m15c_training_trace": file_record(
            spec["m15c_scientific_identity"]["training_trace"]["path"]),
        "m15c_result_audit": file_record(
            spec["m15c_scientific_identity"]["result_audit"]["path"]),
        "m16a_result": file_record(
            spec["m16a_scientific_identity"]["result"]["path"]),
        "m16a_manifest": file_record(
            spec["m16a_scientific_identity"]["manifest"]["path"]),
        "m16a_result_audit": file_record(
            spec["m16a_scientific_identity"]["result_audit"]["path"]),
        "pre_execution_audit": file_record(PREAUDIT_PATH),
        "source_batch_spec": file_record(
            spec["frozen_inputs"]["source_batch_spec"]["path"]),
        "m15a_target_closure": file_record(
            spec["frozen_inputs"]["m15a_target_closure"]["path"]),
        "dependency_records": dependency_records(m15c_spec),
        "output": str(args.output),
        "output_root_absent_at_binding": True,
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    if sha256_file(PREAUDIT_PATH) != PREAUDIT_SHA256:
        raise ValueError("R1 preexecution audit SHA drifted")
    preaudit = json_file(PREAUDIT_PATH)
    expected_authorized = [
        "implement exactly one future file: tools/run_sttrack_lachtt_m16b_r1_canonical_role_capacity.py",
        "create one new frozen binding for that exact implementation",
        "run exactly one M16b-R1 200-step CPU capacity execution after binding",
        "perform independent post-result audit",
    ]
    if (preaudit.get("overall_verdict") != "PASS" or
            preaudit.get("integrity_verdict") != "PASS" or
            preaudit.get("protocol_verdict") != "PASS" or
            preaudit.get("authorization_boundary", {}).get(
                "authorized_after_pass") != expected_authorized):
        raise ValueError("R1 preexecution authorization drifted")
    if (git_output("branch", "--show-current") !=
            spec["repository"]["branch"] or
            git_output("status", "--porcelain")):
        raise ValueError("repository state drifted")
    if subprocess.run([
            "git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
            spec["repository"]["base_commit"], commit,
    ], check=False).returncode != 0:
        raise ValueError("implementation is outside frozen ancestry")
    diff = git_output(
        "diff", "--name-status",
        spec["repository"]["base_commit"] + ".." + commit).splitlines()
    expected_diff = [
        "A\ttools/run_sttrack_lachtt_m16b_r1_canonical_role_capacity.py"]
    if diff != expected_diff:
        raise ValueError("R1 implementation diff drifted")
    authorizations = binding.get("authorizations", {})
    for name in (
            "pre_execution_audit_passed", "one_capacity_execution",
            "independent_result_audit_after_run"):
        if authorizations.get(name) is not True:
            raise ValueError("binding authorization missing: %s" % name)
    for name in (
            "second_capacity_execution", "sequence_disjoint_execution",
            "formal_training", "tracking_checkpoint", "online_replay",
            "depthtrack_test", "cdtb", "vot_low22", "vot_full127",
            "qwen", "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, commit


def r1_frozen_records(spec, m15c_spec, source_spec, native_records,
                      binding, args):
    records = capacity_frozen_records(
        m15c_spec, source_spec, native_records, binding)
    items = [
        ("r1_incident", spec["incident"]),
        ("r1_plan", spec["plan"]),
        ("r1_spec", {"path": str(args.spec),
                     "sha256": sha256_file(args.spec)}),
        ("r1_preaudit", {"path": str(PREAUDIT_PATH),
                         "sha256": PREAUDIT_SHA256}),
        ("r1_runner", {"path": str(Path(__file__).resolve()),
                       "sha256": sha256_file(Path(__file__).resolve())}),
        ("r1_canonical_model",
         spec["m16a_scientific_identity"]["canonical_model"]),
        ("r1_m15c_spec", spec["m15c_scientific_identity"]["spec"]),
        ("r1_m15c_runner", spec["m15c_scientific_identity"]["runner"]),
        ("r1_m15c_parent", spec["m15c_scientific_identity"]["parent_model"]),
        ("r1_m15c_result", spec["m15c_scientific_identity"]["result"]),
        ("r1_m15c_manifest", spec["m15c_scientific_identity"]["manifest"]),
        ("r1_m15c_trace",
         spec["m15c_scientific_identity"]["training_trace"]),
        ("r1_m15c_audit", spec["m15c_scientific_identity"]["result_audit"]),
        ("r1_m16a_result", spec["m16a_scientific_identity"]["result"]),
        ("r1_m16a_manifest", spec["m16a_scientific_identity"]["manifest"]),
        ("r1_m16a_audit", spec["m16a_scientific_identity"]["result_audit"]),
        ("r1_binding", {"path": str(args.binding),
                        "sha256": sha256_file(args.binding)}),
    ]
    for name, item in items:
        records.append((name, Path(item["path"]), item["sha256"], None))
    return records


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    validate_authorization(spec)
    if args.output != Path(spec["output"]["root"]).resolve():
        raise ValueError("output root drifted from M16b-R1 spec")

    identity = spec["m15c_scientific_identity"]
    for name in ("spec", "runner", "parent_model", "result", "manifest",
                 "training_trace", "result_audit"):
        validate_identity_item(identity[name])
    for name in ("canonical_model", "runner", "result", "manifest",
                 "result_audit"):
        validate_identity_item(spec["m16a_scientific_identity"][name])
    validate_identity_item(spec["incident"])
    validate_identity_item(spec["plan"])
    validate_identity_item(spec["frozen_inputs"]["source_batch_spec"])
    validate_identity_item(spec["frozen_inputs"]["m15a_target_closure"])

    m15c_spec = json_file(identity["spec"]["path"])
    binding, runner_path, commit = validate_binding(args, spec, m15c_spec)
    m15c_result = json_file(identity["result"]["path"])
    m15c_audit = json_file(identity["result_audit"]["path"])
    if (m15c_result.get("accepted") is not False or
            m15c_result.get("decision") !=
            "m15c_r1_fail_stop_without_rescan" or
            sorted(m15c_result.get("failed_conditions", [])) !=
            ["candidate_permutation", "event_permutation"] or
            m15c_audit.get("integrity_verdict") != "PASS"):
        raise ValueError("M15c capacity reference boundary drifted")
    m16a_result = json_file(
        spec["m16a_scientific_identity"]["result"]["path"])
    m16a_audit = json_file(
        spec["m16a_scientific_identity"]["result_audit"]["path"])
    if (m16a_result.get("accepted") is not True or
            m16a_result.get("decision") !=
            "m16a_pass_freeze_m16b_capacity_plan_only" or
            m16a_audit.get("overall_verdict") != "PASS" or
            m16a_audit.get("integrity_verdict") != "PASS"):
        raise ValueError("M16a canonical reference boundary drifted")

    source_spec = json_file(
        spec["frozen_inputs"]["source_batch_spec"]["path"])
    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, native_records = load_native_batch(
        m15c_spec, source_spec)
    if binding.get("selected_native_payloads") != native_records:
        raise ValueError("selected native payloads drifted")
    trajectory_target, trajectory_available, target_rows = \
        load_trajectory_targets(m15c_spec, source_spec)
    records = r1_frozen_records(
        spec, m15c_spec, source_spec, native_records, binding, args)
    before_mismatches, frozen_observed = verify_frozen(records)

    builder = m15c_spec["relation_evidence"]["builder_parameters"]
    differences, block_gates, scalar = build_detached_roi_differences(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(builder["ema_alpha"]),
        epsilon=float(builder["l2_epsilon"]),
        soft_distractor_scale=float(builder["soft_distractor_scale"]),
        native_anchor_top_k=int(builder["native_anchor_top_k"]),
        depth_missing_floor=float(builder["depth_missing_floor"]),
    )
    role_ids = torch.arange(
        CANDIDATE_ROLE_COUNT, dtype=torch.int64).unsqueeze(0).expand(
            len(selected_rows), -1).clone()

    optimization = m15c_spec["optimization"]
    architecture = m15c_spec["architecture"]
    seed = int(optimization["seed"])
    torch.manual_seed(seed)
    model = CanonicalRoleIndependentUtilitySafetyRouter(
        hidden_dim=int(architecture["hidden_dim"]), residual_scale=0.1,
        base_projection_seed=int(builder["base_projection_seed"]))
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
    buffer_count = sum(value.numel() for value in model.buffers())
    forbidden_fragments = tuple(architecture["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if any(fragment in type(module).__name__
               for fragment in forbidden_fragments)
    ]
    initial_state = {name: value.detach().clone()
                     for name, value in model.state_dict().items()}
    initial_state_sha256 = state_digest(model)

    model.eval()
    with torch.no_grad():
        initial_outputs, initial_losses = forward_losses(
            model, differences, block_gates, scalar, batch, role_ids,
            trajectory_target, trajectory_available, m15c_spec)
        initial_metrics = evaluate(
            initial_outputs, initial_losses, batch,
            trajectory_target, trajectory_available)
    trace = [trace_row(0, initial_metrics, 0.0, 0.0, 0, False)]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]))
    utility_projector_coverage = set()
    safety_projector_coverage = set()
    utility_nonprojector_gradient_seen = False
    safety_nonprojector_gradient_seen = False
    steps_completed = 0
    gradient_failure = None
    preclip_values, postclip_values = [], []
    model.train()
    for step in range(1, int(optimization["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, losses = forward_losses(
            model, differences, block_gates, scalar, batch, role_ids,
            trajectory_target, trajectory_available, m15c_spec)
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
        if not (
                nonfinite == 0 and math.isfinite(preclip) and
                preclip > float(optimization["preclip_total_l2_min_exclusive"])
                and preclip <= float(optimization["preclip_total_l2_max"])):
            gradient_failure = {
                "step": step, "phase": "preclip", "norm": preclip,
                "nonfinite": nonfinite,
            }
            break
        maximum = float(optimization["global_gradient_clip"])
        scale_gradients(model, min(1.0, maximum / (preclip + 1e-12)))
        postclip, post_nonfinite, _ = gradient_diagnostics(model, 0)
        if not (
                post_nonfinite == 0 and math.isfinite(postclip) and
                postclip <= float(optimization["postclip_total_l2_max"])):
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
                model, differences, block_gates, scalar, batch, role_ids,
                trajectory_target, trajectory_available, m15c_spec)
            step_metrics = evaluate(
                step_outputs, step_losses, batch,
                trajectory_target, trajectory_available)
        trace.append(trace_row(
            step, step_metrics, preclip, postclip, nonfinite, True))
        model.train()

    model.eval()
    with torch.no_grad():
        final_outputs, final_losses = forward_losses(
            model, differences, block_gates, scalar, batch, role_ids,
            trajectory_target, trajectory_available, m15c_spec)
        final_metrics = evaluate(
            final_outputs, final_losses, batch,
            trajectory_target, trajectory_available)
    loss_ratio = (
        final_metrics["losses"]["total_with_trajectory"] /
        initial_metrics["losses"]["total_with_trajectory"])
    changes = {
        "utility_projectors": changed_named(
            initial_state, model.state_dict(), "utility_projectors."),
        "utility_router": changed_named(
            initial_state, model.state_dict(), "utility_router."),
        "safety_projectors": changed_named(
            initial_state, model.state_dict(), "safety_projectors."),
        "safety_critic": changed_named(
            initial_state, model.state_dict(), "safety_critic."),
    }
    final_state_sha256 = state_digest(model)

    inputs = (differences, block_gates, scalar, batch["candidate_valid"])
    permutation_rows = permutation_checks(
        model, inputs, role_ids, spec["permutation_checks"]["permutations"])
    event_permutation_error = max(
        row["event_error"] for row in permutation_rows)
    candidate_permutation_error = max(
        row["candidate_error"] for row in permutation_rows)

    input_counts = {
        "events": len(selected_rows), "candidates": 48,
        "target_rows": target_rows,
        "available_horizon_records": int(trajectory_available.sum().item()),
        "composition": dict(composition),
        "sequences": sorted(sequences),
    }
    training = {
        "seed": seed,
        "steps_requested": int(optimization["steps"]),
        "steps_completed": steps_completed,
        "gradient_failure": gradient_failure,
        "trace_rows": len(trace),
    }
    gradient_safety = {
        "preclip_max": max(preclip_values) if preclip_values else None,
        "postclip_max": max(postclip_values) if postclip_values else None,
        "all_steps_safe": gradient_failure is None,
    }
    projector_gradient_coverage = {
        "utility": sorted(utility_projector_coverage),
        "safety": sorted(safety_projector_coverage),
    }
    projector_changes = {
        "utility": changes["utility_projectors"],
        "safety": changes["safety_projectors"],
    }
    nonprojector_changes = {
        "utility": changes["utility_router"],
        "safety": changes["safety_critic"],
    }
    state_sha256 = {
        "initial": initial_state_sha256,
        "final": final_state_sha256,
    }
    current_scientific = {
        "input_counts": input_counts,
        "parameter_counts": parameter_counts,
        "parameter_id_intersection": parameter_id_intersection,
        "training": training,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_ratio": loss_ratio,
        "gradient_safety": gradient_safety,
        "projector_gradient_coverage": projector_gradient_coverage,
        "projector_changes": projector_changes,
        "nonprojector_changes": nonprojector_changes,
        "state_sha256": state_sha256,
        "forbidden_module_count": len(forbidden_modules),
        "checkpoint_count": 0,
    }

    reference_trace = load_jsonl_gz(identity["training_trace"]["path"])
    exact_keys = set(spec["trace_parity"]["exact_key_set"])
    trace_key_sets_exact = all(set(row) == exact_keys for row in trace)
    reference_key_sets_exact = all(
        set(row) == exact_keys for row in reference_trace)
    trace_mismatch_indices = [
        index for index, (left, right) in enumerate(
            zip(reference_trace, trace)) if left != right]
    if len(reference_trace) != len(trace):
        trace_mismatch_indices.extend(range(
            min(len(reference_trace), len(trace)),
            max(len(reference_trace), len(trace))))
    final_parity_mismatches = compare_final_result(
        m15c_result, current_scientific,
        spec["final_result_parity"]["required_paths"])
    after_mismatches, _ = verify_frozen(records)

    gates = spec["capacity_gates"]
    horizon_mae = list(final_metrics["trajectory_horizon_mae"].values())
    metric_mae = list(final_metrics["trajectory_metric_mae"].values())
    conditions = {
        "source_hashes_before": len(before_mismatches) == 0,
        "source_hashes_after": len(after_mismatches) == 0,
        "repository_clean": not bool(git_output("status", "--porcelain")),
        "batch_composition": composition == Counter(
            spec["frozen_inputs"]["composition"]),
        "distinct_sequences": len(sequences) == 8,
        "input_counts": (
            len(selected_rows) == int(spec["frozen_inputs"]["events_exact"])
            and target_rows == int(
                spec["frozen_inputs"]["trajectory_target_rows_exact"]) and
            int(trajectory_available.sum().item()) == 144),
        "target_availability": trajectory_available.all().item(),
        "parameter_counts": parameter_counts["total"] == int(
            gates["trainable_parameters_exact"]),
        "new_parameter_count": (
            parameter_counts["total"] -
            int(m15c_result["parameter_counts"]["total"]) ==
            int(gates["new_parameters_exact"])),
        "new_buffer_count": buffer_count == int(gates["new_buffers_exact"]),
        "parameter_id_intersection": parameter_id_intersection == 0,
        "forbidden_modules": len(forbidden_modules) == 0,
        "steps_completed": steps_completed == int(
            spec["optimization"]["optimizer_steps_exact"]),
        "gradient_safety": gradient_failure is None,
        "trace_rows": len(trace) == int(spec["trace_parity"]["rows_exact"]),
        "trace_key_sets_exact": trace_key_sets_exact and
            reference_key_sets_exact,
        "trace_exact_parity": len(trace_mismatch_indices) == int(
            spec["trace_parity"]["row_mismatch_count_exact"]),
        "final_result_exact_parity": len(final_parity_mismatches) == int(
            spec["final_result_parity"]["mismatch_count_exact"]),
        "utility_projector_gradient_coverage":
            len(utility_projector_coverage) == int(
                gates["utility_projector_gradient_coverage_exact"]),
        "safety_projector_gradient_coverage":
            len(safety_projector_coverage) == int(
                gates["safety_projector_gradient_coverage_exact"]),
        "utility_nonprojector_gradient": utility_nonprojector_gradient_seen,
        "safety_nonprojector_gradient": safety_nonprojector_gradient_seen,
        "utility_projectors_changed": changes["utility_projectors"] == int(
            gates["utility_projectors_changed_exact"]),
        "safety_projectors_changed": changes["safety_projectors"] == int(
            gates["safety_projectors_changed_exact"]),
        "utility_nonprojectors_changed": changes["utility_router"] >= 1,
        "safety_nonprojectors_changed": changes["safety_critic"] >= 1,
        "loss_ratio": loss_ratio <= float(gates["loss_ratio_max"]),
        "event_commit": (
            final_metrics["event_commit_correct"] == int(
                gates["event_commit_correct_exact"]) and
            final_metrics["event_commit_total"] == int(
                gates["event_commit_total_exact"])),
        "conditional_rank": (
            final_metrics["conditional_rank_correct"] == int(
                gates["conditional_rank_correct_exact"]) and
            final_metrics["conditional_rank_total"] == int(
                gates["conditional_rank_total_exact"])),
        "benefit": (
            final_metrics["benefit_correct"] == int(
                gates["benefit_correct_exact"]) and
            final_metrics["benefit_total"] == int(
                gates["benefit_total_exact"])),
        "catastrophe_true_positive":
            final_metrics["catastrophe_true_positive"] == int(
                gates["catastrophe_true_positive_exact"]),
        "catastrophe_true_negative":
            final_metrics["catastrophe_true_negative"] == int(
                gates["catastrophe_true_negative_exact"]),
        "trajectory_overall_mae":
            final_metrics["trajectory_overall_mae"] <= float(
                gates["trajectory_overall_mae_max"]),
        "trajectory_horizon_mae": max(horizon_mae) <= float(
            gates["trajectory_each_horizon_mae_max"]),
        "trajectory_metric_mae": max(metric_mae) <= float(
            gates["trajectory_each_metric_mae_max"]),
        "h10_gain_mae": final_metrics["h10_gain_mae"] <= float(
            gates["h10_gain_mae_max"]),
        "permutations_tested": len(permutation_rows) == int(
            spec["permutation_checks"]["permutations_tested_exact"]),
        "all_permutations_torch_equal": all(
            row["torch_equal"] for row in permutation_rows),
        "event_permutation_exact_zero": event_permutation_error == float(
            spec["permutation_checks"]["event_output_error_exact"]),
        "candidate_permutation_exact_zero":
            candidate_permutation_error == float(
                spec["permutation_checks"]["candidate_output_error_exact"]),
        "checkpoint_count": int(gates["checkpoint_count_exact"]) == 0,
        "output_file_set_preregistered": sorted(spec["output"]["files"]) ==
            ["manifest.json", "result.json", "training_trace.jsonl.gz"],
    }
    accepted = all(conditions.values())
    failed_conditions = sorted(
        name for name, passed in conditions.items() if not passed)
    result = {
        "schema": "sttrack-lachtt-m16b-r1-canonical-role-capacity-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": (
            "m16b_r1_pass_freeze_sequence_disjoint_plan_only" if accepted
            else "m16b_r1_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "repository": {
            "path": str(REPOSITORY_ROOT), "commit": commit,
            "branch": spec["repository"]["branch"], "clean": True,
        },
        "runner": file_record(runner_path),
        "canonical_model": file_record(MODEL_PATH),
        "parent_model": file_record(PARENT_MODEL_PATH),
        "pre_execution_audit": file_record(PREAUDIT_PATH),
        "m15c_reference": {
            name: file_record(item["path"])
            for name, item in identity.items() if isinstance(item, dict)
            and "path" in item
        },
        "input_counts": input_counts,
        "input_shapes": {
            "raw_difference": list(differences.shape),
            "block_gate": list(block_gates.shape),
            "scalar": list(scalar.shape),
            "trajectory_target": list(trajectory_target.shape),
            "candidate_role_ids": list(role_ids.shape),
        },
        "candidate_role_contract": spec["candidate_role_contract"],
        "parameter_counts": parameter_counts,
        "new_parameter_count": parameter_counts["total"] - 158753,
        "buffer_count": buffer_count,
        "parameter_id_intersection": parameter_id_intersection,
        "training": training,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_ratio": loss_ratio,
        "gradient_safety": gradient_safety,
        "projector_gradient_coverage": projector_gradient_coverage,
        "projector_changes": projector_changes,
        "nonprojector_changes": nonprojector_changes,
        "state_sha256": state_sha256,
        "trace_parity": {
            "reference": file_record(identity["training_trace"]["path"]),
            "rows_reference": len(reference_trace),
            "rows_current": len(trace),
            "exact_key_set": sorted(exact_keys),
            "reference_key_sets_exact": reference_key_sets_exact,
            "current_key_sets_exact": trace_key_sets_exact,
            "row_mismatch_count": len(trace_mismatch_indices),
            "row_mismatch_indices": trace_mismatch_indices[:20],
        },
        "final_result_parity": {
            "reference": file_record(identity["result"]["path"]),
            "required_paths": spec["final_result_parity"]["required_paths"],
            "explicitly_not_compared":
                spec["final_result_parity"]["explicitly_not_compared"],
            "mismatch_count": len(final_parity_mismatches),
            "mismatches": final_parity_mismatches,
        },
        "permutation_checks": {
            "rows": permutation_rows,
            "event_max_abs_error": event_permutation_error,
            "candidate_max_abs_error": candidate_permutation_error,
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
            "second_capacity_execution": False,
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
        "schema": "sttrack-lachtt-m16b-r1-canonical-role-capacity-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "identity": {
            "incident": file_record(spec["incident"]["path"]),
            "plan": file_record(spec["plan"]["path"]),
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "pre_execution_audit": file_record(PREAUDIT_PATH),
            "repository_commit": commit,
            "runner": file_record(runner_path),
            "canonical_model": file_record(MODEL_PATH),
            "parent_model": file_record(PARENT_MODEL_PATH),
            "m15c_spec": file_record(identity["spec"]["path"]),
            "m15c_result": file_record(identity["result"]["path"]),
            "m15c_trace": file_record(identity["training_trace"]["path"]),
            "dependencies": dependency_records(m15c_spec),
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
        "trace_mismatch_count": len(trace_mismatch_indices),
        "final_result_mismatch_count": len(final_parity_mismatches),
        "event_permutation_error": event_permutation_error,
        "candidate_permutation_error": candidate_permutation_error,
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
        "training_trace": file_record(trace_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
