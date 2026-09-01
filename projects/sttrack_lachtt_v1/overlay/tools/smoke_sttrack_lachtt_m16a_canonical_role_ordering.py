#!/usr/bin/env python3
"""Engineering smoke for exact candidate-role canonicalization."""

import argparse
from collections import Counter
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
from lib.models.sttrack.lachtt_independent_utility_safety import (  # noqa: E402
    IndependentUtilitySafetyRouter,
)
from lib.models.sttrack.lachtt_learned_bounded_roi_association import (  # noqa: E402
    build_detached_roi_differences,
)
from tools.run_sttrack_lachtt_m15c_independent_safety_capacity import (  # noqa: E402
    capacity_frozen_records,
)
from tools.smoke_sttrack_lachtt_m8b_cached import (  # noqa: E402
    atomic_json,
    batch_tensors,
    frozen_records,
    gradient_diagnostics,
    load_closure,
    scale_gradients,
    sha256_file,
    validate_selection,
    verify_frozen,
)
from tools.smoke_sttrack_lachtt_m15b_r2_independent_utility_safety import (  # noqa: E402
    dependency_records,
    file_record,
    git_output,
    json_file,
    load_native_batch,
    load_trajectory_targets,
)


PREAUDIT_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_AUDIT_M16A_CANONICAL_ROLE_ORDERING_PREEXEC_20260901.json")
PREAUDIT_SHA256 = \
    "1e86f3bcb32c4dc7d3f52b869e95f296d073d66fe136c143709854b48ef593a9"
PARENT_MODEL_PATH = REPOSITORY_ROOT / \
    "lib/models/sttrack/lachtt_independent_utility_safety.py"
MODEL_PATH = REPOSITORY_ROOT / \
    "lib/models/sttrack/lachtt_canonical_role_router.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and
                isinstance(right, torch.Tensor) and
                torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                set(left) == set(right) and
                all(tensor_tree_equal(left[key], right[key])
                    for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(tensor_tree_equal(a, b)
                    for a, b in zip(left, right)))
    return left == right


def outputs_equal(left, right):
    return (set(left) == set(right) and
            all(torch.equal(left[name], right[name]) for name in left))


def maximum_output_error(left, right):
    return max(float(torch.max(torch.abs(
        left[name] - right[name])).item()) for name in left)


def router_losses(outputs, batch, trajectory_target,
                  trajectory_available, pairwise_margin,
                  trajectory_weight):
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
        pairwise_margin=float(pairwise_margin),
    )
    mask = trajectory_available.unsqueeze(-1).expand_as(trajectory_target)
    trajectory_l1 = (
        torch.abs(outputs["candidate_trajectory"] - trajectory_target) *
        mask.float()).sum() / mask.float().sum()
    losses = {name: value for name, value in strict.items()}
    losses["trajectory_l1"] = trajectory_l1
    losses["total_with_trajectory"] = (
        strict["total"] + float(trajectory_weight) * trajectory_l1)
    return losses


def gradients_equal(parent, canonical):
    parent_named = dict(parent.named_parameters())
    canonical_named = dict(canonical.named_parameters())
    if set(parent_named) != set(canonical_named):
        return False, []
    mismatches = []
    for name in parent_named:
        left = parent_named[name].grad
        right = canonical_named[name].grad
        if ((left is None) != (right is None) or
                (left is not None and not torch.equal(left, right))):
            mismatches.append(name)
    return len(mismatches) == 0, mismatches


def changed_tensor_count(before, after):
    return sum(not torch.equal(before[name], after[name]) for name in before)


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m16a-canonical-role-ordering-smoke-binding/v1",
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "plan_path": spec["plan"]["path"],
        "plan_sha256": spec["plan"]["sha256"],
        "repository_path": str(REPOSITORY_ROOT),
        "repository_commit": commit,
        "repository_clean": True,
        "runner": file_record(runner),
        "model": file_record(MODEL_PATH),
        "parent_model": file_record(PARENT_MODEL_PATH),
        "pre_execution_audit": file_record(PREAUDIT_PATH),
        "source_batch_spec": file_record(
            spec["frozen_evidence"]["source_batch_spec"]["path"]),
        "m15c_r1_result": file_record(
            spec["frozen_evidence"]["m15c_r1_result"]["path"]),
        "m15c_r1_result_audit": file_record(
            spec["frozen_evidence"]["m15c_r1_result_audit"]["path"]),
        "dependency_records": dependency_records(json_file(
            spec["frozen_evidence"]["m15c_r1_spec"]["path"])),
        "output": str(args.output),
        "output_root_absent_at_binding": True,
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    if sha256_file(PREAUDIT_PATH) != PREAUDIT_SHA256:
        raise ValueError("M16a pre-execution audit hash drifted")
    audit = json_file(PREAUDIT_PATH)
    expected_authorized = [
        "implement exactly two new files: lib/models/sttrack/lachtt_canonical_role_router.py and tools/smoke_sttrack_lachtt_m16a_canonical_role_ordering.py",
        "create a new frozen binding after implementation",
        "run exactly one M16a engineering smoke after binding",
        "perform independent post-result audit",
    ]
    if (audit.get("overall_verdict") != "PASS" or
            audit.get("integrity_verdict") != "PASS" or
            audit.get("protocol_verdict") != "PASS" or
            audit.get("authorization_boundary", {}).get(
                "authorized_after_pass") != expected_authorized):
        raise ValueError("M16a pre-execution authorization drifted")
    if (git_output("branch", "--show-current") !=
            spec["repository"]["branch"] or
            git_output("status", "--porcelain")):
        raise ValueError("repository state drifted")
    if subprocess.run([
            "git", "-C", str(REPOSITORY_ROOT), "merge-base",
            "--is-ancestor", spec["repository"]["base_commit"], commit,
    ], check=False).returncode != 0:
        raise ValueError("M16a implementation is outside frozen ancestry")
    diff = git_output(
        "diff", "--name-status", spec["repository"]["base_commit"] +
        ".." + commit).splitlines()
    expected_diff = sorted(
        "A\t" + name for name in spec["repository"]["allowed_new_files"])
    if sorted(diff) != expected_diff:
        raise ValueError("M16a implementation changed unauthorized files")
    authorizations = binding.get("authorizations", {})
    for name in (
            "pre_execution_plan_audit_passed", "one_engineering_smoke",
            "independent_result_audit_after_run", "m16b_capacity_plan_if_pass"):
        if authorizations.get(name) is not True:
            raise ValueError("binding authorization missing: %s" % name)
    for name in (
            "second_m16a_run", "m16b_capacity_execution",
            "sequence_disjoint_plan", "sequence_disjoint_execution",
            "formal_training", "tracking_checkpoint", "online_replay",
            "depthtrack_test", "cdtb", "vot_low22", "vot_full127",
            "qwen", "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, commit


def m16_frozen_records(spec, m15c_spec, source_spec,
                       native_records, binding, args):
    records = capacity_frozen_records(
        m15c_spec, source_spec, native_records, binding)
    for name, item in (
            ("m16_plan", spec["plan"]),
            ("m16_spec", {"path": str(args.spec),
                           "sha256": sha256_file(args.spec)}),
            ("m16_preaudit", {"path": str(PREAUDIT_PATH),
                               "sha256": PREAUDIT_SHA256}),
            ("m16_parent_model", {"path": str(PARENT_MODEL_PATH),
                                   "sha256": sha256_file(PARENT_MODEL_PATH)}),
            ("m16_model", {"path": str(MODEL_PATH),
                            "sha256": sha256_file(MODEL_PATH)}),
            ("m16_runner", {"path": str(Path(__file__).resolve()),
                             "sha256": sha256_file(Path(__file__).resolve())}),
            ("m16_binding", {"path": str(args.binding),
                              "sha256": sha256_file(args.binding)})):
        records.append((name, Path(item["path"]), item["sha256"], None))
    return records


def invalid_role_rejections(model, inputs, canonical_ids):
    differences, block_gates, scalar, candidate_valid = inputs
    cases = []
    duplicate = canonical_ids.clone()
    duplicate[0] = torch.tensor([0, 0, 2, 3, 4, 5])
    cases.append(("duplicate_or_missing_id", duplicate))
    out_of_range = canonical_ids.clone()
    out_of_range[0, -1] = 6
    cases.append(("out_of_range_id", out_of_range))
    cases.append(("floating_dtype", canonical_ids.float()))
    cases.append(("wrong_shape", canonical_ids[:, :5]))
    mixed_invalid = canonical_ids.clone()
    mixed_invalid[-1] = torch.tensor([0, 1, 2, 3, 4, 4])
    cases.append(("non_permutation_row", mixed_invalid))
    rejected = []
    for name, role_ids in cases:
        try:
            model(differences, block_gates, scalar,
                  candidate_valid, role_ids)
        except (TypeError, ValueError):
            rejected.append(name)
    return rejected


def permutation_checks(model, inputs, canonical_ids, permutations):
    differences, block_gates, scalar, candidate_valid = inputs
    model.eval()
    with torch.no_grad():
        reference = model(
            differences, block_gates, scalar, candidate_valid, canonical_ids)
        rows = []
        for values in permutations:
            permutation = torch.tensor(values, dtype=torch.int64)
            permuted = model(
                differences[:, :, permutation],
                block_gates[:, :, permutation],
                scalar[:, :, permutation], candidate_valid[:, permutation],
                canonical_ids[:, permutation])
            expected = {
                "event_commit_logit": reference["event_commit_logit"],
            }
            for name in (
                    "candidate_rank_logits", "candidate_benefit_logits",
                    "candidate_catastrophe_logits", "candidate_trajectory"):
                expected[name] = reference[name][:, permutation]
            details = {
                name: float(torch.max(torch.abs(
                    expected[name] - permuted[name])).item())
                for name in expected
            }
            rows.append({
                "permutation": values,
                "torch_equal": outputs_equal(expected, permuted),
                "event_error": details["event_commit_logit"],
                "candidate_error": max(
                    value for name, value in details.items()
                    if name != "event_commit_logit"),
                "details": details,
            })
    return rows


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    if args.output != Path(spec["output"]["root"]).resolve():
        raise ValueError("output root drifted from M16a spec")
    if (spec.get("complete") is not True or
            spec.get("created_before_implementation_and_execution") is not True):
        raise ValueError("M16a spec is incomplete")
    authorization = spec["authorization"]
    for name in (
            "independent_preexecution_audit_required",
            "implementation_after_preexecution_audit_pass",
            "one_engineering_smoke_after_binding",
            "independent_result_audit_after_run",
            "m16b_capacity_plan_if_pass"):
        if authorization.get(name) is not True:
            raise ValueError("spec authorization missing: %s" % name)
    for name in (
            "m16b_capacity_execution", "sequence_disjoint_plan",
            "sequence_disjoint_execution", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ValueError("unsafe spec authorization: %s" % name)

    binding, runner_path, commit = validate_binding(args, spec)
    m15c_spec = json_file(spec["frozen_evidence"]["m15c_r1_spec"]["path"])
    source_spec = json_file(
        spec["frozen_evidence"]["source_batch_spec"]["path"])
    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, native_records = load_native_batch(
        m15c_spec, source_spec)
    if binding.get("selected_native_payloads") != native_records:
        raise ValueError("binding selected native payloads drifted")
    trajectory_target, trajectory_available, target_rows = \
        load_trajectory_targets(m15c_spec, source_spec)
    records = m16_frozen_records(
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
    inputs = (differences, block_gates, scalar, batch["candidate_valid"])
    role_ids = torch.arange(
        CANDIDATE_ROLE_COUNT, dtype=torch.int64).unsqueeze(0).expand(
            len(selected_rows), -1).clone()

    seed = int(spec["optimization"]["seed"])
    torch.manual_seed(seed)
    parent = IndependentUtilitySafetyRouter(
        hidden_dim=37, residual_scale=0.1,
        base_projection_seed=int(builder["base_projection_seed"]))
    canonical = CanonicalRoleIndependentUtilitySafetyRouter(
        hidden_dim=37, residual_scale=0.1,
        base_projection_seed=int(builder["base_projection_seed"]))
    canonical.load_state_dict(parent.state_dict(), strict=True)
    parent_initial = {
        name: value.detach().clone()
        for name, value in parent.state_dict().items()}
    canonical_initial = {
        name: value.detach().clone()
        for name, value in canonical.state_dict().items()}

    parent_parameters = sum(
        value.numel() for value in parent.parameters() if value.requires_grad)
    canonical_parameters = sum(
        value.numel() for value in canonical.parameters()
        if value.requires_grad)
    parent_buffers = sum(value.numel() for value in parent.buffers())
    canonical_buffers = sum(value.numel() for value in canonical.buffers())
    forbidden = [
        type(module).__name__ for module in canonical.modules()
        if any(fragment in type(module).__name__
               for fragment in spec["model_contract"]["forbidden_module_fragments"])
    ]

    parent.train()
    canonical.train()
    parent_outputs = parent(
        differences, block_gates, scalar, batch["candidate_valid"])
    canonical_outputs = canonical(
        differences, block_gates, scalar, batch["candidate_valid"], role_ids)
    forward_exact = outputs_equal(parent_outputs, canonical_outputs)
    forward_error = maximum_output_error(parent_outputs, canonical_outputs)
    loss_cfg = m15c_spec["loss"]
    parent_losses = router_losses(
        parent_outputs, batch, trajectory_target, trajectory_available,
        loss_cfg["pairwise_margin"], loss_cfg["trajectory_weight"])
    canonical_losses = router_losses(
        canonical_outputs, batch, trajectory_target, trajectory_available,
        loss_cfg["pairwise_margin"], loss_cfg["trajectory_weight"])
    loss_exact = tensor_tree_equal(parent_losses, canonical_losses)

    optimization = spec["optimization"]
    parent_optimizer = torch.optim.AdamW(
        parent.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]))
    canonical_optimizer = torch.optim.AdamW(
        canonical.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]))
    parent_optimizer.zero_grad(set_to_none=True)
    canonical_optimizer.zero_grad(set_to_none=True)
    parent_losses["total_with_trajectory"].backward()
    canonical_losses["total_with_trajectory"].backward()
    gradient_exact, gradient_mismatches = gradients_equal(parent, canonical)
    parent_preclip, parent_nonfinite, _ = gradient_diagnostics(parent, 0)
    canonical_preclip, canonical_nonfinite, _ = gradient_diagnostics(
        canonical, 0)
    preclip_safe = (
        parent_nonfinite == 0 and canonical_nonfinite == 0 and
        math.isfinite(parent_preclip) and
        math.isfinite(canonical_preclip) and
        parent_preclip > float(
            optimization["preclip_total_l2_min_exclusive"]) and
        canonical_preclip > float(
            optimization["preclip_total_l2_min_exclusive"]) and
        parent_preclip <= float(optimization["preclip_total_l2_max"]) and
        canonical_preclip <= float(optimization["preclip_total_l2_max"]))
    parent_step = False
    canonical_step = False
    if preclip_safe and gradient_exact:
        maximum = float(optimization["global_gradient_clip"])
        scale_gradients(
            parent, min(1.0, maximum / (parent_preclip + 1e-12)))
        scale_gradients(
            canonical, min(1.0, maximum / (canonical_preclip + 1e-12)))
        parent_postclip, parent_post_nonfinite, _ = gradient_diagnostics(
            parent, 0)
        canonical_postclip, canonical_post_nonfinite, _ = \
            gradient_diagnostics(canonical, 0)
        postclip_safe = (
            parent_post_nonfinite == 0 and canonical_post_nonfinite == 0 and
            parent_postclip <= float(optimization["postclip_total_l2_max"]) and
            canonical_postclip <= float(
                optimization["postclip_total_l2_max"]))
        if postclip_safe:
            parent_optimizer.step()
            parent_step = True
            canonical_optimizer.step()
            canonical_step = True
    else:
        parent_postclip = parent_preclip
        canonical_postclip = canonical_preclip
        parent_post_nonfinite = parent_nonfinite
        canonical_post_nonfinite = canonical_nonfinite
        postclip_safe = False

    post_state_exact = tensor_tree_equal(
        parent.state_dict(), canonical.state_dict())
    optimizer_state_exact = tensor_tree_equal(
        parent_optimizer.state_dict(), canonical_optimizer.state_dict())
    changed = changed_tensor_count(
        canonical_initial, canonical.state_dict())
    permutation_rows = permutation_checks(
        canonical, inputs, role_ids,
        spec["permutation_checks"]["permutations"])
    event_permutation_error = max(
        row["event_error"] for row in permutation_rows)
    candidate_permutation_error = max(
        row["candidate_error"] for row in permutation_rows)
    invalid_rejected = invalid_role_rejections(
        canonical, inputs, role_ids)
    after_mismatches, _ = verify_frozen(records)

    nonfinite_count = sum((
        int(not torch.isfinite(value).all().item())
        for value in (
            differences, block_gates, scalar, trajectory_target,
            *parent_outputs.values(), *canonical_outputs.values(),
            *parent_losses.values(), *canonical_losses.values()))) + \
        parent_nonfinite + canonical_nonfinite + \
        parent_post_nonfinite + canonical_post_nonfinite
    gates = spec["gates"]
    conditions = {
        "source_hashes_before": len(before_mismatches) == int(
            gates["source_hash_mismatches_exact"]),
        "source_hashes_after": len(after_mismatches) == int(
            gates["source_hash_mismatches_exact"]),
        "repository_clean": not bool(git_output("status", "--porcelain")),
        "batch_composition": composition == Counter(
            spec["inputs"]["composition"]),
        "distinct_sequences": len(sequences) == 8,
        "target_rows": target_rows == int(
            spec["inputs"]["trajectory_target_rows_exact"]),
        "target_availability": trajectory_available.all().item(),
        "candidate_role_rows_valid": role_ids.shape[0] == int(
            gates["candidate_role_rows_valid_exact"]),
        "invalid_role_cases_rejected": (
            len(invalid_rejected) == int(
                gates["invalid_role_cases_rejected_exact"]) and
            invalid_rejected ==
            spec["candidate_role_contract"]["fail_closed_cases"]),
        "parameter_counts": (
            parent_parameters == int(gates["parent_parameter_count_exact"]) and
            canonical_parameters == int(
                gates["canonical_parameter_count_exact"])),
        "new_parameter_count": (
            canonical_parameters - parent_parameters == int(
                gates["new_parameter_count_exact"])),
        "new_buffer_count": (
            canonical_buffers - parent_buffers == int(
                gates["new_buffer_count_exact"])),
        "state_dict_exact_before": tensor_tree_equal(
            parent_initial, canonical_initial),
        "canonical_forward_exact": forward_exact and forward_error == 0.0,
        "canonical_loss_exact": loss_exact,
        "canonical_gradient_exact": gradient_exact,
        "gradient_safety": preclip_safe and postclip_safe,
        "optimizer_steps": parent_step and canonical_step,
        "canonical_post_step_state_exact": (
            post_state_exact and optimizer_state_exact),
        "canonical_changed_trainable_tensors": changed >= int(
            gates["canonical_changed_trainable_tensors_min"]),
        "permutations_tested": len(permutation_rows) == int(
            gates["permutations_tested_exact"]),
        "all_permutations_torch_equal": all(
            row["torch_equal"] for row in permutation_rows),
        "event_permutation_exact_zero": event_permutation_error == float(
            gates["event_permutation_error_exact"]),
        "candidate_permutation_exact_zero": (
            candidate_permutation_error == float(
                gates["candidate_permutation_error_exact"])),
        "nonfinite_count": nonfinite_count == int(
            gates["nonfinite_inputs_outputs_losses_gradients_exact"]),
        "forbidden_modules": len(forbidden) == 0,
        "checkpoint_count": int(gates["checkpoint_count_exact"]) == 0,
        "output_file_set_preregistered": sorted(spec["output"]["files"]) ==
            sorted(gates["output_file_set_exact"]),
    }
    accepted = all(conditions.values())
    failed_conditions = sorted(
        name for name, passed in conditions.items() if not passed)
    result = {
        "schema": "sttrack-lachtt-m16a-canonical-role-ordering-smoke-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m16a_pass_freeze_m16b_capacity_plan_only"
                     if accepted else "m16a_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "repository": {
            "path": str(REPOSITORY_ROOT), "commit": commit,
            "branch": spec["repository"]["branch"], "clean": True,
        },
        "runner": file_record(runner_path),
        "model": file_record(MODEL_PATH),
        "parent_model": file_record(PARENT_MODEL_PATH),
        "pre_execution_audit": file_record(PREAUDIT_PATH),
        "input_counts": {
            "events": len(selected_rows), "candidates": 48,
            "trajectory_target_rows": target_rows,
            "composition": dict(composition),
            "sequences": sorted(sequences),
        },
        "candidate_role_contract": spec["candidate_role_contract"],
        "parameter_counts": {
            "parent": parent_parameters,
            "canonical": canonical_parameters,
            "new": canonical_parameters - parent_parameters,
            "parent_buffers": parent_buffers,
            "canonical_buffers": canonical_buffers,
            "new_buffers": canonical_buffers - parent_buffers,
        },
        "canonical_parity": {
            "state_dict_before": tensor_tree_equal(
                parent_initial, canonical_initial),
            "forward": forward_exact,
            "forward_max_abs_error": forward_error,
            "loss": loss_exact,
            "gradient": gradient_exact,
            "gradient_mismatches": gradient_mismatches,
            "post_step_model_state": post_state_exact,
            "post_step_optimizer_state": optimizer_state_exact,
        },
        "optimization": {
            "seed": seed,
            "parent_step": parent_step,
            "canonical_step": canonical_step,
            "parent_preclip_l2": parent_preclip,
            "canonical_preclip_l2": canonical_preclip,
            "parent_postclip_l2": parent_postclip,
            "canonical_postclip_l2": canonical_postclip,
            "canonical_changed_tensors": changed,
        },
        "permutation_checks": {
            "rows": permutation_rows,
            "event_max_abs_error": event_permutation_error,
            "candidate_max_abs_error": candidate_permutation_error,
        },
        "invalid_role_cases": {
            "required": spec["candidate_role_contract"]["fail_closed_cases"],
            "rejected": invalid_rejected,
        },
        "nonfinite_count": nonfinite_count,
        "forbidden_modules": forbidden,
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
            "m16b_capacity_plan": accepted,
            "m16b_capacity_execution": False,
            "sequence_disjoint_plan": False,
            "sequence_disjoint_execution": False,
            "tracking_checkpoint": False,
            "depthtrack_test": False, "cdtb": False,
            "vot_low22": False, "vot_full127": False, "qwen": False,
        },
    }

    args.output.mkdir(parents=True)
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m16a-canonical-role-ordering-smoke-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "identity": {
            "plan": file_record(spec["plan"]["path"]),
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "pre_execution_audit": file_record(PREAUDIT_PATH),
            "repository_commit": commit,
            "runner": file_record(runner_path),
            "model": file_record(MODEL_PATH),
            "parent_model": file_record(PARENT_MODEL_PATH),
            "source_batch_spec": file_record(
                spec["frozen_evidence"]["source_batch_spec"]["path"]),
            "selected_native_payloads": native_records,
        },
        "payload": {
            "result": file_record(result_path),
        },
        "unauthorized_actions": {
            "checkpoint_written": False,
            "second_m16a_run": False,
            "m16b_capacity_execution": False,
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
        raise RuntimeError("M16a output file set drifted")
    for path in (result_path, manifest_path):
        path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "failed_conditions": failed_conditions,
        "canonical_parity": result["canonical_parity"],
        "permutation_checks": result["permutation_checks"],
        "invalid_role_cases": result["invalid_role_cases"],
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
