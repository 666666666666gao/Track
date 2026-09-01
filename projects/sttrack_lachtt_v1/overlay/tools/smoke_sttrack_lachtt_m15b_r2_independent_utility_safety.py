#!/usr/bin/env python3
"""One bound engineering step for independent utility and safety paths."""

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
from lib.models.sttrack.lachtt_independent_utility_safety import (  # noqa: E402
    HORIZONS,
    TRAJECTORY_METRICS,
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


PREAUDIT_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_AUDIT_M15B_R3_STORAGE_API_WIRING_RECOVERY_PREEXEC_20260901.json")
PREAUDIT_SHA256 = \
    "615b583aae925e81833d8837fba82ff5961946a0db2ca40e7417e9ff5648b66c"


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


def file_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dependency_records(spec):
    return [{
        "path": str((REPOSITORY_ROOT / item["path"]).resolve()),
        "sha256": item["sha256"],
    } for item in spec["relation_evidence"]["dependencies"]]


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    model = (REPOSITORY_ROOT /
             "lib/models/sttrack/lachtt_independent_utility_safety.py")
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m15b-r3-storage-api-wiring-recovery-binding/v1",
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
        "labeled_actions": file_record(spec["labeled_actions"]["path"]),
        "m15a_target_closure": file_record(
            spec["m15a_target_closure"]["path"]),
        "native_anchor_index": file_record(
            spec["relation_evidence"]["native_anchor_index"]["path"]),
        "native_anchor_manifest": file_record(
            spec["relation_evidence"]["native_anchor_manifest"]["path"]),
        "dependency_records": dependency_records(spec),
        "pre_execution_plan_audit": {
            "path": str(PREAUDIT_PATH),
            "sha256": PREAUDIT_SHA256,
        },
        "output": str(args.output),
        "output_root_absent_at_binding": True,
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("binding mismatch: %s" % name)
    if sha256_file(PREAUDIT_PATH) != PREAUDIT_SHA256:
        raise ValueError("pre-execution audit hash drifted")
    audit = json_file(PREAUDIT_PATH)
    if (str(audit.get("overall_verdict", "")).lower() != "pass" or
            str(audit.get("integrity_verdict", "")).lower() != "pass" or
            audit.get("authorized") is not True):
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
    required_true = (
        "pre_execution_plan_audit_passed",
        "one_engineering_optimizer_step",
        "independent_result_audit_after_run",
        "m15c_capacity_plan_if_pass",
    )
    if any(authorizations.get(name) is not True for name in required_true):
        raise ValueError("binding does not authorize bounded M15b-R2")
    for name in (
            "second_run_or_retry", "m15c_capacity_execution",
            "sequence_disjoint_pilot", "formal_training",
            "tracking_checkpoint", "online_replay", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, model.resolve(), commit


def load_native_batch(spec, source_spec):
    rows = {}
    index_path = Path(
        spec["relation_evidence"]["native_anchor_index"]["path"])
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sequence"] in rows:
                raise ValueError("duplicate native anchor")
            rows[row["sequence"]] = row
    rgb, depth, records = [], [], []
    for event in source_spec["selection"]["events"]:
        row = rows.get(event["sequence"])
        if row is None:
            raise ValueError("missing selected native anchor")
        path = index_path.parent / row["path"]
        observed = file_record(path)
        expected = {
            "path": str(path.resolve()),
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
        if observed != expected:
            raise ValueError("selected native anchor identity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (tuple(payload["native_template_rgb_tokens"].shape) != (64, 768) or
                tuple(payload["native_template_depth_tokens"].shape) !=
                (64, 768)):
            raise ValueError("native anchor shape drifted")
        rgb.append(payload["native_template_rgb_tokens"])
        depth.append(payload["native_template_depth_tokens"])
        records.append({"sequence": event["sequence"], **observed})
    return torch.stack(rgb), torch.stack(depth), records


def frozen_input_records(spec, source_spec, native_records, binding):
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
            ("failed_predecessor_incident",
             spec["failed_predecessor"]["incident"]),
            ("failed_predecessor_audit",
             spec["failed_predecessor"]["audit_json"])):
        records.append((name, Path(item["path"]), item["sha256"], None))
    for item in spec["relation_evidence"]["dependencies"]:
        records.append((
            "dependency:" + item["path"],
            REPOSITORY_ROOT / item["path"], item["sha256"], None))
    records.append(("pre_execution_plan_audit", PREAUDIT_PATH,
                    PREAUDIT_SHA256, None))
    for item in native_records:
        records.append(("native_anchor:" + item["sequence"],
                        Path(item["path"]), item["sha256"], item["bytes"]))
    if binding.get("selected_native_payloads") != native_records:
        raise ValueError("binding selected native payloads drifted")
    return records


def load_trajectory_targets(spec, source_spec):
    branch_order = source_spec["candidate_contract"]["branch_order"]
    event_order = [(
        event["sequence"], int(event["event_id"]),
        int(event["trigger_frame"]))
        for event in source_spec["selection"]["events"]]
    event_indices = {key: index for index, key in enumerate(event_order)}
    horizon_indices = {int(value): index
                       for index, value in enumerate(HORIZONS)}
    metric_names = tuple(TRAJECTORY_METRICS)
    target = torch.zeros(8, 6, 3, 5, dtype=torch.float32)
    available = torch.zeros(8, 6, 3, dtype=torch.bool)
    seen = set()
    row_count = 0
    with gzip.open(spec["m15a_target_closure"]["path"], "rt",
                   encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_count += 1
            event_key = (row["sequence"], int(row["event_id"]),
                         int(row["trigger_frame"]))
            if event_key not in event_indices:
                raise ValueError("target closure contains an unselected event")
            event_index = event_indices[event_key]
            candidate_index = int(row["candidate_index"])
            if (candidate_index < 0 or candidate_index >= 6 or
                    row["branch_id"] != branch_order[candidate_index]):
                raise ValueError("target candidate axis drifted")
            horizon = int(row["horizon"])
            if horizon not in horizon_indices:
                raise ValueError("target horizon drifted")
            horizon_index = horizon_indices[horizon]
            key = (event_index, candidate_index, horizon_index)
            if key in seen:
                raise ValueError("duplicate trajectory target")
            seen.add(key)
            values = [float(row[name]) for name in metric_names]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite trajectory target")
            target[event_index, candidate_index, horizon_index] = \
                torch.tensor(values, dtype=torch.float32)
            available[event_index, candidate_index, horizon_index] = bool(
                row["available"])
    if (row_count != int(spec["m15a_target_closure"]["rows"]) or
            len(seen) != 144 or not available.all().item()):
        raise ValueError("trajectory target closure is incomplete")
    return target, available, row_count


def permutation_errors(model, differences, block_gates, scalar,
                       candidate_valid, seed):
    generator = torch.Generator().manual_seed(int(seed) + 17)
    permutation = torch.randperm(6, generator=generator)
    model.eval()
    with torch.no_grad():
        original = model(differences, block_gates, scalar, candidate_valid)
        permuted = model(
            differences[:, :, permutation],
            block_gates[:, :, permutation],
            scalar[:, :, permutation], candidate_valid[:, permutation])
    details = {
        "event_commit_logit": float(torch.max(torch.abs(
            original["event_commit_logit"] -
            permuted["event_commit_logit"])).item())
    }
    for name in (
            "candidate_rank_logits", "candidate_benefit_logits",
            "candidate_catastrophe_logits", "candidate_trajectory"):
        details[name] = float(torch.max(torch.abs(
            original[name][:, permutation] - permuted[name])).item())
    candidate_error = max(value for name, value in details.items()
                          if name != "event_commit_logit")
    return details["event_commit_logit"], candidate_error, details, permutation


def count_nonzero_finite_gradients(parameters):
    return sum(
        parameter.grad is not None and
        torch.isfinite(parameter.grad).all().item() and
        float(torch.linalg.vector_norm(
            parameter.grad.detach().double()).item()) > 0.0
        for parameter in parameters)


def changed_named(before, current, prefix):
    return sum(
        name.startswith(prefix) and not torch.equal(before[name], value)
        for name, value in current.items())


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.output = args.output.resolve()
    spec = json_file(args.spec)
    if args.output != Path(spec["output"]["root"]).resolve():
        raise ValueError("output root drifted from R3 spec")
    if (spec.get("complete") is not True or
            spec.get("created_before_implementation_and_execution") is not True):
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    for name in (
            "repeat_independent_preexecution_audit_required",
            "implementation_after_preexecution_audit_pass",
            "one_engineering_optimizer_step_after_binding",
            "independent_result_audit_after_run",
            "m15c_capacity_plan_if_pass"):
        if authorization.get(name) is not True:
            raise ValueError("spec authorization missing: %s" % name)
    for name in (
            "m15c_capacity_execution", "sequence_disjoint_pilot",
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
    m15a_result = json_file(spec["m15a_target_closure"]["result"]["path"])
    m15a_audit = json_file(
        spec["m15a_target_closure"]["result_audit"]["path"])
    if (m15a_result.get("accepted") is not True or
            str(m15a_audit.get("overall_verdict", "")).lower() != "pass" or
            str(m15a_audit.get("integrity_verdict", "")).lower() != "pass"):
        raise ValueError("M15a accepted boundary drifted")

    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, native_records = load_native_batch(
        spec, source_spec)
    trajectory_target, trajectory_available, target_rows = \
        load_trajectory_targets(spec, source_spec)
    records = frozen_input_records(
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
        hidden_dim=int(architecture["hidden_dim"]),
        residual_scale=0.1,
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
    utility_relations = model.utility_relations(
        differences, block_gates, scalar)
    safety_relations = model.safety_relations(
        differences, block_gates, scalar)
    relation_storage_shared = (
        utility_relations.data_ptr() == safety_relations.data_ptr())
    event_permutation_error, candidate_permutation_error, \
        permutation_details, permutation = permutation_errors(
            model, differences, block_gates, scalar,
            batch["candidate_valid"], seed)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["optimization"]["learning_rate"]),
        weight_decay=float(spec["optimization"]["weight_decay"]))
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    before_state_sha256 = state_digest(model)
    optimizer.zero_grad(set_to_none=True)
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
    strict_losses = cached_strict_router_loss(
        strict_outputs, batch["event_target"], batch["gain_target"],
        batch["beneficial_target"], batch["catastrophic_target"],
        batch["label_available"], batch["candidate_valid"],
        pairwise_margin=float(spec["loss"]["pairwise_margin"]),
    )
    trajectory_mask = trajectory_available.unsqueeze(-1).expand_as(
        trajectory_target)
    trajectory_l1 = (
        torch.abs(outputs["candidate_trajectory"] - trajectory_target) *
        trajectory_mask.float()).sum() / trajectory_mask.float().sum()
    total_loss = strict_losses["total"] + \
        float(spec["loss"]["trajectory_weight"]) * trajectory_l1
    losses = {name: value for name, value in strict_losses.items()}
    losses["trajectory_l1"] = trajectory_l1
    losses["total_with_trajectory"] = total_loss
    total_loss.backward()

    utility_projector_gradients = count_nonzero_finite_gradients(
        model.utility_projectors.parameters())
    safety_projector_gradients = count_nonzero_finite_gradients(
        model.safety_projectors.parameters())
    utility_nonprojector_gradients = count_nonzero_finite_gradients(
        model.utility_router.parameters())
    safety_nonprojector_gradients = count_nonzero_finite_gradients(
        model.safety_critic.parameters())
    preclip_norm, nonfinite_gradients, top_gradients = gradient_diagnostics(
        model, 12)
    optimization = spec["optimization"]
    preclip_safe = (
        nonfinite_gradients == 0 and math.isfinite(preclip_norm) and
        preclip_norm > float(optimization["preclip_total_l2_min_exclusive"]) and
        preclip_norm <= float(optimization["preclip_total_l2_max"]))
    optimizer_step_executed = False
    if preclip_safe:
        maximum = float(optimization["global_gradient_clip"])
        scale_gradients(model, min(1.0, maximum / (preclip_norm + 1e-12)))
        postclip_norm, postclip_nonfinite, _ = gradient_diagnostics(model, 0)
        postclip_safe = (
            postclip_nonfinite == 0 and math.isfinite(postclip_norm) and
            postclip_norm <= float(optimization["postclip_total_l2_max"]))
        if postclip_safe:
            optimizer.step()
            optimizer_step_executed = True
    else:
        postclip_norm = preclip_norm
        postclip_nonfinite = nonfinite_gradients
        postclip_safe = False

    current_state = model.state_dict()
    changes = {
        "utility_projectors": changed_named(
            before_state, current_state, "utility_projectors."),
        "utility_router": changed_named(
            before_state, current_state, "utility_router."),
        "safety_projectors": changed_named(
            before_state, current_state, "safety_projectors."),
        "safety_critic": changed_named(
            before_state, current_state, "safety_critic."),
    }
    after_state_sha256 = state_digest(model)
    after_mismatches, _ = verify_frozen(records)
    nonfinite_counts = {
        "input": int(not (
            torch.isfinite(differences).all().item() and
            torch.isfinite(block_gates).all().item() and
            torch.isfinite(scalar).all().item())),
        "target": int(not torch.isfinite(trajectory_target).all().item()),
        "output": sum(not torch.isfinite(value).all().item()
                      for value in outputs.values()),
        "loss": sum(not math.isfinite(float(value.detach()))
                    for value in losses.values()),
        "gradient": nonfinite_gradients,
    }
    output_shapes = {name: list(value.shape)
                     for name, value in outputs.items()}
    input_shapes = {
        "raw_difference": list(differences.shape),
        "block_gate": list(block_gates.shape),
        "scalar": list(scalar.shape),
        "trajectory_target": list(trajectory_target.shape),
        "trajectory_available": list(trajectory_available.shape),
    }
    gates = spec["gates"]
    conditions = {
        "source_hash_mismatches_before": len(before_mismatches) == 0,
        "source_hash_mismatches_after": len(after_mismatches) == 0,
        "batch_composition_exact": composition == Counter(
            source_spec["selection"]["composition"]),
        "distinct_sequences_exact": len(sequences) == 8,
        "target_rows_exact": target_rows == 144,
        "all_candidate_horizons_available": trajectory_available.all().item(),
        "raw_difference_shape": input_shapes["raw_difference"] ==
            gates["raw_difference_shape"],
        "block_gate_shape": input_shapes["block_gate"] ==
            gates["block_gate_shape"],
        "scalar_shape": input_shapes["scalar"] == gates["scalar_shape"],
        "raw_difference_bound": float(differences.abs().max().item()) <=
            float(gates["raw_difference_max_abs"]),
        "block_gate_bounds": (
            float(block_gates.min().item()) >= float(gates["block_gate_min"]) and
            float(block_gates.max().item()) <= float(gates["block_gate_max"])),
        "scalar_bound": float(scalar.abs().max().item()) <=
            float(gates["scalar_max_abs"]),
        "all_finite": all(value == 0 for value in nonfinite_counts.values()),
        "output_shapes_exact": (
            output_shapes["event_commit_logit"] == gates["event_output_shape"] and
            all(output_shapes[name] == gates["candidate_scalar_output_shape"]
                for name in ("candidate_rank_logits",
                             "candidate_benefit_logits",
                             "candidate_catastrophe_logits")) and
            output_shapes["candidate_trajectory"] ==
                gates["trajectory_output_shape"]),
        "parameter_counts_exact": parameter_counts == {
            "utility_projectors": 36864,
            "utility_router": 44679,
            "safety_projectors": 36864,
            "safety_critic": 40346,
            "total": int(gates["total_parameters_exact"]),
        },
        "parameter_id_intersection_exact":
            parameter_id_intersection == int(
                gates["utility_safety_parameter_id_intersection_exact"]),
        "relation_storage_independent": not relation_storage_shared,
        "utility_projector_gradients_exact":
            utility_projector_gradients == int(
                gates["utility_projector_nonzero_gradient_tensors_exact"]),
        "safety_projector_gradients_exact":
            safety_projector_gradients == int(
                gates["safety_projector_nonzero_gradient_tensors_exact"]),
        "utility_nonprojector_gradients_min":
            utility_nonprojector_gradients >= int(
                gates["utility_nonprojector_nonzero_gradient_tensors_min"]),
        "safety_nonprojector_gradients_min":
            safety_nonprojector_gradients >= int(
                gates["safety_nonprojector_nonzero_gradient_tensors_min"]),
        "preclip_gradient_norm": preclip_safe,
        "postclip_gradient_norm": postclip_safe,
        "optimizer_step_executed": optimizer_step_executed,
        "utility_projectors_changed_exact": changes["utility_projectors"] ==
            int(gates["utility_projector_changed_tensors_exact"]),
        "safety_projectors_changed_exact": changes["safety_projectors"] ==
            int(gates["safety_projector_changed_tensors_exact"]),
        "utility_nonprojectors_changed_min": changes["utility_router"] >=
            int(gates["utility_nonprojector_changed_tensors_min"]),
        "safety_nonprojectors_changed_min": changes["safety_critic"] >=
            int(gates["safety_nonprojector_changed_tensors_min"]),
        "event_permutation_error": event_permutation_error <=
            float(gates["event_permutation_error_max"]),
        "candidate_permutation_error": candidate_permutation_error <=
            float(gates["candidate_permutation_error_max"]),
        "forbidden_module_count": len(forbidden_modules) <=
            int(gates["forbidden_module_count_max"]),
        "checkpoint_count": int(gates["checkpoint_count_exact"]) == 0,
        "output_file_set_preregistered": sorted(
            spec["output"]["files"]) == sorted(gates["output_file_set_exact"]),
    }
    accepted = all(conditions.values())
    failed_conditions = sorted(
        name for name, passed in conditions.items() if not passed)
    result = {
        "schema": "sttrack-lachtt-m15b-r3-storage-api-wiring-recovery-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m15b_r3_pass_freeze_m15c_plan_only" if accepted
                     else "m15b_r3_fail_stop_without_retry"),
        "claim_ceiling": spec["claim_ceiling"],
        "source_hashes": {
            "labeled_actions": spec["labeled_actions"]["sha256"],
            "m15a_target_closure": spec["m15a_target_closure"]["sha256"],
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
        "input_counts": {
            "events": len(selected_rows), "candidates": 48,
            "target_rows": target_rows, "available_horizon_records":
                int(trajectory_available.sum().item()),
            "sequences": sorted(sequences),
            "composition": dict(composition),
        },
        "input_shapes": input_shapes,
        "input_ranges": {
            "raw_difference_max_abs": float(differences.abs().max().item()),
            "block_gate_min": float(block_gates.min().item()),
            "block_gate_max": float(block_gates.max().item()),
            "scalar_max_abs": float(scalar.abs().max().item()),
        },
        "parameter_counts": parameter_counts,
        "parameter_id_intersection": parameter_id_intersection,
        "relation_storage_shared": relation_storage_shared,
        "output_shapes": output_shapes,
        "nonfinite_counts": nonfinite_counts,
        "loss_components": {name: float(value.detach())
                            for name, value in losses.items()},
        "gradient_norms": {
            "preclip_total_l2": preclip_norm,
            "postclip_total_l2": postclip_norm,
            "postclip_nonfinite": postclip_nonfinite,
            "top": top_gradients,
            "optimizer_step_executed": optimizer_step_executed,
        },
        "projector_gradients": {
            "utility_nonzero_finite_tensors": utility_projector_gradients,
            "safety_nonzero_finite_tensors": safety_projector_gradients,
        },
        "projector_changes": {
            "utility": changes["utility_projectors"],
            "safety": changes["safety_projectors"],
        },
        "nonprojector_gradients": {
            "utility_nonzero_finite_tensors": utility_nonprojector_gradients,
            "safety_nonzero_finite_tensors": safety_nonprojector_gradients,
        },
        "nonprojector_changes": {
            "utility": changes["utility_router"],
            "safety": changes["safety_critic"],
        },
        "state_sha256": {
            "before": before_state_sha256, "after": after_state_sha256,
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
        "inputs": {
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "pre_execution_audit": file_record(PREAUDIT_PATH),
            "selected_native_payloads": native_records,
        },
        "authorization": {
            "independent_result_audit": True,
            "m15c_capacity_plan": accepted,
            "m15c_capacity_execution": False,
            "sequence_disjoint_pilot": False,
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
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m15b-r3-storage-api-wiring-recovery-manifest/v1",
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
            "dependencies": dependency_records(spec),
            "labeled_actions": file_record(spec["labeled_actions"]["path"]),
            "m15a_target_closure": file_record(
                spec["m15a_target_closure"]["path"]),
            "native_anchor_index": file_record(
                spec["relation_evidence"]["native_anchor_index"]["path"]),
            "native_anchor_manifest": file_record(
                spec["relation_evidence"]["native_anchor_manifest"]["path"]),
            "selected_native_payloads": native_records,
        },
        "payload": {"result": file_record(result_path)},
        "unauthorized_actions": {
            "checkpoint_written": False,
            "second_run_or_retry": False,
            "m15c_capacity_execution": False,
            "sequence_disjoint_pilot": False,
            "formal_training": False,
            "online_replay": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)
    actual_files = sorted(path.name for path in args.output.iterdir())
    if actual_files != sorted(spec["output"]["files"]):
        raise RuntimeError("output file set drifted")
    result_path.chmod(0o444)
    manifest_path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "failed_conditions": failed_conditions,
        "parameter_counts": parameter_counts,
        "gradient_norms": result["gradient_norms"],
        "projector_gradients": result["projector_gradients"],
        "projector_changes": result["projector_changes"],
        "permutation_errors": result["permutation_errors"],
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
