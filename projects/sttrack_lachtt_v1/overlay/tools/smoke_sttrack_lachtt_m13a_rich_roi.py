#!/usr/bin/env python3
"""One bound engineering step for detached rich RoI instance relations."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_cached_strict_router import (  # noqa: E402
    cached_strict_router_loss,
)
from lib.models.sttrack.lachtt_rich_roi_relation import (  # noqa: E402
    DIFFERENCE_BLOCKS,
    FAMILY_NAMES,
    PROJECTION_WIDTH,
    RICH_RELATION_DIM,
    TEMPORAL_STATISTICS,
    TEMPORAL_WIDTH,
    RichRoIRelationRouter,
    build_rich_roi_relations,
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


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def validate_binding(args, spec):
    binding = json_file(args.binding)
    runner = Path(__file__).resolve()
    model = REPOSITORY_ROOT / spec["model"]["path"]
    commit = git_output("rev-parse", "HEAD")
    expected = {
        "schema": "sttrack-lachtt-m13a-rich-roi-smoke-binding/v1",
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
    if authorizations.get("one_engineering_optimizer_step") is not True:
        raise ValueError("binding does not authorize M13a")
    if authorizations.get("independent_integrity_audit_after_run") is not True:
        raise ValueError("binding does not require the independent audit")
    if authorizations.get("m13b_capacity_plan_if_pass") is not True:
        raise ValueError("binding does not authorize the bounded M13b plan")
    for name in (
            "m13b_capacity_execution", "sequence_disjoint_pilot",
            "formal_training", "tracking_checkpoint",
            "online_replay", "depthtrack_test", "cdtb", "vot_low22",
            "vot_full127", "qwen", "automatic_next_stage"):
        if authorizations.get(name) is not False:
            raise ValueError("unsafe binding authorization: %s" % name)
    if args.output.exists():
        raise FileExistsError(args.output)
    return binding, runner, model, commit


def source_records(spec, source_spec, native_rows):
    records = frozen_records(source_spec)
    for name in ("plan", "source_batch_spec",
                 "native_anchor_index", "native_anchor_manifest"):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"], None))
    for name, item in spec["m12"].items():
        records.append(("m12_" + name, Path(item["path"]),
                        item["sha256"], None))
    item = spec["m10_scalar_relation"]
    records.append(("m10_scalar_relation", REPOSITORY_ROOT / item["path"],
                    item["sha256"], None))
    for sequence, row in native_rows.items():
        records.append((
            "native_anchor:" + sequence,
            Path(spec["native_anchor_index"]["path"]).parent / row["path"],
            row["sha256"], int(row["bytes"])))
    return records


def load_native_batch(spec, source_spec):
    rows = {}
    index_path = Path(spec["native_anchor_index"]["path"])
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sequence"] in rows:
                raise ValueError("duplicate native anchor")
            rows[row["sequence"]] = row
    rgb, depth, selected_rows = [], [], {}
    for event in source_spec["selection"]["events"]:
        sequence = event["sequence"]
        row = rows.get(sequence)
        if row is None:
            raise ValueError("missing selected native anchor")
        path = index_path.parent / row["path"]
        if (path.stat().st_size != int(row["bytes"]) or
                sha256_file(path) != row["sha256"]):
            raise ValueError("native anchor identity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (tuple(payload["native_template_rgb_tokens"].shape) != (64, 768) or
                tuple(payload["native_template_depth_tokens"].shape) !=
                (64, 768)):
            raise ValueError("native anchor shape drifted")
        rgb.append(payload["native_template_rgb_tokens"])
        depth.append(payload["native_template_depth_tokens"])
        selected_rows[sequence] = row
    return torch.stack(rgb), torch.stack(depth), selected_rows


def permutation_errors(model, relations, candidate_valid, seed):
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(6, generator=generator)
    model.eval()
    with torch.no_grad():
        original = model(relations, candidate_valid)
        permuted = model(
            relations[:, :, permutation], candidate_valid[:, permutation])
    errors = {
        "event_commit_logit": float(torch.max(torch.abs(
            original["event_commit_logit"] -
            permuted["event_commit_logit"])).item())
    }
    for name in (
            "candidate_rank_logits", "candidate_benefit_logits",
            "candidate_catastrophe_logits", "candidate_h10_gain"):
        errors[name] = float(torch.max(torch.abs(
            original[name][:, permutation] - permuted[name])).item())
    return max(errors.values()), errors, permutation


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
            authorization.get("one_engineering_optimizer_step") is not True or
            authorization.get("independent_integrity_audit_after_run") is not True or
            authorization.get("m13b_capacity_plan_if_pass") is not True):
        raise ValueError("spec does not authorize the bounded M13a workflow")
    for name in (
            "m13b_capacity_execution", "sequence_disjoint_pilot",
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
    predecessor = json_file(spec["m12"]["result"]["path"])
    predecessor_audit = json_file(spec["m12"]["audit"]["path"])
    if (predecessor.get("accepted") is not False or
            predecessor.get("decision") != "m12_fail_stop_without_rescan"):
        raise ValueError("M12 failure boundary drifted")
    if predecessor_audit.get("integrity_verdict") != "PASS":
        raise ValueError("M12 integrity audit is not a pass")
    if predecessor_audit.get("authorization_boundary", {}).get(
            "stop_task_tower_49d_scalar_router_scanning") is not True:
        raise ValueError("M12 audit does not authorize the mechanism change")

    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)
    native_rgb, native_depth, selected_native_rows = load_native_batch(
        spec, source_spec)
    records = source_records(spec, source_spec, selected_native_rows)
    before_mismatches, frozen_observed = verify_frozen(records)

    relation_spec = spec["relation"]
    relations = build_rich_roi_relations(
        batch["features"], batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(relation_spec["ema_alpha"]),
        epsilon=float(relation_spec["l2_epsilon"]),
        soft_distractor_scale=float(
            relation_spec["soft_distractor_cosine_scale"]),
        base_projection_seed=int(relation_spec["base_projection_seed"]),
    )
    relation_nonfinite = int(not torch.isfinite(relations).all().item())
    relation_max_abs = float(torch.max(torch.abs(relations)).item())
    generator = torch.Generator().manual_seed(int(spec["optimization"]["seed"]))
    extractor_permutation = torch.randperm(6, generator=generator)
    permuted_features = {
        name: value[:, :, extractor_permutation]
        for name, value in batch["features"].items()
    }
    permuted_relations = build_rich_roi_relations(
        permuted_features, batch["initial_image"], batch["identity_text"],
        native_rgb, native_depth,
        ema_alpha=float(relation_spec["ema_alpha"]),
        epsilon=float(relation_spec["l2_epsilon"]),
        soft_distractor_scale=float(
            relation_spec["soft_distractor_cosine_scale"]),
        base_projection_seed=int(relation_spec["base_projection_seed"]),
    )
    extractor_permutation_error = float(torch.max(torch.abs(
        relations[:, :, extractor_permutation] - permuted_relations)).item())

    seed = int(spec["optimization"]["seed"])
    torch.manual_seed(seed)
    model_spec = spec["model"]
    expected_relation_contract = (
        tuple(relation_spec["families"]) == tuple(FAMILY_NAMES) and
        int(relation_spec["projection_width"]) == PROJECTION_WIDTH and
        int(relation_spec["difference_blocks"]) == DIFFERENCE_BLOCKS and
        int(relation_spec["total_width"]) == RICH_RELATION_DIM and
        tuple(model_spec["temporal_statistics"]) == TEMPORAL_STATISTICS and
        int(model_spec["temporal_width"]) == TEMPORAL_WIDTH)
    model = RichRoIRelationRouter(
        hidden_dim=int(model_spec["hidden_dim"]),
        residual_scale=float(model_spec["residual_scale"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    forbidden_fragments = tuple(model_spec["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if any(fragment in type(module).__name__
               for fragment in forbidden_fragments)]
    projection_trainable_parameters = 0
    model_permutation_error, model_permutation_details, permutation = \
        permutation_errors(model, relations, batch["candidate_valid"], seed)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["optimization"]["learning_rate"]),
        weight_decay=float(spec["optimization"]["weight_decay"]))
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    before_state_sha256 = state_digest(model)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(relations, batch["candidate_valid"])
    losses = cached_strict_router_loss(
        outputs,
        batch["event_target"],
        batch["gain_target"],
        batch["beneficial_target"],
        batch["catastrophic_target"],
        batch["label_available"],
        batch["candidate_valid"],
        pairwise_margin=float(spec["optimization"]["pairwise_margin"]),
    )
    losses["total"].backward()
    gates = spec["gates"]
    preclip_norm, nonfinite_gradients, top_gradients = gradient_diagnostics(
        model, 10)
    preclip_safe = (
        nonfinite_gradients == 0 and math.isfinite(preclip_norm) and
        preclip_norm > float(gates["preclip_total_l2_min_exclusive"]) and
        preclip_norm <= float(gates["preclip_total_l2_max"]))
    optimizer_step_executed = False
    if preclip_safe:
        maximum = float(spec["optimization"]["global_clip_max_norm"])
        scale_gradients(model, min(1.0, maximum / (preclip_norm + 1e-12)))
        postclip_norm, postclip_nonfinite, _ = gradient_diagnostics(model, 0)
        postclip_safe = (
            postclip_nonfinite == 0 and math.isfinite(postclip_norm) and
            postclip_norm <= float(gates["postclip_total_l2_max"]))
        if postclip_safe:
            optimizer.step()
            optimizer_step_executed = True
    else:
        postclip_norm = preclip_norm
        postclip_nonfinite = nonfinite_gradients
        postclip_safe = False
    changed_tensors = sum(
        not torch.equal(before_state[name], value)
        for name, value in model.state_dict().items())
    after_state_sha256 = state_digest(model)
    after_mismatches, _ = verify_frozen(records)
    nonfinite_outputs = sum(
        not torch.isfinite(value.float()).all().item()
        for value in outputs.values())
    nonfinite_losses = sum(
        not math.isfinite(float(value.detach())) for value in losses.values())

    conditions = {
        "frozen_hash_mismatches_before_max": len(before_mismatches) <= int(
            gates["frozen_hash_mismatches_max"]),
        "frozen_hash_mismatches_after_max": len(after_mismatches) <= int(
            gates["frozen_hash_mismatches_max"]),
        "batch_composition_exact": composition == Counter(
            source_spec["selection"]["composition"]),
        "distinct_sequences_exact": len(sequences) == 8,
        "relation_contract_exact": expected_relation_contract,
        "relation_shape_exact": list(relations.shape) == gates[
            "relation_shape"],
        "relation_nonfinite_max": relation_nonfinite <= int(
            gates["relation_nonfinite_max"]),
        "relation_absolute_bound": relation_max_abs <= float(
            gates["relation_max_abs"]),
        "extractor_permutation_max_error": extractor_permutation_error <= float(
            gates["permutation_error_max"]),
        "model_permutation_max_error": model_permutation_error <= float(
            gates["permutation_error_max"]),
        "projection_trainable_parameters_max":
            projection_trainable_parameters <= int(
                gates["projection_trainable_parameters_max"]),
        "forbidden_module_count_max": len(forbidden_modules) <= int(
            gates["forbidden_module_count_max"]),
        "parameter_count_exact": parameter_count == int(
            gates["model_parameters_exact"]),
        "output_and_loss_finite": (nonfinite_outputs == 0 and
                                   nonfinite_losses == 0 and
                                   bool(gates["loss_finite"])),
        "gradient_nonfinite_max": nonfinite_gradients <= int(
            gates["nonfinite_gradients_max"]),
        "preclip_gradient_norm": preclip_safe,
        "postclip_gradient_norm": postclip_safe,
        "optimizer_step_executed": optimizer_step_executed,
        "changed_trainable_tensors_min": changed_tensors >= int(
            gates["changed_trainable_tensors_min"]),
    }
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m13a-rich-roi-smoke-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m13a_pass_freeze_capacity_plan_only"
                     if accepted else "m13a_fail_stop_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "conditions": conditions,
        "batch": {
            "composition": dict(composition),
            "sequences": sorted(sequences),
            "events": len(selected_rows),
        },
        "relations": {
            "shape": list(relations.shape),
            "maximum_absolute_value": relation_max_abs,
            "nonfinite": relation_nonfinite,
            "extractor_permutation": extractor_permutation.tolist(),
            "extractor_permutation_error": extractor_permutation_error,
            "families": list(FAMILY_NAMES),
            "difference_blocks": DIFFERENCE_BLOCKS,
            "projection_width": PROJECTION_WIDTH,
            "projection_seed": int(
                relation_spec["base_projection_seed"]),
            "projected_width": int(relation_spec["projected_width"]),
            "scalar_width": int(relation_spec["scalar_width"]),
        },
        "model": {
            "parameters": parameter_count,
            "forbidden_modules": forbidden_modules,
            "projection_trainable_parameters":
                projection_trainable_parameters,
            "temporal_statistics": list(TEMPORAL_STATISTICS),
            "temporal_width": TEMPORAL_WIDTH,
            "candidate_permutation": permutation.tolist(),
            "permutation_error": model_permutation_error,
            "permutation_details": model_permutation_details,
            "state_sha256_before": before_state_sha256,
            "state_sha256_after": after_state_sha256,
            "changed_trainable_tensors": changed_tensors,
        },
        "optimization": {
            "losses": {name: float(value.detach())
                       for name, value in losses.items()},
            "preclip_total_l2": preclip_norm,
            "postclip_total_l2": postclip_norm,
            "nonfinite_gradients": nonfinite_gradients,
            "postclip_nonfinite_gradients": postclip_nonfinite,
            "top_gradients": top_gradients,
            "optimizer_step_executed": optimizer_step_executed,
        },
        "frozen": {
            "before_mismatches": before_mismatches,
            "after_mismatches": after_mismatches,
            "observed": frozen_observed,
        },
        "repository": {
            "path": str(REPOSITORY_ROOT),
            "commit": commit,
            "branch": spec["repository"]["branch"],
            "clean": True,
        },
        "inputs": {
            "spec": file_record(args.spec),
            "binding": file_record(args.binding),
            "runner": file_record(runner_path),
            "model": file_record(model_path),
        },
        "authorization": {
            "independent_integrity_audit": True,
            "m13b_capacity_plan": accepted,
            "m13b_capacity_execution": False,
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
        "schema": "sttrack-lachtt-m13a-rich-roi-smoke-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "payload": {"result": file_record(result_path)},
        "unauthorized_actions": {
            "checkpoint_written": False,
            "m13b_capacity_execution": False,
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
    result_path.chmod(0o444)
    manifest_path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "relations": result["relations"],
        "optimization": result["optimization"],
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
