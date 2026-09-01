#!/usr/bin/env python3
"""Bound engineering smoke for the M8c stable DeepSets router."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_cached_stable_router import (  # noqa: E402
    CachedStableTwoStageRouter,
    cached_strict_router_loss,
)
from tools.smoke_sttrack_lachtt_m8b_cached import (  # noqa: E402
    atomic_json,
    batch_tensors,
    candidate_permutation_error,
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
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments], text=True).strip()


def validate_binding(args, spec):
    binding = json_file(args.binding)
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain")
    runner = Path(__file__).resolve()
    model = REPOSITORY_ROOT / spec["repository"]["model_path"]
    expected = {
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
    if binding.get("complete") is not True:
        raise ValueError("binding is incomplete")
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError("binding mismatch for %s" % key)
    if not clean or branch != spec["repository"]["branch"]:
        raise ValueError("repository state drifted")
    ancestor = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
         spec["repository"]["base_commit"], commit], check=False).returncode == 0
    if not ancestor:
        raise ValueError("implementation is not descended from frozen base")
    authorizations = binding.get("authorizations", {})
    if authorizations.get("m8c_cached_stable_smoke") is not True:
        raise ValueError("binding does not authorize M8c smoke")
    forbidden = (
        "m8c_formal_training", "tracking_checkpoint", "depthtrack_test",
        "cdtb", "vot_low22", "vot_full127", "qwen",
        "automatic_next_stage")
    if any(authorizations.get(name) is not False for name in forbidden):
        raise ValueError("binding contains an unsafe authorization")
    if args.output.exists():
        raise FileExistsError("output already exists")
    return binding, commit, branch, runner, model


def m8c_frozen_records(spec):
    records = []
    for name in ("plan", "source_batch_spec", "m8b0_closure_manifest"):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"], None))
    for name, item in spec["failed_predecessor"].items():
        if isinstance(item, dict) and "path" in item:
            records.append(("failed_" + name, Path(item["path"]),
                            item["sha256"], None))
    return records


def main():
    args = parse_args()
    started = time.time()
    spec = json_file(args.spec)
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    if (authorization.get("m8c_engineering_implementation") is not True or
            authorization.get("one_bound_m8c_smoke") is not True):
        raise ValueError("spec does not authorize bound M8c smoke")
    if any(authorization.get(name) is not False for name in (
            "m8c_formal_training", "tracking_checkpoint", "depthtrack_test",
            "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage")):
        raise ValueError("spec authorization drifted")
    binding, commit, branch, runner_path, model_path = validate_binding(
        args, spec)
    source_spec = json_file(Path(spec["source_batch_spec"]["path"]))
    if source_spec.get("complete") is not True:
        raise ValueError("source batch spec is incomplete")
    records = m8c_frozen_records(spec) + frozen_records(source_spec)
    before_mismatches, frozen_observed = verify_frozen(records)
    predecessor = json_file(
        Path(spec["failed_predecessor"]["corrected_v2_result"]["path"]))
    if (predecessor.get("accepted") is not False or
            predecessor.get("optimizer_step_executed") is not False):
        raise ValueError("failed predecessor state drifted")
    closure = load_closure(source_spec)
    selected_rows, composition, sequences = validate_selection(
        source_spec, closure)
    batch = batch_tensors(source_spec, selected_rows)

    seed = int(source_spec["optimization"]["seed"])
    torch.manual_seed(seed)
    model = CachedStableTwoStageRouter(
        projection_dim=int(spec["architecture"]["projection_dim"]),
        hidden_dim=int(spec["architecture"]["hidden_dim"]),
        l2_normalization_eps=float(
            spec["architecture"]["l2_normalization_eps"]),
        residual_scale=0.1,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    layer_norm_modules = sum(isinstance(module, nn.LayerNorm)
                             for module in model.modules())
    transformer_modules = sum(isinstance(
        module, (nn.TransformerEncoder, nn.TransformerEncoderLayer))
        for module in model.modules())
    permutation_error, permutation_errors, permutation = \
        candidate_permutation_error(model, batch, seed)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(source_spec["optimization"]["learning_rate"]),
        weight_decay=float(source_spec["optimization"]["weight_decay"]))
    before_state = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    before_state_sha256 = state_digest(model)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        batch["features"], batch["initial_image"],
        batch["identity_text"], batch["candidate_valid"])
    losses = cached_strict_router_loss(
        outputs,
        batch["event_target"],
        batch["gain_target"],
        batch["beneficial_target"],
        batch["catastrophic_target"],
        batch["label_available"],
        batch["candidate_valid"],
        pairwise_margin=float(source_spec["optimization"]["pairwise_margin"]),
    )
    losses["total"].backward()
    gradient_contract = spec["gradient_contract"]
    preclip_norm, nonfinite_gradients, top_gradients = gradient_diagnostics(
        model, gradient_contract["top_gradient_parameters_reported"])
    preclip_safe = (
        nonfinite_gradients == 0 and math.isfinite(preclip_norm) and
        preclip_norm > float(
            gradient_contract["preclip_total_l2_min_exclusive"]) and
        preclip_norm <= float(gradient_contract["preclip_total_l2_max"]))
    optimizer_step_executed = False
    if preclip_safe:
        clip_max = float(gradient_contract["global_clip_max_norm"])
        scale_gradients(model, min(1.0, clip_max / (preclip_norm + 1e-12)))
        postclip_norm, postclip_nonfinite, _ = gradient_diagnostics(model, 0)
        postclip_safe = (
            postclip_nonfinite == 0 and math.isfinite(postclip_norm) and
            postclip_norm <= float(gradient_contract["postclip_total_l2_max"]))
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
    nonfinite_loss_terms = sum(
        not math.isfinite(float(value.detach())) for value in losses.values())
    gates = spec["gates"]
    conditions = {
        "all_source_hashes_exact": len(before_mismatches) == 0,
        "batch_composition_exact": composition == Counter(
            source_spec["selection"]["composition"]),
        "distinct_sequences_exact": len(sequences) == int(
            gates["distinct_sequences_exact"]),
        "fold0_events_max": sum(
            int(event["fold"]) == int(
                source_spec["selection"]["evaluation_outer_fold"])
            for event in source_spec["selection"]["events"]) <= int(
                gates["fold0_events_max"]),
        "parameter_count_max": parameter_count <= int(
            gates["parameter_count_max"]),
        "layer_norm_module_count": layer_norm_modules == int(
            gates["layer_norm_module_count"]),
        "transformer_module_count": transformer_modules == int(
            gates["transformer_module_count"]),
        "permutation_equivariance_max_error": permutation_error <= float(
            gates["permutation_equivariance_max_error"]),
        "nonfinite_outputs_max": nonfinite_outputs <= int(
            gates["nonfinite_outputs_max"]),
        "nonfinite_loss_terms_max": nonfinite_loss_terms <= int(
            gates["nonfinite_loss_terms_max"]),
        "nonfinite_gradients_max": nonfinite_gradients <= int(
            gates["nonfinite_gradients_max"]),
        "preclip_total_l2_min_exclusive": preclip_norm > float(
            gates["preclip_total_l2_min_exclusive"]),
        "preclip_total_l2_max": preclip_norm <= float(
            gates["preclip_total_l2_max"]),
        "postclip_total_l2_max": postclip_norm <= float(
            gates["postclip_total_l2_max"]),
        "optimizer_step_executed": optimizer_step_executed,
        "changed_trainable_tensors_min": changed_tensors >= int(
            gates["changed_trainable_tensors_min"]),
        "state_digest_changed": before_state_sha256 != after_state_sha256,
        "frozen_hash_mismatches_before_max": len(before_mismatches) <= int(
            gates["frozen_hash_mismatches_before_max"]),
        "frozen_hash_mismatches_after_max": len(after_mismatches) <= int(
            gates["frozen_hash_mismatches_after_max"]),
        "tracking_checkpoint_written_false": True,
        "public_state_mutations_false": True,
        "qwen_used_false": True,
        "public_benchmark_run_false": True,
    }
    accepted = all(conditions.values())
    decision = (
        "m8c_stable_smoke_pass_freeze_formal_train_spec_only" if accepted else
        "stop_m8c_stable_smoke_failed")
    result = {
        "schema": "sttrack-lachtt-m8c-stable-deepset-smoke-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": decision,
        "claim_ceiling": spec["claim_ceiling"],
        "repository": {"path": str(REPOSITORY_ROOT), "branch": branch,
                       "commit": commit, "clean": True},
        "failed_predecessor_preclip_total_l2": predecessor[
            "gradient_norm_before_clip"],
        "batch_composition": dict(sorted(composition.items())),
        "sequences": sorted(sequences),
        "parameter_count": parameter_count,
        "layer_norm_module_count": layer_norm_modules,
        "transformer_module_count": transformer_modules,
        "permutation": permutation,
        "permutation_errors": permutation_errors,
        "maximum_permutation_error": permutation_error,
        "losses": {name: float(value.detach())
                   for name, value in losses.items()},
        "gradient_norm_before_clip": preclip_norm,
        "gradient_norm_after_clip": postclip_norm,
        "top_gradients_before_clip": top_gradients,
        "optimizer_step_executed": optimizer_step_executed,
        "changed_trainable_tensors": changed_tensors,
        "state_sha256_before": before_state_sha256,
        "state_sha256_after": after_state_sha256,
        "nonfinite_outputs": nonfinite_outputs,
        "nonfinite_loss_terms": nonfinite_loss_terms,
        "nonfinite_gradients": nonfinite_gradients,
        "frozen_hash_mismatches_before": before_mismatches,
        "frozen_hash_mismatches_after": after_mismatches,
        "conditions": conditions,
        "unauthorized_actions": {
            "original_rgb_depth_opened": False,
            "ground_truth_opened": False,
            "sttrack_checkpoint_loaded": False,
            "clip_checkpoint_loaded": False,
            "tracking_checkpoint_written": False,
            "public_state_mutations": False,
            "depthtrack_test_run": False,
            "cdtb_run": False,
            "vot_low22_run": False,
            "vot_full127_run": False,
            "qwen_used": False,
            "automatic_next_stage": False,
        },
        "elapsed_seconds": time.time() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=args.output.name + ".", dir=str(args.output.parent)))
    try:
        result_path = temporary_root / "result.json"
        manifest_path = temporary_root / "manifest.json"
        atomic_json(result_path, result)
        manifest = {
            "schema": "sttrack-lachtt-m8c-stable-deepset-smoke-manifest/v1",
            "complete": True,
            "accepted": accepted,
            "decision": decision,
            "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
            "binding": {"path": str(args.binding),
                        "sha256": sha256_file(args.binding)},
            "runner": {"path": str(runner_path),
                       "sha256": sha256_file(runner_path)},
            "model": {"path": str(model_path),
                      "sha256": sha256_file(model_path)},
            "repository_commit": commit,
            "frozen_inputs": frozen_observed,
            "outputs": {
                "result.json": {"sha256": sha256_file(result_path),
                                "bytes": result_path.stat().st_size}
            },
            "tracking_checkpoint_written": False,
            "public_benchmark_run": False,
            "qwen_used": False,
            "scientific_scope": spec["claim_ceiling"],
        }
        atomic_json(manifest_path, manifest)
        for path in (result_path, manifest_path):
            path.chmod(0o444)
        os.replace(temporary_root, args.output)
        args.output.chmod(0o555)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    print(json.dumps({
        "accepted": accepted,
        "decision": decision,
        "parameter_count": parameter_count,
        "maximum_permutation_error": permutation_error,
        "gradient_norm_before_clip": preclip_norm,
        "gradient_norm_after_clip": postclip_norm,
        "changed_trainable_tensors": changed_tensors,
        "failed_conditions": sorted(
            name for name, passed in conditions.items() if not passed),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
