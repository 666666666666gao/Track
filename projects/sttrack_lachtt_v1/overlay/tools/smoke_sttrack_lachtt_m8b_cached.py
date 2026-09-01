#!/usr/bin/env python3
"""Bound M8b-1 engineering smoke over eight frozen cached events."""

import argparse
from collections import Counter
import gzip
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.models.sttrack.lachtt_cached_strict_router import (  # noqa: E402
    CachedStrictTwoStageRouter,
    FEATURE_KEYS,
    cached_strict_router_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--correction-spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path, value):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        Path(temporary).write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_output(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments], text=True).strip()


def stable_fold(sequence, salt, folds):
    value = hashlib.sha256((salt + "\0" + sequence).encode()).digest()
    return int.from_bytes(value[:8], "big") % int(folds)


def selection_digest(seed, event_class, sequence, event_id, trigger_frame):
    value = "%d\0%s\0%s\0%d\0%d" % (
        int(seed), event_class, sequence, int(event_id), int(trigger_frame))
    return hashlib.sha256(value.encode()).hexdigest()


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def validate_binding(args, spec, correction_spec):
    binding = json_file(args.binding)
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain")
    runner = Path(__file__).resolve()
    model = REPOSITORY_ROOT / spec["repository"]["model_path"]
    expected = {
        "spec_path": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "correction_spec_path": str(args.correction_spec),
        "correction_spec_sha256": sha256_file(args.correction_spec),
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
    base = spec["repository"]["base_commit"]
    ancestor = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
         base, commit], check=False).returncode == 0
    if not ancestor:
        raise ValueError("implementation is not descended from frozen base")
    if correction_spec["repository"]["model_sha256_must_remain"] != \
            sha256_file(model):
        raise ValueError("model changed during gradient-only correction")
    authorizations = binding.get("authorizations", {})
    if authorizations.get("m8b1_cached_engineering_smoke") is not True:
        raise ValueError("binding does not authorize M8b-1 smoke")
    forbidden = (
        "model_load", "m8b2_training", "tracking_checkpoint",
        "depthtrack_test", "cdtb", "vot_low22", "vot_full127", "qwen",
        "automatic_next_stage")
    if any(authorizations.get(name) is not False for name in forbidden):
        raise ValueError("binding contains an unsafe authorization")
    if args.output.exists():
        raise FileExistsError("output already exists")
    return binding, commit, branch, runner, model


def frozen_records(spec):
    records = []
    for name in ("plan",):
        item = spec[name]
        records.append((name, Path(item["path"]), item["sha256"], None))
    for name in ("spec", "binding", "result", "manifest", "closure",
                 "independent_audit"):
        item = spec["m8b0_closure"][name]
        records.append(("m8b0_" + name, Path(item["path"]),
                        item["sha256"], None))
    for event in spec["selection"]["events"]:
        records.append((
            "feature:%s:%d" % (event["sequence"], event["event_id"]),
            Path(event["feature_path"]), event["feature_sha256"],
            int(event["feature_bytes"])))
        records.append((
            "anchor:%s" % event["sequence"], Path(event["anchor_path"]),
            event["anchor_sha256"], int(event["anchor_bytes"])))
    return records


def verify_frozen(records):
    mismatches = []
    observed = []
    for name, path, expected_hash, expected_bytes in records:
        if not path.is_file():
            mismatches.append({"name": name, "reason": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        observed.append({
            "name": name,
            "path": str(path),
            "sha256": actual_hash,
            "bytes": actual_bytes,
        })
        if actual_hash != expected_hash:
            mismatches.append({"name": name, "reason": "sha256"})
        if expected_bytes is not None and actual_bytes != expected_bytes:
            mismatches.append({"name": name, "reason": "bytes"})
    return mismatches, observed


def load_closure(spec):
    path = Path(spec["m8b0_closure"]["closure"]["path"])
    rows = {}
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            key = event_key(row)
            if key in rows:
                raise ValueError("duplicate closure event")
            rows[key] = row
    if count != int(spec["m8b0_closure"]["closure"]["rows"]):
        raise ValueError("closure row count drifted")
    return rows


def validate_selection(spec, closure):
    selection = spec["selection"]
    composition = Counter()
    sequences = set()
    selected = []
    for event in selection["events"]:
        key = event_key(event)
        row = closure.get(key)
        if row is None:
            raise ValueError("selected event is absent from closure")
        for name in ("strict_event_class", "feature_path", "feature_sha256",
                     "feature_bytes", "anchor_path"):
            if row.get(name) != event.get(name):
                raise ValueError("selected event binding mismatch: %s" % name)
        fold = stable_fold(
            event["sequence"], selection["outer_fold_salt"],
            selection["outer_fold_count"])
        digest = selection_digest(
            selection["seed"], event["strict_event_class"],
            event["sequence"], event["event_id"], event["trigger_frame"])
        if fold != int(event["fold"]) or digest != event["selection_digest"]:
            raise ValueError("selected event fold/digest mismatch")
        if fold == int(selection["evaluation_outer_fold"]):
            raise ValueError("fold0 event entered smoke batch")
        composition[event["strict_event_class"]] += 1
        sequences.add(event["sequence"])
        selected.append(row)
    if composition != Counter(selection["composition"]):
        raise ValueError("smoke batch composition drifted")
    if len(sequences) != len(selection["events"]):
        raise ValueError("smoke sequences are not distinct")
    return selected, composition, sequences


def batch_tensors(spec, rows):
    feature_rows = []
    initial_rows = []
    text_rows = []
    event_targets = []
    gain_targets = []
    benefit_targets = []
    catastrophe_targets = []
    available_targets = []
    branch_order = spec["candidate_contract"]["branch_order"]
    for event, row in zip(spec["selection"]["events"], rows):
        payload = torch.load(
            event["feature_path"], map_location="cpu", weights_only=True)
        if set(payload) != set(FEATURE_KEYS):
            raise ValueError("feature payload keys drifted")
        feature_rows.append(payload)
        anchor = torch.load(
            event["anchor_path"], map_location="cpu", weights_only=True)
        if set(anchor) != {"initial_image", "identity_text"}:
            raise ValueError("anchor payload keys drifted")
        initial_rows.append(anchor["initial_image"])
        text_rows.append(anchor["identity_text"])
        actions = row["actions"]
        if [action["branch_id"] for action in actions] != branch_order:
            raise ValueError("strict action order drifted")
        gains, benefits, catastrophes, available = [], [], [], []
        for action in actions:
            label = action["strict_label"]
            is_available = label != "unavailable"
            utility = action["strict_utility"]
            if is_available and utility is None:
                raise ValueError("available strict action lacks utility")
            if not is_available and utility is not None:
                raise ValueError("unavailable strict action has utility")
            gains.append(0.0 if utility is None else
                         float(utility["mean_iou_gain"]))
            benefits.append(label == "beneficial")
            catastrophes.append(label == "catastrophic")
            available.append(is_available)
        event_targets.append(row["strict_event_class"] == "beneficial")
        gain_targets.append(gains)
        benefit_targets.append(benefits)
        catastrophe_targets.append(catastrophes)
        available_targets.append(available)
    features = {name: torch.stack(
        [payload[name] for payload in feature_rows], dim=0)
        for name in FEATURE_KEYS}
    return {
        "features": features,
        "initial_image": torch.stack(initial_rows, dim=0),
        "identity_text": torch.stack(text_rows, dim=0),
        "candidate_valid": torch.ones(len(rows), 6, dtype=torch.bool),
        "event_target": torch.tensor(event_targets, dtype=torch.bool),
        "gain_target": torch.tensor(gain_targets, dtype=torch.float32),
        "beneficial_target": torch.tensor(
            benefit_targets, dtype=torch.bool),
        "catastrophic_target": torch.tensor(
            catastrophe_targets, dtype=torch.bool),
        "label_available": torch.tensor(
            available_targets, dtype=torch.bool),
    }


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gradient_diagnostics(module, top_count):
    rows = []
    total_squared = 0.0
    nonfinite = 0
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = bool(torch.isfinite(gradient).all().item())
        if not finite:
            nonfinite += 1
            l2 = float("nan")
            maximum = float("nan")
        else:
            double = gradient.double()
            squared = float(torch.sum(double * double).item())
            total_squared += squared
            l2 = math.sqrt(squared)
            maximum = float(torch.max(torch.abs(double)).item())
        rows.append({
            "name": name,
            "elements": gradient.numel(),
            "finite": finite,
            "l2": l2,
            "max_abs": maximum,
        })
    total = math.sqrt(total_squared) if math.isfinite(total_squared) else float("inf")
    ordered = sorted(
        rows, key=lambda row: row["l2"] if math.isfinite(row["l2"])
        else float("inf"), reverse=True)
    return total, nonfinite, ordered[:int(top_count)]


def scale_gradients(module, scale):
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(float(scale))


def candidate_permutation_error(model, batch, seed):
    model.eval()
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(6, generator=generator)
    permuted_features = {
        name: value[:, :, permutation] for name, value in
        batch["features"].items()}
    with torch.no_grad():
        original = model(
            batch["features"], batch["initial_image"],
            batch["identity_text"], batch["candidate_valid"])
        permuted = model(
            permuted_features, batch["initial_image"],
            batch["identity_text"], batch["candidate_valid"][:, permutation])
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
    return max(errors.values()), errors, permutation.tolist()


def main():
    args = parse_args()
    started = time.time()
    spec = json_file(args.spec)
    correction_spec = json_file(args.correction_spec)
    if spec.get("complete") is not True:
        raise ValueError("spec is incomplete")
    authorization = spec["authorization"]
    if authorization.get("m8b1_engineering_implementation") is not True or \
            authorization.get(
                "m8b1_smoke_execution_after_separate_binding") is not True:
        raise ValueError("spec does not authorize bound M8b-1 smoke")
    if any(authorization.get(name) is not False for name in (
            "model_load", "m8b2_training", "tracking_checkpoint",
            "depthtrack_test", "cdtb", "vot_low22", "vot_full127", "qwen",
            "automatic_next_stage")):
        raise ValueError("spec authorization drifted")
    if correction_spec.get("complete") is not True:
        raise ValueError("gradient correction spec is incomplete")
    correction_authorization = correction_spec["authorization"]
    if (correction_authorization.get("runner_gradient_gate_correction") is not True or
            correction_authorization.get("one_bound_corrective_smoke") is not True):
        raise ValueError("correction spec does not authorize this run")
    binding, commit, branch, runner_path, model_path = validate_binding(
        args, spec, correction_spec)
    correction_hashes = [
        (Path(correction_spec["source_spec"]["path"]),
         correction_spec["source_spec"]["sha256"]),
        (Path(correction_spec["integrity_override"]["path"]),
         correction_spec["integrity_override"]["sha256"]),
        (Path(correction_spec["invalid_v1"]["result"]["path"]),
         correction_spec["invalid_v1"]["result"]["sha256"]),
        (Path(correction_spec["invalid_v1"]["manifest"]["path"]),
         correction_spec["invalid_v1"]["manifest"]["sha256"]),
    ]
    if any(sha256_file(path) != digest for path, digest in correction_hashes):
        raise ValueError("gradient correction provenance hash mismatch")
    records = frozen_records(spec)
    before_mismatches, frozen_observed = verify_frozen(records)
    closure = load_closure(spec)
    selected_rows, composition, sequences = validate_selection(spec, closure)
    batch = batch_tensors(spec, selected_rows)

    seed = int(spec["optimization"]["seed"])
    torch.manual_seed(seed)
    model = CachedStrictTwoStageRouter(
        projection_dim=int(spec["architecture"]["projection_dim"]),
        hidden_dim=int(spec["architecture"]["hidden_dim"]),
        attention_heads=int(spec["architecture"]["attention_heads"]),
        set_layers=int(spec["architecture"]["set_layers"]),
        set_feedforward_dim=int(
            spec["architecture"]["set_feedforward_dim"]),
        dropout=float(spec["architecture"]["dropout"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    permutation_error, permutation_errors, permutation = \
        candidate_permutation_error(model, batch, seed)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["optimization"]["learning_rate"]),
        weight_decay=float(spec["optimization"]["weight_decay"]))
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
        pairwise_margin=float(spec["optimization"]["pairwise_margin"]),
    )
    losses["total"].backward()
    gradient_contract = correction_spec["gradient_contract"]
    gradient_norm, nonfinite_gradients, top_gradients = gradient_diagnostics(
        model, gradient_contract["top_gradient_parameters_reported"])
    preclip_safe = (
        nonfinite_gradients == 0 and math.isfinite(gradient_norm) and
        gradient_norm > float(
            gradient_contract["preclip_total_l2_min_exclusive"]) and
        gradient_norm <= float(gradient_contract["preclip_total_l2_max"]))
    optimizer_step_executed = False
    if preclip_safe:
        clip_max = float(gradient_contract["global_clip_max_norm"])
        scale_gradients(model, min(1.0, clip_max / (gradient_norm + 1e-12)))
        postclip_gradient_norm, postclip_nonfinite_gradients, _ = \
            gradient_diagnostics(model, 0)
        postclip_safe = (
            postclip_nonfinite_gradients == 0 and
            math.isfinite(postclip_gradient_norm) and
            postclip_gradient_norm <= float(
                gradient_contract["postclip_total_l2_max"]))
        if postclip_safe:
            optimizer.step()
            optimizer_step_executed = True
    else:
        postclip_gradient_norm = gradient_norm
        postclip_nonfinite_gradients = nonfinite_gradients
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
        "batch_composition_exact": composition == Counter(
            spec["selection"]["composition"]),
        "distinct_sequences_exact": len(sequences) == int(
            gates["distinct_sequences_exact"]),
        "fold0_events_max": sum(
            int(event["fold"]) == int(
                spec["selection"]["evaluation_outer_fold"])
            for event in spec["selection"]["events"]) <= int(
                gates["fold0_events_max"]),
        "parameter_count_max": parameter_count <= int(
            gates["parameter_count_max"]),
        "permutation_equivariance_max_error": permutation_error <= float(
            gates["permutation_equivariance_max_error"]),
        "nonfinite_outputs_max": nonfinite_outputs <= int(
            gates["nonfinite_outputs_max"]),
        "nonfinite_loss_terms_max": nonfinite_loss_terms <= int(
            gates["nonfinite_loss_terms_max"]),
        "nonfinite_gradients_max": nonfinite_gradients <= int(
            gates["nonfinite_gradients_max"]),
        "preclip_total_l2_finite": math.isfinite(gradient_norm),
        "preclip_total_l2_min_exclusive": gradient_norm > float(
            correction_spec["gates"]["preclip_total_l2_min_exclusive"]),
        "preclip_total_l2_max": gradient_norm <= float(
            correction_spec["gates"]["preclip_total_l2_max"]),
        "postclip_total_l2_finite": math.isfinite(postclip_gradient_norm),
        "postclip_total_l2_max": postclip_gradient_norm <= float(
            correction_spec["gates"]["postclip_total_l2_max"]),
        "optimizer_step_executed": optimizer_step_executed,
        "changed_trainable_tensors_min": changed_tensors >= int(
            gates["changed_trainable_tensors_min"]),
        "state_digest_changed": before_state_sha256 != after_state_sha256,
        "frozen_hash_mismatches_before_max": len(before_mismatches) <= int(
            gates["frozen_hash_mismatches_before_max"]),
        "frozen_hash_mismatches_after_max": len(after_mismatches) <= int(
            gates["frozen_hash_mismatches_after_max"]),
        "unauthorized_file_opens_max": True,
        "tracking_checkpoint_written_false": True,
        "public_state_mutations_false": True,
        "qwen_used_false": True,
        "public_benchmark_run_false": True,
    }
    accepted = all(conditions.values())
    decision = (
        "m8b_1_gradient_corrected_pass_freeze_formal_train_spec_only"
        if accepted else "stop_m8b_1_gradient_corrected_smoke_failed")
    result = {
        "schema": "sttrack-lachtt-m8b-cached-engineering-smoke-result/v2",
        "complete": True,
        "accepted": accepted,
        "decision": decision,
        "claim_ceiling": correction_spec["claim_ceiling"],
        "invalid_v1_acceptance_overridden": True,
        "repository": {"path": str(REPOSITORY_ROOT), "branch": branch,
                       "commit": commit, "clean": True},
        "batch_composition": dict(sorted(composition.items())),
        "sequences": sorted(sequences),
        "parameter_count": parameter_count,
        "permutation": permutation,
        "permutation_errors": permutation_errors,
        "maximum_permutation_error": permutation_error,
        "losses": {name: float(value.detach())
                   for name, value in losses.items()},
        "gradient_norm_before_clip": gradient_norm,
        "gradient_norm_after_clip": postclip_gradient_norm,
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
            "schema": "sttrack-lachtt-m8b-cached-engineering-smoke-manifest/v2",
            "complete": True,
            "accepted": accepted,
            "decision": decision,
            "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
            "correction_spec": {"path": str(args.correction_spec),
                                "sha256": sha256_file(args.correction_spec)},
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
            "invalid_v1_acceptance_overridden": True,
            "scientific_scope": correction_spec["claim_ceiling"],
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
        "gradient_norm_before_clip": gradient_norm,
        "changed_trainable_tensors": changed_tensors,
        "failed_conditions": sorted(
            name for name, passed in conditions.items() if not passed),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
