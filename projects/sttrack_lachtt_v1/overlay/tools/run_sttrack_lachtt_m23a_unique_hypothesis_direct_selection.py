#!/usr/bin/env python3
"""M23a consumed-fold1 development run for unique RGB-D hypotheses."""

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from pathlib import Path


sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
M22_RUNNER_PATH = REPOSITORY_ROOT / "tools" / (
    "run_sttrack_lachtt_m22a_sequence_disjoint_causal_survival.py")
BRANCH_ORDER = (
    "current_peak0", "current_peak1",
    "last_reliable_peak0", "last_reliable_peak1",
    "velocity_peak0", "velocity_peak1",
)
OUTPUT_FILES = (
    "development_predictions.jsonl.gz",
    "manifest.json",
    "result.json",
    "training_trace.jsonl.gz",
)
SPEC_SCHEMA = "sttrack-lachtt-m23a-unique-hypothesis-direct-selection-spec/v1"
PREAUDIT_SCHEMA = "sttrack-lachtt-m23a-preexecution-audit/v1"
FORBIDDEN_NUMERIC_TARGET_PATHS = {
    Path("/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/"
         "labeled_actions.jsonl.gz").resolve(),
}


class ContractError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--preaudit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": stat_result.st_size,
            "sha256": digest.hexdigest()}


def validate_file_record(record):
    path = Path(record["path"]).resolve()
    if file_record(path) != record:
        raise ContractError("file identity drifted: %s" % path)


def load_json_snapshot(path):
    record = file_record(path)
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if file_record(path) != record:
        raise ContractError("JSON changed while reading: %s" % path)
    return value, record


def load_jsonl_gz(record):
    validate_file_record(record)
    rows = []
    with gzip.open(record["path"], "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    validate_file_record(record)
    return rows


def git_output(*args):
    return subprocess.check_output(
        ("git", "-C", str(REPOSITORY_ROOT), *args),
        text=True).strip()


def stable_int(*items):
    payload = "\0".join(str(item) for item in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_spec(spec, args, spec_record, runner_record):
    if (spec.get("schema") != SPEC_SCHEMA or
            spec.get("complete") is not True or
            spec.get("created_before_execution") is not True):
        raise ContractError("M23a spec identity drifted")
    expected_output = Path(spec["output"]["root"]).resolve()
    if args.output.resolve() != expected_output or args.output.exists():
        raise ContractError("M23a output precondition drifted")
    expected_training = {
        "seed": 20260923,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 12,
        "event_batch_size": 8,
        "optimizer_steps_total": 768,
        "natural_prior_pass": True,
        "sequence_inverse_event_loss_weight": True,
        "class_balanced_resampling": False,
        "scheduler": False,
        "early_stopping": False,
        "augmentation": False,
        "warm_start_checkpoint": False,
        "gradient_clip_norm": 5.0,
        "loss_weights": {
            "benefit_bce": 1.0,
            "catastrophe_bce": 1.0,
            "strict_label_pairwise_rank": 0.5,
        },
    }
    if spec.get("training") != expected_training:
        raise ContractError("M23a training constants drifted")
    expected_architecture = {
        "class": "UniqueHypothesisSelectiveRouter",
        "benefit_projection_seed": 20260923,
        "catastrophe_projection_seed": 20261923,
        "residual_scale": 0.1,
        "catastrophe_penalty": 4.0,
        "expected_parameters": 106434,
        "candidate_count": 6,
        "cached_horizon": 5,
        "exact_bbox_scalars_per_signature": 20,
        "duplicate_relation_aggregation": "arithmetic_mean",
        "duplicate_representative": "lowest_canonical_role_id",
        "utility_safety_parameter_overlap_required": 0,
    }
    if spec.get("architecture") != expected_architecture:
        raise ContractError("M23a architecture constants drifted")
    expected_policy = {
        "candidate_selection": "maximum direct dominance score",
        "dominance": "benefit_probability - 4 * catastrophe_probability",
        "top1_top2_dominance_margin_min": 0.1,
        "benefit_probability_min": 0.8,
        "catastrophe_probability_max": 0.05,
        "otherwise": "abstain and preserve protected branch",
        "threshold_scan": False,
    }
    if spec.get("development_policy") != expected_policy:
        raise ContractError("M23a policy drifted")
    expected_gates = {
        "selected_actions_min": 5,
        "beneficial_actions_min": 4,
        "beneficial_sequences_min": 3,
        "beneficial_precision_min": 0.95,
        "catastrophic_actions_max": 0,
        "selected_mean_true_h10_gain_min": 0.2,
        "selected_branch_aggregate_gt_public": True,
        "all_abstain_is_not_pass": True,
    }
    if spec.get("scientific_gates") != expected_gates:
        raise ContractError("M23a scientific gates drifted")
    supplied_records = (
        list(spec.get("source", {}).get("dependencies", [])) +
        list(spec.get("m22_inputs", {}).values()))
    supplied_paths = {
        Path(record["path"]).resolve() for record in supplied_records
        if isinstance(record, dict) and isinstance(record.get("path"), str)}
    if supplied_paths & FORBIDDEN_NUMERIC_TARGET_PATHS:
        raise ContractError("forbidden fold0/full numeric target was supplied")
    if spec["source"]["runner"] != runner_record:
        raise ContractError("runner/spec identity drifted")
    for record in spec["source"]["dependencies"]:
        validate_file_record(record)
    if Path(spec["source"]["repository"]["path"]).resolve() != REPOSITORY_ROOT:
        raise ContractError("repository path drifted")
    if spec["experiment_plan"]["path"] not in {
            record["path"] for record in spec["source"]["dependencies"]}:
        raise ContractError("experiment plan is not bound")
    return spec_record


def validate_preaudit(preaudit_path, spec_record, runner_record, spec):
    audit, audit_record = load_json_snapshot(preaudit_path)
    repository = spec["source"]["repository"]
    if (audit.get("schema") != PREAUDIT_SCHEMA or
            str(audit.get("overall_verdict", "")).upper() != "PASS" or
            str(audit.get("integrity_verdict", "")).upper() != "PASS" or
            audit.get("authorization", {}).get("m23a_execution") is not True):
        raise ContractError("M23a preaudit did not authorize")
    expected = {
        "spec_sha256": spec_record["sha256"],
        "runner_sha256": runner_record["sha256"],
        "repository_commit": repository["commit"],
    }
    if audit.get("audited_identity") != expected:
        raise ContractError("M23a preaudit identity drifted")
    if audit.get("claim_ceiling") != spec.get("claim_ceiling"):
        raise ContractError("M23a claim ceiling drifted")
    return audit, audit_record


def load_components():
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    models_package = types.ModuleType("lib.models")
    models_package.__path__ = [str(REPOSITORY_ROOT / "lib" / "models")]
    sttrack_package = types.ModuleType("lib.models.sttrack")
    sttrack_package.__path__ = [
        str(REPOSITORY_ROOT / "lib" / "models" / "sttrack")]
    sys.modules["lib.models"] = models_package
    sys.modules["lib.models.sttrack"] = sttrack_package
    import torch
    from tools import (
        run_sttrack_lachtt_m22a_sequence_disjoint_causal_survival as m22)
    m22.load_project_components()
    from lib.models.sttrack.lachtt_unique_hypothesis_selective_router import (
        UniqueHypothesisSelectiveRouter,
    )
    return torch, m22, UniqueHypothesisSelectiveRouter


def event_key(row):
    return row["sequence"], int(row["event_id"]), int(row["trigger_frame"])


def heldout_groups_from_m22(rows):
    groups = {}
    for row in rows:
        key = event_key(row)
        actions = sorted(row["actions"], key=lambda value: int(
            value["candidate_role_id"]))
        if tuple(action["branch_id"] for action in actions) != BRANCH_ORDER:
            raise ContractError("M22 heldout role order drifted")
        groups[key] = [{
            "sequence": row["sequence"],
            "event_id": int(row["event_id"]),
            "trigger_frame": int(row["trigger_frame"]),
            "branch_id": action["branch_id"],
            "candidate_role_id": int(action["candidate_role_id"]),
            "strict_event_class": row["strict_event_class"],
            "strict_label": action["strict_label"],
            "targets": action["actual_trajectory"],
        } for action in actions]
    if len(groups) != 121:
        raise ContractError("M22 heldout prediction count drifted")
    return groups


def load_duplicate_groups(m22, m22_spec, required, targets):
    groups = {}
    for shard in m22_spec["frozen_inputs"]["collection_shards"]:
        ledger_record = shard["event_ledger"]
        for row in m22.load_verified_jsonl(ledger_record):
            key = event_key(row)
            if key not in required:
                continue
            trajectory = row.get("trajectory", [])[:5]
            if len(trajectory) != 5:
                raise ContractError("five-frame trajectory is absent")
            signatures = defaultdict(list)
            for role_id, role in enumerate(BRANCH_ORDER):
                signature = []
                for age in trajectory:
                    branch_map = {branch["name"]: branch
                                  for branch in age["branches"]}
                    if set(branch_map) != set(BRANCH_ORDER):
                        raise ContractError("trajectory role set drifted")
                    signature.extend(float(value)
                                     for value in branch_map[role]["bbox"])
                if len(signature) != 20:
                    raise ContractError("bbox signature width drifted")
                signatures[tuple(signature)].append(role_id)
            target_rows = sorted(targets[key], key=lambda value: int(
                value["candidate_role_id"]))
            event_groups = []
            for members in signatures.values():
                reference = target_rows[members[0]]
                reference_target = json.dumps(
                    reference["targets"], sort_keys=True, separators=(",", ":"))
                for member in members:
                    row_target = target_rows[member]
                    if (row_target["strict_label"] != reference["strict_label"] or
                            json.dumps(row_target["targets"], sort_keys=True,
                                       separators=(",", ":")) != reference_target):
                        raise ContractError("duplicate target conflict")
                event_groups.append(tuple(sorted(members)))
            groups[key] = tuple(sorted(event_groups, key=lambda value: value[0]))
    if set(groups) != required:
        raise ContractError("duplicate group event closure drifted")
    return groups


def aggregate_relation(torch, relation, groups):
    differences, gates, scalar = [value.clone() for value in relation]
    valid = torch.zeros(6, dtype=torch.bool)
    for members in groups:
        representative = members[0]
        valid[representative] = True
        index = torch.tensor(members, dtype=torch.int64)
        differences[:, representative] = differences[:, index].mean(dim=1)
        gates[:, representative] = gates[:, index].mean(dim=1)
        scalar[:, representative] = scalar[:, index].mean(dim=1)
    return differences, gates, scalar, valid


def target_tensors(torch, rows, valid):
    rows = sorted(rows, key=lambda value: int(value["candidate_role_id"]))
    labels = [row["strict_label"] for row in rows]
    return {
        "beneficial": torch.tensor(
            [label == "beneficial" for label in labels], dtype=torch.float32),
        "catastrophic": torch.tensor(
            [label == "catastrophic" for label in labels], dtype=torch.float32),
        "label_score": torch.tensor([
            1.0 if label == "beneficial" else
            -1.0 if label == "catastrophic" else 0.0
            for label in labels], dtype=torch.float32),
        "gain": torch.tensor([
            float(row["targets"]["10"]["gain"]) for row in rows],
            dtype=torch.float32),
        "branch_mean": torch.tensor([
            float(row["targets"]["10"]["branch_mean_iou"]) for row in rows],
            dtype=torch.float32),
        "public_mean": torch.tensor([
            float(row["targets"]["10"]["public_mean_iou"]) for row in rows],
            dtype=torch.float32),
        "labels": labels,
        "valid": valid,
        "event_class": rows[0]["strict_event_class"],
    }


def epoch_batches(keys, seed, epoch, batch_size):
    ordered = sorted(keys, key=lambda key: (
        stable_int(seed, epoch, *key), key))
    return [ordered[index:index + batch_size]
            for index in range(0, len(ordered), batch_size)]


def candidate_permutation(torch, seed, epoch, step, batch_index, key):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_int(
        seed, epoch, step, batch_index, *key) % (2 ** 63 - 1))
    return torch.randperm(6, generator=generator)


def make_batch(torch, keys, relations, targets, sequence_counts,
               seed, epoch, step):
    fields = defaultdict(list)
    for batch_index, key in enumerate(keys):
        permutation = candidate_permutation(
            torch, seed, epoch, step, batch_index, key)
        relation = relations[key]
        target = targets[key]
        fields["differences"].append(relation[0][:, permutation])
        fields["gates"].append(relation[1][:, permutation])
        fields["scalar"].append(relation[2][:, permutation])
        fields["candidate_valid"].append(target["valid"][permutation])
        fields["role_ids"].append(permutation)
        for name in ("beneficial", "catastrophic", "label_score"):
            fields[name].append(target[name][permutation])
        fields["event_weight"].append(1.0 / sequence_counts[key[0]])
    batch = {
        name: torch.stack(value) for name, value in fields.items()
        if name != "event_weight"}
    batch["event_weight"] = torch.tensor(
        fields["event_weight"], dtype=torch.float32)
    batch["event_weight"] /= batch["event_weight"].sum()
    return batch


def masked_event_mean(torch, value, valid):
    return (value * valid.float()).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def pairwise_event_loss(torch, score, label_score, valid):
    output = []
    zero = score.sum() * 0.0
    for event in range(score.shape[0]):
        terms = []
        for left in range(6):
            for right in range(left + 1, 6):
                if not (valid[event, left] and valid[event, right]):
                    continue
                difference = label_score[event, left] - label_score[event, right]
                if abs(float(difference)) <= 1.0e-12:
                    continue
                signed = difference.sign() * (
                    score[event, left] - score[event, right])
                terms.append(torch.nn.functional.softplus(-signed))
        output.append(torch.stack(terms).mean() if terms else zero)
    return torch.stack(output)


def forward_loss(torch, model, batch, weights):
    outputs = model(
        batch["differences"], batch["gates"], batch["scalar"],
        batch["candidate_valid"], batch["role_ids"])
    benefit_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["benefit_logit"], batch["beneficial"], reduction="none")
    catastrophe_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["catastrophe_logit"], batch["catastrophic"], reduction="none")
    benefit = (masked_event_mean(
        torch, benefit_raw, batch["candidate_valid"]) *
        batch["event_weight"]).sum()
    catastrophe = (masked_event_mean(
        torch, catastrophe_raw, batch["candidate_valid"]) *
        batch["event_weight"]).sum()
    rank_events = pairwise_event_loss(
        torch, outputs["dominance_score"], batch["label_score"],
        batch["candidate_valid"])
    rank = (rank_events * batch["event_weight"]).sum()
    losses = {
        "benefit_bce": benefit,
        "catastrophe_bce": catastrophe,
        "strict_label_pairwise_rank": rank,
    }
    total = sum(float(weights[name]) * value for name, value in losses.items())
    if not math.isfinite(float(total.detach())):
        raise ContractError("M23a loss is non-finite")
    return outputs, {**losses, "total": total}


def identity_batch(torch, keys, relations, targets):
    return {
        "differences": torch.stack([relations[key][0] for key in keys]),
        "gates": torch.stack([relations[key][1] for key in keys]),
        "scalar": torch.stack([relations[key][2] for key in keys]),
        "candidate_valid": torch.stack([targets[key]["valid"] for key in keys]),
        "role_ids": torch.arange(6, dtype=torch.int64).expand(len(keys), -1),
    }


def model_outputs(torch, model, batch):
    with torch.no_grad():
        return model(
            batch["differences"], batch["gates"], batch["scalar"],
            batch["candidate_valid"], batch["role_ids"])


def policy_decision(torch, outputs, index, valid, policy):
    score = outputs["dominance_score"][index].masked_fill(~valid, -1.0e9)
    valid_count = int(valid.sum())
    if valid_count < 2:
        raise ContractError("M23a requires at least two unique hypotheses")
    top = torch.topk(score, k=2).indices
    role = int(top[0])
    margin = float(score[top[0]] - score[top[1]])
    benefit = float(outputs["benefit_probability"][index, role])
    catastrophe = float(outputs["catastrophe_probability"][index, role])
    accepted = (
        margin >= float(policy["top1_top2_dominance_margin_min"]) and
        benefit >= float(policy["benefit_probability_min"]) and
        catastrophe <= float(policy["catastrophe_probability_max"]))
    return {
        "top_role_id": role,
        "dominance": float(score[role]),
        "margin": margin,
        "benefit_probability": benefit,
        "catastrophe_probability": catastrophe,
        "selected_role_id": role if accepted else None,
    }


def permutation_audit(torch, model, relation, target, trials=8):
    base_batch = identity_batch(torch, ["event"], {"event": relation},
                                {"event": target})
    base = model_outputs(torch, model, base_batch)
    maximum = 0.0
    selection_mismatch = 0
    for trial in range(trials):
        permutation = candidate_permutation(
            torch, 20260923, 999, trial, 0, ("audit", trial, 0))
        inverse = torch.argsort(permutation)
        batch = {
            "differences": relation[0][:, permutation].unsqueeze(0),
            "gates": relation[1][:, permutation].unsqueeze(0),
            "scalar": relation[2][:, permutation].unsqueeze(0),
            "candidate_valid": target["valid"][permutation].unsqueeze(0),
            "role_ids": permutation.unsqueeze(0),
        }
        output = model_outputs(torch, model, batch)
        for name in base:
            restored = output[name][:, inverse]
            maximum = max(maximum, float((restored - base[name]).abs().max()))
            if not torch.equal(restored, base[name]):
                selection_mismatch += 1
    return {"maximum_absolute_error": maximum,
            "non_equal_outputs": selection_mismatch}


def event_permutation_audit(torch, model, keys, relations, targets, trials=8):
    base_batch = identity_batch(torch, keys, relations, targets)
    base = model_outputs(torch, model, base_batch)
    maximum = 0.0
    non_equal = 0
    for trial in range(trials):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_int(
            20260923, "event-permutation-audit", trial) % (2 ** 63 - 1))
        permutation = torch.randperm(len(keys), generator=generator)
        inverse = torch.argsort(permutation)
        permuted_keys = [keys[int(index)] for index in permutation]
        output = model_outputs(
            torch, model, identity_batch(
                torch, permuted_keys, relations, targets))
        for name in base:
            restored = output[name][inverse]
            maximum = max(
                maximum, float((restored - base[name]).abs().max()))
            if not torch.equal(restored, base[name]):
                non_equal += 1
    return {"maximum_absolute_error": maximum,
            "non_equal_outputs": non_equal}


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_jsonl_gz(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(
                    row, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False) + "\n").encode(
                        "utf-8"))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def publish(output, result, trace, predictions):
    temporary = Path(tempfile.mkdtemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)))
    try:
        atomic_jsonl_gz(temporary / "training_trace.jsonl.gz", trace)
        atomic_jsonl_gz(temporary / "development_predictions.jsonl.gz", predictions)
        atomic_json(temporary / "result.json", result)
        manifest = {
            "schema": "sttrack-lachtt-m23a-output-manifest/v1",
            "complete": True,
            "accepted": result["accepted"],
            "decision": result["decision"],
            "files": {},
        }
        for name in ("training_trace.jsonl.gz", "development_predictions.jsonl.gz",
                     "result.json"):
            manifest["files"][name] = file_record(temporary / name)
            manifest["files"][name]["path"] = str(output / name)
        atomic_json(temporary / "manifest.json", manifest)
        if tuple(sorted(path.name for path in temporary.iterdir())) != tuple(
                sorted(OUTPUT_FILES)):
            raise ContractError("M23a output set drifted")
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.chmod(0o555)
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.preaudit = args.preaudit.resolve()
    args.output = args.output.resolve()
    started = time.time()
    runner_record = file_record(Path(__file__).resolve())
    spec, spec_record = load_json_snapshot(args.spec)
    validate_spec(spec, args, spec_record, runner_record)
    audit, audit_record = validate_preaudit(
        args.preaudit, spec_record, runner_record, spec)
    repository = spec["source"]["repository"]
    if (git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("branch", "--show-current") != repository["branch"] or
            git_output("status", "--porcelain")):
        raise ContractError("repository identity drifted before M23a")

    torch, m22, Router = load_components()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(spec["training"]["seed"]))

    dependencies = {record["path"]: record
                    for record in spec["source"]["dependencies"]}
    m22_spec = m22.load_verified_json(spec["m22_inputs"]["spec"])
    delayed_target_path = Path(
        m22_spec["delayed_heldout_inputs"]["labeled_actions"]["path"]
    ).resolve()
    if delayed_target_path not in FORBIDDEN_NUMERIC_TARGET_PATHS:
        raise ContractError("M22 delayed numeric target identity drifted")
    m22_binding = m22.load_verified_json(spec["m22_inputs"]["binding"])
    m22_result = m22.load_verified_json(spec["m22_inputs"]["result"])
    m22_audit = m22.load_verified_json(spec["m22_inputs"]["result_audit"])
    if (m22_result.get("engineering_pass") is not True or
            m22_result.get("scientific_pass") is not False or
            m22_result.get("scientific_summary", {}).get("selected_actions") != 0 or
            str(m22_audit.get("overall_verdict", "")).upper() != "PASS"):
        raise ContractError("M22 prerequisite result drifted")
    m22_predictions = load_jsonl_gz(spec["m22_inputs"]["predictions"])
    heldout_groups = heldout_groups_from_m22(m22_predictions)

    m22.validate_frozen_receipts(m22_spec)
    collection, sequence_anchors = m22.load_collection_index(m22_spec)
    _, split_entries = m22.load_split_ledger(m22_spec)
    training_groups, _, _ = m22.load_training_targets(m22_spec, split_entries)
    training_sequences = {key[0] for key in training_groups}
    heldout_sequences = {key[0] for key in heldout_groups}
    if (len(training_groups) != 507 or len(heldout_groups) != 121 or
            len(training_sequences) != 76 or len(heldout_sequences) != 20 or
            training_sequences & heldout_sequences):
        raise ContractError("M23a split/count contract drifted")
    for key in heldout_groups:
        if (split_entries[key]["partition"] != "heldout" or
                split_entries[key]["strict_h10_available"] is not True):
            raise ContractError("M23a heldout source drifted")

    all_groups = {**training_groups, **heldout_groups}
    duplicate_groups = load_duplicate_groups(
        m22, m22_spec, set(all_groups), all_groups)
    unique_counts = {key: len(value) for key, value in duplicate_groups.items()}
    if (min(unique_counts.values()) != 2 or
            max(unique_counts.values()) != 6 or
            sum(value < 6 for value in unique_counts.values()) != 580 or
            sum(unique_counts[key] for key in training_groups) != 2106 or
            sum(unique_counts[key] for key in heldout_groups) != 500):
        raise ContractError("M23a unique hypothesis counts drifted")

    native_index = m22.load_native_index(m22_spec)
    required_sequences = sorted({key[0] for key in split_entries})
    clip_binding, _ = m22.validate_anchor_binding(
        m22_binding, sequence_anchors, native_index, required_sequences)
    clip_cache, native_cache, loaded_features = {}, {}, {}
    relations = {}
    targets = {}
    for key in sorted(all_groups):
        relation = m22.relation_for_event(
            m22_spec, collection, sequence_anchors, native_index,
            clip_binding, key, clip_cache, native_cache, loaded_features)
        relation = aggregate_relation(
            torch, relation, duplicate_groups[key])
        relations[key] = relation[:3]
        targets[key] = target_tensors(
            torch, all_groups[key], relation[3])

    architecture = spec["architecture"]
    model = Router(
        benefit_projection_seed=architecture["benefit_projection_seed"],
        catastrophe_projection_seed=architecture["catastrophe_projection_seed"],
        residual_scale=architecture["residual_scale"],
        catastrophe_penalty=architecture["catastrophe_penalty"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    utility_parameters = list(model.utility_parameters())
    safety_parameters = list(model.safety_parameters())
    parameter_overlap = len(
        {id(value) for value in utility_parameters} &
        {id(value) for value in safety_parameters})
    if (parameter_count != architecture["expected_parameters"] or
            parameter_overlap != 0 or
            any(parameter.dtype != torch.float32 or
                parameter.device.type != "cpu" for parameter in model.parameters())):
        raise ContractError("M23a model identity drifted")
    initial_state = m22.state_digest(model)

    training = spec["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"])
    sequence_counts = Counter(key[0] for key in training_groups)
    trace = []
    step = 0
    model.train()
    for epoch in range(training["epochs"]):
        batches = epoch_batches(
            list(training_groups), training["seed"], epoch,
            training["event_batch_size"])
        if len(batches) != 64:
            raise ContractError("M23a epoch batch count drifted")
        for epoch_step, keys in enumerate(batches):
            step += 1
            batch = make_batch(
                torch, keys, relations, targets, sequence_counts,
                training["seed"], epoch, epoch_step)
            optimizer.zero_grad(set_to_none=True)
            outputs, losses = forward_loss(
                torch, model, batch, training["loss_weights"])
            if any(not torch.isfinite(value).all().item()
                   for value in outputs.values()):
                raise ContractError("M23a output is non-finite")
            losses["total"].backward()
            preclip, nonfinite, _ = m22.gradient_diagnostics(model, 0)
            if nonfinite or not math.isfinite(preclip) or preclip <= 0 or preclip > 1000:
                raise ContractError("M23a preclip gradient gate failed")
            maximum = training["gradient_clip_norm"]
            m22.scale_gradients(model, min(1.0, maximum / (preclip + 1e-12)))
            postclip, post_nonfinite, _ = m22.gradient_diagnostics(model, 0)
            if post_nonfinite or not math.isfinite(postclip) or postclip > 5.000001:
                raise ContractError("M23a postclip gradient gate failed")
            optimizer.step()
            trace.append({
                "record_type": "optimizer_step",
                "global_step": step,
                "epoch": epoch,
                "epoch_step": epoch_step,
                "batch_size": len(keys),
                "losses": {name: float(value.detach())
                           for name, value in losses.items()},
                "preclip_total_l2": preclip,
                "postclip_total_l2": postclip,
                "optimizer_step_executed": True,
            })
    if step != 768 or len(trace) != 768:
        raise ContractError("M23a optimizer step count drifted")
    final_state = m22.state_digest(model)

    model.eval()
    heldout_keys = sorted(heldout_groups)
    predictions = []
    selected_rows = []
    for start in range(0, len(heldout_keys), 32):
        keys = heldout_keys[start:start + 32]
        batch = identity_batch(torch, keys, relations, targets)
        outputs = model_outputs(torch, model, batch)
        for index, key in enumerate(keys):
            decision = policy_decision(
                torch, outputs, index, targets[key]["valid"],
                spec["development_policy"])
            selected_role = decision["selected_role_id"]
            selected_label = (targets[key]["labels"][selected_role]
                              if selected_role is not None else "abstain")
            row = {
                "record_type": "fold1_development_prediction",
                "sequence": key[0], "event_id": key[1],
                "trigger_frame": key[2],
                "strict_event_class": targets[key]["event_class"],
                "unique_hypotheses": unique_counts[key],
                "decision": decision,
                "selected_strict_label": selected_label,
                "selected_actual_h10_gain": (
                    float(targets[key]["gain"][selected_role])
                    if selected_role is not None else 0.0),
                "selected_actual_h10_branch_mean_iou": (
                    float(targets[key]["branch_mean"][selected_role])
                    if selected_role is not None else None),
                "selected_actual_h10_public_mean_iou": (
                    float(targets[key]["public_mean"][selected_role])
                    if selected_role is not None else None),
            }
            predictions.append(row)
            if selected_role is not None:
                selected_rows.append(row)

    beneficial = [row for row in selected_rows
                  if row["selected_strict_label"] == "beneficial"]
    catastrophic = [row for row in selected_rows
                    if row["selected_strict_label"] == "catastrophic"]
    neutral = [row for row in selected_rows
               if row["selected_strict_label"] == "neutral"]
    selected_count = len(selected_rows)
    precision = len(beneficial) / selected_count if selected_count else 0.0
    mean_gain = (sum(row["selected_actual_h10_gain"] for row in selected_rows) /
                 selected_count if selected_count else None)
    branch_aggregate = (sum(row["selected_actual_h10_branch_mean_iou"]
                            for row in selected_rows) / selected_count
                        if selected_count else None)
    public_aggregate = (sum(row["selected_actual_h10_public_mean_iou"]
                            for row in selected_rows) / selected_count
                        if selected_count else None)
    summary = {
        "selected_actions": selected_count,
        "beneficial_actions": len(beneficial),
        "neutral_actions": len(neutral),
        "catastrophic_actions": len(catastrophic),
        "beneficial_sequences": len({row["sequence"] for row in beneficial}),
        "selected_sequences": len({row["sequence"] for row in selected_rows}),
        "beneficial_precision": precision,
        "selected_mean_true_h10_gain": mean_gain,
        "selected_branch_aggregate_h10_mean_iou": branch_aggregate,
        "selected_public_aggregate_h10_mean_iou": public_aggregate,
    }
    gates = spec["scientific_gates"]
    scientific_conditions = {
        "selected_actions_min": selected_count >= gates["selected_actions_min"],
        "beneficial_actions_min": len(beneficial) >= gates["beneficial_actions_min"],
        "beneficial_sequences_min": len({row["sequence"] for row in beneficial}) >=
            gates["beneficial_sequences_min"],
        "beneficial_precision_min": precision >= gates["beneficial_precision_min"],
        "catastrophic_actions_max": len(catastrophic) <=
            gates["catastrophic_actions_max"],
        "selected_mean_true_h10_gain_min": mean_gain is not None and
            mean_gain >= gates["selected_mean_true_h10_gain_min"],
        "selected_branch_aggregate_gt_public": branch_aggregate is not None and
            public_aggregate is not None and branch_aggregate > public_aggregate,
        "all_abstain_is_not_pass": selected_count > 0,
    }
    audit_key = heldout_keys[0]
    candidate_permutation_result = permutation_audit(
        torch, model, relations[audit_key], targets[audit_key])
    event_permutation_result = event_permutation_audit(
        torch, model, heldout_keys[:16], relations, targets)
    permutation = {
        "candidate": candidate_permutation_result,
        "event": event_permutation_result,
    }
    engineering_conditions = {
        "optimizer_steps_exact": step == 768,
        "trace_rows_exact": len(trace) == 768,
        "parameter_count_exact": parameter_count == 106434,
        "utility_safety_parameter_overlap_zero": parameter_overlap == 0,
        "candidate_permutation_exact":
            candidate_permutation_result["maximum_absolute_error"] == 0.0,
        "candidate_permutation_all_equal":
            candidate_permutation_result["non_equal_outputs"] == 0,
        "event_permutation_exact":
            event_permutation_result["maximum_absolute_error"] == 0.0,
        "event_permutation_all_equal":
            event_permutation_result["non_equal_outputs"] == 0,
        "train_fold1_sequence_overlap_zero": not (training_sequences & heldout_sequences),
        "unique_hypothesis_counts_exact": (
            sum(unique_counts[key] for key in training_groups) == 2106 and
            sum(unique_counts[key] for key in heldout_groups) == 500),
        "model_state_changed": initial_state != final_state,
        "repository_clean": not git_output("status", "--porcelain"),
        "repository_commit_exact": git_output("rev-parse", "HEAD") == repository["commit"],
        "no_checkpoint": True,
        "no_public_benchmark": True,
        "no_qwen": True,
        "fold0_numeric_targets_not_opened": True,
        "delayed_full_target_source_not_opened": True,
    }
    engineering_pass = all(engineering_conditions.values())
    scientific_pass = all(scientific_conditions.values())
    accepted = engineering_pass and scientific_pass
    decision = ("m23a_pass_authorize_plan_only_for_fold0"
                if accepted else
                "m23a_fail_stop_direct_unique_hypothesis_family_without_scan")
    result = {
        "schema": "sttrack-lachtt-m23a-result/v1",
        "complete": True,
        "accepted": accepted,
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "claim_ceiling": spec["claim_ceiling"],
        "elapsed_seconds": time.time() - started,
        "identity": {
            "spec": spec_record,
            "runner": runner_record,
            "preexecution_audit": audit_record,
            "repository": repository,
        },
        "data": {
            "dataset": "DepthTrack Train only",
            "training_events": len(training_groups),
            "training_sequences": len(training_sequences),
            "fold1_events": len(heldout_groups),
            "fold1_sequences": len(heldout_sequences),
            "training_unique_hypotheses": sum(
                unique_counts[key] for key in training_groups),
            "fold1_unique_hypotheses": sum(
                unique_counts[key] for key in heldout_groups),
            "fold0_numeric_targets_opened": False,
            "delayed_full_target_source_opened": False,
            "fold1_target_source": spec["m22_inputs"]["predictions"],
        },
        "model": {
            "class": architecture["class"],
            "parameter_count": parameter_count,
            "utility_safety_parameter_overlap": parameter_overlap,
            "initial_state_sha256": initial_state,
            "final_state_sha256": final_state,
        },
        "training": {
            "steps_completed": step,
            "trace_rows": len(trace),
            "first_total_loss": trace[0]["losses"]["total"],
            "last_total_loss": trace[-1]["losses"]["total"],
        },
        "permutation_audit": permutation,
        "development_summary": summary,
        "engineering_conditions": engineering_conditions,
        "scientific_conditions": scientific_conditions,
        "failed_engineering_conditions": sorted(
            name for name, value in engineering_conditions.items() if not value),
        "failed_scientific_conditions": sorted(
            name for name, value in scientific_conditions.items() if not value),
        "authorization": {
            "fold0_plan_only": accepted,
            "fold0_execution": False,
            "tracking_checkpoint": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
            "automatic_next_stage": False,
        },
    }

    for record in spec["source"]["dependencies"]:
        validate_file_record(record)
    if (file_record(Path(__file__).resolve()) != runner_record or
            file_record(args.spec) != spec_record or
            file_record(args.preaudit) != audit_record or
            git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("status", "--porcelain")):
        raise ContractError("M23a source changed during execution")
    publish(args.output, result, trace, predictions)
    print(json.dumps({
        "accepted": accepted,
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "development_summary": summary,
        "steps_completed": step,
    }, sort_keys=True))
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
