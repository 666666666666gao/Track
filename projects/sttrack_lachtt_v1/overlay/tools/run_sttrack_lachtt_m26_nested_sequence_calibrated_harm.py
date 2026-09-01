#!/usr/bin/env python3
"""M26 nested sequence-calibrated counterfactual harm guard."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILES = (
    "manifest.json",
    "oof_predictions.jsonl.gz",
    "result.json",
    "training_trace.jsonl.gz",
)
SPEC_SCHEMA = "sttrack-lachtt-m26-nested-sequence-calibrated-harm-spec/v1"
PREFLIGHT_SCHEMA = "sttrack-lachtt-m26-preflight-binding/v1"
PREAUDIT_SCHEMA = "sttrack-lachtt-m26-preexecution-audit/v1"
FOLDS = (2, 3, 4, 5)
CALIBRATION_FOLDS = {2: 3, 3: 2, 4: 5, 5: 4}
FIT_FOLDS = {2: (4, 5), 3: (4, 5), 4: (2, 3), 5: (2, 3)}
EXPECTED_FOLD_COUNTS = {
    2: {"sequences": 20, "events": 132, "unique": 530,
        "calibration_events": 103, "fit_events": 272, "steps": 408},
    3: {"sequences": 17, "events": 103, "unique": 434,
        "calibration_events": 132, "fit_events": 272, "steps": 408},
    4: {"sequences": 18, "events": 73, "unique": 306,
        "calibration_events": 199, "fit_events": 235, "steps": 360},
    5: {"sequences": 21, "events": 199, "unique": 836,
        "calibration_events": 73, "fit_events": 235, "steps": 360},
}
FORBIDDEN_INPUT_KEYS = {
    "predictions", "result", "result_audit", "fold1", "fold0",
    "delayed_targets", "full_targets",
}
FORBIDDEN_NUMERIC_TARGET_PATHS = {
    Path("/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/"
         "labeled_actions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m22a_sequence_disjoint_causal_survival_v1_20260901/"
         "heldout_predictions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m23a_r2_unique_hypothesis_direct_selection_v1_20260902/"
         "development_predictions.jsonl.gz").resolve(),
}
EXPECTED_CLAIM_CEILING = {
    "scope": "DepthTrack Train folds 2-5 post-M25 development only",
    "permitted": "one fixed 1536-step M26 nested calibrated harm result",
    "forbidden": [
        "new tracking checkpoint",
        "DepthTrack Test claim",
        "CDTB claim",
        "VOT low22 claim",
        "VOT full127 claim",
        "public benchmark claim",
        "automatic next-stage execution",
    ],
}
RUNTIME_AUDIT = {
    "forbidden_file_opens": [],
    "network_connects": [],
}


def runtime_audit_hook(event, args):
    if event == "open":
        target = args[0]
        if isinstance(target, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(target)).resolve()
            if resolved in FORBIDDEN_NUMERIC_TARGET_PATHS:
                RUNTIME_AUDIT["forbidden_file_opens"].append(str(resolved))
    elif event == "socket.connect":
        RUNTIME_AUDIT["network_connects"].append(str(args))


sys.addaudithook(runtime_audit_hook)


class ContractError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--preaudit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def git_output(*args):
    return subprocess.check_output(
        ("git", "-C", str(REPOSITORY_ROOT), *args), text=True).strip()


def load_components():
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    import torch
    from tools import (
        run_sttrack_lachtt_m25_sequence_pooled_lofo_direct_router as m25,
    )
    torch_module, m22, m23, _ = m25.load_components()
    if torch_module is not torch:
        raise ContractError("torch module identity drifted")
    from lib.models.sttrack.lachtt_nested_calibrated_counterfactual_harm import (
        NestedCalibratedCounterfactualHarmRouter,
    )
    return torch, m22, m23, m25, NestedCalibratedCounterfactualHarmRouter


def validate_spec(spec, args, spec_record, runner_record, m23):
    if (spec.get("schema") != SPEC_SCHEMA or
            spec.get("complete") is not True or
            spec.get("created_before_execution") is not True):
        raise ContractError("M26 spec identity drifted")
    if (args.output.resolve() != Path(spec["output"]["root"]).resolve() or
            args.output.exists()):
        raise ContractError("M26 output precondition drifted")
    expected_training = {
        "seed": 20260926,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs_per_member": 12,
        "event_batch_size": 8,
        "member_source_folds": [2, 3, 4, 5],
        "optimizer_steps_per_member": {
            "2": 408, "3": 408, "4": 360, "5": 360},
        "optimizer_steps_total": 1536,
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
            "signed_harm_smooth_l1": 1.0,
            "strict_label_pairwise_rank": 0.5,
        },
    }
    if spec.get("training") != expected_training:
        raise ContractError("M26 training constants drifted")
    expected_architecture = {
        "class": "NestedCalibratedCounterfactualHarmRouter",
        "nested_outer_models": 4,
        "benefit_projection_seed": 20260923,
        "residual_scale": 0.1,
        "expected_parameters_per_member": 53808,
        "expected_utility_parameters_per_member": 53217,
        "expected_safety_parameters_per_member": 591,
        "candidate_count": 6,
        "cached_horizon": 5,
        "harm_horizons": [3, 5, 10],
        "harm_target": "public_mean_iou - branch_mean_iou",
        "harm_scalar_summary": ["last", "mean", "minimum", "maximum"],
        "harm_input_dimension": 196,
        "harm_head": "single_linear_196_to_3_tanh",
        "exact_bbox_scalars_per_signature": 20,
        "duplicate_relation_aggregation": "arithmetic_mean",
        "duplicate_representative": "lowest_canonical_role_id",
        "utility_safety_parameter_overlap_required": 0,
        "identical_initial_state_required": True,
    }
    if spec.get("architecture") != expected_architecture:
        raise ContractError("M26 architecture constants drifted")
    expected_policy = {
        "evaluation": "nested sequence-disjoint fit-calibration-evaluation",
        "models_per_event": 1,
        "candidate_selection": "maximum benefit probability",
        "top1_top2_benefit_margin_min": 0.1,
        "benefit_probability_min": 0.8,
        "calibration_residual": "true_harm - predicted_harm",
        "within_sequence_residual_quantile": 0.9,
        "across_sequence_residual_quantile": 0.9,
        "calibrated_harm_upper_max_each_horizon": 0.0,
        "fit_folds": {str(fold): list(FIT_FOLDS[fold]) for fold in FOLDS},
        "calibration_folds": {
            str(fold): CALIBRATION_FOLDS[fold] for fold in FOLDS},
        "otherwise": "abstain and preserve protected branch",
        "threshold_scan": False,
    }
    if spec.get("oof_policy") != expected_policy:
        raise ContractError("M26 OOF policy drifted")
    expected_gates = {
        "selected_actions_min": 8,
        "beneficial_actions_min": 8,
        "beneficial_sequences_min": 5,
        "beneficial_precision_min": 0.95,
        "catastrophic_actions_max": 0,
        "selected_mean_true_h10_gain_min": 0.2,
        "selected_branch_aggregate_gt_public": True,
        "selected_actions_each_evaluation_fold_min": 1,
        "all_abstain_is_not_pass": True,
    }
    if spec.get("scientific_gates") != expected_gates:
        raise ContractError("M26 scientific gates drifted")
    if spec.get("claim_ceiling") != EXPECTED_CLAIM_CEILING:
        raise ContractError("M26 claim ceiling drifted")
    if set(spec.get("m22_inputs", {})) != {"spec", "binding"}:
        raise ContractError("M26 input set expanded beyond frozen Train sources")
    if FORBIDDEN_INPUT_KEYS & set(spec.get("m22_inputs", {})):
        raise ContractError("forbidden numeric target input was supplied")
    supplied_records = (list(spec.get("source", {}).get("dependencies", [])) +
                        list(spec["m22_inputs"].values()))
    supplied_paths = {
        Path(record["path"]).resolve() for record in supplied_records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if supplied_paths & FORBIDDEN_NUMERIC_TARGET_PATHS:
        raise ContractError("forbidden numeric target path was supplied")
    for record in supplied_records:
        m23.validate_file_record(record)
    if spec["source"]["runner"] != runner_record:
        raise ContractError("runner/spec identity drifted")
    if Path(spec["source"]["repository"]["path"]).resolve() != REPOSITORY_ROOT:
        raise ContractError("repository path drifted")
    if spec["experiment_plan"]["path"] not in {
            record["path"] for record in spec["source"]["dependencies"]}:
        raise ContractError("experiment plan is not bound")


def validate_preflight(path, spec_record, runner_record, spec, m23):
    binding, record = m23.load_json_snapshot(path)
    expected = {
        "spec_sha256": spec_record["sha256"],
        "runner_sha256": runner_record["sha256"],
        "repository_commit": spec["source"]["repository"]["commit"],
    }
    if (binding.get("schema") != PREFLIGHT_SCHEMA or
            binding.get("complete") is not True or
            binding.get("created_before_execution") is not True or
            binding.get("bound_identity") != expected or
            binding.get("authorization", {}).get("preaudit_only") is not True or
            binding.get("authorization", {}).get("execution") is not False):
        raise ContractError("M26 preflight binding drifted")
    return record


def validate_preaudit(path, spec_record, runner_record, preflight_record,
                      spec, m23):
    audit, record = m23.load_json_snapshot(path)
    expected = {
        "spec_sha256": spec_record["sha256"],
        "runner_sha256": runner_record["sha256"],
        "preflight_sha256": preflight_record["sha256"],
        "repository_commit": spec["source"]["repository"]["commit"],
    }
    if (audit.get("schema") != PREAUDIT_SCHEMA or
            str(audit.get("overall_verdict", "")).upper() != "PASS" or
            str(audit.get("integrity_verdict", "")).upper() != "PASS" or
            audit.get("authorization", {}).get("m26_execution") is not True or
            audit.get("audited_identity") != expected or
            audit.get("claim_ceiling") != spec.get("claim_ceiling")):
        raise ContractError("M26 preaudit did not authorize exact execution")
    return record


def target_tensors(torch, m23, rows, valid):
    target = m23.target_tensors(torch, rows, valid)
    ordered = sorted(rows, key=lambda value: int(value["candidate_role_id"]))
    target["harm"] = torch.tensor([
        [float(row["targets"][str(horizon)]["public_mean_iou"]) -
         float(row["targets"][str(horizon)]["branch_mean_iou"])
         for horizon in (3, 5, 10)]
        for row in ordered
    ], dtype=torch.float32)
    if (tuple(target["harm"].shape) != (6, 3) or
            not torch.isfinite(target["harm"]).all().item() or
            float(target["harm"].min()) < -1.0 or
            float(target["harm"].max()) > 1.0):
        raise ContractError("M26 signed harm target drifted")
    return target


def make_batch(torch, m23, keys, relations, targets, sequence_counts,
               seed, epoch, step):
    fields = defaultdict(list)
    for batch_index, key in enumerate(keys):
        permutation = m23.candidate_permutation(
            torch, seed, epoch, step, batch_index, key)
        relation = relations[key]
        target = targets[key]
        fields["differences"].append(relation[0][:, permutation])
        fields["gates"].append(relation[1][:, permutation])
        fields["scalar"].append(relation[2][:, permutation])
        fields["candidate_valid"].append(target["valid"][permutation])
        fields["role_ids"].append(permutation)
        for name in ("beneficial", "label_score", "harm"):
            fields[name].append(target[name][permutation])
        fields["event_weight"].append(1.0 / sequence_counts[key[0]])
    batch = {
        name: torch.stack(value) for name, value in fields.items()
        if name != "event_weight"}
    batch["event_weight"] = torch.tensor(
        fields["event_weight"], dtype=torch.float32)
    batch["event_weight"] /= batch["event_weight"].sum()
    return batch


def full_candidate_permutation_audit(torch, m23, model, keys, relations,
                                     targets, trials=8):
    base = m23.model_outputs(
        torch, model, m23.identity_batch(torch, keys, relations, targets))
    maximum = 0.0
    non_equal = 0
    for trial in range(trials):
        fields = defaultdict(list)
        inverse = []
        for batch_index, key in enumerate(keys):
            permutation = m23.candidate_permutation(
                torch, 20260926, 999, trial, batch_index, key)
            relation = relations[key]
            fields["differences"].append(relation[0][:, permutation])
            fields["gates"].append(relation[1][:, permutation])
            fields["scalar"].append(relation[2][:, permutation])
            fields["candidate_valid"].append(
                targets[key]["valid"][permutation])
            fields["role_ids"].append(permutation)
            inverse.append(torch.argsort(permutation))
        batch = {name: torch.stack(value)
                 for name, value in fields.items()}
        output = m23.model_outputs(torch, model, batch)
        for name in base:
            restored = torch.stack([
                output[name][index][inverse[index]]
                for index in range(len(keys))
            ])
            maximum = max(
                maximum, float((restored - base[name]).abs().max()))
            if not torch.equal(restored, base[name]):
                non_equal += 1
    return {
        "events_audited": len(keys),
        "trials": trials,
        "maximum_absolute_error": maximum,
        "non_equal_outputs": non_equal,
    }


def forward_loss(torch, m23, model, batch, weights):
    outputs = model(
        batch["differences"], batch["gates"], batch["scalar"],
        batch["candidate_valid"], batch["role_ids"])
    benefit_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["benefit_logit"], batch["beneficial"], reduction="none")
    benefit = (m23.masked_event_mean(
        torch, benefit_raw, batch["candidate_valid"]) *
        batch["event_weight"]).sum()
    harm_raw = torch.nn.functional.smooth_l1_loss(
        outputs["predicted_harm"], batch["harm"], reduction="none")
    harm_valid = batch["candidate_valid"].unsqueeze(-1).float()
    harm_per_event = ((harm_raw * harm_valid).sum(dim=(1, 2)) /
                      (harm_valid.sum(dim=(1, 2)).clamp_min(1.0) * 3.0))
    harm = (harm_per_event * batch["event_weight"]).sum()
    rank_events = m23.pairwise_event_loss(
        torch, outputs["benefit_probability"], batch["label_score"],
        batch["candidate_valid"])
    rank = (rank_events * batch["event_weight"]).sum()
    losses = {
        "benefit_bce": benefit,
        "signed_harm_smooth_l1": harm,
        "strict_label_pairwise_rank": rank,
    }
    total = sum(float(weights[name]) * value for name, value in losses.items())
    if not math.isfinite(float(total.detach())):
        raise ContractError("M26 loss is non-finite")
    return outputs, {**losses, "total": total}


def calibration_offset(torch, m23, model, keys, relations, targets, policy):
    by_sequence = defaultdict(lambda: [[], [], []])
    prediction_rows = []
    for start in range(0, len(keys), 32):
        batch_keys = keys[start:start + 32]
        batch = m23.identity_batch(torch, batch_keys, relations, targets)
        outputs = m23.model_outputs(torch, model, batch)
        for index, key in enumerate(batch_keys):
            valid = targets[key]["valid"]
            predicted = outputs["predicted_harm"][index]
            actual = targets[key]["harm"]
            for role in range(6):
                if not bool(valid[role]):
                    continue
                residual = actual[role] - predicted[role]
                for horizon_index in range(3):
                    by_sequence[key[0]][horizon_index].append(
                        float(residual[horizon_index]))
                prediction_rows.append((predicted[role].clone(),
                                        actual[role].clone()))
    if len(by_sequence) < 2 or not prediction_rows:
        raise ContractError("M26 calibration set is empty")
    sequence_quantiles = []
    within = float(policy["within_sequence_residual_quantile"])
    across = float(policy["across_sequence_residual_quantile"])
    for sequence in sorted(by_sequence):
        sequence_quantiles.append(torch.tensor([
            float(torch.quantile(torch.tensor(
                by_sequence[sequence][index], dtype=torch.float64), within))
            for index in range(3)
        ], dtype=torch.float64))
    offset = torch.quantile(
        torch.stack(sequence_quantiles), across, dim=0).float()
    covered = torch.zeros(3, dtype=torch.int64)
    total = torch.zeros(3, dtype=torch.int64)
    for predicted, actual in prediction_rows:
        covered += (actual <= predicted + offset).to(dtype=torch.int64)
        total += 1
    return offset, {
        "sequences": len(by_sequence),
        "valid_hypotheses": len(prediction_rows),
        "within_sequence_quantile": within,
        "across_sequence_quantile": across,
        "offset_h3_h5_h10": [float(value) for value in offset],
        "empirical_upper_coverage_h3_h5_h10": [
            float(covered[index]) / float(total[index])
            for index in range(3)],
    }


def policy_decision(torch, outputs, index, valid, offset, policy):
    score = outputs["benefit_probability"][index].masked_fill(
        ~valid, -1.0e9)
    if int(valid.sum()) < 2:
        raise ContractError("M26 requires at least two unique hypotheses")
    top = torch.topk(score, k=2).indices
    role = int(top[0])
    margin = float(score[top[0]] - score[top[1]])
    benefit = float(outputs["benefit_probability"][index, role])
    predicted_harm = outputs["predicted_harm"][index, role]
    calibrated = predicted_harm + offset
    accepted = (
        margin >= float(policy["top1_top2_benefit_margin_min"]) and
        benefit >= float(policy["benefit_probability_min"]) and
        bool((calibrated <= float(
            policy["calibrated_harm_upper_max_each_horizon"])).all()))
    return {
        "top_role_id": role,
        "benefit_probability": benefit,
        "margin": margin,
        "predicted_harm_h3_h5_h10": [
            float(value) for value in predicted_harm],
        "calibration_offset_h3_h5_h10": [float(value) for value in offset],
        "calibrated_harm_upper_h3_h5_h10": [
            float(value) for value in calibrated],
        "selected_role_id": role if accepted else None,
    }


def publish(output, result, trace, predictions, m23):
    temporary = Path(tempfile.mkdtemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)))
    try:
        m23.atomic_jsonl_gz(temporary / "training_trace.jsonl.gz", trace)
        m23.atomic_jsonl_gz(temporary / "oof_predictions.jsonl.gz", predictions)
        m23.atomic_json(temporary / "result.json", result)
        manifest = {
            "schema": "sttrack-lachtt-m26-output-manifest/v1",
            "complete": True,
            "accepted": result["accepted"],
            "decision": result["decision"],
            "files": {},
        }
        for name in ("training_trace.jsonl.gz", "oof_predictions.jsonl.gz",
                     "result.json"):
            manifest["files"][name] = m23.file_record(temporary / name)
            manifest["files"][name]["path"] = str(output / name)
        m23.atomic_json(temporary / "manifest.json", manifest)
        if tuple(sorted(path.name for path in temporary.iterdir())) != tuple(
                sorted(OUTPUT_FILES)):
            raise ContractError("M26 output set drifted")
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.chmod(0o555)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.preflight = args.preflight.resolve()
    args.preaudit = args.preaudit.resolve()
    args.output = args.output.resolve()
    started = time.time()

    torch, m22, m23, m25, Router = load_components()
    runner_record = m23.file_record(Path(__file__).resolve())
    spec, spec_record = m23.load_json_snapshot(args.spec)
    validate_spec(spec, args, spec_record, runner_record, m23)
    preflight_record = validate_preflight(
        args.preflight, spec_record, runner_record, spec, m23)
    preaudit_record = validate_preaudit(
        args.preaudit, spec_record, runner_record, preflight_record, spec, m23)

    repository = spec["source"]["repository"]
    if (git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("branch", "--show-current") != repository["branch"] or
            git_output("status", "--porcelain")):
        raise ContractError("repository identity drifted before M26")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    m22_spec = m22.load_verified_json(spec["m22_inputs"]["spec"])
    delayed_target_path = Path(
        m22_spec["delayed_heldout_inputs"]["labeled_actions"]["path"]
    ).resolve()
    if delayed_target_path != Path(
            "/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/"
            "labeled_actions.jsonl.gz").resolve():
        raise ContractError("M22 delayed numeric target identity drifted")
    m22_binding = m22.load_verified_json(spec["m22_inputs"]["binding"])
    m22.validate_frozen_receipts(m22_spec)
    collection, sequence_anchors = m22.load_collection_index(m22_spec)
    _, split_entries = m22.load_split_ledger(m22_spec)
    training_groups, heldout_commitment, target_record_counts = (
        m22.load_training_targets(m22_spec, split_entries))
    if (heldout_commitment.get("numeric_targets_serialized") is not False or
            int(target_record_counts.get("action_target", -1)) != 3042 or
            int(target_record_counts.get(
                "heldout_target_commitment", -1)) != 1 or
            len(training_groups) != 507):
        raise ContractError("M26 stripped target contract drifted")

    fold_groups = {fold: {} for fold in FOLDS}
    for key, rows in training_groups.items():
        entry = split_entries[key]
        fold = int(entry["fold"])
        if entry["partition"] != "training" or fold not in FOLDS:
            raise ContractError("M26 training fold identity drifted")
        fold_groups[fold][key] = rows
    fold_sequences = {
        fold: {key[0] for key in groups}
        for fold, groups in fold_groups.items()}
    sequence_fold_overlap_zero = all(
        not (fold_sequences[left] & fold_sequences[right])
        for index, left in enumerate(FOLDS)
        for right in FOLDS[index + 1:])
    if not sequence_fold_overlap_zero:
        raise ContractError("M26 sequence folds overlap")
    for fold in FOLDS:
        expected = EXPECTED_FOLD_COUNTS[fold]
        if (len(fold_groups[fold]) != expected["events"] or
                len(fold_sequences[fold]) != expected["sequences"]):
            raise ContractError("M26 per-fold census drifted")

    duplicate_groups = m23.load_duplicate_groups(
        m22, m22_spec, set(training_groups), training_groups)
    unique_counts = {key: len(groups)
                     for key, groups in duplicate_groups.items()}
    if (min(unique_counts.values()) != 2 or
            max(unique_counts.values()) != 6 or
            sum(value < 6 for value in unique_counts.values()) != 467 or
            sum(unique_counts.values()) != 2106):
        raise ContractError("M26 unique hypothesis census drifted")
    for fold in FOLDS:
        if sum(unique_counts[key] for key in fold_groups[fold]) != (
                EXPECTED_FOLD_COUNTS[fold]["unique"]):
            raise ContractError("M26 per-fold unique census drifted")

    native_index = m22.load_native_index(m22_spec)
    required_sequences = sorted({key[0] for key in split_entries})
    clip_binding, _ = m22.validate_anchor_binding(
        m22_binding, sequence_anchors, native_index, required_sequences)
    clip_cache, native_cache, loaded_features = {}, {}, {}
    relations, targets = {}, {}
    for key in sorted(training_groups):
        relation = m22.relation_for_event(
            m22_spec, collection, sequence_anchors, native_index,
            clip_binding, key, clip_cache, native_cache, loaded_features)
        relation = m23.aggregate_relation(
            torch, relation, duplicate_groups[key])
        relations[key] = relation[:3]
        targets[key] = target_tensors(
            torch, m23, training_groups[key], relation[3])

    architecture = spec["architecture"]
    training = spec["training"]
    models, optimizers = {}, {}
    initial_states, final_states = {}, {}
    parameter_counts, utility_counts, safety_counts, overlaps = {}, {}, {}, {}
    for fold in FOLDS:
        torch.manual_seed(int(training["seed"]))
        model = Router(
            benefit_projection_seed=architecture["benefit_projection_seed"],
            residual_scale=architecture["residual_scale"])
        utility = list(model.utility_parameters())
        safety = list(model.safety_parameters())
        count = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
        utility_count = sum(parameter.numel() for parameter in utility)
        safety_count = sum(parameter.numel() for parameter in safety)
        overlap = len({id(value) for value in utility} &
                      {id(value) for value in safety})
        if (count != architecture["expected_parameters_per_member"] or
                utility_count != architecture[
                    "expected_utility_parameters_per_member"] or
                safety_count != architecture[
                    "expected_safety_parameters_per_member"] or
                overlap != 0 or
                any(parameter.dtype != torch.float32 or
                    parameter.device.type != "cpu"
                    for parameter in model.parameters())):
            raise ContractError("M26 model identity drifted")
        models[fold] = model
        optimizers[fold] = torch.optim.AdamW(
            model.parameters(), lr=training["learning_rate"],
            weight_decay=training["weight_decay"])
        initial_states[fold] = m22.state_digest(model)
        parameter_counts[fold] = count
        utility_counts[fold] = utility_count
        safety_counts[fold] = safety_count
        overlaps[fold] = overlap
    if len(set(initial_states.values())) != 1:
        raise ContractError("M26 models did not share identical initialization")

    trace = []
    global_step = 0
    member_steps = {}
    for eval_fold in FOLDS:
        model = models[eval_fold]
        optimizer = optimizers[eval_fold]
        fit_folds = FIT_FOLDS[eval_fold]
        calibration_fold = CALIBRATION_FOLDS[eval_fold]
        fit_groups = {
            key: rows for fit_fold in fit_folds
            for key, rows in fold_groups[fit_fold].items()}
        expected = EXPECTED_FOLD_COUNTS[eval_fold]
        if (len(fit_groups) != expected["fit_events"] or
                len(fold_groups[calibration_fold]) !=
                expected["calibration_events"]):
            raise ContractError("M26 nested event count drifted")
        sequence_counts = Counter(key[0] for key in fit_groups)
        member_step = 0
        model.train()
        for epoch in range(training["epochs_per_member"]):
            batches = m23.epoch_batches(
                list(fit_groups), training["seed"], epoch,
                training["event_batch_size"])
            if len(batches) != math.ceil(
                    len(fit_groups) / training["event_batch_size"]):
                raise ContractError("M26 epoch batch count drifted")
            for epoch_step, keys in enumerate(batches):
                member_step += 1
                global_step += 1
                batch = make_batch(
                    torch, m23, keys, relations, targets, sequence_counts,
                    training["seed"], epoch, epoch_step)
                optimizer.zero_grad(set_to_none=True)
                outputs, losses = forward_loss(
                    torch, m23, model, batch, training["loss_weights"])
                if any(not torch.isfinite(value).all().item()
                       for value in outputs.values()):
                    raise ContractError("M26 output is non-finite")
                losses["total"].backward()
                preclip, nonfinite, _ = m22.gradient_diagnostics(model, 0)
                if (nonfinite or not math.isfinite(preclip) or
                        preclip <= 0 or preclip > 1000):
                    raise ContractError("M26 preclip gradient gate failed")
                maximum = training["gradient_clip_norm"]
                m22.scale_gradients(
                    model, min(1.0, maximum / (preclip + 1.0e-12)))
                postclip, post_nonfinite, _ = m22.gradient_diagnostics(model, 0)
                if (post_nonfinite or not math.isfinite(postclip) or
                        postclip > 5.000001):
                    raise ContractError("M26 postclip gradient gate failed")
                optimizer.step()
                trace.append({
                    "record_type": "optimizer_step",
                    "global_step": global_step,
                    "evaluation_fold": eval_fold,
                    "calibration_fold": calibration_fold,
                    "fit_folds": list(fit_folds),
                    "member_step": member_step,
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "batch_size": len(keys),
                    "losses": {name: float(value.detach())
                               for name, value in losses.items()},
                    "preclip_total_l2": preclip,
                    "postclip_total_l2": postclip,
                    "optimizer_step_executed": True,
                })
        member_steps[eval_fold] = member_step
        final_states[eval_fold] = m22.state_digest(model)
    if (global_step != training["optimizer_steps_total"] or
            len(trace) != training["optimizer_steps_total"] or
            any(member_steps[fold] != EXPECTED_FOLD_COUNTS[fold]["steps"]
                for fold in FOLDS)):
        raise ContractError("M26 optimizer step count drifted")

    for model in models.values():
        model.eval()
    calibration = {}
    offsets = {}
    for eval_fold in FOLDS:
        calibration_fold = CALIBRATION_FOLDS[eval_fold]
        keys = sorted(fold_groups[calibration_fold])
        offset, diagnostics = calibration_offset(
            torch, m23, models[eval_fold], keys, relations, targets,
            spec["oof_policy"])
        if diagnostics["sequences"] != len(fold_sequences[calibration_fold]):
            raise ContractError("M26 calibration sequence count drifted")
        offsets[eval_fold] = offset
        calibration[str(eval_fold)] = {
            "calibration_fold": calibration_fold,
            **diagnostics,
        }

    predictions = []
    for eval_fold in FOLDS:
        fit_folds = FIT_FOLDS[eval_fold]
        calibration_fold = CALIBRATION_FOLDS[eval_fold]
        keys_for_fold = sorted(fold_groups[eval_fold])
        for start in range(0, len(keys_for_fold), 32):
            keys = keys_for_fold[start:start + 32]
            batch = m23.identity_batch(torch, keys, relations, targets)
            outputs = m23.model_outputs(torch, models[eval_fold], batch)
            for index, key in enumerate(keys):
                decision = policy_decision(
                    torch, outputs, index, targets[key]["valid"],
                    offsets[eval_fold], spec["oof_policy"])
                selected_role = decision["selected_role_id"]
                predictions.append({
                    "record_type": "nested_sequence_calibrated_harm_prediction",
                    "evaluation_fold": eval_fold,
                    "calibration_fold": calibration_fold,
                    "fit_folds": list(fit_folds),
                    "sequence": key[0],
                    "event_id": key[1],
                    "trigger_frame": key[2],
                    "strict_event_class": targets[key]["event_class"],
                    "unique_hypotheses": unique_counts[key],
                    "decision": decision,
                    "selected_role_id": selected_role,
                    "selected_strict_label": (
                        targets[key]["labels"][selected_role]
                        if selected_role is not None else "abstain"),
                    "selected_actual_h10_gain": (
                        float(targets[key]["gain"][selected_role])
                        if selected_role is not None else 0.0),
                    "selected_actual_h10_branch_mean_iou": (
                        float(targets[key]["branch_mean"][selected_role])
                        if selected_role is not None else None),
                    "selected_actual_h10_public_mean_iou": (
                        float(targets[key]["public_mean"][selected_role])
                        if selected_role is not None else None),
                    "selected_actual_harm_h3_h5_h10": (
                        [float(value) for value in targets[key]["harm"][
                            selected_role]]
                        if selected_role is not None else None),
                })

    summary = m25.fold_summary(predictions)
    per_fold_summary = {
        str(fold): m25.fold_summary([
            row for row in predictions if row["evaluation_fold"] == fold])
        for fold in FOLDS}
    gates = spec["scientific_gates"]
    scientific_conditions = {
        "selected_actions_min": summary["selected_actions"] >=
            gates["selected_actions_min"],
        "beneficial_actions_min": summary["beneficial_actions"] >=
            gates["beneficial_actions_min"],
        "beneficial_sequences_min": summary["beneficial_sequences"] >=
            gates["beneficial_sequences_min"],
        "beneficial_precision_min": summary["beneficial_precision"] >=
            gates["beneficial_precision_min"],
        "catastrophic_actions_max": summary["catastrophic_actions"] <=
            gates["catastrophic_actions_max"],
        "selected_mean_true_h10_gain_min": (
            summary["selected_mean_true_h10_gain"] is not None and
            summary["selected_mean_true_h10_gain"] >=
            gates["selected_mean_true_h10_gain_min"]),
        "selected_branch_aggregate_gt_public": (
            summary["selected_branch_aggregate_h10_mean_iou"] is not None and
            summary["selected_public_aggregate_h10_mean_iou"] is not None and
            summary["selected_branch_aggregate_h10_mean_iou"] >
            summary["selected_public_aggregate_h10_mean_iou"]),
        "selected_actions_each_evaluation_fold_min": all(
            per_fold_summary[str(fold)]["selected_actions"] >=
            gates["selected_actions_each_evaluation_fold_min"]
            for fold in FOLDS),
        "all_abstain_is_not_pass": summary["selected_actions"] > 0,
    }

    permutation = {}
    for fold in FOLDS:
        keys = sorted(fold_groups[fold])
        candidate = full_candidate_permutation_audit(
            torch, m23, models[fold], keys, relations, targets)
        event = m23.event_permutation_audit(
            torch, models[fold], keys, relations, targets)
        event["events_audited"] = len(keys)
        event["trials"] = 8
        permutation[str(fold)] = {"candidate": candidate, "event": event}
    permutation_exact = all(
        section["maximum_absolute_error"] == 0.0 and
        section["non_equal_outputs"] == 0
        for member in permutation.values() for section in member.values())
    nested_overlap_zero = all(
        not (set(FIT_FOLDS[fold]) & {
            fold, CALIBRATION_FOLDS[fold]}) and
        CALIBRATION_FOLDS[fold] != fold and
        set(FIT_FOLDS[fold]) | {fold, CALIBRATION_FOLDS[fold]} == set(FOLDS)
        for fold in FOLDS)
    prediction_partition_exact = all(
        row["evaluation_fold"] not in row["fit_folds"] and
        row["calibration_fold"] not in row["fit_folds"] and
        row["evaluation_fold"] != row["calibration_fold"] and
        len(row["fit_folds"]) == 2
        for row in predictions)
    training_target_partition_exact = (
        len(training_groups) == 507 and
        int(target_record_counts.get("action_target", -1)) == 3042 and
        int(target_record_counts.get("heldout_target_commitment", -1)) == 1 and
        heldout_commitment.get("numeric_targets_serialized") is False and
        all(split_entries[key]["partition"] == "training" and
            int(split_entries[key]["fold"]) in FOLDS
            for key in training_groups))
    output_contract_excludes_checkpoint = (
        set(OUTPUT_FILES) == {
            "manifest.json", "oof_predictions.jsonl.gz", "result.json",
            "training_trace.jsonl.gz"} and
        not any(Path(name).suffix in {".ckpt", ".pth", ".pt"}
                for name in OUTPUT_FILES))
    qwen_module_not_loaded = not any(
        "qwen" in name.lower() for name in sys.modules)
    engineering_conditions = {
        "model_count_exact": len(models) == 4,
        "optimizer_steps_exact": global_step == 1536,
        "trace_rows_exact": len(trace) == 1536,
        "member_step_counts_exact": all(
            member_steps[fold] == EXPECTED_FOLD_COUNTS[fold]["steps"]
            for fold in FOLDS),
        "parameter_count_exact": all(
            parameter_counts[fold] == 53808 for fold in FOLDS),
        "utility_parameter_count_exact": all(
            utility_counts[fold] == 53217 for fold in FOLDS),
        "safety_parameter_count_exact": all(
            safety_counts[fold] == 591 for fold in FOLDS),
        "utility_safety_parameter_overlap_zero": all(
            overlaps[fold] == 0 for fold in FOLDS),
        "identical_initial_states": len(set(initial_states.values())) == 1,
        "all_member_states_changed": all(
            initial_states[fold] != final_states[fold] for fold in FOLDS),
        "candidate_event_permutation_exact": permutation_exact,
        "sequence_fold_overlap_zero": sequence_fold_overlap_zero,
        "nested_fold_partition_overlap_zero": nested_overlap_zero,
        "prediction_partition_exact": prediction_partition_exact,
        "prediction_rows_exact": len(predictions) == 507,
        "unique_hypothesis_counts_exact": sum(unique_counts.values()) == 2106,
        "repository_clean": not git_output("status", "--porcelain"),
        "repository_commit_exact": (
            git_output("rev-parse", "HEAD") == repository["commit"]),
        "training_target_partition_exact": training_target_partition_exact,
        "forbidden_numeric_target_path_access_zero": not RUNTIME_AUDIT[
            "forbidden_file_opens"],
        "network_connect_access_zero": not RUNTIME_AUDIT["network_connects"],
        "qwen_module_not_loaded": qwen_module_not_loaded,
        "output_contract_excludes_checkpoint": (
            output_contract_excludes_checkpoint),
    }
    engineering_pass = all(engineering_conditions.values())
    scientific_pass = all(scientific_conditions.values())
    accepted = engineering_pass and scientific_pass
    decision = (
        "m26_pass_authorize_separate_plan_only_for_later_validation"
        if accepted else
        "m26_fail_stop_nested_sequence_calibrated_harm_without_scan")
    result = {
        "schema": "sttrack-lachtt-m26-result/v1",
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
            "preflight": preflight_record,
            "preexecution_audit": preaudit_record,
            "repository": repository,
        },
        "data": {
            "dataset": "DepthTrack Train only",
            "development_events": len(training_groups),
            "development_sequences": len(set().union(*fold_sequences.values())),
            "unique_hypotheses": sum(unique_counts.values()),
            "folds": {
                str(fold): {
                    "evaluation_events": len(fold_groups[fold]),
                    "evaluation_sequences": len(fold_sequences[fold]),
                    "evaluation_unique_hypotheses": sum(
                        unique_counts[key] for key in fold_groups[fold]),
                    "calibration_fold": CALIBRATION_FOLDS[fold],
                    "calibration_events": len(fold_groups[
                        CALIBRATION_FOLDS[fold]]),
                    "fit_folds": list(FIT_FOLDS[fold]),
                    "fit_events": sum(
                        len(fold_groups[value]) for value in FIT_FOLDS[fold]),
                } for fold in FOLDS},
            "target_partitions_loaded": ["training"],
            "target_folds_loaded": list(FOLDS),
            "runtime_forbidden_file_open_count": len(
                RUNTIME_AUDIT["forbidden_file_opens"]),
            "runtime_forbidden_file_open_paths": sorted(set(
                RUNTIME_AUDIT["forbidden_file_opens"])),
            "runtime_network_connect_count": len(
                RUNTIME_AUDIT["network_connects"]),
            "qwen_module_loaded": not qwen_module_not_loaded,
            "output_contract_files": list(OUTPUT_FILES),
        },
        "model": {
            "class": architecture["class"],
            "nested_outer_models": len(models),
            "parameter_count_per_member": {
                str(fold): parameter_counts[fold] for fold in FOLDS},
            "utility_parameter_count_per_member": {
                str(fold): utility_counts[fold] for fold in FOLDS},
            "safety_parameter_count_per_member": {
                str(fold): safety_counts[fold] for fold in FOLDS},
            "utility_safety_parameter_overlap": {
                str(fold): overlaps[fold] for fold in FOLDS},
            "initial_state_sha256": {
                str(fold): initial_states[fold] for fold in FOLDS},
            "final_state_sha256": {
                str(fold): final_states[fold] for fold in FOLDS},
        },
        "training": {
            "steps_completed": global_step,
            "trace_rows": len(trace),
            "steps_per_member": {
                str(fold): member_steps[fold] for fold in FOLDS},
            "first_total_loss_per_member": {
                str(fold): next(row["losses"]["total"] for row in trace
                                if row["evaluation_fold"] == fold)
                for fold in FOLDS},
            "last_total_loss_per_member": {
                str(fold): next(row["losses"]["total"]
                                for row in reversed(trace)
                                if row["evaluation_fold"] == fold)
                for fold in FOLDS},
        },
        "calibration": calibration,
        "permutation_audit": permutation,
        "oof_summary": summary,
        "per_fold_summary": per_fold_summary,
        "engineering_conditions": engineering_conditions,
        "scientific_conditions": scientific_conditions,
        "failed_engineering_conditions": sorted(
            name for name, value in engineering_conditions.items()
            if not value),
        "failed_scientific_conditions": sorted(
            name for name, value in scientific_conditions.items()
            if not value),
        "authorization": {
            "later_validation_plan_only": accepted,
            "later_validation_execution": False,
            "consumed_fold1_execution": False,
            "fold0_plan": False,
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

    if (m23.file_record(args.spec) != spec_record or
            m23.file_record(args.preflight) != preflight_record or
            m23.file_record(args.preaudit) != preaudit_record or
            m23.file_record(Path(__file__).resolve()) != runner_record or
            git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("branch", "--show-current") != repository["branch"] or
            git_output("status", "--porcelain")):
        raise ContractError("M26 source changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish(args.output, result, trace, predictions, m23)
    print(json.dumps({
        "accepted": accepted,
        "decision": decision,
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "selected_actions": summary["selected_actions"],
        "beneficial_actions": summary["beneficial_actions"],
        "catastrophic_actions": summary["catastrophic_actions"],
        "optimizer_steps": global_step,
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
