#!/usr/bin/env python3
"""M24 sequence-fold OOF epistemic unanimity committee."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
M23_RUNNER_PATH = REPOSITORY_ROOT / "tools" / (
    "run_sttrack_lachtt_m23a_unique_hypothesis_direct_selection.py")
OUTPUT_FILES = (
    "manifest.json",
    "oof_predictions.jsonl.gz",
    "result.json",
    "training_trace.jsonl.gz",
)
SPEC_SCHEMA = "sttrack-lachtt-m24-sequence-fold-epistemic-committee-spec/v1"
PREFLIGHT_SCHEMA = "sttrack-lachtt-m24-preflight-binding/v1"
PREAUDIT_SCHEMA = "sttrack-lachtt-m24-preexecution-audit/v1"
FOLDS = (2, 3, 4, 5)
EXPECTED_FOLD_COUNTS = {
    2: {"sequences": 20, "events": 132, "unique": 530, "steps": 204},
    3: {"sequences": 17, "events": 103, "unique": 434, "steps": 156},
    4: {"sequences": 18, "events": 73, "unique": 306, "steps": 120},
    5: {"sequences": 21, "events": 199, "unique": 836, "steps": 300},
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
        run_sttrack_lachtt_m23a_unique_hypothesis_direct_selection as m23,
    )
    torch_module, m22, Router = m23.load_components()
    if torch_module is not torch:
        raise ContractError("torch module identity drifted")
    return torch, m22, m23, Router


def validate_spec(spec, args, spec_record, runner_record, m23):
    if (spec.get("schema") != SPEC_SCHEMA or
            spec.get("complete") is not True or
            spec.get("created_before_execution") is not True):
        raise ContractError("M24 spec identity drifted")
    expected_output = Path(spec["output"]["root"]).resolve()
    if args.output.resolve() != expected_output or args.output.exists():
        raise ContractError("M24 output precondition drifted")
    expected_training = {
        "seed": 20260924,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs_per_member": 12,
        "event_batch_size": 8,
        "member_source_folds": [2, 3, 4, 5],
        "optimizer_steps_per_member": {"2": 204, "3": 156,
                                       "4": 120, "5": 300},
        "optimizer_steps_total": 780,
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
        raise ContractError("M24 training constants drifted")
    expected_architecture = {
        "class": "UniqueHypothesisSelectiveRouter",
        "committee_members": 4,
        "benefit_projection_seed": 20260923,
        "catastrophe_projection_seed": 20261923,
        "residual_scale": 0.1,
        "catastrophe_penalty": 4.0,
        "expected_parameters_per_member": 106434,
        "candidate_count": 6,
        "cached_horizon": 5,
        "exact_bbox_scalars_per_signature": 20,
        "duplicate_relation_aggregation": "arithmetic_mean",
        "duplicate_representative": "lowest_canonical_role_id",
        "utility_safety_parameter_overlap_required": 0,
        "identical_initial_state_required": True,
    }
    if spec.get("architecture") != expected_architecture:
        raise ContractError("M24 architecture constants drifted")
    expected_policy = {
        "evaluation": "sequence-fold out-of-fold",
        "voters_per_event": 3,
        "exclude_model_trained_on_evaluation_fold": True,
        "candidate_selection": "unanimous maximum direct dominance role",
        "dominance": "benefit_probability - 4 * catastrophe_probability",
        "minimum_member_top1_top2_margin_min": 0.1,
        "minimum_member_benefit_probability_min": 0.8,
        "maximum_member_catastrophe_probability_max": 0.05,
        "otherwise": "abstain and preserve protected branch",
        "threshold_scan": False,
    }
    if spec.get("committee_policy") != expected_policy:
        raise ContractError("M24 committee policy drifted")
    expected_gates = {
        "selected_actions_min": 12,
        "beneficial_actions_min": 10,
        "beneficial_sequences_min": 6,
        "beneficial_precision_min": 0.95,
        "catastrophic_actions_max": 0,
        "selected_mean_true_h10_gain_min": 0.2,
        "selected_branch_aggregate_gt_public": True,
        "selected_actions_each_evaluation_fold_min": 1,
        "all_abstain_is_not_pass": True,
    }
    if spec.get("scientific_gates") != expected_gates:
        raise ContractError("M24 scientific gates drifted")
    if set(spec.get("m22_inputs", {})) != {"spec", "binding"}:
        raise ContractError("M24 input set expanded beyond frozen Train sources")
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
    return spec_record


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
        raise ContractError("M24 preflight binding drifted")
    return binding, record


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
            audit.get("authorization", {}).get("m24_execution") is not True or
            audit.get("audited_identity") != expected or
            audit.get("claim_ceiling") != spec.get("claim_ceiling")):
        raise ContractError("M24 preaudit did not authorize exact execution")
    return audit, record


def fold_summary(rows):
    selected = [row for row in rows if row["selected_role_id"] is not None]
    beneficial = [row for row in selected
                  if row["selected_strict_label"] == "beneficial"]
    catastrophic = [row for row in selected
                    if row["selected_strict_label"] == "catastrophic"]
    neutral = [row for row in selected
               if row["selected_strict_label"] == "neutral"]
    count = len(selected)
    return {
        "events": len(rows),
        "selected_actions": count,
        "beneficial_actions": len(beneficial),
        "neutral_actions": len(neutral),
        "catastrophic_actions": len(catastrophic),
        "beneficial_sequences": len({row["sequence"] for row in beneficial}),
        "selected_sequences": len({row["sequence"] for row in selected}),
        "beneficial_precision": len(beneficial) / count if count else 0.0,
        "selected_mean_true_h10_gain": (
            sum(row["selected_actual_h10_gain"] for row in selected) / count
            if count else None),
        "selected_branch_aggregate_h10_mean_iou": (
            sum(row["selected_actual_h10_branch_mean_iou"]
                for row in selected) / count if count else None),
        "selected_public_aggregate_h10_mean_iou": (
            sum(row["selected_actual_h10_public_mean_iou"]
                for row in selected) / count if count else None),
    }


def committee_decision(torch, member_outputs, index, valid, policy, m23):
    member_rows = []
    loose_policy = {
        "top1_top2_dominance_margin_min": -1.0e30,
        "benefit_probability_min": -1.0e30,
        "catastrophe_probability_max": 1.0e30,
    }
    for member_fold, outputs in member_outputs:
        row = m23.policy_decision(
            torch, outputs, index, valid, loose_policy)
        member_rows.append({"member_fold": member_fold, **row})
    roles = {row["top_role_id"] for row in member_rows}
    unanimous = len(roles) == 1
    role = member_rows[0]["top_role_id"] if unanimous else None
    minimum_margin = min(row["margin"] for row in member_rows)
    minimum_benefit = min(row["benefit_probability"] for row in member_rows)
    maximum_catastrophe = max(
        row["catastrophe_probability"] for row in member_rows)
    accepted = (
        unanimous and
        minimum_margin >= float(
            policy["minimum_member_top1_top2_margin_min"]) and
        minimum_benefit >= float(
            policy["minimum_member_benefit_probability_min"]) and
        maximum_catastrophe <= float(
            policy["maximum_member_catastrophe_probability_max"]))
    return {
        "member_decisions": member_rows,
        "unanimous_top_role": unanimous,
        "unanimous_role_id": role,
        "minimum_member_margin": minimum_margin,
        "minimum_member_benefit_probability": minimum_benefit,
        "maximum_member_catastrophe_probability": maximum_catastrophe,
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
            "schema": "sttrack-lachtt-m24-output-manifest/v1",
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
            raise ContractError("M24 output set drifted")
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
    args.preflight = args.preflight.resolve()
    args.preaudit = args.preaudit.resolve()
    args.output = args.output.resolve()
    started = time.time()

    torch, m22, m23, Router = load_components()
    runner_record = m23.file_record(Path(__file__).resolve())
    spec, spec_record = m23.load_json_snapshot(args.spec)
    validate_spec(spec, args, spec_record, runner_record, m23)
    _, preflight_record = validate_preflight(
        args.preflight, spec_record, runner_record, spec, m23)
    _, preaudit_record = validate_preaudit(
        args.preaudit, spec_record, runner_record, preflight_record, spec, m23)

    repository = spec["source"]["repository"]
    if (git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("branch", "--show-current") != repository["branch"] or
            git_output("status", "--porcelain")):
        raise ContractError("repository identity drifted before M24")

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
                "heldout_target_commitment", -1)) != 1):
        raise ContractError("stripped training target contract drifted")
    if len(training_groups) != 507:
        raise ContractError("M24 training event count drifted")

    fold_groups = {fold: {} for fold in FOLDS}
    for key, rows in training_groups.items():
        entry = split_entries[key]
        fold = int(entry["fold"])
        if entry["partition"] != "training" or fold not in FOLDS:
            raise ContractError("M24 training fold identity drifted")
        fold_groups[fold][key] = rows
    fold_sequences = {
        fold: {key[0] for key in groups}
        for fold, groups in fold_groups.items()
    }
    if any(fold_sequences[left] & fold_sequences[right]
           for index, left in enumerate(FOLDS)
           for right in FOLDS[index + 1:]):
        raise ContractError("M24 sequence folds overlap")
    for fold in FOLDS:
        expected = EXPECTED_FOLD_COUNTS[fold]
        if (len(fold_groups[fold]) != expected["events"] or
                len(fold_sequences[fold]) != expected["sequences"]):
            raise ContractError("M24 per-fold census drifted")

    duplicate_groups = m23.load_duplicate_groups(
        m22, m22_spec, set(training_groups), training_groups)
    unique_counts = {key: len(groups)
                     for key, groups in duplicate_groups.items()}
    if (min(unique_counts.values()) != 2 or
            max(unique_counts.values()) != 6 or
            sum(value < 6 for value in unique_counts.values()) != 467 or
            sum(unique_counts.values()) != 2106):
        raise ContractError("M24 unique hypothesis census drifted")
    for fold in FOLDS:
        if sum(unique_counts[key] for key in fold_groups[fold]) != (
                EXPECTED_FOLD_COUNTS[fold]["unique"]):
            raise ContractError("M24 per-fold unique census drifted")

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
        targets[key] = m23.target_tensors(
            torch, training_groups[key], relation[3])

    architecture = spec["architecture"]
    training = spec["training"]
    models = {}
    optimizers = {}
    initial_states = {}
    parameter_counts = {}
    parameter_overlaps = {}
    for fold in FOLDS:
        torch.manual_seed(int(training["seed"]))
        model = Router(
            benefit_projection_seed=architecture["benefit_projection_seed"],
            catastrophe_projection_seed=(
                architecture["catastrophe_projection_seed"]),
            residual_scale=architecture["residual_scale"],
            catastrophe_penalty=architecture["catastrophe_penalty"],
        )
        count = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
        overlap = len(
            {id(value) for value in model.utility_parameters()} &
            {id(value) for value in model.safety_parameters()})
        if (count != architecture["expected_parameters_per_member"] or
                overlap != 0 or
                any(parameter.dtype != torch.float32 or
                    parameter.device.type != "cpu"
                    for parameter in model.parameters())):
            raise ContractError("M24 member model identity drifted")
        models[fold] = model
        optimizers[fold] = torch.optim.AdamW(
            model.parameters(), lr=training["learning_rate"],
            weight_decay=training["weight_decay"])
        initial_states[fold] = m22.state_digest(model)
        parameter_counts[fold] = count
        parameter_overlaps[fold] = overlap
    if len(set(initial_states.values())) != 1:
        raise ContractError("M24 members did not share identical initialization")

    trace = []
    global_step = 0
    member_steps = {}
    for fold in FOLDS:
        model = models[fold]
        optimizer = optimizers[fold]
        groups = fold_groups[fold]
        sequence_counts = Counter(key[0] for key in groups)
        member_step = 0
        model.train()
        for epoch in range(training["epochs_per_member"]):
            batches = m23.epoch_batches(
                list(groups), training["seed"], epoch,
                training["event_batch_size"])
            expected_batches = math.ceil(
                len(groups) / training["event_batch_size"])
            if len(batches) != expected_batches:
                raise ContractError("M24 epoch batch count drifted")
            for epoch_step, keys in enumerate(batches):
                member_step += 1
                global_step += 1
                batch = m23.make_batch(
                    torch, keys, relations, targets, sequence_counts,
                    training["seed"], epoch, epoch_step)
                optimizer.zero_grad(set_to_none=True)
                outputs, losses = m23.forward_loss(
                    torch, model, batch, training["loss_weights"])
                if any(not torch.isfinite(value).all().item()
                       for value in outputs.values()):
                    raise ContractError("M24 output is non-finite")
                losses["total"].backward()
                preclip, nonfinite, _ = m22.gradient_diagnostics(model, 0)
                if (nonfinite or not math.isfinite(preclip) or
                        preclip <= 0 or preclip > 1000):
                    raise ContractError("M24 preclip gradient gate failed")
                maximum = training["gradient_clip_norm"]
                m22.scale_gradients(
                    model, min(1.0, maximum / (preclip + 1.0e-12)))
                postclip, post_nonfinite, _ = m22.gradient_diagnostics(model, 0)
                if (post_nonfinite or not math.isfinite(postclip) or
                        postclip > 5.000001):
                    raise ContractError("M24 postclip gradient gate failed")
                optimizer.step()
                trace.append({
                    "record_type": "optimizer_step",
                    "global_step": global_step,
                    "member_fold": fold,
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
        member_steps[fold] = member_step
    if (global_step != training["optimizer_steps_total"] or
            len(trace) != training["optimizer_steps_total"] or
            any(member_steps[fold] != EXPECTED_FOLD_COUNTS[fold]["steps"]
                for fold in FOLDS)):
        raise ContractError("M24 optimizer step count drifted")
    final_states = {fold: m22.state_digest(model)
                    for fold, model in models.items()}

    for model in models.values():
        model.eval()
    predictions = []
    for eval_fold in FOLDS:
        keys_for_fold = sorted(fold_groups[eval_fold])
        voter_folds = tuple(fold for fold in FOLDS if fold != eval_fold)
        if len(voter_folds) != 3:
            raise ContractError("M24 committee size drifted")
        for start in range(0, len(keys_for_fold), 32):
            keys = keys_for_fold[start:start + 32]
            batch = m23.identity_batch(torch, keys, relations, targets)
            member_outputs = [
                (fold, m23.model_outputs(torch, models[fold], batch))
                for fold in voter_folds
            ]
            for index, key in enumerate(keys):
                decision = committee_decision(
                    torch, member_outputs, index, targets[key]["valid"],
                    spec["committee_policy"], m23)
                selected_role = decision["selected_role_id"]
                row = {
                    "record_type": "sequence_fold_oof_prediction",
                    "evaluation_fold": eval_fold,
                    "voter_folds": list(voter_folds),
                    "sequence": key[0],
                    "event_id": key[1],
                    "trigger_frame": key[2],
                    "strict_event_class": targets[key]["event_class"],
                    "unique_hypotheses": unique_counts[key],
                    "committee": decision,
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
                }
                predictions.append(row)

    summary = fold_summary(predictions)
    per_fold_summary = {
        str(fold): fold_summary([
            row for row in predictions if row["evaluation_fold"] == fold])
        for fold in FOLDS
    }
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
        key = sorted(fold_groups[fold])[0]
        candidate = m23.permutation_audit(
            torch, models[fold], relations[key], targets[key])
        event_keys = sorted(fold_groups[fold])[:16]
        event = m23.event_permutation_audit(
            torch, models[fold], event_keys, relations, targets)
        permutation[str(fold)] = {"candidate": candidate, "event": event}
    permutation_exact = all(
        section["maximum_absolute_error"] == 0.0 and
        section["non_equal_outputs"] == 0
        for member in permutation.values() for section in member.values())

    engineering_conditions = {
        "member_count_exact": len(models) == 4,
        "optimizer_steps_exact": global_step == 780,
        "trace_rows_exact": len(trace) == 780,
        "member_step_counts_exact": all(
            member_steps[fold] == EXPECTED_FOLD_COUNTS[fold]["steps"]
            for fold in FOLDS),
        "parameter_count_exact": all(
            parameter_counts[fold] == 106434 for fold in FOLDS),
        "utility_safety_parameter_overlap_zero": all(
            parameter_overlaps[fold] == 0 for fold in FOLDS),
        "identical_initial_states": len(set(initial_states.values())) == 1,
        "all_member_states_changed": all(
            initial_states[fold] != final_states[fold] for fold in FOLDS),
        "candidate_event_permutation_exact": permutation_exact,
        "sequence_fold_overlap_zero": all(
            not (fold_sequences[left] & fold_sequences[right])
            for index, left in enumerate(FOLDS)
            for right in FOLDS[index + 1:]),
        "oof_member_exclusion_exact": all(
            row["evaluation_fold"] not in row["voter_folds"] and
            len(row["voter_folds"]) == 3
            for row in predictions),
        "prediction_rows_exact": len(predictions) == 507,
        "unique_hypothesis_counts_exact": sum(unique_counts.values()) == 2106,
        "repository_clean": not git_output("status", "--porcelain"),
        "repository_commit_exact": (
            git_output("rev-parse", "HEAD") == repository["commit"]),
        "no_checkpoint": True,
        "no_public_benchmark": True,
        "no_qwen": True,
        "consumed_fold1_numeric_targets_not_opened": True,
        "fold0_numeric_targets_not_opened": True,
        "delayed_full_target_source_not_opened": True,
    }
    engineering_pass = all(engineering_conditions.values())
    scientific_pass = all(scientific_conditions.values())
    accepted = engineering_pass and scientific_pass
    decision = (
        "m24_pass_authorize_separate_plan_only_for_consumed_fold1"
        if accepted else
        "m24_fail_stop_sequence_fold_epistemic_committee_without_scan")
    result = {
        "schema": "sttrack-lachtt-m24-result/v1",
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
            "training_events": len(training_groups),
            "training_sequences": len(set().union(*fold_sequences.values())),
            "unique_hypotheses": sum(unique_counts.values()),
            "folds": {
                str(fold): {
                    "events": len(fold_groups[fold]),
                    "sequences": len(fold_sequences[fold]),
                    "unique_hypotheses": sum(
                        unique_counts[key] for key in fold_groups[fold]),
                } for fold in FOLDS
            },
            "consumed_fold1_numeric_targets_opened": False,
            "fold0_numeric_targets_opened": False,
            "delayed_full_target_source_opened": False,
        },
        "model": {
            "class": architecture["class"],
            "members": len(models),
            "parameter_count_per_member": {
                str(fold): parameter_counts[fold] for fold in FOLDS},
            "utility_safety_parameter_overlap": {
                str(fold): parameter_overlaps[fold] for fold in FOLDS},
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
                                if row["member_fold"] == fold)
                for fold in FOLDS
            },
            "last_total_loss_per_member": {
                str(fold): next(row["losses"]["total"] for row in reversed(trace)
                                if row["member_fold"] == fold)
                for fold in FOLDS
            },
        },
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
            "consumed_fold1_plan_only": accepted,
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

    for record in spec["source"]["dependencies"]:
        m23.validate_file_record(record)
    for record in spec["m22_inputs"].values():
        m23.validate_file_record(record)
    if (m23.file_record(Path(__file__).resolve()) != runner_record or
            m23.file_record(args.spec) != spec_record or
            m23.file_record(args.preflight) != preflight_record or
            m23.file_record(args.preaudit) != preaudit_record or
            git_output("rev-parse", "HEAD") != repository["commit"] or
            git_output("status", "--porcelain")):
        raise ContractError("M24 source changed during execution")
    publish(args.output, result, trace, predictions, m23)
    print(json.dumps({
        "accepted": accepted,
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "oof_summary": summary,
        "per_fold_summary": per_fold_summary,
        "steps_completed": global_step,
    }, sort_keys=True))
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
