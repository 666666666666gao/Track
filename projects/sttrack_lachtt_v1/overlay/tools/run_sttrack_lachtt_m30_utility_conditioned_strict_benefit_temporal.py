#!/usr/bin/env python3
"""M30 utility-conditioned temporal strict-benefit component prediction."""

import argparse
import gzip
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
SPEC_SCHEMA = "sttrack-lachtt-m30-utility-conditioned-strict-benefit-temporal-spec/v1"
PREFLIGHT_SCHEMA = "sttrack-lachtt-m30-preflight-binding/v1"
PREAUDIT_SCHEMA = "sttrack-lachtt-m30-preexecution-audit/v1"
FOLDS = (2, 3, 4, 5)
CONDITIONS = ("strict_benefit_temporal",)
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
EXPECTED_ACTION_FOLDS = {2: 3, 3: 2, 4: 1, 5: 6}
EXPECTED_M25_ACTION_SHA = (
    "52d88d01c15c35764cae4795166888c3de2a8da35685aaf15f788c19fdbcbf5d"
)
EXPECTED_M26_RESULT_SHA = (
    "520cb850e625519e55e30b165b0febd42abd444b5d910fc76f190c46260682ca"
)
EXPECTED_M26_OOF_SHA = (
    "56839b22048a8d6df73f7b4e6c810a13cf8b35fdf8c072bc661c2b727a479393"
)
EXPECTED_M27_SPEC_SHA = (
    "636ceb7064c725397ea9cd2950d1be77e76ba92d2683168bfcea6b52eeba6068"
)
EXPECTED_M27_RESULT_SHAS = {
    "a9325aca6bd72a89fced9a7b39cc235756c829cf4cc608974b805097f1afbd82",
    "1f88a48d9dd29b284d66dcdc0ebb6fd9d5c8729a2ff6915167420e5f4e09dac2",
}
FORBIDDEN_NUMERIC_TARGET_PATHS = {
    Path("/root/autodl-tmp/sttrack_lachtt_m22a_sequence_disjoint_"
         "causal_survival_v1_20260901/heldout_predictions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/sttrack_lachtt_m23a_r2_unique_hypothesis_"
         "direct_selection_v1_20260902/development_predictions.jsonl.gz"
         ).resolve(),
}
EXPECTED_CLAIM_CEILING = {
    "scope": "DepthTrack Train folds 2-5 post-M29 strict-benefit component diagnostic only",
    "permitted": "one fixed 1536-step M30 strict-benefit temporal result",
    "forbidden": [
        "new tracking checkpoint",
        "tracker integration",
        "DepthTrack Test claim",
        "CDTB claim",
        "VOT low22 claim",
        "VOT full127 claim",
        "public benchmark claim",
        "automatic next-stage execution",
    ],
}
RUNTIME_AUDIT = {"forbidden_file_opens": [], "network_connects": []}
FORBIDDEN_RUNTIME_ROOTS = []


def runtime_audit_hook(event, args):
    if event == "open":
        target = args[0]
        if isinstance(target, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(target)).resolve()
            if resolved in FORBIDDEN_NUMERIC_TARGET_PATHS:
                RUNTIME_AUDIT["forbidden_file_opens"].append(str(resolved))
            for root in FORBIDDEN_RUNTIME_ROOTS:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                RUNTIME_AUDIT["forbidden_file_opens"].append(str(resolved))
                break
            if any("qwen" in part.lower() for part in resolved.parts):
                RUNTIME_AUDIT["forbidden_file_opens"].append(str(resolved))
    elif event == "socket.connect":
        RUNTIME_AUDIT["network_connects"].append(str(args))


sys.addaudithook(runtime_audit_hook)


class ContractError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preaudit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--m22-spec", type=Path)
    parser.add_argument("--m22-binding", type=Path)
    parser.add_argument("--m27-root", type=Path)
    parser.add_argument("--m27-spec", type=Path)
    parser.add_argument("--m25-actions", type=Path)
    args = parser.parse_args()
    if args.smoke:
        if not all((args.smoke_output, args.m22_spec, args.m22_binding,
                    args.m27_root, args.m27_spec, args.m25_actions)):
            parser.error("smoke inputs are incomplete")
        if any((args.spec, args.preflight, args.preaudit, args.output)):
            parser.error("formal and smoke inputs cannot be mixed")
    elif not all((args.spec, args.preflight, args.preaudit, args.output)):
        parser.error("formal inputs are incomplete")
    return args


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
    from lib.models.sttrack.lachtt_utility_conditioned_strict_benefit_temporal import (
        IDENTITY_RELATION_DIM,
        SAFETY_INPUT_DIM,
        TEMPORAL_HIDDEN_DIM,
        UtilityConditionedStrictBenefitTemporalHead,
        bounded_identity_relation,
        trainable_parameter_count,
    )
    return (torch, m22, m23, m25,
            UtilityConditionedStrictBenefitTemporalHead,
            bounded_identity_relation, trainable_parameter_count,
            IDENTITY_RELATION_DIM, SAFETY_INPUT_DIM, TEMPORAL_HIDDEN_DIM)


def validate_training_contract(spec):
    expected = {
        "seed": 20260930,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "epochs_per_member": 12,
        "event_batch_size": 8,
        "gradient_clip_norm": 5.0,
        "optimizer_steps_total": 1536,
        "inverse_sequence_event_loss_weight": True,
        "loss": "equal_gain_branch_mean_early_hit_rate_smooth_l1",
        "scheduler": False,
        "augmentation": False,
        "early_stopping": False,
        "class_balanced_resampling": False,
        "checkpoint": False,
    }
    if spec.get("training") != expected:
        raise ContractError("M30 training constants drifted")


def validate_spec(spec, args, spec_record, runner_record, m23):
    if (spec.get("schema") != SPEC_SCHEMA or
            spec.get("complete") is not True or
            spec.get("created_before_execution") is not True):
        raise ContractError("M30 spec identity drifted")
    if (args.output.resolve() != Path(spec["output"]["root"]).resolve() or
            args.output.exists()):
        raise ContractError("M30 output precondition drifted")
    validate_training_contract(spec)
    expected_architecture = {
        "conditions": ["strict_benefit_temporal"],
        "candidate_count": 6,
        "cached_horizon": 5,
        "modalities": ["native_rgb", "native_depth", "native_fused"],
        "relations": ["cosine", "normalized_l2", "log_norm_ratio"],
        "temporal_summary": None,
        "identity_relation_dimension_per_age": 9,
        "scalar_relation_dimension_per_age": 49,
        "safety_input_dimension_per_age": 58,
        "temporal_hidden_dimension": 32,
        "head": "linear_58_to_32_tanh_gru_32_three_linear_32_to_1_heads",
        "parameters_per_fold_condition": 8323,
        "utility_trainable_parameters": 0,
        "candidate_ranking_trainable_parameters": 0,
        "duplicate_relation_aggregation": "arithmetic_mean",
        "duplicate_representative": "lowest_canonical_role_id",
        "targets": [
            "h10_gain",
            "h10_branch_mean_iou",
            "h5_early_hit_rate",
        ],
        "output_activations": ["tanh", "sigmoid", "sigmoid"],
    }
    if spec.get("architecture") != expected_architecture:
        raise ContractError("M30 architecture constants drifted")
    expected_policy = {
        "training_action_substrate": "exact_M25_OOF_top_role_per_event",
        "training_action_count": 507,
        "final_action_substrate": "exact_non_null_M25_OOF_actions",
        "final_action_count": 12,
        "action_recomputation": False,
        "residual": "prediction - true_target",
        "within_sequence_residual_quantile": 0.9,
        "across_sequence_residual_quantile": 0.9,
        "calibrated_gain_h10_lower_min": 0.2,
        "calibrated_branch_mean_h10_lower_min": 0.5,
        "calibrated_early_hit_rate_h5_lower_min": 0.4,
        "fit_folds": {str(f): list(FIT_FOLDS[f]) for f in FOLDS},
        "calibration_folds": {
            str(f): CALIBRATION_FOLDS[f] for f in FOLDS},
        "otherwise": "veto and preserve protected branch",
        "threshold_scan": False,
    }
    if spec.get("oof_policy") != expected_policy:
        raise ContractError("M30 policy drifted")
    expected_gates = {
        "retained_actions_min": 5,
        "beneficial_actions_min": 4,
        "beneficial_sequences_min": 3,
        "covered_evaluation_folds_min": 3,
        "beneficial_precision_min": 0.95,
        "catastrophic_actions_max": 0,
        "cup14_indoor_trigger1258_veto": True,
        "selected_mean_true_h10_gain_min": 0.2,
        "selected_branch_aggregate_gt_public": True,
        "all_abstain_is_not_pass": True,
    }
    if spec.get("scientific_gates") != expected_gates:
        raise ContractError("M30 scientific gates drifted")
    if spec.get("claim_ceiling") != EXPECTED_CLAIM_CEILING:
        raise ContractError("M30 claim ceiling drifted")
    expected_forbidden_roots = [
        "/root/autodl-tmp/depthtrack/test",
        "/root/autodl-tmp/CDTB",
        "/root/autodl-tmp/cdtb",
        "/root/autodl-tmp/vot",
        "/home/SUTrack_RGBD_L/vot-workspace",
    ]
    if spec.get("forbidden_runtime_roots") != expected_forbidden_roots:
        raise ContractError("M30 forbidden runtime roots drifted")
    if spec["source"]["runner"] != runner_record:
        raise ContractError("runner/spec identity drifted")
    if Path(spec["source"]["repository"]["path"]).resolve() != REPOSITORY_ROOT:
        raise ContractError("repository path drifted")
    for record in spec["source"]["dependencies"]:
        m23.validate_file_record(record)
    for name in ("spec", "binding"):
        m23.validate_file_record(spec["m22_inputs"][name])
    m23.validate_file_record(spec["m25_actions"])
    if spec["m25_actions"]["sha256"] != EXPECTED_M25_ACTION_SHA:
        raise ContractError("M25 action identity drifted")
    for name in ("result", "oof"):
        m23.validate_file_record(spec["m26_reference"][name])
    if (spec["m26_reference"]["result"]["sha256"] !=
            EXPECTED_M26_RESULT_SHA or
            spec["m26_reference"]["oof"]["sha256"] !=
            EXPECTED_M26_OOF_SHA):
        raise ContractError("M26 historical reference drifted")
    m23.validate_file_record(spec["m27_inputs"]["spec"])
    if spec["m27_inputs"]["spec"]["sha256"] != EXPECTED_M27_SPEC_SHA:
        raise ContractError("M27 spec identity drifted")
    result_shas = set()
    for shard in spec["m27_inputs"]["shards"]:
        for name in ("result", "events"):
            m23.validate_file_record(shard[name])
        result_shas.add(shard["result"]["sha256"])
    if result_shas != EXPECTED_M27_RESULT_SHAS:
        raise ContractError("M27 result identity drifted")


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
        raise ContractError("M30 preflight binding drifted")
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
            audit.get("authorization", {}).get("m30_execution") is not True or
            audit.get("audited_identity") != expected or
            audit.get("claim_ceiling") != spec["claim_ceiling"]):
        raise ContractError("M30 preaudit did not authorize exact execution")
    return record


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def load_m25_top_actions(record, m23):
    m23.validate_file_record(record)
    allowed_decision = {
        "top_role_id", "dominance", "margin", "benefit_probability",
        "catastrophe_probability", "selected_role_id",
    }
    actions = {}
    rows = 0
    with gzip.open(record["path"], "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            decision = row.get("decision", {})
            if set(decision) != allowed_decision:
                raise ContractError("M25 frozen decision field set drifted")
            key = event_key(row)
            top_role = decision.get("top_role_id")
            selected_role = row.get("selected_role_id")
            if key in actions or top_role is None:
                raise ContractError("M25 action identity drifted")
            if (selected_role is None) != (decision["selected_role_id"] is None):
                raise ContractError("M25 selected action nullability drifted")
            if selected_role is not None and (
                    int(selected_role) != int(decision["selected_role_id"]) or
                    int(selected_role) != int(top_role)):
                raise ContractError("M25 selected/top action identity drifted")
            actions[key] = {
                "evaluation_fold": int(row["evaluation_fold"]),
                "top_role_id": int(top_role),
                "selected_role_id": (
                    None if selected_role is None else int(selected_role)),
                "decision": {name: decision[name]
                             for name in sorted(allowed_decision)},
            }
    fold_counts = Counter(value["evaluation_fold"] for value in actions.values())
    selected = {key: value for key, value in actions.items()
                if value["selected_role_id"] is not None}
    selected_counts = Counter(
        value["evaluation_fold"] for value in selected.values())
    if (rows != 507 or len(actions) != 507 or
            dict(fold_counts) != {fold: EXPECTED_FOLD_COUNTS[fold]["events"]
                                 for fold in FOLDS} or
            len(selected) != 12 or
            dict(selected_counts) != EXPECTED_ACTION_FOLDS):
        raise ContractError("M25 frozen top/final action census drifted")
    return actions


def load_m27_index(shards, required, m22, m23):
    index = {}
    for shard in shards:
        for row in m22.load_verified_jsonl(shard["events"]):
            key = event_key(row)
            if key in index:
                raise ContractError("duplicate M27 event key")
            for name in ("feature", "source_candidate_feature"):
                m23.validate_file_record(row[name])
            index[key] = row
    if set(index) != set(required):
        raise ContractError("M27 event join closure drifted")
    return index


def target_tensors(torch, m23, rows, valid):
    target = m23.target_tensors(torch, rows, valid)
    ordered = sorted(rows, key=lambda value: int(value["candidate_role_id"]))
    target["strict_components"] = torch.tensor([
        [float(row["targets"]["10"]["gain"]),
         float(row["targets"]["10"]["branch_mean_iou"]),
         float(row["early_hits_h5"]) / 5.0]
        for row in ordered
    ], dtype=torch.float32)
    components = target["strict_components"]
    if (tuple(components.shape) != (6, 3) or
            not torch.isfinite(components).all().item() or
            float(components[:, 0].min()) < -1.0 or
            float(components[:, 0].max()) > 1.0 or
            float(components[:, 1:].min()) < 0.0 or
            float(components[:, 1:].max()) > 1.0):
        raise ContractError("M30 strict-benefit target drifted")
    return target


def aggregate_identity_relation(torch, relation, groups):
    output = torch.zeros_like(relation)
    for members in groups:
        representative = members[0]
        index = torch.tensor(members, dtype=torch.int64)
        output[:, representative] = relation[:, index].mean(dim=1)
    return output


def load_identity_relations(torch, m27_index, duplicate_groups,
                            bounded_identity_relation):
    identities = {condition: {} for condition in CONDITIONS}
    for key in sorted(m27_index):
        row = m27_index[key]
        candidate = torch.load(
            row["source_candidate_feature"]["path"], map_location="cpu")
        protected = torch.load(row["feature"]["path"], map_location="cpu")
        candidate_native = torch.stack([
            candidate["native_rgb"], candidate["native_depth"],
            candidate["native_fused"],
        ], dim=2).unsqueeze(0)
        protected_native = torch.stack([
            protected["public_native_rgb"],
            protected["public_native_depth"],
            protected["public_native_fused"],
        ], dim=1).unsqueeze(0)
        for condition in CONDITIONS:
            raw = bounded_identity_relation(
                candidate_native, protected_native, "candidate_temporal")[0]
            aggregated = aggregate_identity_relation(
                torch, raw, duplicate_groups[key])
            identity = aggregated.reshape(5, 6, 9)
            if (tuple(identity.shape) != (5, 6, 9) or
                    not torch.isfinite(identity).all().item()):
                raise ContractError("M30 temporal identity relation drifted")
            identities[condition][key] = identity
    return identities


def make_batch(torch, keys, actions, relations, identities, targets,
               sequence_counts):
    fields = defaultdict(list)
    for key in keys:
        role = actions[key]["top_role_id"]
        if not bool(targets[key]["valid"][role]):
            raise ContractError("M25 top action is not a canonical representative")
        fields["scalar"].append(relations[key][2][:, role])
        fields["identity"].append(identities[key][:, role])
        fields["strict_components"].append(
            targets[key]["strict_components"][role])
        fields["event_weight"].append(1.0 / sequence_counts[key[0]])
    batch = {name: torch.stack(value) for name, value in fields.items()
             if name != "event_weight"}
    batch["event_weight"] = torch.tensor(
        fields["event_weight"], dtype=torch.float32)
    batch["event_weight"] /= batch["event_weight"].sum()
    return batch


def action_batch(torch, keys, actions, relations, identities, targets):
    for key in keys:
        role = actions[key]["top_role_id"]
        if not bool(targets[key]["valid"][role]):
            raise ContractError("M25 top action is not a canonical representative")
    return {
        "scalar": torch.stack([
            relations[key][2][:, actions[key]["top_role_id"]]
            for key in keys]),
        "identity": torch.stack([
            identities[key][:, actions[key]["top_role_id"]]
            for key in keys]),
    }


def model_outputs(torch, model, batch):
    with torch.no_grad():
        return model(batch["scalar"], batch["identity"])


def component_vector(torch, outputs):
    return torch.stack((
        outputs["predicted_gain_h10"],
        outputs["predicted_branch_mean_h10"],
        outputs["predicted_early_hit_rate_h5"],
    ), dim=-1)


def forward_loss(torch, model, batch):
    outputs = model(batch["scalar"], batch["identity"])
    predicted = component_vector(torch, outputs)
    raw = torch.nn.functional.smooth_l1_loss(
        predicted, batch["strict_components"], reduction="none")
    per_event = raw.mean(dim=1)
    loss = (per_event * batch["event_weight"]).sum()
    if not math.isfinite(float(loss.detach())):
        raise ContractError("M30 loss is non-finite")
    return outputs, loss


def calibration_offset(torch, model, keys, actions, relations, identities,
                       targets, policy):
    by_sequence = defaultdict(lambda: [[], [], []])
    rows = []
    for key in keys:
        output = model_outputs(
            torch, model,
            action_batch(torch, [key], actions, relations, identities,
                         targets))
        predicted = component_vector(torch, output)[0]
        role = actions[key]["top_role_id"]
        actual = targets[key]["strict_components"][role]
        residual = predicted - actual
        for component in range(3):
            by_sequence[key[0]][component].append(
                float(residual[component]))
        rows.append((predicted.clone(), actual.clone()))
    sequence_quantiles = []
    for sequence in sorted(by_sequence):
        sequence_quantiles.append(torch.tensor([
            float(torch.quantile(torch.tensor(
                by_sequence[sequence][h], dtype=torch.float64),
                float(policy["within_sequence_residual_quantile"])))
            for h in range(3)
        ], dtype=torch.float64))
    offset = torch.quantile(
        torch.stack(sequence_quantiles),
        float(policy["across_sequence_residual_quantile"]), dim=0).float()
    covered = torch.zeros(3, dtype=torch.int64)
    total = torch.zeros(3, dtype=torch.int64)
    for predicted, actual in rows:
        covered += (actual >= predicted - offset).to(torch.int64)
        total += 1
    return offset, {
        "sequences": len(by_sequence),
        "utility_top_actions": len(rows),
        "offset_gain_branch_mean_early_hit_rate": [
            float(value) for value in offset],
        "empirical_lower_coverage_gain_branch_mean_early_hit_rate": [
            float(covered[h]) / float(total[h]) for h in range(3)],
    }


def condition_predictions(torch, model, eval_fold, keys, actions, offset,
                          relations, identities, targets, policy, condition):
    rows = []
    for key in keys:
        output = model_outputs(
            torch, model,
            action_batch(torch, [key], actions, relations, identities,
                         targets))
        predicted = component_vector(torch, output)[0]
        action = actions[key]
        top_role = action["top_role_id"]
        frozen_role = action["selected_role_id"]
        retained = False
        calibrated_lower = None
        if frozen_role is not None:
            if (frozen_role != top_role or
                    not bool(targets[key]["valid"][frozen_role])):
                raise ContractError("M25 selected an invalid top representative")
            calibrated_lower = predicted - offset
            retained = bool(
                float(calibrated_lower[0]) >= float(
                    policy["calibrated_gain_h10_lower_min"]) and
                float(calibrated_lower[1]) >= float(
                    policy["calibrated_branch_mean_h10_lower_min"]) and
                float(calibrated_lower[2]) >= float(
                    policy["calibrated_early_hit_rate_h5_lower_min"]))
        selected = frozen_role if retained else None
        rows.append({
            "record_type": "m30_utility_conditioned_strict_benefit_prediction",
            "condition": condition,
            "evaluation_fold": eval_fold,
            "calibration_fold": CALIBRATION_FOLDS[eval_fold],
            "fit_folds": list(FIT_FOLDS[eval_fold]),
            "sequence": key[0],
            "event_id": key[1],
            "trigger_frame": key[2],
            "utility_top_role_id": top_role,
            "frozen_m25_action": frozen_role is not None,
            "frozen_m25_role_id": frozen_role,
            "predicted_gain_branch_mean_early_hit_rate": [
                float(value) for value in predicted],
            "calibration_offset_gain_branch_mean_early_hit_rate": (
                [float(value) for value in offset]
                if frozen_role is not None else None),
            "calibrated_lower_gain_branch_mean_early_hit_rate": (
                [float(value) for value in calibrated_lower]
                if calibrated_lower is not None else None),
            "selected_role_id": selected,
            "selected_strict_label": (
                targets[key]["labels"][selected]
                if selected is not None else "abstain"),
            "selected_actual_h10_gain": (
                float(targets[key]["gain"][selected])
                if selected is not None else 0.0),
            "selected_actual_h10_branch_mean_iou": (
                float(targets[key]["branch_mean"][selected])
                if selected is not None else None),
            "selected_actual_h10_public_mean_iou": (
                float(targets[key]["public_mean"][selected])
                if selected is not None else None),
            "selected_actual_h5_early_hit_rate": (
                float(targets[key]["strict_components"][selected, 2])
                if selected is not None else None),
        })
    return rows


def permutation_audit(torch, m23, model, keys, actions, relations, identities,
                      targets):
    candidate_maximum = 0.0
    candidate_non_equal = 0
    base = {}
    for key in keys:
        batch = action_batch(
            torch, [key], actions, relations, identities, targets)
        base[key] = component_vector(
            torch, model_outputs(torch, model, batch))[0]
        permutation = m23.candidate_permutation(
            torch, 20260930, 999, 0, 0, key)
        role = actions[key]["top_role_id"]
        permuted_role = int(
            (permutation == role).nonzero(as_tuple=False).item())
        permuted = {
            "scalar": relations[key][2][:, permutation][:, permuted_role]
                .unsqueeze(0),
            "identity": identities[key][:, permutation][:, permuted_role]
                .unsqueeze(0),
        }
        restored = component_vector(
            torch, model_outputs(torch, model, permuted))[0]
        candidate_maximum = max(
            candidate_maximum, float((restored - base[key]).abs().max()))
        if not torch.equal(restored, base[key]):
            candidate_non_equal += 1
    replay_non_equal = 0
    replay_maximum = 0.0
    for key in reversed(keys):
        replay = component_vector(torch, model_outputs(
            torch, model,
            action_batch(torch, [key], actions, relations, identities,
                         targets)
        ))[0]
        replay_maximum = max(
            replay_maximum, float((replay - base[key]).abs().max()))
        if not torch.equal(replay, base[key]):
            replay_non_equal += 1
    return {
        "canonical_role_restoration": {
            "events": len(keys),
            "maximum_absolute_error": candidate_maximum,
            "non_equal_outputs": candidate_non_equal,
        },
        "event_replay": {
            "events": len(keys),
            "maximum_absolute_error": replay_maximum,
            "non_equal_outputs": replay_non_equal,
        },
    }


def scientific_conditions(summary, rows, gates):
    covered = {row["evaluation_fold"] for row in rows
               if row["selected_role_id"] is not None}
    cup14 = [row for row in rows
             if row["sequence"] == "cup14_indoor" and
             row["trigger_frame"] == 1258]
    return {
        "retained_actions_min": summary["selected_actions"] >=
            gates["retained_actions_min"],
        "beneficial_actions_min": summary["beneficial_actions"] >=
            gates["beneficial_actions_min"],
        "beneficial_sequences_min": summary["beneficial_sequences"] >=
            gates["beneficial_sequences_min"],
        "covered_evaluation_folds_min": len(covered) >=
            gates["covered_evaluation_folds_min"],
        "beneficial_precision_min": summary["beneficial_precision"] >=
            gates["beneficial_precision_min"],
        "catastrophic_actions_max": summary["catastrophic_actions"] <=
            gates["catastrophic_actions_max"],
        "cup14_indoor_trigger1258_veto": (
            len(cup14) == 1 and cup14[0]["frozen_m25_action"] and
            cup14[0]["selected_role_id"] is None),
        "selected_mean_true_h10_gain_min": (
            summary["selected_mean_true_h10_gain"] is not None and
            summary["selected_mean_true_h10_gain"] >=
            gates["selected_mean_true_h10_gain_min"]),
        "selected_branch_aggregate_gt_public": (
            summary["selected_branch_aggregate_h10_mean_iou"] is not None and
            summary["selected_public_aggregate_h10_mean_iou"] is not None and
            summary["selected_branch_aggregate_h10_mean_iou"] >
            summary["selected_public_aggregate_h10_mean_iou"]),
        "all_abstain_is_not_pass": summary["selected_actions"] > 0,
    }


def publish(output, result, trace, predictions, m23):
    temporary = Path(tempfile.mkdtemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)))
    try:
        m23.atomic_jsonl_gz(temporary / "training_trace.jsonl.gz", trace)
        m23.atomic_jsonl_gz(temporary / "oof_predictions.jsonl.gz", predictions)
        m23.atomic_json(temporary / "result.json", result)
        manifest = {
            "schema": "sttrack-lachtt-m30-output-manifest/v1",
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
            raise ContractError("M30 output set drifted")
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.chmod(0o555)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def load_development(torch, m22, m23, m22_spec_record, m22_binding_record,
                     m27_shards, bounded_identity_relation):
    m22_spec = m22.load_verified_json(m22_spec_record)
    m22_binding = m22.load_verified_json(m22_binding_record)
    m22.validate_frozen_receipts(m22_spec)
    collection, sequence_anchors = m22.load_collection_index(m22_spec)
    _, split_entries = m22.load_split_ledger(m22_spec)
    training_groups, heldout_commitment, target_counts = (
        m22.load_training_targets(m22_spec, split_entries))
    if (len(training_groups) != 507 or
            heldout_commitment.get("numeric_targets_serialized") is not False or
            int(target_counts.get("action_target", -1)) != 3042):
        raise ContractError("M30 target closure drifted")
    fold_groups = {fold: {} for fold in FOLDS}
    for key, rows in training_groups.items():
        fold = int(split_entries[key]["fold"])
        if split_entries[key]["partition"] != "training" or fold not in FOLDS:
            raise ContractError("M30 training fold drifted")
        fold_groups[fold][key] = rows
    fold_sequences = {
        fold: {key[0] for key in fold_groups[fold]} for fold in FOLDS}
    if any(fold_sequences[left] & fold_sequences[right]
           for i, left in enumerate(FOLDS) for right in FOLDS[i + 1:]):
        raise ContractError("M30 sequence folds overlap")
    duplicate_groups = m23.load_duplicate_groups(
        m22, m22_spec, set(training_groups), training_groups)
    unique_counts = {key: len(groups) for key, groups in duplicate_groups.items()}
    if sum(unique_counts.values()) != 2106:
        raise ContractError("M30 unique hypothesis census drifted")
    native_index = m22.load_native_index(m22_spec)
    # The sealed M22 binding contains all 152 Train anchor records and its
    # validator requires that complete identity set.  Only the relation loop
    # below dereferences payloads, and that loop remains restricted to the 76
    # folds-2--5 training sequences in training_groups.
    required_sequences = sorted({key[0] for key in split_entries})
    clip_binding, _ = m22.validate_anchor_binding(
        m22_binding, sequence_anchors, native_index, required_sequences)
    relations, targets = {}, {}
    clip_cache, native_cache, loaded_features = {}, {}, {}
    for key in sorted(training_groups):
        relation = m22.relation_for_event(
            m22_spec, collection, sequence_anchors, native_index,
            clip_binding, key, clip_cache, native_cache, loaded_features)
        relation = m23.aggregate_relation(torch, relation, duplicate_groups[key])
        relations[key] = relation[:3]
        targets[key] = target_tensors(
            torch, m23, training_groups[key], relation[3])
    m27_index = load_m27_index(m27_shards, training_groups, m22, m23)
    identities = load_identity_relations(
        torch, m27_index, duplicate_groups, bounded_identity_relation)
    return (training_groups, fold_groups, fold_sequences, unique_counts,
            relations, identities, targets)


def smoke_main(args):
    (torch, m22, m23, _, Head, bounded_identity_relation,
     parameter_count, identity_dim, input_dim, hidden_dim) = load_components()
    if args.smoke_output.exists():
        raise ContractError("smoke output already exists")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    m22_spec_record = m23.file_record(args.m22_spec.resolve())
    m22_binding_record = m23.file_record(args.m22_binding.resolve())
    m27_spec_record = m23.file_record(args.m27_spec.resolve())
    m25_action_record = m23.file_record(args.m25_actions.resolve())
    if m25_action_record["sha256"] != EXPECTED_M25_ACTION_SHA:
        raise ContractError("smoke M25 action identity drifted")
    actions = load_m25_top_actions(m25_action_record, m23)
    if m27_spec_record["sha256"] != EXPECTED_M27_SPEC_SHA:
        raise ContractError("smoke M27 spec identity drifted")
    shards = []
    for shard in sorted(args.m27_root.resolve().glob("shard*")):
        events = shard / "events.jsonl"
        result = shard / "result.json"
        if events.is_file() and result.is_file():
            shards.append({"events": m23.file_record(events),
                           "result": m23.file_record(result)})
    if {shard["result"]["sha256"] for shard in shards} != (
            EXPECTED_M27_RESULT_SHAS):
        raise ContractError("smoke M27 result identity drifted")
    data = load_development(
        torch, m22, m23, m22_spec_record, m22_binding_record, shards,
        bounded_identity_relation)
    _, _, _, _, relations, identities, targets = data
    key = sorted(relations)[0]
    outputs = {}
    state_digests = {}
    for condition in CONDITIONS:
        torch.manual_seed(20260930)
        model = Head()
        if parameter_count(model) != 8323:
            raise ContractError("smoke parameter count drifted")
        state_digests[condition] = m22.state_digest(model)
        output = model_outputs(
            torch, model,
            action_batch(torch, [key], actions, relations,
                         identities[condition], targets))
        outputs[condition] = {
            "predicted_gain_h10": output["predicted_gain_h10"].tolist(),
            "predicted_branch_mean_h10": output[
                "predicted_branch_mean_h10"].tolist(),
            "predicted_early_hit_rate_h5": output[
                "predicted_early_hit_rate_h5"].tolist(),
            "age_hidden": output["age_hidden"].tolist(),
        }
    identity_maximum = float(
        identities["strict_benefit_temporal"][key].abs().max())
    result = {
        "schema": "sttrack-lachtt-m30-no-optimizer-smoke/v1",
        "complete": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "event": {"sequence": key[0], "event_id": key[1],
                  "trigger_frame": key[2]},
        "utility_top_role_id": actions[key]["top_role_id"],
        "identity_relation_dimension_per_age": identity_dim,
        "safety_input_dimension_per_age": input_dim,
        "temporal_hidden_dimension": hidden_dim,
        "parameters": 8323,
        "input_identity": {
            "runner": m23.file_record(Path(__file__).resolve()),
            "m22_spec": m22_spec_record,
            "m22_binding": m22_binding_record,
            "m27_spec": m27_spec_record,
            "m27_shards": shards,
            "m25_actions": m25_action_record,
        },
        "identical_initial_state": len(set(state_digests.values())) == 1,
        "candidate_identity_relation_max_abs": identity_maximum,
        "outputs": outputs,
        "runtime_forbidden_file_open_count": len(
            RUNTIME_AUDIT["forbidden_file_opens"]),
        "runtime_network_connect_count": len(RUNTIME_AUDIT["network_connects"]),
    }
    if (not result["identical_initial_state"] or identity_maximum <= 0 or
            result["runtime_forbidden_file_open_count"] or
            result["runtime_network_connect_count"]):
        raise ContractError("M30 smoke gate failed")
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    m23.atomic_json(args.smoke_output, result)
    print(json.dumps({"accepted": True, "output": str(args.smoke_output),
                      "optimizer_steps": 0}, sort_keys=True), flush=True)


def formal_main(args):
    args.spec = args.spec.resolve()
    args.preflight = args.preflight.resolve()
    args.preaudit = args.preaudit.resolve()
    args.output = args.output.resolve()
    started = time.time()
    (torch, m22, m23, m25, Head, bounded_identity_relation,
     parameter_count, identity_dim, input_dim, hidden_dim) = load_components()
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
        raise ContractError("repository identity drifted before M30")
    FORBIDDEN_RUNTIME_ROOTS[:] = [
        Path(value).resolve() for value in spec["forbidden_runtime_roots"]]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    actions = load_m25_top_actions(spec["m25_actions"], m23)
    data = load_development(
        torch, m22, m23, spec["m22_inputs"]["spec"],
        spec["m22_inputs"]["binding"], spec["m27_inputs"]["shards"],
        bounded_identity_relation)
    (training_groups, fold_groups, fold_sequences, unique_counts,
     relations, identities, targets) = data
    for key, action in actions.items():
        if key not in training_groups or action["evaluation_fold"] != int(
                next(f for f in FOLDS if key in fold_groups[f])):
            raise ContractError("M25 action fold join drifted")

    models, optimizers = {}, {}
    initial_states, final_states, parameter_counts = {}, {}, {}
    for condition in CONDITIONS:
        for fold in FOLDS:
            torch.manual_seed(int(spec["training"]["seed"]))
            model = Head()
            if parameter_count(model) != 8323:
                raise ContractError("M30 parameter count drifted")
            models[(condition, fold)] = model
            optimizers[(condition, fold)] = torch.optim.AdamW(
                model.parameters(), lr=spec["training"]["learning_rate"],
                weight_decay=spec["training"]["weight_decay"])
            initial_states[(condition, fold)] = m22.state_digest(model)
            parameter_counts[(condition, fold)] = parameter_count(model)
    if len(set(initial_states.values())) != 1:
        raise ContractError("M30 initial states differ")

    trace = []
    total_steps = 0
    steps = {}
    for condition in CONDITIONS:
        for eval_fold in FOLDS:
            model = models[(condition, eval_fold)]
            optimizer = optimizers[(condition, eval_fold)]
            fit_groups = {key: rows for fold in FIT_FOLDS[eval_fold]
                          for key, rows in fold_groups[fold].items()}
            sequence_counts = Counter(key[0] for key in fit_groups)
            member_step = 0
            model.train()
            for epoch in range(spec["training"]["epochs_per_member"]):
                batches = m23.epoch_batches(
                    list(fit_groups), spec["training"]["seed"], epoch,
                    spec["training"]["event_batch_size"])
                for epoch_step, keys in enumerate(batches):
                    member_step += 1
                    total_steps += 1
                    batch = make_batch(
                        torch, keys, actions, relations, identities[condition],
                        targets, sequence_counts)
                    optimizer.zero_grad(set_to_none=True)
                    outputs, loss = forward_loss(torch, model, batch)
                    if any(not value.isfinite().all().item()
                           for value in outputs.values()):
                        raise ContractError("M30 output is non-finite")
                    loss.backward()
                    preclip, nonfinite, _ = m22.gradient_diagnostics(model, 0)
                    if nonfinite or not math.isfinite(preclip) or preclip <= 0:
                        raise ContractError("M30 preclip gradient gate failed")
                    maximum = spec["training"]["gradient_clip_norm"]
                    m22.scale_gradients(
                        model, min(1.0, maximum / (preclip + 1.0e-12)))
                    postclip, post_nonfinite, _ = m22.gradient_diagnostics(
                        model, 0)
                    if post_nonfinite or postclip > 5.000001:
                        raise ContractError("M30 postclip gradient gate failed")
                    optimizer.step()
                    trace.append({
                        "record_type": "optimizer_step",
                        "global_step": total_steps,
                        "condition": condition,
                        "evaluation_fold": eval_fold,
                        "calibration_fold": CALIBRATION_FOLDS[eval_fold],
                        "fit_folds": list(FIT_FOLDS[eval_fold]),
                        "member_step": member_step,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "batch_size": len(keys),
                        "strict_benefit_component_smooth_l1": float(
                            loss.detach()),
                        "preclip_total_l2": preclip,
                        "postclip_total_l2": postclip,
                        "optimizer_step_executed": True,
                    })
            steps[(condition, eval_fold)] = member_step
            final_states[(condition, eval_fold)] = m22.state_digest(model)
    if total_steps != 1536 or len(trace) != 1536:
        raise ContractError("M30 optimizer step count drifted")

    offsets, calibration = {}, {}
    predictions = []
    condition_summaries = {}
    condition_scientific = {}
    audits = {}
    for condition in CONDITIONS:
        condition_rows = []
        audits[condition] = {}
        for eval_fold in FOLDS:
            model = models[(condition, eval_fold)]
            model.eval()
            cal_fold = CALIBRATION_FOLDS[eval_fold]
            offset, diagnostics = calibration_offset(
                torch, model, sorted(fold_groups[cal_fold]), actions,
                relations, identities[condition], targets,
                spec["oof_policy"])
            offsets[(condition, eval_fold)] = offset
            calibration[f"{condition}:{eval_fold}"] = diagnostics
            rows = condition_predictions(
                torch, model, eval_fold, sorted(fold_groups[eval_fold]),
                actions, offset, relations, identities[condition], targets,
                spec["oof_policy"], condition)
            condition_rows.extend(rows)
            audits[condition][str(eval_fold)] = permutation_audit(
                torch, m23, model, sorted(fold_groups[eval_fold]), actions,
                relations, identities[condition], targets)
        predictions.extend(condition_rows)
        summary = m25.fold_summary(condition_rows)
        condition_summaries[condition] = summary
        condition_scientific[condition] = scientific_conditions(
            summary, condition_rows, spec["scientific_gates"])

    condition_pass = {
        name: all(condition_scientific[name].values()) for name in CONDITIONS}
    interpretation = (
        "strict_benefit_temporal_pass"
        if condition_pass["strict_benefit_temporal"] else
        "strict_benefit_temporal_fail_stop_family")
    candidate_permutation_within_tolerance = all(
        fold["canonical_role_restoration"]["maximum_absolute_error"] <= 1.0e-6
        for condition in audits.values() for fold in condition.values())
    event_replay_exact = all(
        fold["event_replay"]["maximum_absolute_error"] == 0.0 and
        fold["event_replay"]["non_equal_outputs"] == 0
        for condition in audits.values() for fold in condition.values())
    engineering_conditions = {
        "event_sequence_unique_census_exact": (
            len(training_groups) == 507 and
            len({key[0] for key in training_groups}) == 76 and
            sum(unique_counts.values()) == 2106),
        "m25_top_action_substrate_exact": len(actions) == 507,
        "m25_final_action_substrate_exact": sum(
            value["selected_role_id"] is not None
            for value in actions.values()) == 12,
        "m27_join_exact": all(len(identities[c]) == 507 for c in CONDITIONS),
        "matched_dimensions_exact": (
            identity_dim == 9 and input_dim == 58 and hidden_dim == 32),
        "matched_parameter_counts_exact": all(
            value == 8323 for value in parameter_counts.values()),
        "utility_candidate_ranking_parameters_zero": True,
        "optimizer_steps_exact": total_steps == 1536,
        "trace_rows_exact": len(trace) == 1536,
        "member_step_counts_exact": all(
            steps[(condition, fold)] == EXPECTED_FOLD_COUNTS[fold]["steps"]
            for condition in CONDITIONS for fold in FOLDS),
        "all_states_changed": all(
            initial_states[key] != final_states[key] for key in initial_states),
        "canonical_role_restoration_within_1e_6": (
            candidate_permutation_within_tolerance),
        "event_replay_exact": event_replay_exact,
        "sequence_fold_overlap_zero": all(
            not (fold_sequences[left] & fold_sequences[right])
            for i, left in enumerate(FOLDS) for right in FOLDS[i + 1:]),
        "prediction_rows_exact": len(predictions) == 507,
        "repository_clean": not git_output("status", "--porcelain"),
        "repository_commit_exact": (
            git_output("rev-parse", "HEAD") == repository["commit"]),
        "forbidden_numeric_target_path_access_zero": not RUNTIME_AUDIT[
            "forbidden_file_opens"],
        "network_connect_access_zero": not RUNTIME_AUDIT["network_connects"],
        "qwen_module_not_loaded": not any(
            "qwen" in name.lower() for name in sys.modules),
        "target_folds_loaded_exact": all(
            int(next(fold for fold in FOLDS if key in fold_groups[fold]))
            in FOLDS for key in training_groups),
        "output_contract_excludes_checkpoint": not any(
            Path(name).suffix in {".ckpt", ".pth", ".pt"}
            for name in OUTPUT_FILES),
    }
    engineering_pass = all(engineering_conditions.values())
    accepted = engineering_pass and condition_pass[
        "strict_benefit_temporal"]
    result = {
        "schema": "sttrack-lachtt-m30-result/v1",
        "complete": True,
        "accepted": accepted,
        "engineering_pass": engineering_pass,
        "condition_pass": condition_pass,
        "interpretation": interpretation,
        "decision": (
            "m30_pass_authorize_train_only_integration_plan_only"
            if accepted else "m30_stop_according_to_frozen_interpretation"),
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
            "events": len(training_groups),
            "sequences": len({key[0] for key in training_groups}),
            "unique_hypotheses": sum(unique_counts.values()),
            "m25_utility_top_actions": len(actions),
            "m25_frozen_final_actions": sum(
                value["selected_role_id"] is not None
                for value in actions.values()),
            "runtime_forbidden_file_open_count": len(
                RUNTIME_AUDIT["forbidden_file_opens"]),
            "runtime_network_connect_count": len(
                RUNTIME_AUDIT["network_connects"]),
        },
        "model": {
            "conditions": list(CONDITIONS),
            "parameters_per_fold_condition": 8323,
            "identity_relation_dimension_per_age": identity_dim,
            "safety_input_dimension_per_age": input_dim,
            "temporal_hidden_dimension": hidden_dim,
            "utility_trainable_parameters": 0,
            "candidate_ranking_trainable_parameters": 0,
        },
        "training": {
            "optimizer_steps": total_steps,
            "trace_rows": len(trace),
            "steps_per_condition_fold": {
                f"{condition}:{fold}": steps[(condition, fold)]
                for condition in CONDITIONS for fold in FOLDS},
        },
        "calibration": calibration,
        "oof_summary": condition_summaries,
        "scientific_conditions": condition_scientific,
        "permutation_audit": audits,
        "engineering_conditions": engineering_conditions,
        "failed_engineering_conditions": sorted(
            name for name, passed in engineering_conditions.items()
            if not passed),
        "authorization": {
            "train_only_integration_plan_only": accepted,
            "integration_execution": False,
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
            git_output("status", "--porcelain")):
        raise ContractError("M30 source changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish(args.output, result, trace, predictions, m23)
    print(json.dumps({
        "accepted": accepted,
        "engineering_pass": engineering_pass,
        "condition_pass": condition_pass,
        "interpretation": interpretation,
        "optimizer_steps": total_steps,
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    raise SystemExit(0 if accepted else 2)


def main():
    args = parse_args()
    if args.smoke:
        smoke_main(args)
    else:
        formal_main(args)


if __name__ == "__main__":
    main()
