#!/usr/bin/env python3
"""Audit final M5 predictions on the training folds only."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.models.sttrack.lachtt_rollout_association import (
    LanguageAnchoredDenseAssociation,
)
from lib.models.sttrack.lachtt_setwise_association import (
    SetwiseCandidateAssociation,
)
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from tools.run_sttrack_lachtt_recursive_pilot import (
    atomic_json,
    build_context,
    load_schedule,
    sha256_file,
    stable_fold,
    valid_window,
)
from tools.run_sttrack_lachtt_setwise_pilot import run_event, select_action


def summarize(values):
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("invalid prediction distribution")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.heads = args.heads.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    config = Path(spec["base_model"]["config"]["path"])
    checkpoint = Path(spec["base_model"]["checkpoint"]["path"])
    update_config_from_file(str(config))
    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("tracking checkpoint strict load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    dense = LanguageAnchoredDenseAssociation().cuda().eval()
    setwise = SetwiseCandidateAssociation().cuda().eval()
    saved = torch.load(str(args.heads), map_location="cpu")
    dense.load_state_dict(saved["dense"], strict=True)
    setwise.load_state_dict(saved["setwise"], strict=True)
    before = {
        "dense": {key: value.detach().clone()
                  for key, value in dense.state_dict().items()},
        "setwise": {key: value.detach().clone()
                    for key, value in setwise.state_dict().items()},
    }
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    trace_paths = [Path(row["path"])
                   for row in spec["data"]["protected_trace_shards"]]
    rows_by_sequence, events = load_schedule(trace_paths, 4)
    fold_count = int(spec["data"]["outer_fold_count"])
    heldout = int(spec["data"]["evaluation_outer_fold"])
    events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", fold_count) != heldout]
    events = events[:int(spec["data"]["training_event_limit"])]
    contexts = {}
    dataset_root = Path(spec["data"]["dataset_root"])
    language = Path(spec["inputs"]["language_manifest"]["path"])
    clip_model = Path(spec["inputs"]["clip"]["path"])

    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language, clip_model)
        return contexts[sequence]

    identity = torch.arange(6, device="cuda")
    valid_events = 0
    unavailable_events = 0
    selected = []
    beneficial_capacity = 0
    catastrophic_capacity = 0
    distributions = {key: [] for key in (
        "top_candidate_selection_probability",
        "abstain_probability",
        "top_candidate_margin",
        "top_candidate_beneficial_probability",
        "top_candidate_catastrophic_probability",
        "top_candidate_predicted_gain",
    )}
    threshold_pass = {key: 0 for key in (
        "candidate_beats_abstain",
        "candidate_selection_probability",
        "selection_probability_margin",
        "beneficial_probability",
        "catastrophic_probability",
        "predicted_gain",
        "all_conditions",
    )}
    policy = spec["evaluation_action_policy"]
    for event in events:
        item = context(event["sequence"])
        frame = event["trigger_frame"]
        if not valid_window(item.gt, frame, 4):
            unavailable_events += 1
            continue
        with torch.no_grad():
            trace = run_event(
                network, dense, setwise, None, preprocessor, item, frame, 4,
                keep_rate, window, identity, False,
                spec["training"].get("loss_balance", {}))
        valid_events += 1
        beneficial_capacity += int(any(
            action["actual_beneficial"] for action in trace["actions"]))
        catastrophic_capacity += int(any(
            action["actual_catastrophic"] for action in trace["actions"]))
        action, top_name, _ = select_action(trace, policy)
        top = max(trace["actions"], key=lambda row:
                  (row["selection_probability"], row["name"]))
        candidate_probabilities = sorted(
            (row["selection_probability"] for row in trace["actions"]),
            reverse=True)
        competitor = max(trace["abstain_probability"],
                         candidate_probabilities[1])
        margin = top["selection_probability"] - competitor
        checks = {
            "candidate_beats_abstain":
                top["selection_probability"] > trace["abstain_probability"],
            "candidate_selection_probability":
                top["selection_probability"] >=
                policy["candidate_selection_probability_min"],
            "selection_probability_margin":
                margin >= policy["selection_probability_margin_min"],
            "beneficial_probability":
                top["beneficial_probability"] >=
                policy["beneficial_probability_min"],
            "catastrophic_probability":
                top["catastrophic_probability"] <=
                policy["catastrophic_probability_max"],
            "predicted_gain":
                top["predicted_gain"] >= policy["predicted_gain_min"],
        }
        for key, value in checks.items():
            threshold_pass[key] += int(value)
        threshold_pass["all_conditions"] += int(all(checks.values()))
        distributions["top_candidate_selection_probability"].append(
            top["selection_probability"])
        distributions["abstain_probability"].append(
            trace["abstain_probability"])
        distributions["top_candidate_margin"].append(margin)
        distributions["top_candidate_beneficial_probability"].append(
            top["beneficial_probability"])
        distributions["top_candidate_catastrophic_probability"].append(
            top["catastrophic_probability"])
        distributions["top_candidate_predicted_gain"].append(
            top["predicted_gain"])
        if action is not None:
            label = ("catastrophic" if action["actual_catastrophic"] else
                     "beneficial" if action["actual_beneficial"] else
                     "neutral")
            selected.append({
                "sequence": event["sequence"],
                "trigger_frame": frame,
                "candidate": action["name"],
                "label": label,
                "actual_gain": action["actual_gain"],
                "top_name": top_name,
                "selection_margin": margin,
            })
    if any(not torch.equal(before[group][key], module.state_dict()[key])
           for group, module in (("dense", dense), ("setwise", setwise))
           for key in before[group]):
        raise RuntimeError("read-only audit mutated heads")
    stage = ("m6" if spec["schema"].startswith("sttrack-lachtt-m6-")
             else "m5")
    result = {
        "schema": "sttrack-lachtt-%s-training-fold-prediction-audit/v1" %
                  stage,
        "complete": True,
        "train_only": True,
        "heldout_fold_opened": False,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "spec_sha256": sha256_file(args.spec),
        "heads_sha256": sha256_file(args.heads),
        "scheduled_events": len(events),
        "valid_events": valid_events,
        "unavailable_events": unavailable_events,
        "beneficial_capacity_events": beneficial_capacity,
        "catastrophic_capacity_events": catastrophic_capacity,
        "selected_actions": len(selected),
        "beneficial_actions": sum(row["label"] == "beneficial"
                                  for row in selected),
        "neutral_actions": sum(row["label"] == "neutral"
                               for row in selected),
        "catastrophic_actions": sum(row["label"] == "catastrophic"
                                    for row in selected),
        "threshold_pass_counts": threshold_pass,
        "top_candidate_distributions": {
            key: summarize(values) for key, values in distributions.items()
        },
        "selected_details": selected,
        "elapsed_seconds": time.time() - started,
        "checkpoint_written": False,
        "qwen_used": False,
        "vot_run": False,
        "automatic_next_stage": False,
    }
    atomic_json(args.output, result)
    args.output.chmod(0o444)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
