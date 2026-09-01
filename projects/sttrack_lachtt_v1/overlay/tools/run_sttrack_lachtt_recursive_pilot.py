#!/usr/bin/env python3
"""Run the pre-registered Train-only M3 recursive association pilot."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.models.sttrack.lachtt_rollout_association import (
    LanguageAnchoredDenseAssociation,
)
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.tracker.sttrack_lachtt_observation import ClipCandidateEncoder
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from tools.run_sttrack_lachtt_train152_collection import read_initial_bbox
from tools.smoke_sttrack_lachtt_multicenter_recursive_training import (
    frames_for,
    language_for,
    network_forward,
    one_rollout,
    read_gt,
    read_rgbd,
    sha256_file,
    template_bank,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl_gz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stable_fold(sequence, salt, folds):
    value = hashlib.sha256((salt + "\0" + sequence).encode()).digest()
    return int.from_bytes(value[:8], "big") % folds


def event_order(event):
    value = "%s\0%d" % (event["sequence"], event["trigger_frame"])
    return hashlib.sha256(value.encode()).hexdigest()


def load_schedule(paths, horizon):
    rows_by_sequence = defaultdict(list)
    events = []
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        if (trace.get("complete") is not True or
                trace.get("ground_truth_used_after_initialization") is not False or
                trace.get("metric_computed") is not False):
            raise ValueError("protected schedule is not sealed GT-free data")
        for row in trace["rows"]:
            rows_by_sequence[row["sequence"]].append(row)
        del trace
    for sequence, rows in rows_by_sequence.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("protected sequence rows are incomplete")
        for row in rows:
            shadow = row.get("risk_recovery_shadow")
            frame = int(row["frame_index"])
            if (isinstance(shadow, dict) and shadow.get("event_started") and
                    frame + horizon <= len(rows)):
                events.append({
                    "sequence": sequence,
                    "trigger_frame": frame,
                    "event_id": int(shadow["event_id"]),
                    "trigger_reasons": list(shadow.get("trigger_reasons", [])),
                })
    events.sort(key=event_order)
    return rows_by_sequence, events


def valid_window(gt, trigger, horizon):
    if trigger < 1 or trigger + horizon > len(gt):
        return False
    value = gt[trigger:trigger + horizon]
    return bool(np.isfinite(value).all() and (value[:, 2:] > 0.0).all())


class SequenceContext:
    pass


def build_context(sequence, dataset_root, rows, network, preprocessor,
                  keep_rate, language_manifest, clip_model):
    root = dataset_root / sequence
    colors, depths = frames_for(root)
    gt = read_gt(root / "groundtruth.txt")
    if len(colors) != len(rows) or len(depths) != len(rows):
        raise ValueError("sequence/trace frame length mismatch: %s" % sequence)
    if len(gt) < len(rows):
        raise ValueError("ground truth is shorter than frames: %s" % sequence)
    gt_tail_rows_ignored = len(gt) - len(rows)
    gt = gt[:len(rows)]
    initial_bbox = read_initial_bbox(root / "groundtruth.txt")
    initial_image, _ = read_rgbd(colors[0], depths[0])
    template_patch, _, _ = sample_target(
        initial_image, initial_bbox, 2.0, output_sz=128)
    template = preprocessor.process(template_patch).detach()
    templates = [template, template]
    neutral_patch, _, _ = sample_target(
        initial_image, initial_bbox, 4.0, output_sz=256)
    neutral = preprocessor.process(neutral_patch).detach()
    anchor_output = network_forward(
        network, templates, neutral, None, keep_rate)
    context = SequenceContext()
    context.colors = colors
    context.depths = depths
    context.gt = gt
    context.rows = rows
    context.templates = templates
    context.anchor_rgb = template_bank(
        anchor_output["candidate_features"]["template_rgb_tokens"])
    context.anchor_depth = template_bank(
        anchor_output["candidate_features"]["template_depth_tokens"])
    encoder = ClipCandidateEncoder(
        clip_model, initial_image, initial_bbox,
        language_for(language_manifest, sequence))
    context.anchor_text = encoder.text_feature.detach()
    context.gt_tail_rows_ignored = gt_tail_rows_ignored
    return context


def classify_action(action, protected):
    candidate = action["ious"]
    delta = float(np.mean(candidate) - np.mean(protected))
    new_low = any(c <= 0.1 and p > 0.1
                  for c, p in zip(candidate, protected))
    if delta <= -0.25 or new_low:
        label = "catastrophic"
    elif delta >= 0.05:
        label = "beneficial"
    else:
        label = "neutral"
    return label, delta, new_low


def select_action(actions, policy):
    ranked = sorted(
        actions,
        key=lambda row: (float(row["hazard_probability"]),
                         -float(row["refined_response"]), row["name"]))
    best = ranked[0]
    margin = (float(ranked[1]["hazard_probability"]) -
              float(best["hazard_probability"]))
    eligible = bool(
        float(best["hazard_probability"]) <=
        float(policy["eligible_hazard_max"]) and
        margin >= float(policy["required_hazard_margin_to_runner_up"]) and
        float(best["refined_response"]) >=
        float(policy["required_refined_response_min"]))
    return (best if eligible else None), margin


def validate_binding(spec_path, spec, binding_path, output):
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    clean = not subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        text=True).strip()
    expected = {
        "spec_sha256": sha256_file(spec_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "repository_commit": commit,
        "output": str(output),
    }
    base_commit = spec["repository"]["commit"]
    ancestor = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
         base_commit, commit], check=False).returncode == 0
    if (binding.get("complete") is not True or
            binding.get("created_before_training") is not True or
            binding.get("pilot_training_authorized") is not True or
            any(binding.get(key) != value for key, value in expected.items()) or
            not ancestor or not clean):
        raise ValueError("pilot binding, repository, or output is invalid")
    return binding


def main():
    args = parse_args()
    started = time.time()
    for name in ("spec", "binding", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if (spec.get("complete") is not True or
            spec.get("created_before_training") is not True or
            spec["authorization"]["pilot_runner_implementation"] is not True or
            spec["authorization"]["pilot_training_before_runner_binding_amendment"] is not False):
        raise ValueError("invalid pilot specification")
    binding = validate_binding(args.spec, spec, args.binding, args.output)

    seed = int(spec["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Path(spec["model"]["config"]["path"])
    checkpoint = Path(spec["model"]["checkpoint"]["path"])
    clip_model = Path(spec["model"]["clip"]["path"])
    language_manifest = Path(spec["inputs"]["language_manifest"]["path"])
    for record, path in ((spec["model"]["config"], config),
                         (spec["model"]["checkpoint"], checkpoint),
                         (spec["model"]["clip"], clip_model),
                         (spec["inputs"]["language_manifest"],
                          language_manifest)):
        if sha256_file(path) != record["sha256"]:
            raise ValueError("frozen input hash mismatch")
    trace_paths = [Path(row["path"])
                   for row in spec["data"]["protected_trace_shards"]]
    for row, path in zip(spec["data"]["protected_trace_shards"], trace_paths):
        if sha256_file(path) != row["sha256"]:
            raise ValueError("protected trace hash mismatch")

    update_config_from_file(str(config))
    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("official checkpoint strict load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    head = LanguageAnchoredDenseAssociation().cuda().train()
    if sum(value.numel() for value in head.parameters()) != int(
            spec["model"]["parameter_count"]):
        raise RuntimeError("association parameter count drifted")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]))
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    horizon = int(spec["model"]["horizon"])
    rows_by_sequence, events = load_schedule(trace_paths, horizon)
    folds = int(spec["data"]["outer_fold_count"])
    eval_fold = int(spec["data"]["evaluation_outer_fold"])
    train_events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", folds) != eval_fold]
    eval_events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", folds) == eval_fold]
    train_events = train_events[:int(spec["data"]["training_event_limit"])]
    eval_events = eval_events[:int(spec["data"]["evaluation_event_limit"])]
    if ({event["sequence"] for event in train_events} &
            {event["sequence"] for event in eval_events}):
        raise RuntimeError("pilot sequence split leaked")

    dataset_root = Path(spec["data"]["dataset_root"])
    contexts = {}
    alignment_audits = []
    unavailable = []
    train_traces = []
    torch.cuda.reset_peak_memory_stats()

    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language_manifest, clip_model)
            if contexts[sequence].gt_tail_rows_ignored:
                alignment_audits.append({
                    "sequence": sequence,
                    "gt_tail_rows_ignored":
                        contexts[sequence].gt_tail_rows_ignored,
                    "frame_and_trace_rows": len(contexts[sequence].rows),
                })
        return contexts[sequence]

    for epoch in range(int(spec["training"]["epochs"])):
        for event in train_events:
            item = context(event["sequence"])
            frame = int(event["trigger_frame"])
            if not valid_window(item.gt, frame, horizon):
                unavailable.append({"role": "train", "epoch": epoch, **event})
                continue
            trace = one_rollout(
                network, head, optimizer, preprocessor, item.templates,
                item.anchor_rgb, item.anchor_depth, item.anchor_text,
                item.colors, item.depths, item.gt, item.rows, frame, horizon,
                keep_rate, window, train=True)
            train_traces.append({
                "epoch": epoch, **event,
                **{key: trace[key] for key in
                   ("loss", "dense_loss", "rank_loss", "survival_loss",
                    "hazard_loss", "gradient_norm")},
            })
    if not train_traces or any(not math.isfinite(row["loss"])
                               for row in train_traces):
        raise RuntimeError("pilot produced no finite training updates")

    before_eval = {name: value.detach().clone()
                   for name, value in head.named_parameters()}
    head.eval()
    evaluations = []
    for event in eval_events:
        item = context(event["sequence"])
        frame = int(event["trigger_frame"])
        if not valid_window(item.gt, frame, horizon):
            unavailable.append({"role": "evaluation", **event})
            continue
        trace = one_rollout(
            network, head, None, preprocessor, item.templates,
            item.anchor_rgb, item.anchor_depth, item.anchor_text,
            item.colors, item.depths, item.gt, item.rows, frame, horizon,
            keep_rate, window, train=False)
        selected, hazard_margin = select_action(
            trace["actions"], spec["frozen_action_policy"])
        row = {
            **event,
            "outer_fold": eval_fold,
            "protected_ious": trace["protected_ious"],
            "actions": trace["actions"],
            "selected": None,
            "hazard_margin": hazard_margin,
            "label": "abstain",
            "mean_iou_delta": 0.0,
            "new_low_overlap": False,
        }
        if selected is not None:
            label, delta, new_low = classify_action(
                selected, trace["protected_ious"])
            row.update({
                "selected": selected["name"], "label": label,
                "mean_iou_delta": delta, "new_low_overlap": new_low,
            })
        evaluations.append(row)
    if any(not torch.equal(before_eval[name], value.detach())
           for name, value in head.named_parameters()):
        raise RuntimeError("evaluation mutated association parameters")

    selected = [row for row in evaluations if row["selected"] is not None]
    beneficial = [row for row in selected if row["label"] == "beneficial"]
    catastrophic = [row for row in selected
                    if row["label"] == "catastrophic"]
    gate = {
        "protected_trace_mutations": 0,
        "selected_actions": len(selected),
        "beneficial_actions": len(beneficial),
        "beneficial_sequences": len({row["sequence"] for row in beneficial}),
        "catastrophic_actions": len(catastrophic),
    }
    conditions = {
        "protected_trace_mutations_max": gate["protected_trace_mutations"] <= 0,
        "selected_actions_min": gate["selected_actions"] >= 2,
        "beneficial_actions_min": gate["beneficial_actions"] >= 2,
        "beneficial_sequences_min": gate["beneficial_sequences"] >= 2,
        "catastrophic_actions_max": gate["catastrophic_actions"] <= 0,
    }
    passed = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m3-recursive-pilot-result/v1",
        "complete": True,
        "accepted": passed,
        "decision": ("full_nested_oof_spec_only_authorized" if passed else
                     "stop_pilot_no_oof_no_replay_no_public_benchmark"),
        "repository_commit": binding["repository_commit"],
        "spec_sha256": binding["spec_sha256"],
        "binding_sha256": sha256_file(args.binding),
        "runner_sha256": binding["runner_sha256"],
        "seed": seed,
        "epochs": int(spec["training"]["epochs"]),
        "horizon": horizon,
        "scheduled_train_events": len(train_events),
        "training_updates": len(train_traces),
        "scheduled_evaluation_events": len(eval_events),
        "evaluated_events": len(evaluations),
        "unavailable_records": len(unavailable),
        "alignment_audits": alignment_audits,
        "train_sequences": len({event["sequence"] for event in train_events}),
        "evaluation_sequences": len({event["sequence"] for event in eval_events}),
        "sequence_overlap": 0,
        "gate": gate,
        "conditions": conditions,
        "loss_first": train_traces[0]["loss"],
        "loss_last": train_traces[-1]["loss"],
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started,
        "head_checkpoint_only": True,
        "tracking_checkpoint_written": False,
        "protected_commit": False,
        "future_text_used": False,
        "qwen_used": False,
        "depthtrack_test_run": False,
        "cdtb_run": False,
        "vot_low22_run": False,
        "vot_full127_run": False,
        "automatic_next_stage": False,
    }
    args.output.mkdir(parents=True)
    atomic_json(args.output / "result.json", result)
    atomic_jsonl_gz(args.output / "evaluation_events.jsonl.gz", evaluations)
    atomic_jsonl_gz(args.output / "training_trace.jsonl.gz", train_traces)
    atomic_json(args.output / "unavailable.json", unavailable)
    atomic_torch(args.output / "association_head_only.pt", {
        "schema": "sttrack-lachtt-association-head-only/v1",
        "repository_commit": binding["repository_commit"],
        "spec_sha256": binding["spec_sha256"],
        "state_dict": {name: value.detach().cpu()
                       for name, value in head.state_dict().items()},
    })
    manifest = {
        "schema": "sttrack-lachtt-m3-recursive-pilot-manifest/v1",
        "complete": True,
        "result": {"path": str(args.output / "result.json"),
                   "sha256": sha256_file(args.output / "result.json")},
        "evaluation": {"path": str(args.output / "evaluation_events.jsonl.gz"),
                       "sha256": sha256_file(args.output / "evaluation_events.jsonl.gz")},
        "training_trace": {"path": str(args.output / "training_trace.jsonl.gz"),
                           "sha256": sha256_file(args.output / "training_trace.jsonl.gz")},
        "head": {"path": str(args.output / "association_head_only.pt"),
                 "sha256": sha256_file(args.output / "association_head_only.pt")},
        "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
        "binding": {"path": str(args.binding),
                    "sha256": sha256_file(args.binding)},
    }
    atomic_json(args.output / "manifest.json", manifest)
    for path in args.output.iterdir():
        path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
