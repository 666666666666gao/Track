#!/usr/bin/env python3
"""Real Train-fold engineering smoke for the M6 gate-aligned objective."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
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
from tools.run_sttrack_lachtt_setwise_pilot import run_event


def loss_configuration(spec):
    configuration = dict(spec["training"]["loss_balance"])
    reduction = configuration["selection_reduction"]
    configuration["normalize_selection_event_weight"] = not (
        reduction.startswith("single-event CE multiplied"))
    configuration["gate_aligned_margins"] = spec["training"][
        "gate_aligned_margins"]
    configuration["setwise_losses"] = spec["training"]["setwise_losses"]
    return configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-events", default=32, type=int)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    seed = int(spec["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    config = Path(spec["base_model"]["config"]["path"])
    checkpoint = Path(spec["base_model"]["checkpoint"]["path"])
    clip_model = Path(spec["inputs"]["clip"]["path"])
    language = Path(spec["inputs"]["language_manifest"]["path"])
    for path, expected in (
            (config, spec["base_model"]["config"]["sha256"]),
            (checkpoint, spec["base_model"]["checkpoint"]["sha256"]),
            (clip_model, spec["inputs"]["clip"]["sha256"]),
            (language, spec["inputs"]["language_manifest"]["sha256"])):
        if sha256_file(path) != expected:
            raise ValueError("frozen input hash mismatch: %s" % path)
    trace_paths = [Path(row["path"])
                   for row in spec["data"]["protected_trace_shards"]]
    for path, row in zip(trace_paths,
                         spec["data"]["protected_trace_shards"]):
        if sha256_file(path) != row["sha256"]:
            raise ValueError("protected trace hash mismatch")
    update_config_from_file(str(config))
    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("tracking checkpoint strict load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    dense = LanguageAnchoredDenseAssociation().cuda().train()
    setwise = SetwiseCandidateAssociation().cuda().train()
    optimizer = torch.optim.AdamW(
        list(dense.parameters()) + list(setwise.parameters()),
        lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]))
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    rows_by_sequence, events = load_schedule(trace_paths, 4)
    fold_count = int(spec["data"]["outer_fold_count"])
    heldout = int(spec["data"]["evaluation_outer_fold"])
    events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", fold_count) != heldout]
    events = events[:int(spec["data"]["training_event_limit"])]
    contexts = {}
    dataset_root = Path(spec["data"]["dataset_root"])

    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language, clip_model)
        return contexts[sequence]

    identity = torch.arange(6, device="cuda")
    configuration = loss_configuration(spec)
    selected_event = None
    inspected = 0
    for event in events[:args.maximum_events]:
        item = context(event["sequence"])
        frame = event["trigger_frame"]
        if not valid_window(item.gt, frame, 4):
            continue
        inspected += 1
        with torch.no_grad():
            trace = run_event(
                network, dense, setwise, None, preprocessor, item, frame, 4,
                keep_rate, window, identity, False, configuration)
        if any(action["actual_beneficial"] for action in trace["actions"]):
            selected_event = event
            break
    if selected_event is None:
        raise RuntimeError("no beneficial Train-fold event in smoke window")
    before = {
        "dense": {key: value.detach().clone()
                  for key, value in dense.state_dict().items()},
        "setwise": {key: value.detach().clone()
                    for key, value in setwise.state_dict().items()},
    }
    item = context(selected_event["sequence"])
    trace = run_event(
        network, dense, setwise, optimizer, preprocessor, item,
        selected_event["trigger_frame"], 4, keep_rate, window, identity, True,
        configuration)
    changed = sum(
        not torch.equal(before[group][key], module.state_dict()[key])
        for group, module in (("dense", dense), ("setwise", setwise))
        for key in before[group])
    numeric = {key: value for key, value in trace.items()
               if key.startswith("setwise_") or key in
               ("loss", "dense_total", "gradient_norm")}
    if (changed == 0 or
            any(not math.isfinite(float(value)) for value in numeric.values()) or
            not any(action["actual_beneficial"]
                    for action in trace["actions"]) or
            trace["setwise_beneficial_gate"] <= 0.0 or
            trace["setwise_catastrophic_gate"] <= 0.0):
        raise RuntimeError("M6 real-event engineering update failed")
    result = {
        "schema": "sttrack-lachtt-m6-gate-aligned-engineering-smoke/v1",
        "complete": True,
        "accepted": True,
        "scientific_scope": "single beneficial DepthTrack Train event finite forward/backward only",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "spec_sha256": sha256_file(args.spec),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "inspected_valid_events": inspected,
        "event": selected_event,
        "outer_fold": stable_fold(
            selected_event["sequence"], "sttrack-lachtt-outer-v1",
            fold_count),
        "beneficial_candidates": sum(
            action["actual_beneficial"] for action in trace["actions"]),
        "catastrophic_candidates": sum(
            action["actual_catastrophic"] for action in trace["actions"]),
        "metrics": numeric,
        "changed_parameter_tensors": changed,
        "elapsed_seconds": time.time() - started,
        "checkpoint_written": False,
        "tracking_checkpoint_written": False,
        "qwen_used": False,
        "vot_run": False,
        "automatic_next_stage": False,
    }
    atomic_json(args.output / "result.json", result)
    (args.output / "result.json").chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
