#!/usr/bin/env python3
"""One-minibatch engineering smoke for the M7 balanced trainer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
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
from tools.run_sttrack_lachtt_fullset_balanced import (
    attach_event_classes,
    balanced_event_batches,
    load_event_classes,
    run_event,
    stable_event_permutation,
)
from tools.run_sttrack_lachtt_recursive_pilot import (
    atomic_json,
    build_context,
    load_schedule,
    sha256_file,
    stable_fold,
    valid_window,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    seed = int(spec["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = Path(spec["base_model"]["config"]["path"])
    checkpoint = Path(spec["base_model"]["checkpoint"]["path"])
    clip_model = Path(spec["inputs"]["clip"]["path"])
    language = Path(spec["inputs"]["language_manifest"]["path"])
    label_path = Path(spec["data"]["labeled_actions"]["path"])
    frozen = [
        (config, spec["base_model"]["config"]["sha256"]),
        (checkpoint, spec["base_model"]["checkpoint"]["sha256"]),
        (clip_model, spec["inputs"]["clip"]["sha256"]),
        (language, spec["inputs"]["language_manifest"]["sha256"]),
        (label_path, spec["data"]["labeled_actions"]["sha256"]),
    ]
    if any(sha256_file(path) != digest for path, digest in frozen):
        raise ValueError("frozen input hash mismatch")
    traces = [Path(row["path"])
              for row in spec["data"]["protected_trace_shards"]]
    if any(sha256_file(path) != row["sha256"]
           for path, row in zip(traces,
                                spec["data"]["protected_trace_shards"])):
        raise ValueError("protected trace hash mismatch")

    update_config_from_file(str(config))
    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("checkpoint strict load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    dense = LanguageAnchoredDenseAssociation().cuda().train()
    setwise = SetwiseCandidateAssociation().cuda().train()
    parameters = list(dense.parameters()) + list(setwise.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]))
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    horizon = int(spec["architecture"]["horizon"])
    loss_balance = dict(spec["training"]["loss_balance"])
    loss_balance["gate_aligned_margins"] = spec["training"][
        "gate_aligned_margins"]
    loss_balance["setwise_losses"] = spec["training"]["setwise_losses"]

    rows_by_sequence, events = load_schedule(traces, horizon)
    events = attach_event_classes(events, load_event_classes(label_path))
    fold_count = int(spec["data"]["outer_fold_count"])
    eval_fold = int(spec["data"]["evaluation_outer_fold"])
    train_events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", fold_count) != eval_fold
        and event["event_class"] != "unavailable"]
    batch = balanced_event_batches(
        train_events, spec["training"]["event_minibatch_composition"],
        1, seed, 0)[0]
    if Counter(event["event_class"] for event in batch) != Counter(
            spec["training"]["event_minibatch_composition"]):
        raise RuntimeError("smoke batch composition drifted")

    contexts = {}
    dataset_root = Path(spec["data"]["dataset_root"])
    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language, clip_model)
        return contexts[sequence]

    before = [value.detach().clone() for value in
              list(dense.state_dict().values()) +
              list(setwise.state_dict().values())]
    traces_out = []
    torch.cuda.reset_peak_memory_stats()
    for index, event in enumerate(batch):
        item = context(event["sequence"])
        frame = int(event["trigger_frame"])
        if not valid_window(item.gt, frame, horizon):
            raise RuntimeError("available smoke event became unavailable")
        permutation = stable_event_permutation(
            seed, 0, event["sequence"], frame, 6, torch.device("cuda"))
        trace = run_event(
            network, dense, setwise, optimizer, preprocessor, item, frame,
            horizon, keep_rate, window, permutation, True, loss_balance,
            loss_divisor=len(batch), zero_grad=(index == 0),
            optimizer_step=(index == len(batch) - 1))
        traces_out.append({
            "sequence": event["sequence"],
            "event_id": int(event["event_id"]),
            "event_class": event["event_class"],
            "loss": trace["loss"],
            "gradient_norm": trace["gradient_norm"],
        })
    after = list(dense.state_dict().values()) + list(setwise.state_dict().values())
    changed = sum(not torch.equal(left, right)
                  for left, right in zip(before, after))
    if changed <= 0 or not all(np.isfinite(row["loss"])
                               for row in traces_out):
        raise RuntimeError("M7 smoke did not produce a finite update")
    final_gradient = traces_out[-1]["gradient_norm"]
    if not np.isfinite(final_gradient) or final_gradient <= 0.0:
        raise RuntimeError("M7 smoke final gradient is invalid")

    result = {
        "schema": "sttrack-lachtt-m7-balanced-engineering-smoke/v1",
        "complete": True,
        "accepted": True,
        "spec_sha256": sha256_file(args.spec),
        "batch_size": len(batch),
        "batch_composition": dict(Counter(
            event["event_class"] for event in batch)),
        "changed_tensors": changed,
        "final_gradient_norm": final_gradient,
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "events": traces_out,
        "protected_commit": False,
        "tracking_checkpoint_written": False,
        "qwen_used": False,
        "public_benchmark_run": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
