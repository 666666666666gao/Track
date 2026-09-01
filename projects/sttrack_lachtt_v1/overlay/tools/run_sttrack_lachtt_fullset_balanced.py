#!/usr/bin/env python3
"""Run the pre-registered M7 full-event balanced Train-only experiment."""

import argparse
import copy
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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
    setwise_association_loss,
)
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.tracker.sttrack_lachtt_observation import (
    RecursiveBranch,
    bbox_iou,
    decode_nms_candidates,
    relative_geometry,
    split_query_state,
    stack_query_states,
)
from lib.test.utils.hann import hann2d
from tools.run_sttrack_lachtt_recursive_pilot import (
    atomic_json,
    atomic_jsonl_gz,
    atomic_torch,
    build_context,
    load_schedule,
    sha256_file,
    stable_fold,
    valid_window,
)
from tools.run_sttrack_lachtt_train152_collection import (
    SOURCE_NAMES,
    event_priors,
    repeated_templates,
)
from tools.smoke_sttrack_lachtt_multicenter_recursive_training import (
    batched_search,
    network_forward,
    target_maps_and_losses,
)
from tools.smoke_sttrack_lachtt_recursive_training import read_rgbd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def stable_event_permutation(seed, epoch, sequence, frame, candidates,
                             device):
    payload = "%d\0%d\0%s\0%d" % (seed, epoch, sequence, frame)
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],
                           "big")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value)
    return torch.randperm(candidates, generator=generator).to(device)


def load_event_classes(path):
    labels = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["sequence"], int(row["event_id"]),
                   int(row["trigger_frame"]))
            labels[key].append(row["label"])
    classes = {}
    for key, values in labels.items():
        if "beneficial" in values:
            classes[key] = "beneficial"
        elif "catastrophic" in values:
            classes[key] = "catastrophic"
        elif "neutral" in values:
            classes[key] = "neutral"
        else:
            classes[key] = "unavailable"
    return classes


def attach_event_classes(events, classes):
    attached = []
    for event in events:
        key = (event["sequence"], int(event["event_id"]),
               int(event["trigger_frame"]))
        if key not in classes:
            raise ValueError("event missing from frozen Gate-A labels: %r" %
                             (key,))
        attached.append({**event, "event_class": classes[key]})
    return attached


def balanced_event_batches(events, composition, steps, seed, epoch):
    pools = defaultdict(list)
    for event in events:
        pools[event["event_class"]].append(event)
    expected_classes = set(composition)
    if set(pools) != expected_classes or any(not pools[name]
                                              for name in expected_classes):
        raise ValueError("balanced event pools are incomplete")
    offsets = {name: 0 for name in expected_classes}
    for name in sorted(expected_classes):
        payload = "%d\0%d\0%s" % (seed, epoch, name)
        class_seed = int.from_bytes(
            hashlib.sha256(payload.encode()).digest()[:8], "big")
        random.Random(class_seed).shuffle(pools[name])
    batches = []
    for step in range(steps):
        batch = []
        for name in sorted(expected_classes):
            pool = pools[name]
            for _ in range(int(composition[name])):
                batch.append(pool[offsets[name] % len(pool)])
                offsets[name] += 1
        payload = "%d\0%d\0%d\0within" % (seed, epoch, step)
        batch_seed = int.from_bytes(
            hashlib.sha256(payload.encode()).digest()[:8], "big")
        random.Random(batch_seed).shuffle(batch)
        batches.append(batch)
    return batches


def association_all(head, output, anchor_rgb, anchor_depth, anchor_text,
                    validity):
    features = output["candidate_features"]
    batch = features["search_rgb_tokens"].shape[0]
    return head.forward_all(
        features["search_rgb_tokens"].detach(),
        features["search_depth_tokens"].detach(),
        features["search_fused_tokens"].detach(),
        anchor_rgb.expand(batch, -1, -1),
        anchor_depth.expand(batch, -1, -1),
        anchor_text.expand(batch, -1), validity)


def candidate_observations(hidden, hazards, candidates, priors):
    embeddings, scalars = [], []
    for candidate in candidates:
        source = int(candidate["source_index"])
        row = int(candidate["grid_row"])
        column = int(candidate["grid_column"])
        side = int(round(math.sqrt(hidden.shape[1])))
        embeddings.append(hidden[source, row * side + column])
        hazard = torch.sigmoid(hazards[source, 0, row, column])
        geometry = relative_geometry(candidate["bbox"], priors[source])
        fixed = torch.tensor([
            float(candidate["score"]), float(candidate["margin"]),
            float(candidate["entropy"]), *geometry,
        ], device=hidden.device, dtype=torch.float32)
        scalars.append(torch.cat((fixed[:3], hazard[None], fixed[3:])))
    return torch.stack(embeddings), torch.stack(scalars)


def outcome_targets(branch_ious, protected_ious, device):
    gains, beneficial, catastrophic = [], [], []
    for index in range(len(branch_ious[0])):
        candidate = [branch_ious[age][index]
                     for age in range(len(branch_ious))]
        delta = float(np.mean(candidate) - np.mean(protected_ious))
        new_low = any(value <= 0.1 and protected > 0.1
                      for value, protected in zip(candidate, protected_ious))
        gains.append(delta)
        catastrophic.append(delta <= -0.25 or new_low)
        beneficial.append(delta >= 0.05 and not new_low)
    return (torch.tensor([gains], device=device, dtype=torch.float32),
            torch.tensor([beneficial], device=device, dtype=torch.bool),
            torch.tensor([catastrophic], device=device, dtype=torch.bool))


def run_event(network, dense, setwise, optimizer, preprocessor, context,
              trigger, horizon, keep_rate, window, permutation, train,
              loss_balance=None, loss_divisor=1.0, zero_grad=True,
              optimizer_step=True):
    rows = context.rows
    gt = context.gt
    public_before = json.dumps(rows[trigger:trigger + horizon],
                               sort_keys=True, allow_nan=False)
    image, raw_depth = read_rgbd(
        context.colors[trigger], context.depths[trigger])
    priors = event_priors(rows, trigger, image.shape, gt[0].tolist())
    search, resize, validity = batched_search(
        preprocessor, image, raw_depth, priors, 6.0, 256)
    output = network_forward(
        network, repeated_templates(context.templates[0], 3), search, None,
        keep_rate)
    logits, hazards, hidden = association_all(
        dense, output, context.anchor_rgb, context.anchor_depth,
        context.anchor_text, validity)
    dense_loss, rank_loss, survival_loss, _ = target_maps_and_losses(
        logits, gt[trigger].tolist(), priors, resize)
    refined = dense.refine_score(output["score_map"], logits, 0.20, True)
    candidates = decode_nms_candidates(
        window * refined, output["size_map"], output["offset_map"], priors,
        resize, image.shape, 256, peaks_per_prior=2, nms_kernel=3)
    age0_candidates = copy.deepcopy(candidates)
    features, scalars = candidate_observations(
        hidden, hazards, candidates, priors)
    feature_ages, scalar_ages = [features], [scalars]
    source_queries = split_query_state(output["track_query_before"])
    branches = []
    for candidate in candidates:
        source = int(candidate["source_index"])
        branches.append(RecursiveBranch(
            name="%s_peak%d" % (SOURCE_NAMES[source],
                                  int(candidate["peak_rank"])),
            source_name=SOURCE_NAMES[source],
            peak_rank=int(candidate["peak_rank"]),
            bbox=list(candidate["bbox"]),
            query_state=[value.detach().clone()
                         for value in source_queries[source]]))
    branch_ious = [[bbox_iou(branch.bbox, gt[trigger].tolist())
                    for branch in branches]]
    hazard_logits = [torch.stack([
        hazards[int(candidate["source_index"]), 0,
                int(candidate["grid_row"]), int(candidate["grid_column"])]
        for candidate in candidates])]
    dense_losses, rank_losses, survival_losses = (
        [dense_loss], [rank_loss], [survival_loss])

    for age in range(1, horizon):
        frame = trigger + age
        image, raw_depth = read_rgbd(
            context.colors[frame], context.depths[frame])
        priors = [list(branch.bbox) for branch in branches]
        search, resize, validity = batched_search(
            preprocessor, image, raw_depth, priors, 4.0, 256)
        query = stack_query_states([branch.query_state for branch in branches])
        output = network_forward(
            network, repeated_templates(context.templates[0], len(branches)),
            search, query, keep_rate)
        logits, hazards, hidden = association_all(
            dense, output, context.anchor_rgb, context.anchor_depth,
            context.anchor_text, validity)
        dl, rl, sl, _ = target_maps_and_losses(
            logits, gt[frame].tolist(), priors, resize)
        dense_losses.append(dl); rank_losses.append(rl); survival_losses.append(sl)
        refined = dense.refine_score(output["score_map"], logits, 0.20, True)
        candidates = decode_nms_candidates(
            window * refined, output["size_map"], output["offset_map"],
            priors, resize, image.shape, 256, peaks_per_prior=1,
            nms_kernel=3)
        features, scalars = candidate_observations(
            hidden, hazards, candidates, priors)
        feature_ages.append(features); scalar_ages.append(scalars)
        hazard_logits.append(torch.stack([
            hazards[int(candidate["source_index"]), 0,
                    int(candidate["grid_row"]), int(candidate["grid_column"])]
            for candidate in candidates]))
        query_states = split_query_state(output["track_query_before"])
        branches = [RecursiveBranch(
            name=branch.name, source_name=branch.source_name,
            peak_rank=branch.peak_rank, bbox=list(candidate["bbox"]),
            query_state=query_states[index])
            for index, (branch, candidate) in enumerate(zip(branches,
                                                             candidates))]
        branch_ious.append([bbox_iou(branch.bbox, gt[frame].tolist())
                            for branch in branches])

    protected_ious = [bbox_iou(rows[trigger + age]["public_bbox"],
                               gt[trigger + age].tolist())
                      for age in range(horizon)]
    iou_tensor = torch.tensor(branch_ious, device=logits.device)
    hazard_target = torch.stack([
        (iou_tensor[age:] <= 0.1).any(dim=0).float()
        for age in range(horizon)])
    dense_hazard_loss = F.binary_cross_entropy_with_logits(
        torch.stack(hazard_logits), hazard_target)
    dense_total = (torch.stack(dense_losses).mean() +
                   torch.stack(rank_losses).mean() +
                   torch.stack(survival_losses).mean() + dense_hazard_loss)
    trajectory_features = torch.stack(feature_ages)[None, :, permutation]
    trajectory_scalars = torch.stack(scalar_ages)[None, :, permutation]
    valid = torch.ones(1, len(branches), dtype=torch.bool,
                       device=logits.device)
    gain_target, beneficial_target, catastrophic_target = outcome_targets(
        branch_ious, protected_ious, logits.device)
    gain_target = gain_target[:, permutation]
    beneficial_target = beneficial_target[:, permutation]
    catastrophic_target = catastrophic_target[:, permutation]
    set_outputs = setwise(trajectory_features, trajectory_scalars, valid)
    loss_balance = loss_balance or {}
    gate_aligned = loss_balance.get("gate_aligned_margins") or {}
    gate_aligned = ({
        "beneficial_positive_logit_floor": float(gate_aligned[
            "beneficial_positive_logit_floor"]),
        "beneficial_negative_logit_ceiling": float(gate_aligned[
            "beneficial_negative_logit_ceiling"]),
        "catastrophic_positive_logit_floor": float(gate_aligned[
            "catastrophic_positive_logit_floor"]),
        "catastrophic_negative_logit_ceiling": float(gate_aligned[
            "catastrophic_negative_logit_ceiling"]),
        "beneficial_gain_floor": float(gate_aligned[
            "beneficial_gain_floor"]),
    } if gate_aligned else None)
    setwise_weights = loss_balance.get("setwise_losses", {})
    set_losses = setwise_association_loss(
        set_outputs, gain_target, beneficial_target, catastrophic_target,
        valid, rank_margin=0.10,
        selection_beneficial_event_weight=float(loss_balance.get(
            "selection_beneficial_event_weight", 1.0)),
        beneficial_bce_positive_weight=float(loss_balance.get(
            "beneficial_bce_positive_weight", 1.0)),
        catastrophic_bce_positive_weight=float(loss_balance.get(
            "catastrophic_bce_positive_weight", 1.0)),
        gain_beneficial_candidate_weight=float(loss_balance.get(
            "gain_beneficial_candidate_weight", 1.0)),
        pairwise_beneficial_candidate_weight=float(loss_balance.get(
            "pairwise_beneficial_candidate_weight", 1.0)),
        normalize_selection_event_weight=bool(loss_balance.get(
            "normalize_selection_event_weight", True)),
        gate_aligned_margins=gate_aligned,
        beneficial_gate_weight=float(setwise_weights.get(
            "beneficial_gate_margin", 0.0)),
        catastrophic_gate_weight=float(setwise_weights.get(
            "catastrophic_gate_margin", 0.0)),
        gain_gate_weight=float(setwise_weights.get(
            "gain_gate_margin", 0.0)))
    total = set_losses["total"] + 0.25 * dense_total
    if train:
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)
        (total / float(loss_divisor)).backward()
        if optimizer_step:
            gradient = float(torch.nn.utils.clip_grad_norm_(
                list(dense.parameters()) + list(setwise.parameters()),
                5.0).item())
            optimizer.step()
        else:
            gradient = 0.0
    else:
        gradient = 0.0
    inverse = torch.argsort(permutation)
    probabilities = F.softmax(set_outputs["selection_logits"], dim=1)
    selection_candidates = probabilities[:, :-1][:, inverse][0]
    actions = []
    for index, (branch, candidate) in enumerate(zip(branches,
                                                     age0_candidates)):
        permuted_index = int(inverse[index].item())
        actions.append({
            "name": branch.name,
            "refined_response": float(candidate["score"]),
            "selection_probability": float(selection_candidates[index].item()),
            "beneficial_probability": float(torch.sigmoid(
                set_outputs["beneficial_logits"][0, permuted_index]).item()),
            "catastrophic_probability": float(torch.sigmoid(
                set_outputs["catastrophic_logits"][0, permuted_index]).item()),
            "predicted_gain": float(
                set_outputs["gain"][0, permuted_index].item()),
            "actual_gain": float(gain_target[0, permuted_index].item()),
            "actual_beneficial": bool(
                beneficial_target[0, permuted_index].item()),
            "actual_catastrophic": bool(
                catastrophic_target[0, permuted_index].item()),
            "ious": [float(branch_ious[age][index])
                     for age in range(horizon)],
        })
    public_after = json.dumps(rows[trigger:trigger + horizon],
                              sort_keys=True, allow_nan=False)
    if public_before != public_after:
        raise RuntimeError("protected trace mutated")
    return {
        "loss": float(total.detach().item()),
        "dense_total": float(dense_total.detach().item()),
        **{"setwise_%s" % key: float(value.detach().item())
           for key, value in set_losses.items()},
        "gradient_norm": gradient,
        "actions": actions,
        "abstain_probability": float(probabilities[0, -1].item()),
        "protected_ious": protected_ious,
    }


def select_action(trace, policy):
    entries = [(row["selection_probability"], row["name"], row)
               for row in trace["actions"]]
    entries.append((trace["abstain_probability"], "abstain", None))
    entries.sort(key=lambda item: (-item[0], item[1]))
    best_probability, best_name, action = entries[0]
    margin = best_probability - entries[1][0]
    if (action is None or
            best_probability < policy["candidate_selection_probability_min"] or
            margin < policy["selection_probability_margin_min"] or
            action["beneficial_probability"] <
            policy["beneficial_probability_min"] or
            action["catastrophic_probability"] >
            policy["catastrophic_probability_max"] or
            action["predicted_gain"] < policy["predicted_gain_min"]):
        return None, best_name, margin
    return action, best_name, margin


def validate_binding(spec_path, spec, binding_path, output):
    binding = json.loads(binding_path.read_text())
    commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    clean = not subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        text=True).strip()
    ancestor = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor",
         spec["repository"]["base_commit"], commit], check=False).returncode == 0
    expected = {"spec_sha256": sha256_file(spec_path),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "repository_commit": commit, "output": str(output)}
    if (binding.get("complete") is not True or
            binding.get("m7_training_authorized") is not True or
            any(binding.get(key) != value for key, value in expected.items()) or
            not ancestor or not clean):
        raise ValueError("M7 binding/repository mismatch")
    return binding


def main():
    args = parse_args()
    started = time.time()
    for name in ("spec", "binding", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text())
    binding = validate_binding(args.spec, spec, args.binding, args.output)
    seed = int(spec["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = Path(spec["base_model"]["config"]["path"])
    checkpoint = Path(spec["base_model"]["checkpoint"]["path"])
    clip_model = Path(spec["inputs"]["clip"]["path"])
    language = Path(spec["inputs"]["language_manifest"]["path"])
    label_path = Path(spec["data"]["labeled_actions"]["path"])
    gate_a_path = Path(spec["data"]["gate_a_result"]["path"])
    frozen = [(config, spec["base_model"]["config"]["sha256"]),
              (checkpoint, spec["base_model"]["checkpoint"]["sha256"]),
              (clip_model, spec["inputs"]["clip"]["sha256"]),
              (language, spec["inputs"]["language_manifest"]["sha256"]),
              (label_path, spec["data"]["labeled_actions"]["sha256"]),
              (gate_a_path, spec["data"]["gate_a_result"]["sha256"])]
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
    for parameter in network.parameters(): parameter.requires_grad_(False)
    dense = LanguageAnchoredDenseAssociation().cuda().train()
    setwise = SetwiseCandidateAssociation().cuda().train()
    optimizer = torch.optim.AdamW(
        list(dense.parameters()) + list(setwise.parameters()),
        lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]))
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    horizon = int(spec["architecture"]["horizon"])
    loss_balance = dict(spec["training"].get("loss_balance", {}))
    loss_balance["normalize_selection_event_weight"] = bool(
        loss_balance.get("normalize_selection_event_weight", False))
    loss_balance["gate_aligned_margins"] = spec["training"].get(
        "gate_aligned_margins", {})
    loss_balance["setwise_losses"] = spec["training"].get(
        "setwise_losses", {})
    rows_by_sequence, events = load_schedule(traces, horizon)
    classes = load_event_classes(label_path)
    events = attach_event_classes(events, classes)
    fold_count = int(spec["data"]["outer_fold_count"])
    eval_fold = int(spec["data"]["evaluation_outer_fold"])
    train_events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", fold_count) != eval_fold
        and event["event_class"] != "unavailable"]
    eval_events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", fold_count) == eval_fold
        and event["event_class"] != "unavailable"]
    expected_train = spec["data"]["training_available_events"]
    expected_eval = spec["data"]["evaluation_available_events"]
    train_counts = Counter(event["event_class"] for event in train_events)
    eval_counts = Counter(event["event_class"] for event in eval_events)
    if (len(train_events) != int(expected_train["total"]) or
            len(eval_events) != int(expected_eval["total"]) or
            any(train_counts[name] != int(expected_train[name])
                for name in ("beneficial", "catastrophic", "neutral")) or
            any(eval_counts[name] != int(expected_eval[name])
                for name in ("beneficial", "catastrophic", "neutral"))):
        raise RuntimeError("frozen M7 event-class counts drifted")
    contexts, alignment, unavailable, training = {}, [], [], []
    dataset_root = Path(spec["data"]["dataset_root"])
    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language, clip_model)
            if contexts[sequence].gt_tail_rows_ignored:
                alignment.append({"sequence": sequence,
                                  "gt_tail_rows_ignored":
                                  contexts[sequence].gt_tail_rows_ignored})
        return contexts[sequence]
    torch.cuda.reset_peak_memory_stats()
    optimizer_updates = 0
    minibatch_size = int(spec["training"]["event_minibatch_size"])
    composition = spec["training"]["event_minibatch_composition"]
    steps_per_epoch = int(spec["training"]["steps_per_epoch"])
    for epoch in range(int(spec["training"]["epochs"])):
        batches = balanced_event_batches(
            train_events, composition, steps_per_epoch, seed, epoch)
        for step, batch in enumerate(batches):
            if len(batch) != minibatch_size:
                raise RuntimeError("M7 minibatch size drifted")
            for batch_index, event in enumerate(batch):
                item = context(event["sequence"])
                frame = event["trigger_frame"]
                if not valid_window(item.gt, frame, horizon):
                    raise RuntimeError(
                        "frozen available training event became unavailable")
                permutation = stable_event_permutation(
                    seed, epoch, event["sequence"], frame, 6,
                    torch.device("cuda"))
                trace = run_event(
                    network, dense, setwise, optimizer, preprocessor, item,
                    frame, horizon, keep_rate, window, permutation, True,
                    loss_balance, loss_divisor=minibatch_size,
                    zero_grad=(batch_index == 0),
                    optimizer_step=(batch_index == minibatch_size - 1))
                training.append({
                    "epoch": epoch, "optimizer_step": step,
                    "batch_index": batch_index, **event,
                    **{key: trace[key] for key in trace
                       if key.startswith("setwise_") or key in
                       ("loss", "dense_total", "gradient_norm")}})
            optimizer_updates += 1
    dense.eval(); setwise.eval()
    before = {"dense": {k:v.detach().clone() for k,v in dense.state_dict().items()},
              "setwise": {k:v.detach().clone() for k,v in setwise.state_dict().items()}}
    evaluations = []
    identity = torch.arange(6, device="cuda")
    for event in eval_events:
        item = context(event["sequence"]); frame = event["trigger_frame"]
        if not valid_window(item.gt, frame, horizon):
            raise RuntimeError(
                "frozen available evaluation event became unavailable")
        trace = run_event(network, dense, setwise, None, preprocessor, item,
                          frame, horizon, keep_rate, window, identity, False,
                          loss_balance)
        action, top_name, margin = select_action(
            trace, spec["evaluation_action_policy"])
        row = {**event, "actions": trace["actions"],
               "abstain_probability": trace["abstain_probability"],
               "top_name": top_name, "selection_margin": margin,
               "selected": None, "label": "abstain", "gain": 0.0}
        if action is not None:
            label = ("catastrophic" if action["actual_catastrophic"] else
                     "beneficial" if action["actual_beneficial"] else "neutral")
            row.update({"selected": action["name"], "label": label,
                        "gain": action["actual_gain"]})
        evaluations.append(row)
    if (len(training) != int(spec["training"]["event_forwards_total"]) or
            optimizer_updates != int(
                spec["training"]["optimizer_updates_total"]) or
            len(evaluations) != int(expected_eval["total"])):
        raise RuntimeError("frozen M7 execution counts drifted")
    if any(not torch.equal(before[group][key], module.state_dict()[key])
           for group, module in (("dense", dense), ("setwise", setwise))
           for key in before[group]):
        raise RuntimeError("evaluation mutated parameters")
    selected = [row for row in evaluations if row["selected"]]
    beneficial = [row for row in selected if row["label"] == "beneficial"]
    catastrophic = [row for row in selected if row["label"] == "catastrophic"]
    precision = len(beneficial) / max(1, len(selected))
    gate = {"selected_actions": len(selected),
            "beneficial_actions": len(beneficial),
            "beneficial_sequences": len(set(row["sequence"] for row in beneficial)),
            "catastrophic_actions": len(catastrophic),
            "beneficial_precision": precision,
            "protected_trace_mutations": 0}
    conditions = {
        "selected_actions_min": gate["selected_actions"] >= 5,
        "beneficial_actions_min": gate["beneficial_actions"] >= 4,
        "beneficial_sequences_min": gate["beneficial_sequences"] >= 3,
        "beneficial_precision_min": precision >= 0.95,
        "catastrophic_actions_max": len(catastrophic) == 0,
        "protected_trace_mutations_max": True}
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m7-fullset-balanced-result/v1",
        "complete": True, "accepted": accepted,
        "decision": ("complete_sequence_train_oof_spec_only_authorized"
                     if accepted else
                     "stop_m7_no_threshold_scan_no_public_benchmark"),
        "repository_commit": binding["repository_commit"],
        "spec_sha256": binding["spec_sha256"],
        "binding_sha256": sha256_file(args.binding),
        "runner_sha256": binding["runner_sha256"],
        "training_event_forwards": len(training),
        "training_optimizer_updates": optimizer_updates,
        "training_event_class_counts": dict(train_counts),
        "evaluation_event_class_counts": dict(eval_counts),
        "evaluated_events": len(evaluations),
        "train_sequences": len(set(e["sequence"] for e in train_events)),
        "evaluation_sequences": len(set(e["sequence"] for e in eval_events)),
        "sequence_overlap": len(set(e["sequence"] for e in train_events) &
                                set(e["sequence"] for e in eval_events)),
        "excluded_unavailable_events": spec["data"][
            "excluded_unavailable_events"],
        "unavailable_records": len(unavailable),
        "alignment_audits": alignment, "gate": gate,
        "conditions": conditions,
        "loss_first": training[0]["loss"], "loss_last": training[-1]["loss"],
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time()-started,
        "head_checkpoint_only": True, "tracking_checkpoint_written": False,
        "protected_commit": False, "future_text_used": False,
        "qwen_used": False, "depthtrack_test_run": False,
        "cdtb_run": False, "vot_low22_run": False,
        "vot_full127_run": False, "automatic_next_stage": False}
    args.output.mkdir(parents=True)
    atomic_json(args.output/"result.json", result)
    atomic_jsonl_gz(args.output/"training_trace.jsonl.gz", training)
    atomic_jsonl_gz(args.output/"evaluation_events.jsonl.gz", evaluations)
    atomic_json(args.output/"unavailable.json", unavailable)
    atomic_torch(args.output/"heads_only.pt", {
        "schema":"sttrack-lachtt-m7-heads-only/v1",
        "dense":{k:v.detach().cpu() for k,v in dense.state_dict().items()},
        "setwise":{k:v.detach().cpu() for k,v in setwise.state_dict().items()},
        "spec_sha256":binding["spec_sha256"]})
    manifest={"schema":"sttrack-lachtt-m7-fullset-balanced-manifest/v1",
              "complete":True}
    for name in ("result.json","training_trace.jsonl.gz",
                 "evaluation_events.jsonl.gz","unavailable.json","heads_only.pt"):
        manifest[name]={"path":str(args.output/name),
                        "sha256":sha256_file(args.output/name)}
    atomic_json(args.output/"manifest.json",manifest)
    for path in args.output.iterdir(): path.chmod(0o444)
    args.output.chmod(0o555)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
