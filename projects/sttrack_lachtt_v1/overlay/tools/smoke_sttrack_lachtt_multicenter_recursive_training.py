#!/usr/bin/env python3
"""Train-only multi-center H4 predicted-crop association smoke."""

import argparse
import copy
import json
import math
from pathlib import Path
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
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.tracker.sttrack_lachtt_observation import (
    ClipCandidateEncoder,
    RecursiveBranch,
    bbox_iou,
    decode_nms_candidates,
    split_query_state,
    stack_query_states,
)
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.heapmap_utils import generate_heatmap
from tools.run_sttrack_lachtt_train152_collection import (
    EXPECTED_CHECKPOINT_SHA256,
    SOURCE_NAMES,
    event_priors,
    read_initial_bbox,
    repeated_templates,
)
from tools.smoke_sttrack_lachtt_recursive_training import (
    atomic_json,
    depth_validity,
    frames_for,
    language_for,
    map_prediction,
    read_gt,
    read_rgbd,
    sha256_file,
    template_bank,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--language-manifest", required=True, type=Path)
    parser.add_argument("--clip-model", required=True, type=Path)
    parser.add_argument("--protected-trace", required=True, type=Path)
    parser.add_argument("--sequence", default="egg_indoor")
    parser.add_argument("--trigger-frame", type=int, default=39)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def trace_rows(path, sequence):
    trace = json.loads(path.read_text(encoding="utf-8"))
    if (trace.get("complete") is not True or
            trace.get("ground_truth_used_after_initialization") is not False or
            trace.get("metric_computed") is not False):
        raise ValueError("protected trace is not sealed GT-free evidence")
    rows = [row for row in trace["rows"] if row["sequence"] == sequence]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows or [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("protected rows are not a complete sequence")
    return rows


def batched_search(preprocessor, image, raw_depth, priors, factor, size):
    tensors, resize_factors, validities = [], [], []
    for prior in priors:
        patch, resize, _ = sample_target(image, prior, factor, output_sz=size)
        tensors.append(preprocessor.process(patch))
        resize_factors.append(float(resize))
        validities.append(depth_validity(raw_depth, prior, factor, size, 16))
    return (torch.cat(tensors, dim=0), resize_factors,
            torch.cat(validities, dim=0))


def target_maps_and_losses(logits, gt_box, priors, resize_factors):
    batch, _, side, _ = logits.shape
    targets, inside, centers = [], [], []
    for index in range(batch):
        transformed = transform_image_to_crop(
            torch.tensor(gt_box, dtype=torch.float32, device="cuda"),
            torch.tensor(priors[index], dtype=torch.float32, device="cuda"),
            resize_factors[index],
            torch.tensor([256.0, 256.0], device="cuda"), normalize=True)
        center = transformed[:2] + 0.5 * transformed[2:]
        valid = bool((center >= 0.0).all().item() and
                     (center <= 1.0).all().item() and
                     (transformed[2:] > 0.0).all().item())
        if valid:
            target = generate_heatmap([transformed[None]], 256, 16)[0][0]
            cell = torch.round(center * side).long().clamp(0, side - 1)
            centers.append((int(cell[0].item()), int(cell[1].item())))
        else:
            target = torch.zeros(side, side, device="cuda")
            centers.append(None)
        targets.append(target)
        inside.append(valid)
    target = torch.stack(targets, dim=0).unsqueeze(1)
    probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weights = torch.pow(1.0 - target, 4.0) * negative
    positive_loss = -(
        torch.log(probability) * torch.pow(1.0 - probability, 2.0) *
        positive).sum() / positive.sum().clamp_min(1.0)
    negative_loss = -(
        torch.log(1.0 - probability) * torch.pow(probability, 2.0) *
        negative_weights).sum() / negative_weights.sum().clamp_min(1.0)
    dense = positive_loss + negative_loss
    ranks, supports = [], []
    for index, center in enumerate(centers):
        if center is None:
            continue
        x, y = center
        positive = logits[index, 0, y, x]
        mask = torch.ones(side, side, dtype=torch.bool, device="cuda")
        mask[max(0, y - 1):min(side, y + 2),
             max(0, x - 1):min(side, x + 2)] = False
        negative = logits[index, 0][mask].max()
        ranks.append(F.relu(0.20 - positive + negative))
        supports.append(-torch.log(probability[index, 0, y, x]))
    zero = logits.sum() * 0.0
    rank = torch.stack(ranks).mean() if ranks else zero
    survival = torch.stack(supports).mean() if supports else zero
    return dense, rank, survival, inside


def network_forward(network, templates, search, query, keep_rate):
    with torch.no_grad():
        output = network.forward(
            template=templates, search=[search], track_query_before=query,
            keep_rate=keep_rate, return_candidate_features=True)[0]
    output["track_query_before"] = [
        value.detach().clone() for value in output["track_query_before"]]
    return output


def association_forward(head, output, anchor_rgb, anchor_depth,
                        anchor_text, validity):
    features = output["candidate_features"]
    batch = features["search_rgb_tokens"].shape[0]
    return head.forward_with_hazard(
        features["search_rgb_tokens"].detach(),
        features["search_depth_tokens"].detach(),
        features["search_fused_tokens"].detach(),
        anchor_rgb.expand(batch, -1, -1),
        anchor_depth.expand(batch, -1, -1),
        anchor_text.expand(batch, -1), validity)


def one_rollout(network, head, optimizer, preprocessor, templates,
                anchor_rgb, anchor_depth, anchor_text, colors, depths, gt,
                rows, trigger, horizon, keep_rate, window, train=True):
    public_before = json.dumps(rows[trigger:trigger + horizon],
                               sort_keys=True, allow_nan=False)
    image, raw_depth = read_rgbd(colors[trigger], depths[trigger])
    priors = event_priors(rows, trigger, image.shape, gt[0].tolist())
    search, resize, validity = batched_search(
        preprocessor, image, raw_depth, priors, 6.0, 256)
    output = network_forward(
        network, repeated_templates(templates[0], 3), search, None, keep_rate)
    logits, source_hazard = association_forward(
        head, output, anchor_rgb, anchor_depth, anchor_text, validity)
    dense, rank, survival, inside = target_maps_and_losses(
        logits, gt[trigger].tolist(), priors, resize)
    refined = head.refine_score(output["score_map"], logits, 0.20, True)
    candidates = decode_nms_candidates(
        window * refined, output["size_map"], output["offset_map"],
        priors, resize, image.shape, 256, peaks_per_prior=2,
        nms_kernel=3)
    if len(candidates) != 6:
        raise RuntimeError("age0 must produce six branches")
    age0_candidates = copy.deepcopy(candidates)
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
                         for value in source_queries[source]],
        ))
    unique_age0 = len({tuple(round(value, 4) for value in branch.bbox)
                       for branch in branches})
    if unique_age0 < 4:
        raise RuntimeError("multi-center age0 action diversity is too low")
    hazard_logits = [torch.stack([
        source_hazard[int(candidate["source_index"]), 0,
                      int(candidate["grid_row"]),
                      int(candidate["grid_column"])]
        for candidate in candidates])]
    branch_ious = [[bbox_iou(branch.bbox, gt[trigger].tolist())
                    for branch in branches]]
    frame_rows = [{
        "age": 0, "frame_index": trigger,
        "inside_crops": int(sum(inside)),
        "branch_ious": branch_ious[0],
        "branches": [branch.name for branch in branches],
    }]
    dense_losses, rank_losses, survival_losses = [dense], [rank], [survival]

    for age in range(1, horizon):
        frame = trigger + age
        image, raw_depth = read_rgbd(colors[frame], depths[frame])
        priors = [list(branch.bbox) for branch in branches]
        search, resize, validity = batched_search(
            preprocessor, image, raw_depth, priors, 4.0, 256)
        query = stack_query_states([branch.query_state for branch in branches])
        output = network_forward(
            network, repeated_templates(templates[0], len(branches)),
            search, query, keep_rate)
        logits, hazards = association_forward(
            head, output, anchor_rgb, anchor_depth, anchor_text, validity)
        dense, rank, survival, inside = target_maps_and_losses(
            logits, gt[frame].tolist(), priors, resize)
        dense_losses.append(dense)
        rank_losses.append(rank)
        survival_losses.append(survival)
        refined = head.refine_score(output["score_map"], logits, 0.20, True)
        candidates = decode_nms_candidates(
            window * refined, output["size_map"], output["offset_map"],
            priors, resize, image.shape, 256, peaks_per_prior=1,
            nms_kernel=3)
        hazard_logits.append(torch.stack([
            hazards[int(candidate["source_index"]), 0,
                    int(candidate["grid_row"]),
                    int(candidate["grid_column"])]
            for candidate in candidates]))
        query_states = split_query_state(output["track_query_before"])
        next_branches = []
        for index, (branch, candidate) in enumerate(zip(branches, candidates)):
            next_branches.append(RecursiveBranch(
                name=branch.name, source_name=branch.source_name,
                peak_rank=branch.peak_rank, bbox=list(candidate["bbox"]),
                query_state=query_states[index]))
        branches = next_branches
        ious = [bbox_iou(branch.bbox, gt[frame].tolist())
                for branch in branches]
        branch_ious.append(ious)
        frame_rows.append({
            "age": age, "frame_index": frame,
            "inside_crops": int(sum(inside)), "branch_ious": ious,
            "branches": [branch.name for branch in branches],
        })

    iou_tensor = torch.tensor(branch_ious, device="cuda")
    hazard_targets = []
    for age in range(horizon):
        hazard_targets.append(
            (iou_tensor[age:] <= 0.1).any(dim=0).float())
    hazard_target = torch.stack(hazard_targets, dim=0)
    hazard_logit = torch.stack(hazard_logits, dim=0)
    hazard_loss = F.binary_cross_entropy_with_logits(
        hazard_logit, hazard_target)
    dense_loss = torch.stack(dense_losses).mean()
    rank_loss = torch.stack(rank_losses).mean()
    survival_loss = torch.stack(survival_losses).mean()
    total = dense_loss + rank_loss + survival_loss + hazard_loss
    if train:
        if optimizer is None:
            raise ValueError("training rollout requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            head.parameters(), 5.0).item())
        optimizer.step()
    else:
        gradient_norm = 0.0
    public_after = json.dumps(rows[trigger:trigger + horizon],
                              sort_keys=True, allow_nan=False)
    if public_before != public_after:
        raise RuntimeError("protected trace was mutated")
    protected_ious = [
        bbox_iou(rows[trigger + age]["public_bbox"],
                 gt[trigger + age].tolist())
        for age in range(horizon)]
    hazard_probabilities = torch.sigmoid(hazard_logits[0]).detach().cpu()
    actions = []
    for index, (branch, candidate) in enumerate(zip(branches,
                                                     age0_candidates)):
        actions.append({
            "name": branch.name,
            "source": branch.source_name,
            "peak_rank": branch.peak_rank,
            "refined_response": float(candidate["score"]),
            "hazard_probability": float(hazard_probabilities[index].item()),
            "ious": [float(branch_ious[age][index])
                     for age in range(horizon)],
        })
    return {
        "loss": float(total.detach().item()),
        "dense_loss": float(dense_loss.detach().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "survival_loss": float(survival_loss.detach().item()),
        "hazard_loss": float(hazard_loss.detach().item()),
        "gradient_norm": gradient_norm,
        "unique_age0_boxes": unique_age0,
        "frames": frame_rows,
        "actions": actions,
        "protected_ious": protected_ious,
        "protected_trace_exact": True,
    }


def main():
    args = parse_args()
    started = time.time()
    for name in ("checkpoint", "config", "dataset_root",
                 "language_manifest", "clip_model", "protected_trace",
                 "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.horizon != 4 or args.steps != 3:
        raise ValueError("multi-center smoke is frozen to H4 and three steps")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official checkpoint hash mismatch")
    update_config_from_file(str(args.config))
    rows = trace_rows(args.protected_trace, args.sequence)
    root = args.dataset_root / args.sequence
    colors, depths = frames_for(root)
    gt = read_gt(root / "groundtruth.txt")
    if (args.trigger_frame <= 0 or
            args.trigger_frame + args.horizon > len(colors)):
        raise ValueError("trigger/horizon is outside sequence")
    window_gt = gt[args.trigger_frame:args.trigger_frame + args.horizon]
    if (not np.isfinite(window_gt).all() or
            (window_gt[:, 2:] <= 0.0).any()):
        raise ValueError("risk-event H4 ground truth is unavailable")
    shadow = rows[args.trigger_frame].get("risk_recovery_shadow")
    if not isinstance(shadow, dict) or not shadow.get("event_started"):
        raise ValueError("requested frame is not a sealed risk event")

    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(args.checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    head = LanguageAnchoredDenseAssociation().cuda().train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {name: value.detach().clone()
              for name, value in head.named_parameters()}
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    initial_bbox = read_initial_bbox(root / "groundtruth.txt")
    initial_image, _ = read_rgbd(colors[0], depths[0])
    template_patch, _, _ = sample_target(
        initial_image, initial_bbox, 2.0, output_sz=128)
    template = preprocessor.process(template_patch).detach()
    templates = [template, template]
    neutral_patch, _, _ = sample_target(
        initial_image, initial_bbox, 4.0, output_sz=256)
    neutral = preprocessor.process(neutral_patch).detach()
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    anchor_output = network_forward(
        network, templates, neutral, None, keep_rate)
    anchor_rgb = template_bank(
        anchor_output["candidate_features"]["template_rgb_tokens"])
    anchor_depth = template_bank(
        anchor_output["candidate_features"]["template_depth_tokens"])
    clip_encoder = ClipCandidateEncoder(
        args.clip_model, initial_image, initial_bbox,
        language_for(args.language_manifest, args.sequence))
    anchor_text = clip_encoder.text_feature.detach()
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    torch.cuda.reset_peak_memory_stats()
    traces = []
    for step in range(args.steps):
        trace = one_rollout(
            network, head, optimizer, preprocessor, templates,
            anchor_rgb, anchor_depth, anchor_text, colors, depths, gt,
            rows, args.trigger_frame, args.horizon, keep_rate, window)
        trace["step"] = step
        traces.append(trace)
    changed = sum(not torch.equal(before[name], value.detach())
                  for name, value in head.named_parameters())
    finite = all(math.isfinite(trace[key]) for trace in traces
                 for key in ("loss", "dense_loss", "rank_loss",
                             "survival_loss", "hazard_loss",
                             "gradient_norm"))
    if changed == 0 or not finite:
        raise RuntimeError("multi-center training update is invalid")
    result = {
        "schema": "sttrack-lachtt-multicenter-recursive-smoke/v1",
        "complete": True, "accepted": True,
        "scientific_scope": "engineering smoke only; no effectiveness claim",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "sequence": args.sequence,
        "trigger_frame": args.trigger_frame,
        "event_id": int(shadow["event_id"]),
        "horizon": args.horizon,
        "steps": args.steps,
        "branches": 6,
        "sources": list(SOURCE_NAMES),
        "association_parameters": sum(
            value.numel() for value in head.parameters()),
        "changed_parameter_tensors": changed,
        "traces": traces,
        "protected_trace_sha256": sha256_file(args.protected_trace),
        "protected_trace_exact": True,
        "candidate_committed_to_protected": False,
        "checkpoint_written": False,
        "future_text_used": False,
        "qwen_used": False,
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started,
        "association_source_sha256": sha256_file(
            REPOSITORY_ROOT /
            "lib/models/sttrack/lachtt_rollout_association.py"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "depthtrack_test_run": False,
        "cdtb_run": False,
        "vot_low22_run": False,
        "vot_full127_run": False,
        "automatic_next_stage": False,
    }
    args.output.mkdir(parents=True)
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
