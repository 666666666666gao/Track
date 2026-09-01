#!/usr/bin/env python3
"""Train-only H4 predicted-crop recursive association engineering smoke."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
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
    bbox_iou,
)
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import clip_box
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


EXPECTED_CHECKPOINT_SHA256 = (
    "cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--language-manifest", required=True, type=Path)
    parser.add_argument("--clip-model", required=True, type=Path)
    parser.add_argument("--sequence", default="lock01_wild")
    parser.add_argument("--anchor-frame", type=int, default=382)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def language_for(path, sequence):
    found = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("sequence_name") == sequence:
                found.append(" ".join(str(row.get("language", "")).split()))
    if len(found) != 1 or not found[0]:
        raise ValueError("sequence language is missing or duplicated")
    return found[0]


def frames_for(root):
    colors = sorted((root / "color").glob("*"))
    depths = sorted((root / "depth").glob("*"))
    if not colors or len(colors) != len(depths):
        raise ValueError("RGB/depth frames are misaligned")
    return colors, depths


def read_rgbd(color, depth):
    image = get_rgbd_frame(str(color), str(depth), dtype="rgbcolormap",
                           depth_clip=True)
    raw = cv2.imread(str(depth), -1)
    if image is None or image.shape[2] != 6 or raw is None:
        raise ValueError("invalid RGB-D frame")
    return image, raw


def read_gt(path):
    values = np.genfromtxt(str(path), delimiter=",", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("invalid ground truth")
    return values


def template_bank(value):
    if list(value.shape[:1]) != [1] or value.shape[-1] != 768:
        raise ValueError("template token shape drifted")
    if value.shape[1] != 128:
        raise ValueError("expected two 8x8 template banks")
    return value.reshape(1, 2, 64, 768).mean(dim=1).detach()


def depth_validity(raw_depth, prior, factor, output_size, token_side):
    validity = np.repeat(
        ((raw_depth > 0).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    crop, _, _ = sample_target(validity, prior, factor, output_sz=output_size)
    value = torch.from_numpy(crop[:, :, 0]).float().cuda()[None, None] / 255.0
    value = F.interpolate(value, size=(token_side, token_side), mode="nearest")
    return value.flatten(2).transpose(1, 2).clamp(0.0, 1.0)


def map_prediction(prediction, prior, resize_factor, search_size,
                   image_shape):
    cx, cy, width, height = [float(value) for value in prediction]
    cx *= search_size / resize_factor
    cy *= search_size / resize_factor
    width *= search_size / resize_factor
    height *= search_size / resize_factor
    previous_x = prior[0] + 0.5 * prior[2]
    previous_y = prior[1] + 0.5 * prior[3]
    half = 0.5 * search_size / resize_factor
    mapped = [cx + previous_x - half - 0.5 * width,
              cy + previous_y - half - 0.5 * height,
              width, height]
    return [float(value) for value in
            clip_box(mapped, image_shape[0], image_shape[1], margin=10)]


def one_forward(network, preprocessor, templates, image, raw_depth, prior,
                query, keep_rate, search_factor, search_size):
    patch, resize_factor, _ = sample_target(
        image, prior, search_factor, output_sz=search_size)
    search = preprocessor.process(patch)
    with torch.no_grad():
        output = network.forward(
            template=templates, search=[search], track_query_before=query,
            keep_rate=keep_rate, return_candidate_features=True)[0]
    validity = depth_validity(
        raw_depth, prior, search_factor, search_size,
        int(round(math.sqrt(output["candidate_features"][
            "search_rgb_tokens"].shape[1]))))
    output["track_query_before"] = [
        value.detach().clone() for value in output["track_query_before"]]
    return output, float(resize_factor), validity


def rollout(network, association, optimizer, preprocessor, templates,
            anchor_rgb, anchor_depth, anchor_text, colors, depths, gt,
            anchor_frame, horizon, keep_rate, output_window, train):
    protected_state = gt[anchor_frame].astype(float).tolist()
    tentative_state = copy.deepcopy(protected_state)
    protected_query = None
    tentative_query = None
    focal = FocalLoss()
    losses, rows = [], []
    initial_protected = copy.deepcopy(protected_state)
    for age in range(1, horizon + 1):
        frame = anchor_frame + age
        image, raw_depth = read_rgbd(colors[frame], depths[frame])
        protected, protected_resize, _ = one_forward(
            network, preprocessor, templates, image, raw_depth,
            protected_state, protected_query, keep_rate, 4.0, 256)
        protected_query = protected["track_query_before"]
        disabled = association.refine_score(
            protected["score_map"], torch.zeros_like(protected["score_map"]),
            enabled=False)
        if not torch.equal(disabled, protected["score_map"]):
            raise RuntimeError("disabled association changed protected score")
        protected_response = output_window * protected["score_map"]
        protected_box = network.box_head.cal_bbox(
            protected_response, protected["size_map"],
            protected["offset_map"])[0]
        protected_state = map_prediction(
            protected_box, protected_state, protected_resize, 256,
            image.shape)

        tentative, tentative_resize, validity = one_forward(
            network, preprocessor, templates, image, raw_depth,
            tentative_state, tentative_query, keep_rate, 4.0, 256)
        tentative_query = tentative["track_query_before"]
        feature = tentative["candidate_features"]
        logits = association(
            feature["search_rgb_tokens"].detach(),
            feature["search_depth_tokens"].detach(),
            feature["search_fused_tokens"].detach(),
            anchor_rgb, anchor_depth, anchor_text, validity)
        gt_box = transform_image_to_crop(
            torch.tensor(gt[frame], dtype=torch.float32, device="cuda"),
            torch.tensor(tentative_state, dtype=torch.float32,
                         device="cuda"),
            tentative_resize, torch.tensor([256.0, 256.0], device="cuda"),
            normalize=True)
        target = generate_heatmap([gt_box[None]], 256, 16)[0].unsqueeze(1)
        association_probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        loss = focal(association_probability, target)
        losses.append(loss)
        refined = association.refine_score(
            tentative["score_map"], logits, weight=0.20, enabled=True)
        response = output_window * refined
        tentative_box = network.box_head.cal_bbox(
            response, tentative["size_map"], tentative["offset_map"])[0]
        tentative_state = map_prediction(
            tentative_box.detach(), tentative_state, tentative_resize, 256,
            image.shape)
        truth = gt[frame].astype(float).tolist()
        rows.append({
            "age": age, "frame_index": frame,
            "protected_iou": float(bbox_iou(protected_state, truth)),
            "tentative_iou": float(bbox_iou(tentative_state, truth)),
            "association_loss": float(loss.detach().item()),
        })
    total = torch.stack(losses).mean()
    gradient_norm = 0.0
    if train:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            association.parameters(), 5.0).item())
        optimizer.step()
    if initial_protected == protected_state:
        raise RuntimeError("protected branch did not recursively advance")
    return float(total.detach().item()), gradient_norm, rows


def main():
    args = parse_args()
    started = time.time()
    for name in ("checkpoint", "config", "dataset_root",
                 "language_manifest", "clip_model", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.horizon != 4 or args.steps != 3:
        raise ValueError("engineering smoke is frozen to H4 and three steps")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official checkpoint hash mismatch")
    update_config_from_file(str(args.config))
    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(args.checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    association = LanguageAnchoredDenseAssociation().cuda().train()
    optimizer = torch.optim.AdamW(
        association.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {name: value.detach().clone()
              for name, value in association.named_parameters()}
    root = args.dataset_root / args.sequence
    colors, depths = frames_for(root)
    gt = read_gt(root / "groundtruth.txt")
    if args.anchor_frame < 0 or args.anchor_frame + args.horizon >= len(colors):
        raise ValueError("anchor/horizon is outside sequence")
    used_gt = gt[args.anchor_frame:args.anchor_frame + args.horizon + 1]
    if (not np.isfinite(used_gt).all() or
            (used_gt[:, 2:] <= 0.0).any()):
        raise ValueError("anchor/rollout window has unavailable ground truth")
    anchor_image, _ = read_rgbd(
        colors[args.anchor_frame], depths[args.anchor_frame])
    anchor_bbox = gt[args.anchor_frame].astype(float).tolist()
    template_patch, _, _ = sample_target(
        anchor_image, anchor_bbox, 2.0, output_sz=128)
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    template = preprocessor.process(template_patch).detach()
    templates = [template, template]
    neutral_patch, _, _ = sample_target(
        anchor_image, anchor_bbox, 4.0, output_sz=256)
    neutral = preprocessor.process(neutral_patch).detach()
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    with torch.no_grad():
        anchor_output = network.forward(
            template=templates, search=[neutral], track_query_before=None,
            keep_rate=keep_rate, return_candidate_features=True)[0]
    anchor_rgb = template_bank(
        anchor_output["candidate_features"]["template_rgb_tokens"])
    anchor_depth = template_bank(
        anchor_output["candidate_features"]["template_depth_tokens"])
    clip_encoder = ClipCandidateEncoder(
        args.clip_model, anchor_image, anchor_bbox,
        language_for(args.language_manifest, args.sequence))
    anchor_text = clip_encoder.text_feature.detach()
    output_window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    torch.cuda.reset_peak_memory_stats()
    traces = []
    for step in range(args.steps):
        loss, gradient, rows = rollout(
            network, association, optimizer, preprocessor, templates,
            anchor_rgb, anchor_depth, anchor_text, colors, depths, gt,
            args.anchor_frame, args.horizon, keep_rate, output_window,
            train=True)
        traces.append({"step": step, "loss": loss,
                       "gradient_norm": gradient, "frames": rows})
    changed = sum(not torch.equal(before[name], value.detach())
                  for name, value in association.named_parameters())
    if changed == 0 or any(not math.isfinite(row["loss"]) or
                           not math.isfinite(row["gradient_norm"])
                           for row in traces):
        raise RuntimeError("recursive training smoke did not update safely")
    result = {
        "schema": "sttrack-lachtt-recursive-training-smoke/v1",
        "complete": True,
        "accepted": True,
        "scientific_scope": "engineering smoke only; not an effectiveness result",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "sequence": args.sequence,
        "anchor_frame": args.anchor_frame,
        "horizon": args.horizon,
        "steps": args.steps,
        "association_parameters": sum(
            value.numel() for value in association.parameters()),
        "changed_parameter_tensors": changed,
        "traces": traces,
        "protected_state_independent": True,
        "tentative_uses_own_predicted_crop": True,
        "disabled_score_exact_parity": True,
        "candidate_committed_to_protected": False,
        "checkpoint_written": False,
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": sha256_file(args.checkpoint),
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
