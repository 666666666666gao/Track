#!/usr/bin/env python3
"""Bounded STTrack checkpoint and candidate-feature smoke.

This is intentionally not a benchmark evaluation.  It reads ground truth only
for the first-frame initialization box, then runs three recursive predictions
without calculating IoU or any public-set metric.
"""

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import clip_box


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_init_bbox(path):
    first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    values = [float(value) for value in first.replace("\t", ",").split(",")]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid first-frame initialization box")
    return values


def resolve_frames(sequence, count=4):
    sequence = Path(sequence)
    colors = sorted((sequence / "color").glob("*"))[:count]
    depth_dirs = [sequence / "depth", sequence / "depth_raw"]
    depth_dir = next((path for path in depth_dirs if path.is_dir()), None)
    if depth_dir is None:
        raise FileNotFoundError("sequence has no depth/depth_raw directory")
    depths = sorted(depth_dir.glob("*"))[:count]
    if len(colors) != count or len(depths) != count:
        raise RuntimeError("bounded smoke requires four RGB-D frames")
    return list(zip(colors, depths))


def map_box_back(pred_box, previous_state, search_size, resize_factor):
    cx_prev = previous_state[0] + 0.5 * previous_state[2]
    cy_prev = previous_state[1] + 0.5 * previous_state[3]
    cx, cy, width, height = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [
        cx_real - 0.5 * width,
        cy_real - 0.5 * height,
        width,
        height,
    ]


def tensor_record(tensor):
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached).all().item()),
        "mean_abs": float(detached.abs().float().mean().item()),
    }


def main():
    args = parse_args()
    started = time.time()
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint = Path(args.checkpoint).resolve()
    sequence = Path(args.sequence).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    update_config_from_file(args.config)
    if not bool(cfg.MODEL.TSG.FIX_QUERY_WINDOW):
        raise RuntimeError("M0 requires the fixed query-window configuration")

    checkpoint_sha = sha256_file(checkpoint)
    expected_sha = (
        "cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98"
    )
    if checkpoint_sha != expected_sha:
        raise RuntimeError("official checkpoint SHA-256 mismatch")

    network = build_sttrack(cfg, training=False)
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    incompatible = network.load_state_dict(checkpoint_data["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict load returned incompatible keys")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    frames = resolve_frames(sequence, count=4)
    state = read_init_bbox(sequence / "groundtruth.txt")
    images = [
        get_rgbd_frame(
            str(color), str(depth), dtype="rgbcolormap", depth_clip=True
        )
        for color, depth in frames
    ]
    if any(image is None or image.ndim != 3 or image.shape[2] != 6 for image in images):
        raise RuntimeError("RGB-D loader did not return HxWx6 inputs")

    template_patch, _, _ = sample_target(
        images[0],
        state,
        cfg.TEST.TEMPLATE_FACTOR,
        output_sz=cfg.TEST.TEMPLATE_SIZE,
    )
    template = preprocessor.process(template_patch)
    template_list = [template] * cfg.DATA.TEMPLATE.NUMBER
    template_before = template.detach().clone()
    query_state = None
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    output_window = hann2d(
        torch.tensor(
            [
                cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE,
                cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE,
            ]
        ).long(),
        centered=True,
    ).cuda()

    torch.cuda.reset_peak_memory_stats()
    frame_records = []
    with torch.no_grad():
        for frame_offset, image in enumerate(images[1:], start=1):
            height, width, _ = image.shape
            search_patch, resize_factor, _ = sample_target(
                image,
                state,
                cfg.TEST.SEARCH_FACTOR,
                output_sz=cfg.TEST.SEARCH_SIZE,
            )
            search = preprocessor.process(search_patch)
            outputs = network.forward(
                template=template_list,
                search=[search],
                track_query_before=query_state,
                keep_rate=keep_rate,
                return_candidate_features=True,
            )[0]
            query_state = outputs["track_query_before"]
            if any(query.shape[1] != cfg.MODEL.TSG.TRACK_QUERY_OLD for query in query_state):
                raise RuntimeError("query history length drifted during recursive smoke")

            candidate = outputs["candidate_features"]
            tensor_keys = [
                "template_rgb_tokens",
                "template_depth_tokens",
                "search_rgb_tokens",
                "search_depth_tokens",
                "search_fused_tokens",
                "track_query_rgb",
                "track_query_depth",
            ]
            tensor_records = {key: tensor_record(candidate[key]) for key in tensor_keys}
            if not all(record["finite"] for record in tensor_records.values()):
                raise RuntimeError("non-finite candidate feature tensor")
            if candidate["search_rgb_tokens"].shape != candidate["search_depth_tokens"].shape:
                raise RuntimeError("RGB/Depth candidate token shapes disagree")

            response = output_window * outputs["score_map"]
            predicted_boxes = network.box_head.cal_bbox(
                response, outputs["size_map"], outputs["offset_map"]
            ).view(-1, 4)
            predicted_box = (
                predicted_boxes.mean(dim=0)
                * cfg.TEST.SEARCH_SIZE
                / resize_factor
            ).tolist()
            state = clip_box(
                map_box_back(
                    predicted_box,
                    state,
                    cfg.TEST.SEARCH_SIZE,
                    resize_factor,
                ),
                height,
                width,
                margin=10,
            )
            if len(state) != 4 or not all(math.isfinite(value) for value in state):
                raise RuntimeError("non-finite recursive prediction box")
            frame_records.append(
                {
                    "frame_offset": frame_offset,
                    "bbox": [float(value) for value in state],
                    "response_max": float(response.max().item()),
                    "query_lengths": [int(value.shape[1]) for value in query_state],
                    "candidate_features": tensor_records,
                    "rgb_depth_search_mean_abs_difference": float(
                        (
                            candidate["search_rgb_tokens"]
                            - candidate["search_depth_tokens"]
                        )
                        .abs()
                        .float()
                        .mean()
                        .item()
                    ),
                }
            )

    template_unchanged = bool(torch.equal(template, template_before))
    if not template_unchanged:
        raise RuntimeError("candidate-feature smoke mutated the immutable template")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = None
    result = {
        "schema_version": "sttrack-fallback-m0-smoke-result-v1",
        "status": "pass",
        "scientific_scope": "four-frame engineering smoke; first-frame GT initialization only; no IoU or benchmark metric",
        "repository_commit": git_commit,
        "config": str(Path(args.config).resolve()),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
            "strict_load_missing_keys": list(incompatible.missing_keys),
            "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        },
        "sequence": sequence.name,
        "input_shape": list(images[0].shape),
        "fixed_query_window": True,
        "template_unchanged": template_unchanged,
        "frame_records": frame_records,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "elapsed_seconds": time.time() - started,
        "baseline_evaluation_run": False,
        "depthtrack_test_run": False,
        "cdtb_run": False,
        "vot_run": False,
        "new_metric": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
