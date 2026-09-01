#!/usr/bin/env python3
"""Collect protected STTrack candidate transactions on DepthTrack Train.

The public schedule is reused from a sealed GT-free STTrack trace.  Only the
first ground-truth line is read to construct the immutable template.  Six
candidate branches are created at each risk event and recursively rolled out
for ten frames without changing any public state or calculating a metric.
"""

import argparse
from collections import defaultdict
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.tracker.sttrack_lachtt_observation import (
    ClipCandidateEncoder,
    RecursiveBranch,
    decode_nms_candidates,
    finite_bbox,
    pool_candidate_tokens,
    raw_depth_rois,
    relative_geometry,
    split_query_state,
    stack_query_states,
)
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import clip_box


EXPECTED_CHECKPOINT_SHA256 = (
    "cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98"
)
SOURCE_NAMES = ("current", "last_reliable", "velocity")
FEATURE_AGES = 5
ROLLOUT_AGES = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--language-manifest", required=True, type=Path)
    parser.add_argument("--clip-model", required=True, type=Path)
    parser.add_argument("--protected-trace", required=True, type=Path)
    parser.add_argument("--protected-plan", required=True, type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path, value):
    path = Path(path)
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


def load_languages(path):
    result = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = str(row.get("sequence_name", "")).strip()
            language = " ".join(str(row.get("language", "")).split())
            if not name or not language or name in result:
                raise ValueError("language manifest is not one-to-one")
            result[name] = language
    return result


def read_initial_bbox(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        line = stream.readline()
    values = [float(value) for value in line.strip().replace("\t", ",").split(",")]
    result = finite_bbox(values)
    if result is None:
        raise ValueError("malformed first-frame bbox")
    return result


def resolve_frames(sequence_root):
    colors = sorted(path for path in (sequence_root / "color").iterdir()
                    if path.is_file())
    depth_root = next((path for path in
                       (sequence_root / "depth", sequence_root / "depth_raw")
                       if path.is_dir()), None)
    if depth_root is None:
        raise FileNotFoundError("sequence has no depth directory")
    depths = sorted(path for path in depth_root.iterdir() if path.is_file())
    if not colors or len(colors) != len(depths):
        raise ValueError("RGB/depth frame count mismatch")
    return colors, depths


def read_rgbd(color_path, depth_path):
    image = get_rgbd_frame(str(color_path), str(depth_path),
                           dtype="rgbcolormap", depth_clip=True)
    raw_depth = cv2.imread(str(depth_path), -1)
    if (image is None or image.ndim != 3 or image.shape[2] != 6 or
            raw_depth is None or raw_depth.ndim != 2):
        raise ValueError("invalid RGB-D frame")
    return image, raw_depth


def safe_public_row(row):
    score = row.get("public_score")
    shadow = row.get("risk_recovery_shadow")
    risk = shadow.get("risk") if isinstance(shadow, dict) else None
    return bool(
        score is not None and float(score) >= 0.30 and
        isinstance(risk, dict) and
        float(risk.get("center_jump_scale", float("inf"))) <= 0.75 and
        float(risk.get("absolute_log_area_change", float("inf"))) <= 0.70)


def extrapolate_velocity(history, frame_index):
    if len(history) < 2:
        return list(history[-1][1])
    (previous_index, previous), (last_index, last) = history[-2:]
    delta = max(1, last_index - previous_index)
    horizon = max(1, frame_index - last_index)
    result = [last[position] +
              (last[position] - previous[position]) / delta * horizon
              for position in range(4)]
    result[2] = max(10.0, result[2])
    result[3] = max(10.0, result[3])
    return result


def event_priors(rows, frame_index, image_shape, init_bbox):
    if frame_index <= 0:
        raise ValueError("risk event cannot occur at initialization")
    current = finite_bbox(rows[frame_index - 1]["public_bbox"])
    if current is None:
        raise ValueError("malformed public state before event")
    reliable = [(0, list(init_bbox))]
    for index in range(1, frame_index):
        if safe_public_row(rows[index]):
            bbox = finite_bbox(rows[index]["public_bbox"])
            if bbox is not None:
                reliable.append((index, bbox))
    last_reliable = list(reliable[-1][1])
    velocity = extrapolate_velocity(reliable, frame_index)
    height, width = image_shape[:2]
    return [
        list(clip_box(current, height, width, margin=10)),
        list(clip_box(last_reliable, height, width, margin=10)),
        list(clip_box(velocity, height, width, margin=10)),
    ]


def repeated_templates(template, batch):
    return [template.repeat(batch, 1, 1, 1),
            template.repeat(batch, 1, 1, 1)]


def collect_observations(output, candidates, priors, resize_factors,
                         image, raw_depth, public_row, search_size,
                         clip_encoder):
    features = output["candidate_features"]
    native = {}
    for key, name in (("search_rgb_tokens", "native_rgb"),
                      ("search_depth_tokens", "native_depth"),
                      ("search_fused_tokens", "native_fused")):
        value = pool_candidate_tokens(
            features[key], candidates, priors, resize_factors, search_size)
        native[name] = value.detach().cpu().half()
    source_indexes = [int(candidate["source_index"])
                      for candidate in candidates]
    query_rgb = features["track_query_rgb"][source_indexes].mean(dim=1)
    query_depth = features["track_query_depth"][source_indexes].mean(dim=1)
    bboxes = [candidate["bbox"] for candidate in candidates]
    clip_features = clip_encoder.encode(image, bboxes)
    public_bbox = finite_bbox(public_row["public_bbox"])
    public_score = public_row.get("public_score")
    scalars = []
    for candidate in candidates:
        source = int(candidate["source_index"])
        scalars.append([
            float(candidate["score"]), float(candidate["margin"]),
            float(candidate["entropy"]),
            0.0 if public_score is None else float(public_score),
            *relative_geometry(candidate["bbox"], public_bbox),
            *relative_geometry(candidate["bbox"], priors[source]),
            float(resize_factors[source]),
        ])
    return {
        **native,
        "query_rgb": query_rgb.detach().cpu().half(),
        "query_depth": query_depth.detach().cpu().half(),
        "clip_image": clip_features.detach().cpu().half(),
        "raw_depth": raw_depth_rois(raw_depth, bboxes).half(),
        "scalars": torch.tensor(scalars, dtype=torch.float32),
    }


def merge_feature_ages(values):
    keys = sorted(values[0])
    if any(sorted(value) != keys for value in values):
        raise RuntimeError("feature keys drifted across rollout")
    return {key: torch.stack([value[key] for value in values], dim=0)
            for key in keys}


def validate_feature_block(features):
    expected = {
        "clip_image": [FEATURE_AGES, 6, 768],
        "native_depth": [FEATURE_AGES, 6, 768],
        "native_fused": [FEATURE_AGES, 6, 768],
        "native_rgb": [FEATURE_AGES, 6, 768],
        "query_depth": [FEATURE_AGES, 6, 768],
        "query_rgb": [FEATURE_AGES, 6, 768],
        "raw_depth": [FEATURE_AGES, 6, 2, 16, 16],
        "scalars": [FEATURE_AGES, 6, 15],
    }
    if sorted(features) != sorted(expected):
        raise RuntimeError("candidate feature schema drifted")
    for name, shape in expected.items():
        value = features[name]
        if list(value.shape) != shape:
            raise RuntimeError("candidate feature shape drifted: %s" % name)
        if not torch.isfinite(value.float()).all().item():
            raise RuntimeError("candidate feature is non-finite: %s" % name)
    validity = features["raw_depth"][:, :, 1].float()
    if validity.min().item() < 0.0 or validity.max().item() > 1.0:
        raise RuntimeError("depth validity mask is outside [0, 1]")


def run_event(network, preprocessor, template, output_window, keep_rate,
              colors, depths, rows, frame_index, init_bbox, clip_encoder,
              search_size):
    first_image, first_depth = read_rgbd(colors[frame_index], depths[frame_index])
    priors = event_priors(rows, frame_index, first_image.shape, init_bbox)
    search_tensors, resize_factors = [], []
    for prior in priors:
        patch, resize_factor, _ = sample_target(
            first_image, prior, 6.0, output_sz=search_size)
        search_tensors.append(preprocessor.process(patch))
        resize_factors.append(float(resize_factor))
    search = torch.cat(search_tensors, dim=0)
    with torch.no_grad():
        output = network.forward(
            template=repeated_templates(template, len(priors)),
            search=[search], track_query_before=None, keep_rate=keep_rate,
            return_candidate_features=True)[0]
    response = output_window * output["score_map"]
    candidates = decode_nms_candidates(
        response, output["size_map"], output["offset_map"], priors,
        resize_factors, first_image.shape, search_size,
        peaks_per_prior=2, nms_kernel=3)
    if len(candidates) != 6:
        raise RuntimeError("age-0 action count is not six")
    query_states = split_query_state(output["track_query_before"])
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
                         for value in query_states[source]],
        ))
    feature_ages = [collect_observations(
        output, candidates, priors, resize_factors, first_image, first_depth,
        rows[frame_index], search_size, clip_encoder)]
    trajectory = [{
        "age": 0,
        "frame_index": frame_index,
        "branches": [{
            "name": branch.name, "source": branch.source_name,
            "peak_rank": branch.peak_rank,
            "bbox": list(candidate["bbox"]),
            "score": float(candidate["score"]),
            "margin": float(candidate["margin"]),
            "entropy": float(candidate["entropy"]),
        } for branch, candidate in zip(branches, candidates)],
    }]
    for age in range(1, ROLLOUT_AGES):
        image, raw_depth = read_rgbd(
            colors[frame_index + age], depths[frame_index + age])
        priors = [list(branch.bbox) for branch in branches]
        search_tensors, resize_factors = [], []
        for prior in priors:
            patch, resize_factor, _ = sample_target(
                image, prior, 4.0, output_sz=search_size)
            search_tensors.append(preprocessor.process(patch))
            resize_factors.append(float(resize_factor))
        search = torch.cat(search_tensors, dim=0)
        query_input = stack_query_states(
            [branch.query_state for branch in branches])
        with torch.no_grad():
            output = network.forward(
                template=repeated_templates(template, len(branches)),
                search=[search], track_query_before=query_input,
                keep_rate=keep_rate, return_candidate_features=True)[0]
        response = output_window * output["score_map"]
        candidates = decode_nms_candidates(
            response, output["size_map"], output["offset_map"], priors,
            resize_factors, image.shape, search_size,
            peaks_per_prior=1, nms_kernel=3)
        if len(candidates) != len(branches):
            raise RuntimeError("recursive action count drifted")
        query_states = split_query_state(output["track_query_before"])
        next_branches = []
        for index, (branch, candidate) in enumerate(zip(branches, candidates)):
            next_branches.append(RecursiveBranch(
                name=branch.name, source_name=branch.source_name,
                peak_rank=branch.peak_rank, bbox=list(candidate["bbox"]),
                query_state=query_states[index]))
        branches = next_branches
        if age < FEATURE_AGES:
            feature_ages.append(collect_observations(
                output, candidates, priors, resize_factors, image, raw_depth,
                rows[frame_index + age], search_size, clip_encoder))
        trajectory.append({
            "age": age,
            "frame_index": frame_index + age,
            "branches": [{
                "name": branch.name, "source": branch.source_name,
                "peak_rank": branch.peak_rank,
                "bbox": list(candidate["bbox"]),
                "score": float(candidate["score"]),
                "margin": float(candidate["margin"]),
                "entropy": float(candidate["entropy"]),
            } for branch, candidate in zip(branches, candidates)],
        })
    if len(feature_ages) != FEATURE_AGES or len(trajectory) != ROLLOUT_AGES:
        raise RuntimeError("rollout horizon contract failed")
    features = merge_feature_ages(feature_ages)
    validate_feature_block(features)
    return features, trajectory


def validate_spec(args):
    if args.smoke:
        if args.spec is not None:
            raise ValueError("smoke must not impersonate a formal frozen run")
        return None
    if args.spec is None or not args.spec.is_file():
        raise ValueError("formal collection requires a frozen spec")
    if args.sequence or args.max_events is not None:
        raise ValueError("formal collection cannot use smoke limits")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if (spec.get("complete") is not True or
            spec.get("created_before_collection") is not True or
            spec.get("repository_commit") != subprocess.check_output(
                ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
                text=True).strip()):
        raise ValueError("formal spec state mismatch")
    if args.dataset_root != Path(spec["dataset_root"]).resolve():
        raise ValueError("formal dataset root differs from frozen spec")
    expected_output = (Path(spec["collection_root"]).resolve() /
                       ("shard%d" % args.shard))
    if args.output != expected_output:
        raise ValueError("formal output is outside the frozen shard root")
    bindings = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "language_manifest": args.language_manifest,
        "clip_model": args.clip_model,
        "protected_trace_shard_%d" % args.shard: args.protected_trace,
        "protected_plan": args.protected_plan,
        "runner": Path(__file__).resolve(),
        "observation": (REPOSITORY_ROOT /
                        "lib/test/tracker/sttrack_lachtt_observation.py"),
        "model": REPOSITORY_ROOT / "lib/models/sttrack/sttrack.py",
    }
    records = spec.get("bindings", {})
    for name, path in bindings.items():
        expected = records.get(name)
        if not isinstance(expected, dict) or expected.get("sha256") != sha256_file(path):
            raise ValueError("formal binding mismatch: %s" % name)
    return spec


def main():
    args = parse_args()
    started = time.time()
    for name in ("checkpoint", "config", "dataset_root", "language_manifest",
                 "clip_model",
                 "protected_trace", "protected_plan", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.spec is not None:
        args.spec = args.spec.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    formal_spec = validate_spec(args)
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official STTrack checkpoint hash mismatch")
    update_config_from_file(str(args.config))
    if (not bool(cfg.MODEL.TSG.FIX_QUERY_WINDOW) or
            float(cfg.TEST.SEARCH_FACTOR) != 4.0 or
            int(cfg.TEST.SEARCH_SIZE) != 256 or
            int(cfg.DATA.TEMPLATE.NUMBER) != 2):
        raise ValueError("STTrack fixed-query/config contract mismatch")
    protected = json.loads(args.protected_trace.read_text(encoding="utf-8"))
    if (protected.get("complete") is not True or
            protected.get("ground_truth_used_after_initialization") is not False or
            protected.get("metric_computed") is not False or
            int(protected.get("shard", -1)) != args.shard):
        raise ValueError("protected trace is not a valid GT-free schedule")
    plan = json.loads(args.protected_plan.read_text(encoding="utf-8"))
    plan_shard = next((item for item in plan["shards"]
                       if int(item["shard"]) == args.shard), None)
    if plan_shard is None:
        raise ValueError("protected plan has no requested shard")
    selected_sequences = list(plan_shard["sequences"])
    if args.sequence:
        unknown = set(args.sequence) - set(selected_sequences)
        if unknown:
            raise ValueError("unknown smoke sequence: %s" % sorted(unknown))
        selected_sequences = list(args.sequence)
    rows_by_sequence = defaultdict(list)
    for row in protected["rows"]:
        if row["sequence"] in selected_sequences:
            rows_by_sequence[row["sequence"]].append(row)
    for name in selected_sequences:
        rows_by_sequence[name].sort(key=lambda row: int(row["frame_index"]))
    languages = load_languages(args.language_manifest)

    network = build_sttrack(cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(args.checkpoint), map_location="cpu")["net"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("official checkpoint strict load failed")
    network = network.cuda().eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    feature_side = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
    output_window = hann2d(torch.tensor([feature_side, feature_side]).long(),
                           centered=True).cuda()
    torch.cuda.reset_peak_memory_stats()

    args.output.mkdir(parents=True)
    (args.output / "events").mkdir()
    (args.output / "anchors").mkdir()
    metadata_path = args.output / "events.jsonl.partial"
    event_count = 0
    sequence_counts = {}
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for sequence_name in selected_sequences:
            rows = rows_by_sequence[sequence_name]
            sequence_root = args.dataset_root / sequence_name
            colors, depths = resolve_frames(sequence_root)
            if len(rows) != len(colors) or sequence_name not in languages:
                if sequence_name not in languages:
                    raise ValueError("missing sequence language")
                raise ValueError("protected rows/frame count mismatch")
            init_bbox = read_initial_bbox(sequence_root / "groundtruth.txt")
            initial_image, _ = read_rgbd(colors[0], depths[0])
            template_patch, _, _ = sample_target(
                initial_image, init_bbox, cfg.TEST.TEMPLATE_FACTOR,
                output_sz=cfg.TEST.TEMPLATE_SIZE)
            template = preprocessor.process(template_patch).detach()
            template_before = template.clone()
            clip_encoder = ClipCandidateEncoder(
                args.clip_model,
                initial_image, init_bbox, languages[sequence_name])
            anchor_path = args.output / "anchors" / (sequence_name + ".pt")
            atomic_torch_save(anchor_path, clip_encoder.anchor_record())
            starts = []
            for row in rows:
                shadow = row.get("risk_recovery_shadow")
                if (isinstance(shadow, dict) and shadow.get("event_started") and
                        int(row["frame_index"]) + ROLLOUT_AGES <= len(rows)):
                    starts.append(int(row["frame_index"]))
            if args.max_events is not None:
                starts = starts[:max(0, args.max_events - event_count)]
            sequence_event_count = 0
            for frame_index in starts:
                features, trajectory = run_event(
                    network, preprocessor, template, output_window, keep_rate,
                    colors, depths, rows, frame_index, init_bbox, clip_encoder,
                    int(cfg.TEST.SEARCH_SIZE))
                event_id = int(rows[frame_index]["risk_recovery_shadow"]["event_id"])
                relative = Path("events") / (
                    "%s_event%04d_frame%06d.pt" %
                    (sequence_name, event_id, frame_index))
                feature_path = args.output / relative
                atomic_torch_save(feature_path, features)
                row = {
                    "sequence": sequence_name,
                    "event_id": event_id,
                    "trigger_frame": frame_index,
                    "feature_path": str(relative),
                    "feature_sha256": sha256_file(feature_path),
                    "feature_bytes": feature_path.stat().st_size,
                    "anchor_path": str(anchor_path.relative_to(args.output)),
                    "public": [{
                        "age": age,
                        "frame_index": frame_index + age,
                        "bbox": [float(value) for value in
                                 rows[frame_index + age]["public_bbox"]],
                        "score": rows[frame_index + age].get("public_score"),
                    } for age in range(ROLLOUT_AGES)],
                    "trajectory": trajectory,
                }
                metadata.write(json.dumps(row, sort_keys=True,
                                          allow_nan=False) + "\n")
                metadata.flush()
                os.fsync(metadata.fileno())
                event_count += 1
                sequence_event_count += 1
                print(json.dumps({
                    "sequence": sequence_name, "event": event_id,
                    "trigger_frame": frame_index,
                    "completed_events": event_count,
                }, sort_keys=True), flush=True)
                if args.max_events is not None and event_count >= args.max_events:
                    break
            sequence_counts[sequence_name] = sequence_event_count
            if not torch.equal(template, template_before):
                raise RuntimeError("immutable first template was mutated")
            del clip_encoder
            torch.cuda.empty_cache()
            if args.max_events is not None and event_count >= args.max_events:
                break
    final_metadata = args.output / "events.jsonl"
    os.replace(metadata_path, final_metadata)
    if (formal_spec is not None and event_count !=
            int(formal_spec["expected_event_counts"][str(args.shard)])):
        raise RuntimeError("formal event count differs from frozen spec")
    manifest = {
        "schema": "sttrack-lachtt-train152-collection/v1",
        "complete": True,
        "accepted": True,
        "smoke": bool(args.smoke),
        "scientific_scope": (
            "GT-free protected candidate rollout; first-frame initialization "
            "box only; no metric and no public-state commit"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "shard": args.shard,
        "event_count": event_count,
        "sequence_event_counts": sequence_counts,
        "feature_ages": FEATURE_AGES,
        "rollout_ages": ROLLOUT_AGES,
        "branches_per_event": 6,
        "source_names": list(SOURCE_NAMES),
        "ground_truth_used_after_initialization": False,
        "future_ground_truth_opened": False,
        "metric_computed": False,
        "candidate_committed_to_public_tracker": False,
        "future_frame_text_used": False,
        "qwen_used": False,
        "immutable_template": True,
        "independent_query_state": True,
        "metadata": file_record(final_metadata),
        "checkpoint": file_record(args.checkpoint),
        "config": file_record(args.config),
        "language_manifest": file_record(args.language_manifest),
        "clip_model": file_record(args.clip_model),
        "protected_trace": file_record(args.protected_trace),
        "protected_plan": file_record(args.protected_plan),
        "runner": file_record(Path(__file__).resolve()),
        "observation": file_record(
            REPOSITORY_ROOT /
            "lib/test/tracker/sttrack_lachtt_observation.py"),
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started,
        "depthtrack_test_run": False,
        "cdtb_run": False,
        "vot_low22_run": False,
        "vot_full127_run": False,
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
