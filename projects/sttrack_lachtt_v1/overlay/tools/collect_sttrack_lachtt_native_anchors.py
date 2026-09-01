#!/usr/bin/env python3
"""Collect immutable first-frame native STTrack RGB/depth identity tokens.

Only the first ground-truth row is read.  One first-frame forward pass is used
per sequence; no future frame, metric, candidate commit, or public benchmark is
opened.  Duplicate template token banks are averaged into one 8x8 bank.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.test.tracker.data_utils import PreprocessorMM
from lib.train.data.processing_utils import sample_target
from tools.run_sttrack_lachtt_train152_collection import (
    EXPECTED_CHECKPOINT_SHA256,
    atomic_torch_save,
    read_initial_bbox,
    read_rgbd,
    repeated_templates,
    resolve_frames,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--protected-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
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


def selected_sequences(plan):
    names = []
    for shard in plan["shards"]:
        names.extend(shard["sequences"])
    if len(names) != len(set(names)):
        raise ValueError("protected plan repeats a sequence")
    return sorted(names)


def validate_formal(args):
    if args.smoke:
        if args.spec is not None:
            raise ValueError("smoke cannot impersonate formal collection")
        return None
    if args.spec is None or not args.spec.is_file() or args.sequence:
        raise ValueError("formal collection requires one frozen spec and no filter")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    if (spec.get("complete") is not True or
            spec.get("created_before_collection") is not True or
            spec.get("repository_commit") != commit or
            args.output != Path(spec["output_root"]).resolve()):
        raise ValueError("formal native-anchor spec state mismatch")
    paths = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "dataset_root_marker": args.dataset_root / ".",
        "protected_plan": args.protected_plan,
        "collector": Path(__file__).resolve(),
        "model": REPOSITORY_ROOT / "lib/models/sttrack/sttrack.py",
    }
    for name, path in paths.items():
        expected = spec["bindings"].get(name)
        if name == "dataset_root_marker":
            if expected.get("path") != str(args.dataset_root):
                raise ValueError("dataset root binding mismatch")
        elif expected.get("sha256") != sha256_file(path):
            raise ValueError("formal native-anchor binding mismatch: %s" % name)
    return spec


def collapse_template_bank(value, template_number):
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 768:
        raise RuntimeError("native template token rank/width drifted")
    token_count = int(value.shape[1])
    if token_count % template_number:
        raise RuntimeError("template token count is not divisible by templates")
    per_template = token_count // template_number
    if per_template != 64:
        raise RuntimeError("expected an 8x8 native template token bank")
    bank = value.reshape(1, template_number, per_template, 768).mean(dim=1)[0]
    if not torch.isfinite(bank).all().item():
        raise RuntimeError("native template token bank is non-finite")
    return bank.detach().cpu().half()


def main():
    args = parse_args()
    started = time.time()
    for name in ("checkpoint", "config", "dataset_root", "protected_plan",
                 "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.spec is not None:
        args.spec = args.spec.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    formal_spec = validate_formal(args)
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official STTrack checkpoint hash mismatch")
    update_config_from_file(str(args.config))
    if (not bool(cfg.MODEL.TSG.FIX_QUERY_WINDOW) or
            int(cfg.DATA.TEMPLATE.NUMBER) != 2 or
            int(cfg.TEST.TEMPLATE_SIZE) != 128 or
            int(cfg.TEST.SEARCH_SIZE) != 256 or
            float(cfg.TEST.SEARCH_FACTOR) != 4.0):
        raise ValueError("STTrack native-anchor config contract mismatch")
    plan = json.loads(args.protected_plan.read_text(encoding="utf-8"))
    names = selected_sequences(plan)
    if args.sequence:
        unknown = set(args.sequence) - set(names)
        if unknown:
            raise ValueError("unknown sequence filter: %s" % sorted(unknown))
        names = list(args.sequence)
    if formal_spec is not None and len(names) != int(formal_spec["sequence_count"]):
        raise ValueError("formal sequence count drifted")

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
    torch.cuda.reset_peak_memory_stats()

    args.output.mkdir(parents=True)
    (args.output / "anchors").mkdir()
    partial = args.output / "index.jsonl.partial"
    rows = []
    with partial.open("w", encoding="utf-8") as stream:
        for sequence in names:
            root = args.dataset_root / sequence
            colors, depths = resolve_frames(root)
            bbox = read_initial_bbox(root / "groundtruth.txt")
            image, _ = read_rgbd(colors[0], depths[0])
            template_patch, _, _ = sample_target(
                image, bbox, cfg.TEST.TEMPLATE_FACTOR,
                output_sz=cfg.TEST.TEMPLATE_SIZE)
            search_patch, _, _ = sample_target(
                image, bbox, cfg.TEST.SEARCH_FACTOR,
                output_sz=cfg.TEST.SEARCH_SIZE)
            template = preprocessor.process(template_patch).detach()
            search = preprocessor.process(search_patch).detach()
            with torch.no_grad():
                output = network.forward(
                    template=repeated_templates(template, 1), search=[search],
                    track_query_before=None, keep_rate=keep_rate,
                    return_candidate_features=True)[0]
            feature = output["candidate_features"]
            rgb = collapse_template_bank(
                feature["template_rgb_tokens"], int(cfg.DATA.TEMPLATE.NUMBER))
            depth = collapse_template_bank(
                feature["template_depth_tokens"], int(cfg.DATA.TEMPLATE.NUMBER))
            value = {
                "native_template_rgb_tokens": rgb,
                "native_template_depth_tokens": depth,
                "native_template_rgb_mean": rgb.float().mean(dim=0).half(),
                "native_template_depth_mean": depth.float().mean(dim=0).half(),
            }
            path = args.output / "anchors" / (sequence + ".pt")
            atomic_torch_save(path, value)
            row = {
                "sequence": sequence,
                "path": str(path.relative_to(args.output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
            }
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({"sequence": sequence,
                              "completed": len(rows)}, sort_keys=True), flush=True)
    index = args.output / "index.jsonl"
    os.replace(partial, index)
    manifest = {
        "schema": "sttrack-lachtt-native-anchors/v1",
        "complete": True,
        "accepted": True,
        "smoke": bool(args.smoke),
        "scientific_scope": (
            "DepthTrack Train immutable first-frame native RGB/depth identity "
            "tokens only; no future frame, metric, commit, or public benchmark"),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "sequence_count": len(rows),
        "template_token_bank_shape": [64, 768],
        "ground_truth_rows_read_per_sequence": 1,
        "future_ground_truth_opened": False,
        "future_frame_opened": False,
        "metric_computed": False,
        "candidate_committed_to_public_tracker": False,
        "qwen_used": False,
        "index": record(index),
        "checkpoint": record(args.checkpoint),
        "config": record(args.config),
        "protected_plan": record(args.protected_plan),
        "collector": record(Path(__file__).resolve()),
        "model": record(REPOSITORY_ROOT / "lib/models/sttrack/sttrack.py"),
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
