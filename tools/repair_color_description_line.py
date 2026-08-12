#!/usr/bin/env python3
"""Regenerate one invalid frame description with Qwen2.5-VL and Qwen3."""

from __future__ import print_function

import argparse
import os
import sys


TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

from annotate_color_descriptions import (
    DEFAULT_QWEN25_MODEL,
    DEFAULT_QWEN3_MODEL,
    CORRECTED_FILENAME,
    Qwen25Captioner,
    Qwen3Corrector,
    _atomic_write_bytes,
    _is_valid_corrected_text,
    collect_sequence_inputs,
    validate_runtime_options,
)


DEFAULT_DATA_ROOT = r"D:\steam\RGB-D-L-work\datasets\DepthTrackTrain"
DEFAULT_DESCRIPTION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "color_desc", "depthtrack_train"))


def replace_description_line(path, frame_index, description, expected_frame_count):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if len(lines) != int(expected_frame_count):
        raise ValueError(
            "Description/frame count mismatch for {}: {} lines, {} frames".format(
                path, len(lines), expected_frame_count))
    if frame_index < 0 or frame_index >= len(lines):
        raise IndexError("Frame index {} is outside {} lines".format(frame_index, len(lines)))
    description = str(description or "").strip()
    if not description:
        raise ValueError("Replacement description is empty")
    if "\n" in description or "\r" in description:
        raise ValueError("Replacement description must contain exactly one line")
    lines[frame_index] = description
    _atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--description-root", default=DEFAULT_DESCRIPTION_ROOT)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--qwen25-model", default=DEFAULT_QWEN25_MODEL)
    parser.add_argument("--qwen3-model", default=DEFAULT_QWEN3_MODEL)
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_runtime_options(args)
    sequence_root = os.path.join(args.data_root, "sequences")
    if not os.path.isdir(sequence_root):
        sequence_root = args.data_root
    sequence_dir = os.path.join(sequence_root, args.sequence)
    frames, boxes = collect_sequence_inputs(
        sequence_dir, required_frame_ids=[args.frame_index])
    output_path = os.path.join(args.description_root, args.sequence, CORRECTED_FILENAME)
    if args.dry_run:
        print("Would repair {} frame {} in {}".format(args.sequence, args.frame_index, output_path))
        return 0

    captioner = Qwen25Captioner(
        args.qwen25_model, args.gpu_memory, args.cpu_memory, args.load_in_4bit)
    try:
        draft = captioner.describe(frames[args.frame_index], boxes[args.frame_index], args.sequence)
    finally:
        captioner.close()
    corrector = Qwen3Corrector(
        args.qwen3_model, args.gpu_memory, args.cpu_memory, args.load_in_4bit)
    try:
        corrected = corrector.correct(draft)
    finally:
        corrector.close()
    if not _is_valid_corrected_text(corrected):
        raise RuntimeError("Qwen3 returned an invalid correction: {}".format(corrected))
    replace_description_line(output_path, args.frame_index, corrected, len(frames))
    print("Qwen2.5 draft: {}".format(draft))
    print("Qwen3 correction: {}".format(corrected))
    print("Updated {} frame {}".format(args.sequence, args.frame_index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
