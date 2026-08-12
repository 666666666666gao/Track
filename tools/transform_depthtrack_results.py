#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKING_ROOT = os.path.join(PROJECT_ROOT, "tracking")
for path in (PROJECT_ROOT, TRACKING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from lib.test.analysis.depthtrack_prre import evaluate_depthtrack_prre, format_depthtrack_prre
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.utils.load_text import load_text


def _result_dir(root, tracker, parameter, epoch):
    return os.path.join(root, tracker, "{}_{}".format(parameter, epoch))


def _read_boxes(path):
    arr = np.asarray(load_text(path, delimiter=("\t", ",", " "), dtype=np.float64))
    return arr.reshape(-1, 4)


def _write_boxes(path, boxes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, boxes, fmt="%.6f", delimiter="\t")


def _scale_boxes(boxes, scale_w, scale_h):
    out = boxes.copy()
    cx = boxes[:, 0] + 0.5 * boxes[:, 2]
    cy = boxes[:, 1] + 0.5 * boxes[:, 3]
    out[:, 2] = np.maximum(boxes[:, 2] * scale_w, 1e-6)
    out[:, 3] = np.maximum(boxes[:, 3] * scale_h, 1e-6)
    out[:, 0] = cx - 0.5 * out[:, 2]
    out[:, 1] = cy - 0.5 * out[:, 3]
    return out


def _ema_boxes(boxes, alpha):
    if alpha <= 0.0 or alpha >= 1.0 or len(boxes) <= 1:
        return boxes
    out = boxes.copy()
    for i in range(1, len(out)):
        if np.isfinite(out[i]).all() and np.isfinite(out[i - 1]).all():
            out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="Transform saved DepthTrack result boxes and evaluate.")
    parser.add_argument("--tracker", default="mplt_track")
    parser.add_argument("--dataset_name", default="depthtrack_test")
    parser.add_argument("--results_root", default="output/test/tracking_results")
    parser.add_argument("--source_param", required=True)
    parser.add_argument("--source_epoch", required=True)
    parser.add_argument("--output_param", required=True)
    parser.add_argument("--output_epoch", default="ep0000")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--scale_w", type=float, default=None)
    parser.add_argument("--scale_h", type=float, default=None)
    parser.add_argument("--ema_alpha", type=float, default=1.0,
                        help="Causal box EMA alpha. 1.0 disables smoothing.")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    scale_w = args.scale if args.scale_w is None else args.scale_w
    scale_h = args.scale if args.scale_h is None else args.scale_h
    root = os.path.abspath(args.results_root)
    src_dir = _result_dir(root, args.tracker, args.source_param, args.source_epoch)
    out_dir = _result_dir(root, args.tracker, args.output_param, args.output_epoch)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset = get_dataset(args.dataset_name)
    for seq in dataset:
        src_box = os.path.join(src_dir, seq.name + ".txt")
        if not os.path.isfile(src_box):
            raise FileNotFoundError(src_box)
        boxes = _scale_boxes(_read_boxes(src_box), scale_w, scale_h)
        boxes = _ema_boxes(boxes, args.ema_alpha)
        _write_boxes(os.path.join(out_dir, seq.name + ".txt"), boxes)
        for name in os.listdir(src_dir):
            if name.startswith(seq.name + "_") and name.endswith(".txt"):
                shutil.copy2(os.path.join(src_dir, name), os.path.join(out_dir, name))

    tracker = Tracker(args.tracker, args.output_param, args.dataset_name, args.output_epoch)
    results = evaluate_depthtrack_prre([tracker], dataset, output_json=args.output_json)
    print(format_depthtrack_prre(results))


if __name__ == "__main__":
    main()
