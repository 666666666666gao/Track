#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.test.analysis.depthtrack_prre import evaluate_depthtrack_prre, format_depthtrack_prre
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.utils.load_text import load_text


def _read_array(path, cols=None, default=None, length=None):
    if os.path.isfile(path):
        arr = np.asarray(load_text(path, delimiter=("\t", ",", " "), dtype=np.float64))
        return arr.reshape(-1, cols) if cols else arr.reshape(-1)
    if default is not None and length is not None:
        return np.full(length, default, dtype=np.float64)
    return None


def _write_array(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.ndim == 1:
        np.savetxt(path, arr, fmt="%.6f")
    else:
        np.savetxt(path, arr, fmt="%.6f", delimiter="\t")


def _result_dir(root, tracker, parameter, epoch):
    return os.path.join(root, tracker, "{}_{}".format(parameter, epoch))


def _choose_mask(mode, score_a, score_b, cons_a, cons_b, rgb_a, rgb_b, dep_a, dep_b,
                 ent_a, center_follow_b, delta, min_cons):
    if mode == "score":
        choose = score_b > score_a + delta
    elif mode == "score_cons":
        choose = (score_b + 0.20 * cons_b) > (score_a + 0.20 * cons_a) + delta
    elif mode == "score_depth":
        choose = ((score_b + 0.15 * rgb_b + 0.20 * dep_b) >
                  (score_a + 0.15 * rgb_a + 0.20 * dep_a) + delta)
    elif mode == "score_recover":
        choose = ((score_a < 0.35) | (ent_a > 0.65) | (center_follow_b > 0.5)) & (
            score_b > score_a + delta)
    elif mode == "cons_only":
        choose = cons_b > cons_a + delta
    elif mode == "depth_only":
        choose = dep_b > dep_a + delta
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    return choose & (cons_b >= min_cons)


def main():
    parser = argparse.ArgumentParser(
        description="Build a deterministic two-expert DepthTrack result ensemble.")
    parser.add_argument("--tracker", default="mplt_track")
    parser.add_argument("--dataset_name", default="depthtrack_test")
    parser.add_argument("--results_root", default="output/test/tracking_results")
    parser.add_argument("--expert_a", required=True, help="Expert A parameter name without epoch suffix.")
    parser.add_argument("--epoch_a", required=True)
    parser.add_argument("--expert_b", required=True, help="Expert B parameter name without epoch suffix.")
    parser.add_argument("--epoch_b", required=True)
    parser.add_argument("--output_param", required=True)
    parser.add_argument("--output_epoch", default="ep0000")
    parser.add_argument("--mode", default="score",
                        choices=["score", "score_cons", "score_depth", "score_recover",
                                 "cons_only", "depth_only"])
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--min_cons", type=float, default=0.35)
    parser.add_argument("--score_policy", default="selected", choices=["selected", "max"])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name)
    root = os.path.abspath(args.results_root)
    dir_a = _result_dir(root, args.tracker, args.expert_a, args.epoch_a)
    dir_b = _result_dir(root, args.tracker, args.expert_b, args.epoch_b)
    out_dir = _result_dir(root, args.tracker, args.output_param, args.output_epoch)

    for seq in dataset:
        box_a = _read_array(os.path.join(dir_a, seq.name + ".txt"), cols=4)
        box_b = _read_array(os.path.join(dir_b, seq.name + ".txt"), cols=4)
        if box_a is None or box_b is None:
            raise FileNotFoundError("Missing boxes for sequence {}".format(seq.name))
        n = min(len(box_a), len(box_b))
        box_a, box_b = box_a[:n], box_b[:n]

        score_a = _read_array(os.path.join(dir_a, seq.name + "_best_score.txt"), default=1.0, length=n)[:n]
        score_b = _read_array(os.path.join(dir_b, seq.name + "_best_score.txt"), default=1.0, length=n)[:n]
        cons_a = _read_array(os.path.join(dir_a, seq.name + "_template_consistency.txt"), default=0.5, length=n)[:n]
        cons_b = _read_array(os.path.join(dir_b, seq.name + "_template_consistency.txt"), default=0.5, length=n)[:n]
        rgb_a = _read_array(os.path.join(dir_a, seq.name + "_template_rgb_consistency.txt"), default=0.5, length=n)[:n]
        rgb_b = _read_array(os.path.join(dir_b, seq.name + "_template_rgb_consistency.txt"), default=0.5, length=n)[:n]
        dep_a = _read_array(os.path.join(dir_a, seq.name + "_template_depth_consistency.txt"), default=0.5, length=n)[:n]
        dep_b = _read_array(os.path.join(dir_b, seq.name + "_template_depth_consistency.txt"), default=0.5, length=n)[:n]
        ent_a = _read_array(os.path.join(dir_a, seq.name + "_response_entropy.txt"), default=0.0, length=n)[:n]
        center_follow_b = _read_array(os.path.join(dir_b, seq.name + "_center_follow.txt"), default=0.0, length=n)[:n]

        choose_b = _choose_mask(args.mode, score_a, score_b, cons_a, cons_b, rgb_a, rgb_b,
                                dep_a, dep_b, ent_a, center_follow_b,
                                args.delta, args.min_cons)
        boxes = np.where(choose_b[:, None], box_b, box_a)
        if args.score_policy == "max":
            scores = np.maximum(score_a, score_b)
        else:
            scores = np.where(choose_b, score_b, score_a)

        _write_array(os.path.join(out_dir, seq.name + ".txt"), boxes)
        _write_array(os.path.join(out_dir, seq.name + "_best_score.txt"), scores)

    tracker = Tracker(args.tracker, args.output_param, args.dataset_name, args.output_epoch)
    results = evaluate_depthtrack_prre([tracker], dataset, output_json=args.output_json)
    print(format_depthtrack_prre(results))


if __name__ == "__main__":
    main()
