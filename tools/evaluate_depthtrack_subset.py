#!/usr/bin/env python3
import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKING_ROOT = os.path.join(PROJECT_ROOT, "tracking")
for path in (PROJECT_ROOT, TRACKING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from lib.test.analysis.depthtrack_prre import evaluate_depthtrack_prre, format_depthtrack_prre
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def _result_base(tracker, seq_name):
    return os.path.join(tracker.results_dir, seq_name)


def _clean_sequence_outputs(tracker, sequences):
    suffixes = [
        ".txt",
        "_time.txt",
        "_best_score.txt",
        "_template_consistency.txt",
        "_template_rgb_consistency.txt",
        "_template_depth_consistency.txt",
        "_state_update_ratio.txt",
        "_score_ema.txt",
        "_effective_lost_thr.txt",
        "_effective_damping_thr.txt",
        "_response_entropy.txt",
        "_response_peak_ratio.txt",
        "_response_center_distance.txt",
        "_response_guard.txt",
        "_depth_guard.txt",
        "_center_follow.txt",
        "_window_penalty.txt",
        "_search_factor.txt",
        "_confident_mismatch_search.txt",
        "_candidate_rerank.txt",
        "_candidate_rank.txt",
        "_candidate_score_gain.txt",
        "_candidate_consistency_gain.txt",
        "_candidate_selected_iou.txt",
        "_candidate_oracle_iou.txt",
        "_candidate_oracle_rank.txt",
        "_candidate_oracle_score_ratio.txt",
        "_candidate_oracle_rgb_consistency.txt",
        "_candidate_oracle_depth_consistency.txt",
        "_candidate_oracle_consistency.txt",
        "_candidate_oracle_motion_score.txt",
        "_template_update.txt",
        "_candidate_recovery_trigger.txt",
        "_candidate_stable_block.txt",
        "_candidate_ordinary_trigger.txt",
        "_language_runtime_gate.txt",
        "_all_boxes.txt",
        "_all_scores.txt",
        "_template_update_reason.txt",
        "_template_update_gate_score.txt",
        "_template_update_gate_stable.txt",
        "_template_update_gate_consistency.txt",
        "_template_update_gate_rgb.txt",
        "_template_update_gate_depth.txt",
    ]
    for seq in sequences:
        base = _result_base(tracker, seq.name)
        for suffix in suffixes:
            path = base + suffix
            if os.path.isfile(path):
                os.remove(path)


def main():
    parser = argparse.ArgumentParser(description="Run a small DepthTrack subset evaluation.")
    parser.add_argument("tracker_name", nargs="?", default="mplt_track")
    parser.add_argument("tracker_param", nargs="?", default="vitb_256_mplt_32x1_1e4_depthtrack_15ep_roberta")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--dataset_name", default="depthtrack_test")
    parser.add_argument("--checkpoint_root", default="./output/depthtrack_roberta")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if args.checkpoint_root:
        os.environ["MPLT_CHECKPOINT_ROOT"] = os.path.abspath(args.checkpoint_root)

    dataset = get_dataset(args.dataset_name)
    if args.sequence:
        wanted = set(args.sequence)
        selected = [seq for seq in dataset if seq.name in wanted]
        missing = sorted(wanted - {seq.name for seq in selected})
        if missing:
            raise SystemExit("Unknown sequence(s): {}".format(", ".join(missing)))
    elif args.first > 0:
        selected = list(dataset[:args.first])
    else:
        selected = list(dataset)

    if not selected:
        raise SystemExit("No sequences selected")

    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.epoch)
    if args.clean:
        _clean_sequence_outputs(tracker, selected)

    print("selected sequences: {}".format(", ".join(seq.name for seq in selected)))
    run_dataset(selected, [tracker], debug=False, threads=args.threads, num_gpus=args.num_gpus)
    results = evaluate_depthtrack_prre([tracker], selected, output_json=args.output_json)
    print(format_depthtrack_prre(results))


if __name__ == "__main__":
    main()
