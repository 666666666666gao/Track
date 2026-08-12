import argparse
import os
import sys

import _init_paths

from lib.test.analysis.depthtrack_prre import evaluate_depthtrack_prre, format_depthtrack_prre
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_depthtrack_eval(tracker_name, tracker_param, epochs, dataset_name='depthtrack_test',
                        checkpoint_root=None, threads=0, num_gpus=1, output_json=None,
                        skip_run=False):
    if checkpoint_root:
        os.environ['MPLT_CHECKPOINT_ROOT'] = os.path.abspath(checkpoint_root)

    dataset = get_dataset(dataset_name)
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, str(epoch)) for epoch in epochs]

    if not skip_run:
        for tracker in trackers:
            run_dataset(dataset, [tracker], debug=False, threads=threads, num_gpus=num_gpus)

    results = evaluate_depthtrack_prre(trackers, dataset, output_json=output_json)
    print(format_depthtrack_prre(results))
    return results


def main():
    parser = argparse.ArgumentParser(description='Run DepthTrack test and report Pr/Re/F-score.')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--epochs', nargs='+', required=True)
    parser.add_argument('--dataset_name', type=str, default='depthtrack_test')
    parser.add_argument('--checkpoint_root', type=str, default=None)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--skip_run', action='store_true')
    args = parser.parse_args()

    run_depthtrack_eval(
        args.tracker_name,
        args.tracker_param,
        args.epochs,
        dataset_name=args.dataset_name,
        checkpoint_root=args.checkpoint_root,
        threads=args.threads,
        num_gpus=args.num_gpus,
        output_json=args.output_json,
        skip_run=args.skip_run,
    )


if __name__ == '__main__':
    main()
