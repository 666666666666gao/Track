"""Hand-computed geometry checks, independent of collected experiment GT."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    path = parser.parse_args().source
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location('m53_analysis', path)
    analysis = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(analysis)
    target = np.array([0., 0., 10., 10.])
    miss = [30., 0., 10., 10.]
    near = [1., 0., 10., 10.]
    partial = [2., 0., 10., 10.]

    def view(frame, box, second=None):
        candidates = [dict(bbox=box)]*10
        if second is not None:
            candidates[1] = dict(bbox=second)
        return dict(template_frame=frame, bbox=box, candidates=candidates)

    control = view(50, miss)
    del control['template_frame']
    event = dict(active_template_frame=50, baseline=control,
                 alternatives=[view(10, near), view(20, partial, target.tolist()), view(40, miss)])
    value = analysis.event_capacity(event, target, {50: 0., 10: .7, 20: 0., 40: None})
    all_past, trusted = value['modes']['all_past'], value['modes']['valid_past']
    assert all_past['oracle_top1_iou'] == 9/11 and all_past['union_top10_iou'] == 1.
    assert all_past['best_top1_template_frame'] == 10 and all_past['best_top10_template_frame'] == 20
    assert trusted['oracle_top1_iou'] == trusted['union_top10_iou'] == 9/11
    assert value['past_reads'] == 3 and value['valid_past_reads'] == 1
    assert value['harmful_past_reads'] == 0
    # Keeping the current control must not hide harmful individual past reads.
    control = view(50, target.tolist())
    del control['template_frame']
    healthy = analysis.event_capacity(dict(active_template_frame=50, baseline=control,
                                          alternatives=[view(10, miss)]), target, {50: 0., 10: 1.})
    assert healthy['modes']['valid_past']['oracle_top1_iou'] == 1.
    assert healthy['harmful_past_reads'] == healthy['harmful_valid_past_reads'] == 1
    assert healthy['healthy_past_reads'] == healthy['healthy_valid_past_reads'] == 1
    rows = [dict(sequence='a', valid_gt=True, **value), dict(sequence='b', valid_gt=True, **healthy),
            dict(sequence='a', valid_gt=False)]
    summary = analysis.summarize(rows, 'all_past')
    assert summary['events'] == 3 and summary['valid_events'] == 2 and summary['invalid_gt_events'] == 1
    assert summary['severe_events_recovered'] == 1 and summary['recovered_sequences'] == ['a']
    assert summary['new_candidate_events'] == 1 and summary['current_candidate_rank_improved'] == 0
    assert abs(summary['mean_oracle_top1_iou'] - 10/11) < 1e-15
    assert analysis.summarize([rows[-1]], 'default')['mean_oracle_top1_iou'] is None
    assert not analysis.valid_gt(np.array([0., 0., float('nan'), 10.]))
    assert not analysis.valid_gt(np.array([0., 0., 0., 10.]))
    # A correct current secondary candidate is ranking capacity, not new capacity.
    control = view(50, miss, target.tolist())
    del control['template_frame']
    ranking = analysis.event_capacity(dict(active_template_frame=50, baseline=control,
                                          alternatives=[view(10, near)]), target, {50: 1., 10: 1.})
    rank_summary = analysis.summarize([dict(sequence='c', valid_gt=True, **ranking)], 'all_past')
    assert rank_summary['current_candidate_rank_improved'] == 1 and rank_summary['new_candidate_events'] == 0
    json.dumps([value, healthy, summary, rank_summary], allow_nan=False)
    print(json.dumps(dict(status='PASS', analysis_source_sha256=analysis.sha(path),
                          checks=['separate_top1_top10_views', 'historical_write_gt_filter', 'control_retained',
                                  'individual_healthy_harm', 'invalid_gt_exclusion', 'rank_vs_new_capacity',
                                  'finite_json_serialization'], experiment_gt_opened=False), indent=2))


if __name__ == '__main__':
    main()
