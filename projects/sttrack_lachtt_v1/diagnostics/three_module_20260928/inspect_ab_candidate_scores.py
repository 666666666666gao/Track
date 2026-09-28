"""Read M99 final scores on all cached Train events; no policy or training."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import batch, load_inputs, summarize


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for tag in row['strata']:
            groups[tag].append(row)
    return {tag:dict(events=len(values),
                     quality_argmax_iou50=sum(r['quality_choice_iou'] >= .5 for r in values),
                     quality_argmax_mean_iou=sum(r['quality_choice_iou'] for r in values)/len(values),
                     quality_rescues=sum(r['native_iou'] < .5 <= r['quality_choice_iou'] for r in values),
                     quality_breaks=sum(r['quality_choice_iou'] < .5 <= r['native_iou'] for r in values),
                     native_correct_selection_changed=sum(r['native_iou'] >= .5 and r['selected'] != 0 for r in values),
                     mean_candidate_quality_abs_iou_error=sum(r['quality_abs_iou_error'] for r in values)/len(values))
            for tag,values in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path('/root/autodl-tmp')
    result = json.loads(args.result.read_text())
    assert result['status'] == 'complete_visual_control'
    assert sha(args.weights) == result['final_weights_sha256']
    panels = load_inputs(root/'sttrack_m90_train_states_20260928',
                         root/'sttrack_m98_train_contexts_20260928',
                         root/'sttrack_m95_initial_origins_20260928', False)
    empty = torch.load(root/'sttrack_full152_paired_20260925/text_full152.pt', map_location='cpu')['empty'].cuda().float()
    model = InstanceCandidatePrototype().cuda()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()
    prior_development = {row['key']:row for row in result['development_rows']}
    all_rows, statistics = [], {}
    with torch.no_grad():
        for split in ('fit',) if args.smoke else ('fit', 'development'):
            panel = panels[split]
            rows = []
            count = 64 if args.smoke else len(panel['key'])
            for start in range(0, count, 64):
                indices = torch.arange(start, min(start+64, count))
                output = model(batch(panel, indices, empty, torch.device('cuda')))
                assert torch.equal(output['selection_logits'], output['visual_selection_logits'])
                score = output['selection_logits'].cpu()
                quality = output['quality_logits'].sigmoid().cpu()
                assert bool(torch.isfinite(score).all()) and bool(torch.isfinite(quality).all())
                choices = output['selected_index'].cpu()
                quality_choices = output['quality_logits'].argmax(-1).cpu()
                for offset,index in enumerate(indices.tolist()):
                    iou = panel['iou'][index]
                    native = panel['base_scores'][index]
                    residual = score[offset]-native
                    selected, quality_selected = int(choices[offset]), int(quality_choices[offset])
                    row = dict(key=panel['key'][index], strata=panel['strata'][index],
                               native_iou=float(iou[0]), selected_iou=float(iou[selected]),
                               oracle_iou=float(iou.max()), selected=selected)
                    if split == 'development':
                        assert row == prior_development[row['key']]
                    row.update(split=split, quality_choice=quality_selected,
                               quality_choice_iou=float(iou[quality_selected]),
                               quality_abs_iou_error=float((quality[offset]-iou).abs().mean()),
                               native_gap_to_selected=float(native[0]-native[selected]),
                               learned_residual_advantage=float(residual[selected]-residual[0]),
                               selection_gap_over_native=float(score[offset,selected]-score[offset,0]),
                               native_geometry=panel['geometry'][index,0].tolist(),
                               selected_geometry=panel['geometry'][index,selected].tolist(),
                               candidates=[dict(index=k, box=panel['boxes'][index,k].tolist(),
                                                native_log_hann=float(native[k]),
                                                selection_logit=float(score[offset,k]),
                                                learned_residual=float(residual[k]),
                                                estimated_iou=float(quality[offset,k]), gt_iou=float(iou[k]))
                                           for k in range(10)])
                    rows.append(row)
            selection = summarize(rows)
            if not args.smoke:
                assert selection == result[split]
            statistics[split] = dict(selection=selection, quality_readout=summaries(rows))
            all_rows.extend(rows)
    assert len(all_rows) == (64 if args.smoke else 3039)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'events.jsonl').open('w') as stream:
        for row in all_rows:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    report = dict(status='complete_candidate_score_sanity_only' if args.smoke else 'complete_candidate_score_diagnostic_only',
                  events=len(all_rows), source_sha256=sha(Path(__file__)),
                  final_weights_sha256=result['final_weights_sha256'], statistics=statistics,
                  native_score_is_log_hann_response_not_log_odds=True,
                  quality_argmax_is_readout_only_not_a_new_runtime_policy=True,
                  no_optimization=True, no_checkpoint=True, no_semantic_identity_labels=True,
                  no_tracker_state_or_public_evaluation=True,
                  development_breaks=[r for r in all_rows if r['split']=='development'
                                      and r['native_iou'] >= .5 > r['selected_iou']])
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='development_breaks'}), flush=True)


if __name__ == '__main__':
    main()
