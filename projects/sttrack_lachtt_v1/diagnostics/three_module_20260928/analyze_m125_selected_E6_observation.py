"""CPU-only scoring of a sealed selected-E6 read-only prediction stage."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import time
import numpy as np
from bind_m122_official_initializations import sha


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parsed=parser.parse_args()
    stage=json.loads(parsed.stage.read_text())
    assert stage['status']=='complete_M125_selected_E6_prediction_stage'
    assert stage['current_GT_opened'] is False and stage['optimizer_steps']==0
    assert stage['frozen_before_after_exact']
    assert sha(stage['source_path'])==stage['source_sha256']
    args=SimpleNamespace(**stage['args'])
    args.plan=Path(args.plan);args.suite=Path(args.suite);args.output=Path(args.output)
    assert args.output.resolve()==parsed.stage.parent.resolve()
    assert sha(args.plan)==stage['plan_sha256']
    suite_result_path=args.suite/'result.json'
    assert sha(suite_result_path)==stage['terminal_suite_result_sha256']
    selected_best=stage['selected_best'];plan=stage['plan'];bundle=stage['bundle'];case=stage['case'];frames=stage['frames']
    prediction_path=Path(stage['prediction_path']);prediction_sha=stage['prediction_sha256']
    assert sha(prediction_path)==prediction_sha
    predicted=json.loads(prediction_path.read_text())
    assert predicted['current_GT_opened'] is False and predicted['optimizer_steps']==0
    boxes=predicted['boxes'];scores=predicted['scores'];snapshots=predicted['sampled_candidates']
    folder=Path(plan['dataset_root'])/args.sequence
    started=time.time()
    metric_path = Path(plan['metric_source'])
    assert sha(metric_path) == plan['metric_source_sha256'] == '05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
    spec = importlib.util.spec_from_file_location('locked_M124_selected_best_onset_metric', metric_path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    gt_path = folder / 'groundtruth.txt'
    assert sha(gt_path) == case['gt_sha256']
    gt = metric._load_rows(gt_path, 4)
    reference = Path(plan['output']) / (args.sequence + '.txt')
    reference_scores = Path(plan['output']) / (args.sequence + '_all_scores.txt')
    receipt = json.loads((Path(plan['output']) / 'receipt.json').read_text())
    assert receipt['plan_sha256'] == sha(args.plan)
    assert receipt['final_sha256'] == bundle['final_sha256']
    assert receipt['bank_sha256'] == plan['bank_sha256'] and receipt['bundle_sha256'] == plan['bundle_sha256']
    proof = next(row for row in receipt['sequences'] if row['sequence'] == args.sequence)
    assert sha(reference) == proof['bbox_sha256'] and sha(reference_scores) == proof['confidence_sha256']
    previous_boxes = metric._load_rows(reference, 4)[:len(boxes)]
    previous_scores = metric._load_rows(reference_scores, 1).reshape(-1)[:len(scores)]
    height, width = stage['image_shape']
    actual = np.asarray(boxes)
    overlap, valid = metric._vot_overlaps(actual, gt[:len(boxes)], width, height)
    saved_overlap, saved_valid = metric._vot_overlaps(previous_boxes, gt[:len(boxes)], width, height)
    difference = np.abs(actual - previous_boxes)
    equal_rows = (difference <= 5.01e-7).all(1)
    score_difference = np.abs(np.asarray(scores) - previous_scores)
    first_unequal = np.flatnonzero(~equal_rows)
    diagnostics = []
    for snapshot in snapshots:
        frame = snapshot['frame']
        native = np.asarray(snapshot['native_boxes_float32'])
        refined = np.asarray(snapshot['refined_boxes'])
        repeated_gt = np.repeat(gt[frame:frame + 1], 256, axis=0)
        native_iou, candidate_valid = metric._vot_overlaps(native, repeated_gt, width, height)
        refined_iou, _ = metric._vot_overlaps(refined, repeated_gt, width, height)
        full = snapshot['full_selected']
        base = snapshot['native_selected']
        visual = snapshot['visual_selected']
        no_quality = snapshot['without_quality_selected']
        best = int(refined_iou.argmax())
        ranking = np.argsort(-np.asarray(snapshot['full_response']), kind='stable')
        diagnostics.append(dict(frame=frame, GT_valid=bool(valid[frame]),
            replayed_submitted_IoU=float(overlap[frame]), saved_submitted_IoU=float(saved_overlap[frame]),
            GT_center_in_crop=bool(snapshot['crop_origin'][0] <= gt[frame,0] + .5*gt[frame,2] < snapshot['crop_origin'][0] + snapshot['crop_origin'][2]
                and snapshot['crop_origin'][1] <= gt[frame,1] + .5*gt[frame,3] < snapshot['crop_origin'][1] + snapshot['crop_origin'][2]) if valid[frame] else None,
            native_candidate_best_IoU=float(native_iou.max()), refined_candidate_best_IoU=float(refined_iou.max()),
            native_candidate_IoU_ge_05=int((native_iou >= .5).sum()), refined_candidate_IoU_ge_05=int((refined_iou >= .5).sum()),
            native_score_native_geometry_IoU=float(native_iou[base]), native_score_refined_geometry_IoU=float(refined_iou[base]),
            full_score_native_geometry_IoU=float(native_iou[full]), full_score_refined_geometry_IoU=float(refined_iou[full]),
            Empty_score_refined_geometry_IoU=float(refined_iou[visual]), without_quality_score_refined_geometry_IoU=float(refined_iou[no_quality]),
            best_refined_candidate_score_rank=int(np.flatnonzero(ranking == best)[0]) + 1,
            selected_semantic_delta=float(snapshot['semantic_delta'][full]), best_refined_candidate_semantic_delta=float(snapshot['semantic_delta'][best]),
            selected_quality=float(snapshot['quality_probability'][full]), best_refined_candidate_quality=float(snapshot['quality_probability'][best]),
            native_iou_by_candidate=native_iou.tolist(), refined_iou_by_candidate=refined_iou.tolist()))
    assert sha(prediction_path) == prediction_sha
    result = dict(status='complete_readonly_M125_selected_E6_CDTB_observation', sequence=args.sequence,
        prefix_frames=len(boxes), tracker_calls=len(boxes)-1,
        selected_epoch=selected_best['epoch'], terminal_suite_result_sha256=sha(suite_result_path),
        terminal_suite_root=str(args.suite), required_full_training_and_VOT_terminal=True,
        frame_sampling_scope='Caller-specified posthoc diagnostic panel; never chooses a tracking candidate or resets state.', extra_Empty_decoder_calls=len(snapshots),
        candidate_states=len(snapshots), candidates_per_state=256, seed_scan=False, optimizer_steps=0,
        initialization_GT_bbox_only=True, subsequent_GT_used_after_predictions_sealed=True,
        no_GT_state_reset=True, alternatives_submitted=False, state_after_alternatives_unchanged=True,
        frozen_before_after_exact=True, original_plan_sha256=sha(args.plan), original_final_sha256=bundle['final_sha256'],
        original_bank_sha256=plan['bank_sha256'], metric_source_sha256=sha(metric_path),
        source_sha256=dict(prediction=stage['source_sha256'],analysis=sha(__file__)), replayed_prediction_sha256=prediction_sha,
        official_prediction_sha256=sha(reference), GT_sha256=sha(gt_path),
        bbox_max_abs_difference_from_saved=float(difference.max()),
        score_max_abs_difference_from_saved=float(score_difference.max()),
        all_prefix_boxes_match_saved_rounding=bool(equal_rows.all()),
        first_saved_bbox_difference_frame=None if not len(first_unequal) else int(first_unequal[0]),
        prefix_comparison='actual replay own history; no forcing saved boxes or retrying to get parity',
        native_geometry_representation='saved observation float32 candidate boxes; not exact float64 native output',
        causal_scope='Read-only score/geometry counterfactuals on selected-best own history; alternatives are same replay state only; if saved prefix differs, do not claim original state reproduction; no complete-trajectory rescue or text semantic truth claim',
        NN_elapsed_seconds=stage['elapsed_seconds'], CPU_elapsed_seconds=time.time()-started, elapsed_seconds=stage['elapsed_seconds']+time.time()-started, diagnostics=diagnostics)
    write(args.output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ['status','sequence','prefix_frames','candidate_states',
        'all_prefix_boxes_match_saved_rounding','bbox_max_abs_difference_from_saved','optimizer_steps']}), flush=True)


if __name__ == '__main__':
    main()
