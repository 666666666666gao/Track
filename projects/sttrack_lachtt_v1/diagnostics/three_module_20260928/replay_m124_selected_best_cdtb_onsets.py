"""Read-only selected-best M124 own-history replay and same-state candidate diagnostics.

Subsequent GT is opened only after the requested sequence prefix is sealed.
Alternative scores/boxes never change the submitted state, query or templates.
"""
import argparse
from pathlib import Path
import hashlib
import importlib.util
import json
import time

import numpy as np
import torch

from bind_m122_official_initializations import sha
from m124_epochwise_official_runtime import checked_plan, OfficialDenseTracker
from train_template_write_C import state_digest



def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite_result_path = args.suite / 'result.json'
    suite = json.loads(suite_result_path.read_text())
    assert suite['status'] == 'complete_M124_epochwise_training_best_and_full_VOT'
    assert suite['formal_nine_metrics'] and suite['automatic_retries'] == 0
    assert int((args.suite / 'controller.exit').read_text()) == 0
    selected_best = json.loads((args.suite / 'best.json').read_text())
    assert suite['best'] == selected_best
    plan, bundle = checked_plan(args.plan)
    assert plan['dataset'] == 'cdtb'
    assert bundle['final_sha256'] == selected_best['final_sha256']
    assert args.plan.resolve() == (Path(selected_best['evaluation']) / 'cdtb/plan.json').resolve()
    complete = json.loads((Path(selected_best['evaluation']) / 'all_results.json').read_text())
    assert complete['status'] == 'three_M124_selected_epoch_full_evaluations_complete'
    assert len(complete['results']) == 3
    assert all(row['final_sha256'] == selected_best['final_sha256'] for row in complete['results'])
    assert sha(plan['cases_path']) == plan['cases_sha256']
    cases = json.loads(Path(plan['cases_path']).read_text())
    case = next(row for row in cases if row['sequence'] == args.sequence)
    frames = args.frames
    assert frames == sorted(set(frames)) and min(frames) >= 1
    assert max(frames) < case['frames']
    actor = OfficialDenseTracker(plan, bundle)
    frozen_before = actor.actor.frozen_digest()
    AB_before = state_digest(actor.actor.decoder)
    C_before = state_digest(actor.trusted.writer)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    args.output.mkdir(exist_ok=False)
    folder = Path(plan['dataset_root']) / args.sequence
    boxes = [case['init_bbox']]
    scores = [1.]
    actions = []
    snapshots = []
    started = time.time()
    for frame in range(max(frames) + 1):
        stem = '%08d' % (frame + 1)
        rgb = folder / 'color' / (stem + '.jpg')
        depth = folder / 'depth' / (stem + '.png')
        image = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
        if frame == 0:
            actor.initialize(image, case['init_bbox'], rgb)
            continue
        out, data, action = actor.trusted.step(image)
        boxes.append(action['bbox'])
        scores.append(action['best_score'])
        actions.append(action)
        if frame in frames:
            # Main action has already been submitted. All following alternatives are read-only.
            state_before = list(actor.actor.native.state)
            query_before = [value.clone() for value in actor.actor.native.track_query_before]
            feature_before = actor.actor.selected_feature.clone()
            templates_before = [value.clone() for value in actor.actor.native.z_dict]
            original_query = data['query']
            empty = data['empty_query'][None, None].expand_as(original_query)
            alternate = dict(data, query=empty)
            null_out = actor.actor.decoder(alternate)
            assert not bool(null_out['semantic_delta'].any())
            assert torch.equal(null_out['response'], out['visual_response'])
            assert torch.equal(null_out['raw_boxes'], out['raw_boxes'])
            quality = out['quality_logits'].sigmoid()[0]
            assert bool((quality > 0).all())
            # Use the unrounded native pixel boxes retained by the actor observation path.
            # data boxes are float32; report that representation explicitly rather than calling it exact native output.
            refined = out['boxes'][0]
            response = out['response'][0]
            native_response = data['native_response'][0]
            visual = out['visual_response'][0]
            score_without_quality = response / quality
            values = dict(frame=frame, image_shape=list(image.shape[:2]),
                crop_origin=data['origin'][0].cpu().tolist(), action=action,
                full_selected=int(out['selected_index'][0]), native_selected=int(native_response.argmax()),
                visual_selected=int(visual.argmax()), without_quality_selected=int(score_without_quality.argmax()),
                native_boxes_float32=data['boxes'][0].cpu().tolist(),
                refined_boxes=refined.cpu().tolist(),
                full_response=response.cpu().tolist(), native_response=native_response.cpu().tolist(),
                visual_response=visual.cpu().tolist(), full_response_without_quality=score_without_quality.cpu().tolist(),
                quality_probability=quality.cpu().tolist(),
                semantic_delta=out['semantic_delta'][0].cpu().tolist(),
                phrase_cosine=out['phrase_cosine'][0].cpu().tolist(),
                Empty_semantic_zero=True, Empty_response_equals_same_state_visual_response=True,
                geometry_same_with_Empty=True, RGB_sha256=sha(rgb), Depth_sha256=sha(depth))
            snapshots.append(values)
            assert list(actor.actor.native.state) == state_before
            assert all(torch.equal(left, right) for left, right in zip(actor.actor.native.track_query_before, query_before))
            assert torch.equal(actor.actor.selected_feature, feature_before)
            assert all(torch.equal(template, old) for template, old in zip(actor.actor.native.z_dict, templates_before))
    assert actor.actor.frozen_digest() == frozen_before
    assert state_digest(actor.actor.decoder) == AB_before and state_digest(actor.trusted.writer) == C_before
    checked_plan(args.plan)
    prediction_path = args.output / 'replayed_predictions.json'
    write(prediction_path, dict(sequence=args.sequence, boxes=boxes, scores=scores,
        actions=actions, sampled_candidates=snapshots, current_GT_opened=False, optimizer_steps=0,
        alternatives_submitted=False, state_after_alternatives_unchanged=True))
    prediction_sha = sha(prediction_path)
    # No current/future annotation was used to generate the above predictions or candidate tensors.
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
    height, width = image.shape[:2]
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
    result = dict(status='complete_readonly_M124_selected_best_CDTB_onset_replay', sequence=args.sequence,
        prefix_frames=len(boxes), tracker_calls=len(boxes)-1,
        selected_epoch=selected_best['epoch'], terminal_suite_result_sha256=sha(suite_result_path),
        terminal_suite_root=str(args.suite), required_full_training_and_VOT_terminal=True,
        frame_sampling_scope='Caller-specified posthoc diagnostic panel; never chooses a tracking candidate or resets state.', extra_Empty_decoder_calls=len(snapshots),
        candidate_states=len(snapshots), candidates_per_state=256, seed_scan=False, optimizer_steps=0,
        initialization_GT_bbox_only=True, subsequent_GT_used_after_predictions_sealed=True,
        no_GT_state_reset=True, alternatives_submitted=False, state_after_alternatives_unchanged=True,
        frozen_before_after_exact=True, original_plan_sha256=sha(args.plan), original_final_sha256=bundle['final_sha256'],
        original_bank_sha256=plan['bank_sha256'], metric_source_sha256=sha(metric_path),
        source_sha256=sha(__file__), replayed_prediction_sha256=prediction_sha,
        official_prediction_sha256=sha(reference), GT_sha256=sha(gt_path),
        bbox_max_abs_difference_from_saved=float(difference.max()),
        score_max_abs_difference_from_saved=float(score_difference.max()),
        all_prefix_boxes_match_saved_rounding=bool(equal_rows.all()),
        first_saved_bbox_difference_frame=None if not len(first_unequal) else int(first_unequal[0]),
        prefix_comparison='actual replay own history; no forcing saved boxes or retrying to get parity',
        native_geometry_representation='saved observation float32 candidate boxes; not exact float64 native output',
        causal_scope='Read-only score/geometry counterfactuals on selected-best own history; alternatives are same replay state only; if saved prefix differs, do not claim original state reproduction; no complete-trajectory rescue or text semantic truth claim',
        elapsed_seconds=time.time()-started, diagnostics=diagnostics)
    write(args.output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ['status','sequence','prefix_frames','candidate_states',
        'all_prefix_boxes_match_saved_rounding','bbox_max_abs_difference_from_saved','optimizer_steps']}), flush=True)


if __name__ == '__main__':
    main()
