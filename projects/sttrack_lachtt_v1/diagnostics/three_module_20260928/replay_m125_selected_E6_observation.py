"""Selected E6 prediction stage; CPU GT scoring runs separately after sealing.

Subsequent GT is opened only after the requested sequence prefix is sealed.
Alternative scores/boxes never change the submitted state, query or templates.
"""
import argparse
from pathlib import Path
import hashlib
import json
import time

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
    stage = dict(status='complete_M125_selected_E6_prediction_stage',
        sequence=args.sequence, selected_epoch=selected_best['epoch'], frames=frames,
        args=dict(plan=str(args.plan), sequence=args.sequence, frames=frames, suite=str(args.suite), output=str(args.output)),
        plan=plan, bundle=bundle, case=case, selected_best=selected_best,
        plan_sha256=sha(args.plan), terminal_suite_result_sha256=sha(suite_result_path),
        prediction_path=str(prediction_path), prediction_sha256=prediction_sha,
        source_path=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        image_shape=list(image.shape[:2]), frozen_before_after_exact=True,
        AB_before_sha256=AB_before, C_before_sha256=C_before,
        current_GT_opened=False, optimizer_steps=0, tracker_calls=len(boxes)-1,
        extra_Empty_decoder_calls=len(snapshots), elapsed_seconds=time.time()-started)
    write(args.output / 'prediction_stage.json', stage)
    print(json.dumps({key: stage[key] for key in ['status','sequence','tracker_calls','extra_Empty_decoder_calls']}), flush=True)


if __name__ == '__main__':
    main()
