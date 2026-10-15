"""Summarize archived E6 diagnostics; continuous box coverage is not visibility."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def intersection_area(rectangles):
    return max(0., min(r[2] for r in rectangles) - max(r[0] for r in rectangles)) * max(
        0., min(r[3] for r in rectangles) - max(r[1] for r in rectangles))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.archive.read_bytes()
    archive = json.loads(raw)
    assert archive['status'] == 'actual_complete_M125_R2_replay_primary_inputs'
    assert archive['formal_suite_unchanged'] and archive['NN_calls'] == 0
    files = archive['files']
    for row in files.values():
        assert digest(row['text'].encode()) == row['sha256']
    root = '/root/autodl-tmp/m125_selected_E6_split_replay_R2_20261015'
    terminal = json.loads(files[root + '/result.json']['text'])
    assert files[root + '/controller.exit']['text'].strip() == '0'
    summaries, states = [], []
    for record in terminal['results']:
        sequence = record['sequence']
        prefix = root + '/' + sequence + '/'
        stage = json.loads(files[prefix + 'prediction_stage.json']['text'])
        prediction = json.loads(files[prefix + 'replayed_predictions.json']['text'])
        result = json.loads(files[prefix + 'result.json']['text'])
        assert stage['prediction_sha256'] == result['replayed_prediction_sha256'] == files[
            prefix + 'replayed_predictions.json']['sha256']
        assert result['all_prefix_boxes_match_saved_rounding']
        assert not prediction['current_GT_opened'] and not prediction['alternatives_submitted']
        assert prediction['state_after_alternatives_unchanged'] and prediction['optimizer_steps'] == 0
        gt_path = stage['plan']['dataset_root'] + '/' + sequence + '/groundtruth.txt'
        assert files[gt_path]['sha256'] == stage['case']['gt_sha256'] == result['GT_sha256']
        truth = [[float(value) for value in line.split(',')] for line in files[gt_path]['text'].splitlines()]
        snapshots = {row['frame']: row for row in prediction['sampled_candidates']}
        for diagnostic in result['diagnostics']:
            frame = diagnostic['frame']
            snapshot = snapshots[frame]
            assert len(snapshot['native_boxes_float32']) == len(snapshot['refined_boxes']) == 256
            row = dict(sequence=sequence, frame=frame, GT_valid=diagnostic['GT_valid'],
                       GT_center_in_crop=diagnostic['GT_center_in_crop'],
                       selected_IoU=diagnostic['replayed_submitted_IoU'],
                       native_best_IoU=diagnostic['native_candidate_best_IoU'],
                       refined_best_IoU=diagnostic['refined_candidate_best_IoU'],
                       native_count_ge_05=diagnostic['native_candidate_IoU_ge_05'],
                       refined_count_ge_05=diagnostic['refined_candidate_IoU_ge_05'],
                       Empty_selected_IoU=diagnostic['Empty_score_refined_geometry_IoU'],
                       without_quality_selected_IoU=diagnostic['without_quality_score_refined_geometry_IoU'],
                       selected_quality=diagnostic['selected_quality'],
                       crop_origin=snapshot['crop_origin'], image_shape=snapshot['image_shape'])
            if diagnostic['GT_valid']:
                x, y, width, height = truth[frame]
                gt_rectangle = (x, y, x + width, y + height)
                image_height, image_width = snapshot['image_shape']
                image_rectangle = (0., 0., float(image_width), float(image_height))
                crop_x, crop_y, crop_size = snapshot['crop_origin']
                crop_rectangle = (crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)
                area_in_image = intersection_area([gt_rectangle, image_rectangle])
                assert area_in_image > 0
                area_in_crop = intersection_area([gt_rectangle, image_rectangle, crop_rectangle])
                row.update(GT_box=truth[frame], GT_box_area_in_image=area_in_image,
                           GT_box_area_in_image_and_crop=area_in_crop,
                           GT_box_crop_coverage_fraction=area_in_crop / area_in_image,
                           GT_width_in_256_search=width * 256 / crop_size,
                           GT_height_in_256_search=height * 256 / crop_size)
            states.append(row)
        summaries.append({key: result[key] for key in ['sequence', 'tracker_calls', 'prefix_frames',
            'candidate_states', 'bbox_max_abs_difference_from_saved', 'score_max_abs_difference_from_saved',
            'all_prefix_boxes_match_saved_rounding', 'first_saved_bbox_difference_frame']})
    valid = [row for row in states if row['GT_valid']]
    value = dict(status='complete_static_M125_R2_box_coverage_summary',
        source_sha256=digest(Path(__file__).read_bytes()), archive_sha256=digest(raw),
        formal_suite_result_sha256=archive['formal_suite_result_sha256'],
        scope='Four posthoc CDTB own-history prefixes; 25 selected states, not a dataset rate or first-error causal study.',
        coverage_definition='Continuous rectangular area(GT intersect image intersect crop) / area(GT intersect image); no occlusion, pixel mask, depth validity or instance visibility inference.',
        geometry_metric='IoUs copied from locked official raster-metric diagnostic outputs; this summarizer does not recompute raster IoU.',
        prefix_comparison='Bounding boxes match saved six-decimal rounding; no original hidden-state attestation or alternative-trajectory rescue.',
        counts=dict(sequences=len(summaries), tracker_calls=sum(r['tracker_calls'] for r in summaries),
            prefix_rows=sum(r['prefix_frames'] for r in summaries), candidate_states=len(states),
            candidates_per_geometry=len(states) * 256, GT_valid_states=len(valid),
            GT_invalid_states=len(states) - len(valid),
            valid_states_without_refined_candidate_ge_05=sum(r['refined_count_ge_05'] == 0 for r in valid)),
        sequences=summaries, states=states, NN_calls=0, optimizer_steps=0)
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(value['counts']))
    for row in valid:
        print(row['sequence'], row['frame'], 'coverage', round(row['GT_box_crop_coverage_fraction'], 6),
              'target_px', round(row['GT_width_in_256_search'], 2), round(row['GT_height_in_256_search'], 2),
              'best_IoU', round(row['refined_best_IoU'], 6))


if __name__ == '__main__':
    main()
