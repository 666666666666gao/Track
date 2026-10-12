"""Compare complete captured CPU CDTB diagnoses; never run tracking or training."""
from pathlib import Path
import csv
import hashlib
import json


A = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
OLD = A / 'M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json'
NEW = A / 'M124_epoch03_CDTB_CPU_diagnostic_original_stdout_20261013.json'


def main():
    old, new = json.loads(OLD.read_bytes()), json.loads(NEW.read_bytes())
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == 'aa54318793d5666fa8c5aa7cc831ea6916a0dbed37082a9e2dfdea21b47a90b9'
    assert hashlib.sha256(NEW.read_bytes()).hexdigest() == '04fc5990f7fe6955d464a16e5a2243c2823c8e04095c7aa379b10085d35a79f9'
    assert old['status'] == 'actual_complete_M124_epoch02_CDTB_saved_trajectory_diagnosis'
    assert new['status'] == 'actual_complete_M124_epoch03_CDTB_saved_trajectory_diagnosis'
    assert old['sequences'] == new['sequences'] == 80 and old['frames'] == new['frames'] == 101956
    assert old['metric_source_sha256'] == new['metric_source_sha256'] == '05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
    assert new['analysis_source_sha256'] == '8f6c08fc5aa4d9f0119774622e3968108c978dd7d7a5f99d7541b71c7796ef6a'
    assert new['NN_calls'] == new['GPU_queries'] == new['training_progress_queries'] == 0
    before_rows = {row['sequence']: row for row in old['per_sequence']}
    after_rows = {row['sequence']: row for row in new['per_sequence']}
    assert len(before_rows) == len(after_rows) == 80 and set(before_rows) == set(after_rows)
    rows = []
    keys = ['P_percent', 'R_percent', 'all_saved_boxes_R_percent', 'confidence_R_loss_pp',
            'serious_low_IoU_frames', 'serious_low_IoU_crop_inside_frames', 'serious_low_IoU_crop_outside_frames']
    for name, before in before_rows.items():
        after = after_rows[name]
        assert before['frames'] == after['frames'] and before['valid_GT_frames'] == after['valid_GT_frames']
        assert before['GT_sha256'] == after['GT_sha256'] and before['image_shape'] == after['image_shape']
        row = dict(sequence=name, frames=after['frames'], valid_GT_frames=after['valid_GT_frames'], GT_sha256=after['GT_sha256'])
        for key in keys:
            row['epoch02_' + key] = before[key]
            row['epoch03_' + key] = after[key]
            row['delta_' + key] = after[key] - before[key]
        row['epoch02_long_low_segments'] = len(before['long_low_IoU_segments'])
        row['epoch03_long_low_segments'] = len(after['long_low_IoU_segments'])
        row['macro_all_R_delta_contribution_pp'] = row['delta_all_saved_boxes_R_percent'] / 80
        rows.append(row)
    before, after = old['aggregate'], new['aggregate']
    formal_delta = after['R'] - before['R']
    all_delta = after['all_saved_boxes_R'] - before['all_saved_boxes_R']
    old_loss = before['all_saved_boxes_R'] - before['R']
    new_loss = after['all_saved_boxes_R'] - after['R']
    assert abs(formal_delta - (all_delta - (new_loss - old_loss))) < 1e-10
    assert abs(sum(row['macro_all_R_delta_contribution_pp'] for row in rows) - all_delta) < 1e-8
    ordered = sorted(rows, key=lambda row: row['delta_all_saved_boxes_R_percent'])
    segments = [segment for row in after_rows.values() for segment in row['long_low_IoU_segments']]
    summary = dict(status='actual_local_M124_epoch03_vs_epoch02_CDTB_saved_comparison', sequences=80, frames=101956,
        epoch02=before, epoch03=after, formal_R_delta_pp=formal_delta, all_saved_boxes_R_delta_pp=all_delta,
        epoch02_confidence_R_loss_pp=old_loss, epoch03_confidence_R_loss_pp=new_loss,
        confidence_R_loss_change_pp=new_loss-old_loss, target_R_percent=75.6,
        epoch03_all_saved_boxes_R_gap_to_target_pp=75.6-after['all_saved_boxes_R'],
        largest8_all_R_losses=ordered[:8], largest8_all_R_gains=list(reversed(ordered[-8:])),
        improved_all_R_sequences=sum(row['delta_all_saved_boxes_R_percent'] > 0 for row in rows),
        worsened_all_R_sequences=sum(row['delta_all_saved_boxes_R_percent'] < 0 for row in rows),
        unchanged_all_R_sequences=sum(row['delta_all_saved_boxes_R_percent'] == 0 for row in rows),
        long_low_segments=len(segments), long_low_segment_frames=sum(segment['length'] for segment in segments),
        segment_onsets_center_inside=sum(segment['start_center_in_crop'] for segment in segments),
        segment_onsets_center_outside=sum(not segment['start_center_in_crop'] for segment in segments),
        inside_onsets_later_outside=sum(segment['start_center_in_crop'] and segment['center_outside_frames'] > 0 for segment in segments),
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [OLD, NEW]},
        CPU_diagnostic_input_count=len(new['inputs_sha256']), evaluation_type='real_gt',
        NN_calls=0, GPU_queries=0, server_queries=0, training_progress_queries=0,
        limits='Two own-history saved trajectories from successive trained epochs, not same-state candidate/text/module interventions. Formal per-sequence P/R uses each epoch dataset-wide best-F report threshold. All-box R is fixed-trajectory recall without report-score filtering, not deployable P/F or a recovery experiment. Macro contributions are arithmetic, not a unique network-cause diagnosis. GT validity does not establish physical absence. Crop reconstruction is not a bitwise hidden-state replay. No M124 VOT or full-nine completion claim.')
    out = A / 'M124_epoch03_CDTB_saved_comparison_20261013'
    assert not out.exists()
    out.mkdir()
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (out / 'all80_sequence_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ['status', 'formal_R_delta_pp', 'all_saved_boxes_R_delta_pp',
        'epoch02_confidence_R_loss_pp', 'epoch03_confidence_R_loss_pp', 'confidence_R_loss_change_pp',
        'epoch03_all_saved_boxes_R_gap_to_target_pp', 'improved_all_R_sequences', 'worsened_all_R_sequences',
        'unchanged_all_R_sequences', 'long_low_segments', 'segment_onsets_center_inside', 'segment_onsets_center_outside']}, indent=2))


if __name__ == '__main__':
    main()
