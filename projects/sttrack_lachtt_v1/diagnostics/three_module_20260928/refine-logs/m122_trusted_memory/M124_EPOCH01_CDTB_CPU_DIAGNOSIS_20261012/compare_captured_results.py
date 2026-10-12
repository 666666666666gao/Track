"""Local comparison of captured real-GT CPU diagnoses; no SSH/NN."""
from pathlib import Path
import csv,hashlib,json
A=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
OLD=A/'CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json'
NEW=A/'M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json'
old=json.loads(OLD.read_bytes());new=json.loads(NEW.read_bytes())
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='bcdc6791cf67d71b77e08cacad8b91cefed1c7a89260f98ade04183dfe64de72'
assert old['status']=='actual_complete_CDTB_saved_trajectory_recall_diagnosis'
assert new['status']=='actual_complete_M124_epoch01_CDTB_saved_trajectory_diagnosis'
assert old['sequences']==new['sequences']==80 and old['frames']==new['frames']==101956
assert old['metric_source_sha256']==new['metric_source_sha256']=='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
assert new['NN_calls']==new['GPU_queries']==new['training_progress_queries']==0
old_rows={r['sequence']:r for r in old['per_sequence'] if r['model']=='CurrentC'}
new_rows={r['sequence']:r for r in new['per_sequence']}
assert len(old_rows)==len(new_rows)==80 and set(old_rows)==set(new_rows)
rows=[]
for name in old_rows:
    before,after=old_rows[name],new_rows[name]
    assert before['frames']==after['frames'] and before['valid_GT_frames']==after['valid_GT_frames']
    gt_hashes=[value for path,value in old['inputs_sha256'].items() if path.endswith('/'+name+'/groundtruth.txt')]
    assert len(gt_hashes)==1 and gt_hashes[0]==after['GT_sha256']
    item=dict(sequence=name,frames=after['frames'],valid_GT_frames=after['valid_GT_frames'],GT_sha256=after['GT_sha256'])
    for key in ['P_percent','R_percent','all_saved_boxes_R_percent','confidence_R_loss_pp','serious_low_IoU_frames','serious_low_IoU_crop_inside_frames','serious_low_IoU_crop_outside_frames']:
        item['CurrentC_'+key]=before[key];item['M124_epoch01_'+key]=after[key];item['delta_'+key]=after[key]-before[key]
    item['CurrentC_long_low_segments']=len(before['long_low_IoU_segments'])
    item['M124_epoch01_long_low_segments']=len(after['long_low_IoU_segments'])
    item['macro_all_R_delta_contribution_pp']=item['delta_all_saved_boxes_R_percent']/80.
    rows.append(item)
before=old['aggregate']['CurrentC'];after=new['aggregate']
R_delta=after['R']-before['R'];all_delta=after['all_saved_boxes_R']-before['all_saved_boxes_R']
old_screen_loss=before['all_saved_boxes_R']-before['R'];new_screen_loss=after['all_saved_boxes_R']-after['R']
assert abs(R_delta-(all_delta-(new_screen_loss-old_screen_loss)))<1e-10
assert abs(sum(r['macro_all_R_delta_contribution_pp'] for r in rows)-all_delta)<1e-8
segments=[s for r in new_rows.values() for s in r['long_low_IoU_segments']]
sorted_rows=sorted(rows,key=lambda r:r['delta_all_saved_boxes_R_percent'])
top_worse=sorted_rows[:8];top_better=list(reversed(sorted_rows[-8:]))
summary=dict(status='actual_local_M124_epoch01_vs_CurrentC_CDTB_saved_comparison',sequences=80,frames=101956,CurrentC=before,M124_epoch01=after,formal_R_delta_pp=R_delta,all_saved_boxes_R_delta_pp=all_delta,CurrentC_confidence_R_loss_pp=old_screen_loss,M124_epoch01_confidence_R_loss_pp=new_screen_loss,confidence_R_loss_change_pp=new_screen_loss-old_screen_loss,target_R_percent=75.6,M124_epoch01_all_saved_boxes_R_gap_to_target_pp=75.6-after['all_saved_boxes_R'],largest8_all_R_gains=top_better,largest8_all_R_losses=top_worse,improved_all_R_sequences=sum(r['delta_all_saved_boxes_R_percent']>0 for r in rows),worsened_all_R_sequences=sum(r['delta_all_saved_boxes_R_percent']<0 for r in rows),unchanged_all_R_sequences=sum(r['delta_all_saved_boxes_R_percent']==0 for r in rows),long_low_segments=len(segments),long_low_segment_frames=sum(s['length'] for s in segments),segment_onsets_center_inside=sum(s['start_center_in_crop'] for s in segments),segment_onsets_center_outside=sum(not s['start_center_in_crop'] for s in segments),inside_onsets_later_outside=sum(s['start_center_in_crop'] and s['center_outside_frames']>0 for s in segments),valid_GT_reappearance_events=sum(r['valid_GT_reappearance_count'] for r in new_rows.values()),valid_GT_reappearance_center_outside_events=sum(r['reappearance_center_outside_count'] for r in new_rows.values()),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD,NEW]},NN_calls=0,GPU_queries=0,server_queries=0,training_progress_queries=0,limits='Both own-history saved trajectories with different trained weights/training protocols, not a paired module or text-content intervention. Per-sequence formal values use each model dataset-wide best-F threshold; all-box R comparisons remove this report-confidence difference. Aggregate contribution arithmetic is not deployable recovery or evidence of a unique causal mechanism. GT validity changes do not establish physical disappearance. No result from incomplete epoch02 or M124 VOT.')
out=A/'M124_epoch01_CDTB_saved_comparison_20261012'
assert not out.exists();out.mkdir()
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (out/'all80_sequence_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({key:summary[key] for key in ['status','formal_R_delta_pp','all_saved_boxes_R_delta_pp','CurrentC_confidence_R_loss_pp','M124_epoch01_confidence_R_loss_pp','M124_epoch01_all_saved_boxes_R_gap_to_target_pp','improved_all_R_sequences','worsened_all_R_sequences','unchanged_all_R_sequences','long_low_segments','segment_onsets_center_inside','segment_onsets_center_outside']},indent=2))
