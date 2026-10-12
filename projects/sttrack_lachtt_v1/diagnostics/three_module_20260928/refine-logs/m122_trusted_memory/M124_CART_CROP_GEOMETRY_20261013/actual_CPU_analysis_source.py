from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math
A=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
source=Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING\实验诊断\CDTB定位失败_20261012\onset_rows.json')
payload=json.loads(source.read_bytes())
assert payload['status']=='actual_CPU_saved_CDTB_onset_visual_diagnostic'
case=next(item for item in payload['results'] if item['sequence']=='cart_room_occ_1')
rows=[]
for item in case['rows']:
    assert item['GT_valid']
    previous=item['previous_prediction']
    crop=item['crop']
    assert crop[2]==crop[3] and crop[2]>0
    side=math.ceil(4*math.sqrt(previous[2]*previous[3]))
    expected=[round(previous[0]+previous[2]/2-side/2),round(previous[1]+previous[3]/2-side/2),side,side]
    assert expected==crop,(item['frame'],expected,crop)
    gt=item['GT_bbox']
    center=[(gt[0]+gt[2]/2-crop[0])/side,(gt[1]+gt[3]/2-crop[1])/side]
    contained=crop[0]<=gt[0] and crop[1]<=gt[1] and gt[0]+gt[2]<=crop[0]+side and gt[1]+gt[3]<=crop[1]+side
    rows.append(dict(frame=item['frame'],stored_IoU=item['IoU'],stored_score=item['score'],saved_crop=crop,GT_bbox=gt,GT_center_in_crop=item['GT_center_in_crop'],GT_entire_box_in_crop=contained,GT_center_normalized_to_crop=center,GT_width_fraction_of_crop=gt[2]/side,GT_height_fraction_of_crop=gt[3]/side,crop_matches_saved_prior_box_formula=True))
by_frame={row['frame']:row for row in rows}
summary={}
for before,after in [(175,176),(176,177)]:
    first,last=by_frame[before],by_frame[after]
    summary[str(before)+'_to_'+str(after)]=dict(crop_side_change_percent=100*(last['saved_crop'][2]/first['saved_crop'][2]-1),GT_width_fraction_change_percent=100*(last['GT_width_fraction_of_crop']/first['GT_width_fraction_of_crop']-1),GT_height_fraction_change_percent=100*(last['GT_height_fraction_of_crop']/first['GT_height_fraction_of_crop']-1),after_GT_center_normalized=last['GT_center_normalized_to_crop'],after_GT_entire_box_in_crop=last['GT_entire_box_in_crop'])
result=dict(status='actual_CPU_existing_cart_saved_crop_geometry',computed_at=datetime.now(timezone.utc).isoformat(),model_scope='historical CurrentC/P1 saved cart trajectory, not current M124 weights',sequence='cart_room_occ_1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,summary=summary,scope='Recompute saved crop geometry and GT extent ratios only. The prior-box formula matches the selected saved records; no resampled pixel tensor, attention, candidate ranking, or unique causal root verified. These ratios are geometric spans, not effective token counts.',NN_calls=0,GPU_queries=0,server_queries=0,training_progress_queries=0,human_labels_changed=False,running_training_sources_changed=False)
target=A/'M124_cart_saved_crop_geometry_20261013.json'
assert not target.exists()
target.write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(status=result['status'],checked_saved_frames=len(rows),summary=summary,source_sha256=result['source_sha256']),indent=2))