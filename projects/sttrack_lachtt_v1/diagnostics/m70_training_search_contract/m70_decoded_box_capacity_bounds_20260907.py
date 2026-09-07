"""Conservative geometry-only IoU bounds for the actual sigmoid-size/clipped M70 decoder."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

BASE=Path('/root/autodl-tmp');PARENT=BASE/'sttrack_m65_category_null_support_20260907'
INVENTORY=BASE/'sttrack_m70_recovery_window_inventory_20260907';ROOT=INVENTORY/'training_search_contract'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    assert sha(INVENTORY/'result.json')=='8822782394e73154a034e71d783b8eb4e8b27c3302b37ad94e06d9f029b9f17a'
    frozen=read(INVENTORY/'result.json');assert sha(INVENTORY/'geometry_events.json')==frozen['events_sha256']
    assert not (INVENTORY/'candidate_capacity/result.json').exists()
    manifest=read(PARENT/'integration.json')
    assert sha(PARENT/'integration.json')=='5881ef1594929f0e71b0f7e8da6564e1c4669cbef86b1986b2948d7e0f231a65'
    names=['lib/models/layers/head.py','lib/test/tracker/sttrack_lachtt_observation.py','lib/utils/box_ops.py']
    for n in names:assert sha(PARENT/'code'/n)==manifest['source_sha256'][n]
    head=(PARENT/'code'/names[0]).read_text()
    assert 'return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset' in head
    assert 'torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)' in head
    mapping=(PARENT/'code'/names[1]).read_text();assert 'clip_box(mapped, height, width, margin=10)' in mapping
    events=read(INVENTORY/'geometry_events.json')['events'];assert len(events)==216
    rows=[]
    for e in events:
        x,y,w,h=e['GT_bbox'];W=e['image_width'];H=e['image_height'];assert w>0 and h>0
        visible_w=max(0.,min(x+w,W)-max(x,0.));visible_h=max(0.,min(y+h,H)-max(y,0.))
        sides=dict(local_grid=e['arms']['null']['local']['rectangle'][2],wide=e['arms']['null']['wide']['rectangle'][2],coarse=max(W,H))
        values={}
        for kind,side in sides.items():
            assert side>=10
            size_bound=min(w,side)*min(h,side)/(w*h)
            clipped_bound=min(visible_w,side)*min(visible_h,side)/(w*h)
            assert 0<=clipped_bound<=size_bound<=1
            # A freely positioned rectangle inside the visible GT, with width/height capped at side,
            # attains this intersection/GT-area value if min-size and all learned constraints are relaxed.
            bw=min(visible_w,side);bh=min(visible_h,side);area=bw*bh
            independently_formed_iou=area/(w*h+area-area)
            assert independently_formed_iou==clipped_bound
            values[kind]=dict(side=side,size_only_upper_bound=size_bound,after_image_clipping_upper_bound=clipped_bound,
                size_alone_rules_out_IoU_half=size_bound<.5,image_and_size_rule_out_IoU_half=clipped_bound<.5)
        rows.append(dict(sequence=e['sequence'],frame=e['frame'],tags=e['tags'],
            local_centre_inside=e['arms']['null']['local']['centre_inside'],GT_centre_inside_image=e['GT_vs_image']['centre_inside'],
            GT_wh=[w,h],GT_visible_wh=[visible_w,visible_h],bounds=values))
    summaries={}
    for group in ['all','H10','H10_local_outside','valid_after_invalid','healthy']:
        selected=[r for r in rows if group=='all' or group in r['tags'] or (group=='H10_local_outside' and 'H10' in r['tags'] and not r['local_centre_inside'])]
        summaries[group]=dict(events=len(selected),bounds={k:dict(size_alone_impossible=sum(r['bounds'][k]['size_alone_rules_out_IoU_half'] for r in selected),
            image_and_size_impossible=sum(r['bounds'][k]['image_and_size_rule_out_IoU_half'] for r in selected)) for k in sides})
    impossible=[r for r in rows if any(v['image_and_size_rule_out_IoU_half'] for v in r['bounds'].values())]
    result=dict(status='completed_conservative_decoder_IoU_capacity_bounds',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),geometry_sha256=sha(INVENTORY/'geometry_events.json'),capacity_spec_sha256=sha(INVENTORY/'candidate_capacity/spec.json'),
        bound_model_sources={n:sha(PARENT/'code'/n) for n in names},events=216,summaries=summaries,impossible_events=impossible,
        derivation='Sigmoid normalized widths/heights are below1; after256/resize mapping each is below the window side s. For s>=10, clip_box(margin10) cannot increase either dimension above s. Any reported box is inside the image. Let visible GT dimensions be vw,vh and raw GT dimensions be w,h. Intersection<=min(s,vw)*min(s,vh); union>=w*h. Thus raw continuous IoU<=min(s,vw)*min(s,vh)/(w*h). Ignore all center/offset/feature/visibility constraints, so this is optimistic.',
        equality_policy='Use conservative side s rather than the slightly smaller sigmoid clamp limit; only upper_bound<0.5 is declared impossible.',
        metric_scope='M70 raw continuous rectangle IoU only. This is not VOT image-bounded overlap and does not change the original criterion.',
        claim_limit='A bound>=0.5 does not prove that a correct candidate exists or succeeds; it only fails to exclude one.',
        inference_or_GT_policy_changed=False,new_model_forwards=0,optimizer_steps=0,public_evaluation_allowed=False,independent_model_review_pass=False)
    (ROOT/'decoder_capacity_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
