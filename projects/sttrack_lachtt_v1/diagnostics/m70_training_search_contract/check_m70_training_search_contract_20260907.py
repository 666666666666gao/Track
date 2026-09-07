"""Verify the completed contract audit against actual sampled image headers and resize records."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import py_compile
import numpy as np
from PIL import Image

BASE=Path('/root/autodl-tmp');TRAIN=BASE/'sttrack_m65_category_null_support_20260907'
ROOT=BASE/'sttrack_m70_recovery_window_inventory_20260907/training_search_contract'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    source=BASE/'m70_training_search_contract_audit_20260907.py';py_compile.compile(str(source),doraise=True)
    result=read(ROOT/'result.json');spec=read(ROOT/'spec.json')
    assert result['source_sha256']==spec['source_sha256']==sha(source)
    assert result['spec_sha256']==sha(ROOT/'spec.json') and result['sampled_geometry_sha256']==sha(ROOT/'sampled_geometry.json')
    training=read(TRAIN/'training_spec.json');records=read(ROOT/'sampled_geometry.json')
    by_frame={(r['sequence'],r['frame']):r for r in records['training']}
    cases={r['sequence']:r for r in training['sequence_order']}
    gt={name:np.loadtxt(Path(training['dataset_root'])/name/'groundtruth.txt',delimiter=',').reshape(-1,4) for name in cases}
    initial_size={}
    for name,case in cases.items():
        with Image.open(case['initial_rgb']) as im:initial_size[name]=im.size
    traces=[json.loads(x) for x in (TRAIN/'training/null/sampled_state_trace.jsonl').read_text().splitlines()]
    assert len(traces)==3917
    max_scale_error=0.;checked=0
    for row in traces:
        name=row['sequence'];frame=row['frame_index'];path=Path(training['dataset_root'])/name/'color'/('%08d.jpg'%(frame+1))
        with Image.open(path) as image:assert image.size==initial_size[name]
        if row['label']=='invalid':continue
        saved=by_frame[(name,frame)];expected=gt[name][frame,2:]*row['resize_factor']
        max_scale_error=max(max_scale_error,abs(saved['short']-float(expected.min())),abs(saved['long']-float(expected.max())))
        assert abs(saved['linear']-float(np.sqrt(expected.prod())))<=1e-12
        checked+=1
    assert checked==3660 and max_scale_error<=1e-12
    inside=[r for r in records['training'] if r['label']=='centre_inside']
    assert len(inside)==3194
    parent=read(ROOT.parent/'geometry_events.json')['events']
    lookup={(r['sequence'],r['frame']):r for r in parent}
    max_development_error=0.
    for row in records['development']:
        e=lookup[(row['sequence'],row['frame'])]
        side=max(e['image_width'],e['image_height']) if row['kind']=='coarse' else e['arms']['null']['local']['rectangle'][2]
        expected=np.asarray(e['GT_bbox'][2:])*256/side
        max_development_error=max(max_development_error,abs(row['short']-float(expected.min())),abs(row['long']-float(expected.max())))
    assert len(records['development'])==432 and max_development_error==0
    heavy_padding=sum(r['padding_fraction']>=.4375 for r in inside)
    receipt=dict(status='completed_actual_header_and_recorded_resize_verification',observed_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),audit_result_sha256=sha(ROOT/'result.json'),spec_sha256=sha(ROOT/'spec.json'),
        actual_training_image_headers=3917,dimensions_equal_to_sequence_initialization=True,
        independently_checked_valid_training_sizes=checked,maximum_training_arithmetic_difference=max_scale_error,
        independently_checked_development_sizes=432,maximum_development_arithmetic_difference=max_development_error,
        fit_centre_inside_samples_with_padding_at_least_global_fraction=heavy_padding,
        padding_reference=.4375,padding_interpretation='Whole-image nominal padding is not unprecedented in this adaptation sample; its causal effect remains untested.',
        no_model_forward=True,new_optimizer_steps=0,independent_model_review_pass=False)
    (ROOT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
