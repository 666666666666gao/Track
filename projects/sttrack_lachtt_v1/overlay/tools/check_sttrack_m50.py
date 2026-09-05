"""Check native parity and exact causality of the extra scale template write."""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root; plan=json.loads((root/'spec.json').read_text())
    parent=Path(plan['source_root']); spec=json.loads((parent/'spec.json').read_text())
    repo=Path(spec['repository']); sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import check_binding,sha
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_scale_template import STTrackScaleTemplate
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.train.data.processing_utils import sample_target
    check_binding(parent,spec)
    for name,digest in plan['source_sha256'].items(): assert sha(repo/name)==digest,name
    inputs=json.loads((parent/'inference_inputs.json').read_text())
    eligible=[]
    for case in inputs:
        if case['split']!='fit':continue
        scale=math.sqrt(case['init_bbox'][2]*case['init_bbox'][3])
        scheduled=sum(r['frame_index']%50==0 and r['score']>.75 for r in case['expected_rows'][1:121])
        if scheduled==0:continue
        for r in case['expected_rows'][1:121]:
            now=math.sqrt(r['bbox'][2]*r['bbox'][3]); ratio=max(now/scale,scale/now)
            if r['frame_index']%50==0 and r['score']>.75:scale=now
            elif r['score']>.75 and ratio>=plan['scale_change']:
                eligible.append((r['frame_index'],case['sequence'],case));break
    first,sequence,case=min(eligible,key=lambda x:(x[0],x[1]))
    torch.set_num_threads(1); update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,
                           search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    native=STTrack(params); variant=STTrackScaleTemplate(params,float('inf'))
    data=Path(spec['dataset_root'])/sequence
    def image_at(i):
        return get_rgbd_frame(str(data/'color'/f'{i+1:08d}.jpg'),str(data/'depth'/f'{i+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
    image=image_at(0)
    native.initialize(image,dict(init_bbox=list(case['init_bbox'])))
    variant.initialize(image,dict(init_bbox=list(case['init_bbox'])))
    writes=0
    for i in range(1,121):
        image=image_at(i); a=native.track(image); b=variant.track(image)
        assert a['target_bbox']==b['target_bbox'] and a['best_score']==b['best_score']
        assert all(torch.equal(x,y) for x,y in zip(native.z_dict,variant.z_dict))
        assert all(torch.equal(x,y) for x,y in zip(native.track_query_before,variant.track_query_before))
        writes+=b['template_update']=='native'
    assert writes>=1
    variant.scale_change=plan['scale_change']
    variant.initialize(image_at(0),dict(init_bbox=list(case['init_bbox'])))
    anchor_tensor=variant.z_dict[0].clone(); actual_first=None
    for i in range(1,121):
        image=image_at(i); out=variant.track(image)
        assert torch.equal(anchor_tensor,variant.z_dict[0])
        if actual_first is None:
            expected=case['expected_rows'][i]
            assert out['target_bbox']==expected['bbox'] and out['best_score']==expected['score']
        if out['template_update']=='scale':
            if actual_first is None:actual_first=i
            patch,_,_=sample_target(image,variant.state,2.,output_sz=128)
            assert np.array_equal(patch,variant.z_patch_arr)
            assert torch.equal(variant.preprocessor.process(patch),variant.z_dict[1])
    assert actual_first==first
    result=dict(status='PASS',sequence=sequence,first_scale_write=first,disabled_native_parity_frames=120,
                disabled_native_template_writes=writes,enabled_frames=120,initial_template_unchanged=True,
                extra_template_exact_current_prediction_crop=True,ground_truth_files_opened=False,
                spec_sha256=sha(root/'spec.json'),source_sha256=sha(__file__))
    (root/'contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
