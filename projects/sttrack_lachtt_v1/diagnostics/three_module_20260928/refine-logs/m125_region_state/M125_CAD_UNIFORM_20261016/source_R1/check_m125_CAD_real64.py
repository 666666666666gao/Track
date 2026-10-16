"""Inference-only first-case prefix witness, after the original R0 suite ends."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from bind_m122_official_initializations import sha
from full_dense_tracker import state_digest
from m125_CAD_official_runtime import checked_plan, OfficialProfileTracker


def main(args):
    plan,bundle=checked_plan(args.plan)
    assert plan['dataset']=='depthtrack_test'
    cases=json.loads(Path(plan['cases_path']).read_bytes());case=cases[0]
    assert case['frames']>=65
    args.output.mkdir()
    actor=OfficialProfileTracker(plan,bundle)
    frozen=actor.actor.frozen_digest()
    initial=dict(A_B=state_digest(actor.actor.decoder),C=state_digest(actor.trusted.writer))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    folder=Path(plan['dataset_root'])/case['sequence']
    boxes=[case['init_bbox']];scores=[1.];records=[]
    started=time.time()
    for frame in range(65):
        stem='%08d'%(frame+1)
        rgb=folder/'color'/(stem+'.jpg');depth=folder/'depth'/(stem+'.png')
        image=get_rgbd_frame(str(rgb),str(depth),dtype='rgbcolormap',depth_clip=True)
        if frame==0:
            actor.initialize(image,case['init_bbox'],rgb)
            first_template=actor.actor.native.z_dict[0]
        else:
            out,_,record=actor.trusted.step(image)
            assert out['boxes'].shape==(1,256,4)
            assert record['inference_profile_id']==bundle['inference_profile_id']
            assert actor.actor.native.z_dict[0] is first_template and len(actor.actor.native.z_dict)==2
            boxes.append(record['bbox']);scores.append(record['best_score']);records.append(record)
    boxes=np.asarray(boxes);scores=np.asarray(scores)
    np.savetxt(args.output/'boxes.txt',boxes,fmt='%.6f',delimiter=',')
    np.savetxt(args.output/'scores.txt',scores,fmt='%.6f')
    assert actor.actor.frozen_digest()==frozen
    assert state_digest(actor.actor.decoder)==initial['A_B'] and state_digest(actor.trusted.writer)==initial['C']
    assert all(p.grad is None for model in [actor.actor.decoder,actor.trusted.writer,actor.actor.native.network,actor.actor.clip] for p in model.parameters())
    counts=actor.trusted.counts
    assert counts['tracking_calls']==64
    assert counts['template_writes']==sum(record['template_write'] for record in records)
    assert counts['qualified']==sum(record['profile_write_qualified'] for record in records)
    assert counts['C_evaluations']==sum(record['C_prediction'] is not None for record in records)
    parity=None
    if bundle['inference_profile_id']=='U0':
        reference=json.loads(Path(plan['reference_R0_plan_path']).read_bytes())
        old=Path(reference['output'])
        old_boxes=(old/(case['sequence']+'.txt')).read_text().splitlines()[:65]
        old_scores=(old/(case['sequence']+'_all_scores.txt')).read_text().splitlines()[:65]
        assert old_boxes==(args.output/'boxes.txt').read_text().splitlines()
        assert old_scores==(args.output/'scores.txt').read_text().splitlines()
        parity=dict(all65_box_and_score_rows_match_original_six_decimal_files=True,
            reference_box_sha256=sha(old/(case['sequence']+'.txt')),
            reference_score_sha256=sha(old/(case['sequence']+'_all_scores.txt')),
            scope='saved bbox/score prefix only, not hidden-state bit attestation or full new U0 rerun')
    result=dict(status='complete_M125_CAD_real64_inference_interface_witness',
        inference_profile_id=bundle['inference_profile_id'],inference_profile=bundle['inference_profile'],
        final_sha256=bundle['final_sha256'],bundle_sha256=plan['bundle_sha256'],plan_sha256=sha(args.plan),
        first_case=case['sequence'],track_calls=64,rows_including_initialization=65,candidates_per_state=256,
        template_counts=counts,first_template_object_preserved=True,frozen_before_after_exact=True,
        A_B_C_states_unchanged=True,optimizer_steps=0,subsequent_GT_opened=False,new_checkpoint_created=False,
        U0_original_prefix_parity=parity,records=records,
        elapsed_seconds_including_frozen_checks=time.time()-started,source_sha256=sha(__file__),
        formal_dataset_metrics=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['status','inference_profile_id','track_calls','template_counts','final_sha256']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
