"""Verify complete-state default parity and the nondefault candidate contract."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2026);update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    path=root/'runtime_smoke_initializer.pth';torch.save(dict(model=CandidateSetAssociation().state_dict(),variant='geometry',scope='untrained contract initializer only'),path)
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    baseline=STTrack(params);candidate=STTrackCandidateSet(params,path)
    cases=[x for x in json.loads((root/'inference_inputs.json').read_text()) if x['split']=='fit'][:2]
    def image_at(case,frame):
        folder=Path(spec['dataset_root'])/case['sequence']
        return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
    frames=0
    for case in cases:
        image=image_at(case,0);info=dict(init_bbox=list(case['init_bbox']));baseline.initialize(image,info);candidate.initialize(image,info)
        for frame in range(1,61):
            image=image_at(case,frame);a=baseline.track(image);b=candidate.track(image)
            assert a['target_bbox']==b['target_bbox'] and a['best_score']==b['best_score']
            assert b['association_candidate']==0 and candidate.previous_choice==0
            assert all(torch.equal(x,y) for x,y in zip(baseline.z_dict,candidate.z_dict))
            assert all(torch.equal(x,y) for x,y in zip(baseline.track_query_before,candidate.track_query_before))
            assert np.abs(np.asarray(b['target_bbox'])-case['expected_rows'][frame]['bbox']).max()==0
            frames+=1
    case=cases[0];candidate.initialize(image_at(case,0),dict(init_bbox=list(case['init_bbox'])));candidate.track(image_at(case,1));calls=[]
    def force(*args):
        calls.append(int(args[-1][0]));logits=torch.full((1,11),-20.,device='cuda');logits[:,1]=20.
        return logits,torch.zeros(1,11,11,device='cuda')
    candidate.association.forward=force
    for frame in [2,3]:
        out=candidate.track(image_at(case,frame));item=candidate.previous_set['candidates'][1]
        assert out['association_candidate']==candidate.previous_choice==1
        assert np.abs(np.asarray(out['target_bbox'])-item['bbox']).max()<.001
        assert abs(float(out['best_score'])-item['score'])<1e-7
    assert calls==[0,1],calls
    receipt=dict(status='PASS',frames=frames,default_bbox_score_exact=True,template_query_tensors_exact=True,
        forced_candidate_frames=2,selected_own_box_and_confidence=True,previous_selected_index_propagated=True,
        scope='Initialization and integration test only; no trained-head performance or public metric.')
    (root/'runtime_smoke.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
