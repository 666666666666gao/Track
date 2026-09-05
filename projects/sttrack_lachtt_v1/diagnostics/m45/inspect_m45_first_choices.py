"""Replay the first choice in the two largest M45 full-recursion regressions."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    plan=json.loads((root/'spec.json').read_text());parent=Path(plan['source_root']);spec=json.loads((parent/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import sha,check_binding
    from tools.train_sttrack_m42 import overlaps
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    check_binding(parent,spec)
    audit=json.loads((root/'recursive_result.json').read_text());assert audit['integrity_pass'] and not audit['primary_pass']
    control=json.loads((parent/'recursive_result.json').read_text());assert sha(parent/'recursive_result.json')==plan['control_recursive_result_sha256']
    training=json.loads((root/'geometry_result.json').read_text());assert sha(root/'geometry_final.pth')==training['checkpoint_sha256']
    result=json.loads((root/'recursive_result.json').read_text())
    harms=sorted(result['per_sequence'],key=lambda n:result['per_sequence'][n]['iou_sum']-control['per_sequence']['default'][n]['iou_sum'])[:2]
    assert harms==['egg_indoor','mobilephone02_indoor']
    plans={x['sequence']:x for x in json.loads((parent/'inference_inputs.json').read_text())}
    torch.set_num_threads(1);update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackCandidateSet(params,root/'geometry_final.pth');forward=tracker.association.forward;capture={}
    def observed(*args):
        output=forward(*args);capture['inputs']=[x.detach().clone() for x in args];capture['logits']=output[0].detach().clone();capture['affinity']=output[1].detach().clone()
        return output
    tracker.association.forward=observed;rows=[];frames=0
    for name in harms:
        event=[r for r in audit['first_overrides'] if r['sequence']==name][0]
        stop=event['first_override'];case=plans[name];folder=Path(spec['dataset_root'])/name
        sealed=json.loads((root/'recursive'/(name+'.json')).read_text())['rows']
        def image_at(frame):return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
        tracker.initialize(image_at(0),dict(init_bbox=case['init_bbox']))
        for frame in range(1,stop+1):
            out=tracker.track(image_at(frame));assert out['target_bbox']==sealed[frame]['bbox'] and float(out['best_score'])==sealed[frame]['score'];frames+=1
        # Ground truth is opened only after the complete replay is sealed.
        gt=torch.tensor(np.loadtxt(folder/'groundtruth.txt',delimiter=',')[stop],dtype=torch.float32)
        candidates=tracker.previous_set['candidates'];boxes=torch.tensor([c['bbox'] for c in candidates]);ious=overlaps(boxes,gt)
        current,previous,refs,geometry,scores,selected=capture['inputs'];past=int(selected[0]);chosen=out['association_candidate']
        similarities={}
        for index,label in [(0,'default'),(chosen,'chosen')]:
            similarities[label]={m:float(F.cosine_similarity(current[0,index,k].flatten(),previous[0,past,k].flatten(),dim=0)) for k,m in enumerate(['rgb','depth'])}
        rows.append(dict(sequence=name,frame_zero_based=stop,replay_frames=stop,previous_choice=past,chosen=chosen,
            ground_truth=gt.tolist(),candidate_iou=ious.tolist(),candidate_boxes=boxes.tolist(),candidate_scores=[c['score'] for c in candidates],
            logits=capture['logits'][0].cpu().tolist(),previous_choice_affinity_column=capture['affinity'][0,:,past].cpu().tolist(),
            native_previous_cosine=similarities,expected_default_iou=event['first_default_iou'],expected_selected_iou=event['first_selected_iou'],
            template_update_frame=bool(stop%tracker.update_intervals==0),replay_matches_sealed_trajectory=True))
    check_binding(parent,spec)
    output=dict(status='complete',frames=frames,rows=rows,source_sha256=sha(__file__),recursive_result_sha256=sha(root/'recursive_result.json'),
                scope='Two preselected largest-regression first choices only. GT opened after replay. No training, changed policy, public metric or claim that these explain all failures.')
    (root/'first_choice_diagnosis.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
