"""Full causal trajectories on the frozen Train development sequences."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=['spatial','pooled']); p.add_argument('--parity-smoke', action='store_true')
    args = p.parse_args(); root = args.root
    spec = json.loads((root/'spec.json').read_text()); repo = Path(spec['repository'])
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(root/'inference_inputs.json') == spec['inference_inputs_sha256']
    for name, digest in spec['source_sha256'].items():
        assert sha(repo/name) == digest, name
    sys.path.insert(0, str(repo))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_local_spatial import STTrackLocalSpatial
    from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,
                             search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    cases = json.loads((root/'inference_inputs.json').read_text())
    if args.parity_smoke:
        cases = [c for c in cases if c['split']=='fit'][:2]
        tracker = STTrackLocalSpatial(params,None,parity_smoke=True)
        baseline = STTrack(params)
    else:
        recursive_spec = json.loads((root/'recursive_spec.json').read_text())
        for name, digest in recursive_spec['source_sha256'].items():
            assert sha(repo/name) == digest, name
        result = json.loads((root/'training_result.json').read_text())
        assert result['information_gate_pass']
        checkpoint = root/(args.variant+'_final.pth')
        assert sha(checkpoint) == result['variants'][args.variant]['checkpoint_sha256']
        tracker = STTrackLocalSpatial(params,str(checkpoint))
        cases = [c for c in cases if c['split']=='development_holdout']
    output_dir = root/'recursive'; output_dir.mkdir(exist_ok=True)
    receipts=[]; start=time.time()
    for case in cases:
        folder=Path(spec['dataset_root'])/case['sequence']
        def image_at(frame):
            return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),
                                   dtype='rgbcolormap',depth_clip=True)
        image=image_at(0); tracker.initialize(image,dict(init_bbox=list(case['init_bbox'])))
        if args.parity_smoke:
            baseline.initialize(image,dict(init_bbox=list(case['init_bbox'])))
        frames=61 if args.parity_smoke else len(list((folder/'color').glob('*.jpg')))
        rows=[dict(frame=0,bbox=case['init_bbox'],score=None,choice=0,none=False)]
        for frame in range(1,frames):
            image=image_at(frame); output=tracker.track(image)
            if args.parity_smoke:
                reference=baseline.track(image)
                assert output['target_bbox']==reference['target_bbox'],(case['sequence'],frame,'bbox')
                assert output['best_score']==reference['best_score'],(case['sequence'],frame,'confidence')
                assert output['association_candidate']==0
                assert all(torch.equal(a,b) for a,b in zip(tracker.z_dict,baseline.z_dict))
                assert all(torch.equal(a,b) for a,b in zip(tracker.track_query_before,baseline.track_query_before))
                assert np.max(np.abs(np.array(output['target_bbox'])-case['expected_rows'][frame]['bbox']))==0
            rows.append(dict(frame=frame,bbox=output['target_bbox'],score=float(output['best_score']),
                choice=output['association_candidate'],none=output['association_none']))
        if args.parity_smoke:
            class ForcedSecond(torch.nn.Module):
                def forward(self, candidates, references, scalars):
                    logits=torch.zeros(candidates.shape[0],11,device=candidates.device)
                    logits[:,1]=10.
                    return logits
            prior=list(tracker.state); captured={}; original=tracker.network.forward
            def capture(*pos,**kw):
                value=original(*pos,**kw); captured['output']=value[0]; return value
            tracker.network.forward=capture
            saved_association=tracker.association; tracker.association=ForcedSecond()
            image=image_at(61); probe=tracker.track(image)
            tracker.network.forward=original; tracker.association=saved_association
            import math
            resize=256/math.ceil(math.sqrt(prior[2]*prior[3])*4.)
            value=captured['output']
            candidate=decode_nms_candidates(tracker.output_window*value['score_map'],value['size_map'],value['offset_map'],
                [prior],[resize],image.shape,256,10,3)[1]
            assert probe['association_candidate']==1
            assert np.max(np.abs(np.array(probe['target_bbox'])-candidate['bbox']))<.001
            assert probe['best_score']==candidate['score']
        receipt=dict(sequence=case['sequence'],frames=frames,changes=sum(r['choice']!=0 for r in rows),elapsed=time.time()-start)
        if not args.parity_smoke:
            path=output_dir/(args.variant+'_'+case['sequence']+'.json')
            path.write_text(json.dumps(dict(sequence=case['sequence'],variant=args.variant,rows=rows))+'\n')
            receipt['sha256']=sha(path)
        receipts.append(receipt); print(json.dumps(receipt),flush=True)
    receipt=dict(status='complete',sequences=receipts,elapsed_seconds=time.time()-start,ground_truth_files_opened=False,
                 model_source_sha256=sha(repo/'lib/test/tracker/sttrack_local_spatial.py'),runner_sha256=sha(__file__))
    name='recursive_smoke_receipt.json' if args.parity_smoke else args.variant+'_recursive_receipt.json'
    (root/name).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
