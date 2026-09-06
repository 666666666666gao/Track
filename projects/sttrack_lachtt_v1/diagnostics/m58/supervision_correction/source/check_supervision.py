"""Same-output comparison of native and mismatched dense training labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

root=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
sys.path.insert(0,str(root/'code'));sys.path.insert(0,str(root))
from causal_training import CausalTrainingTracker,supervision as old_loss
from causal_training_native_labels import supervision as native_loss
from lib.config.sttrack.config import cfg,update_config_from_file
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.train.data.processing_utils import transform_image_to_crop
from lib.utils.heapmap_utils import generate_heatmap

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(1)
update_config_from_file(str(root/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
integration=json.loads((root/'integration.json').read_text())
params=SimpleNamespace(cfg=cfg,checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
    base_checkpoint_sha256=integration['native_checkpoint_sha256'],template_factor=2.,template_size=128,
    search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
tracker=CausalTrainingTracker(params,str(root/'native_parity/text_zero.pth'))
bank=torch.load(root/'text_fit.pt',map_location='cpu')
inventory=json.loads((root/'data_inventory.json').read_text())
rows=[];started=time.time()
for name in ['cube04_indoor','chair01_indoor','bag04_indoor']:
    case=next(r for r in inventory['sequences_detail'] if r['sequence']==name)
    assert case['split']=='fit'
    folder=Path(inventory['dataset_root'])/name
    assert sha(folder/'groundtruth.txt')==case['groundtruth_sha256']
    gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',').reshape(-1,4)
    def frame(i):
        return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),
            str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
    index=bank['sequences'].index(name)
    tracker.initialize(frame(0),dict(init_bbox=case['first_box'],text_tokens=bank['tokens'][index],
        text_mask=bank['mask'][index],empty_text=bank['empty']))
    for i in range(1,103):
        with torch.no_grad():
            out,state=tracker.step(frame(i))
            prior=state['previous_bbox'];resize=state['resize_factor']
            a,ad=old_loss(tracker.network,out,gt[i],prior,resize,256)
            b,bd=native_loss(tracker.network,out,gt[i],prior,resize,256)
            if a is None:
                assert b is None
                continue
            target=transform_image_to_crop(torch.tensor(gt[i],dtype=torch.float32),
                torch.tensor(prior),resize,torch.tensor([256,256]),normalize=True)
            centre=(target[:2]+target[2:]/2)*16
            nearest=centre.round()
            interior=bool(((nearest>=0)&(nearest<16)).all()) and bd['label']=='centre_inside'
            if interior:
                native_map=generate_heatmap(target[None,None],patch_size=256,stride=16)[0]
                cell=native_map.flatten().argmax().item()
                assert cell==int(nearest[1])*16+int(nearest[0])
                native_score=float(out['score_map'].flatten()[cell])
                peak_cell=int(out['score_map'].flatten().argmax())
            else:
                native_score=peak_cell=None
            pred=np.array(state['bbox']);truth=gt[i]
            lo=np.maximum(pred[:2],truth[:2]);hi=np.minimum(pred[:2]+pred[2:],truth[:2]+truth[2:])
            intersection=np.maximum(hi-lo,0).prod();iou=intersection/(pred[2:].prod()+truth[2:].prod()-intersection)
            rows.append(dict(sequence=name,frame=i,iou=float(iou),score=state['best_score'],
                old=ad,corrected=bd,corrected_native_cell=nearest.tolist(),
                corrected_cell_raw_score=native_score,predicted_peak_cell=peak_cell,
                native_heatmap_cell_checked=interior))
first=next(x for x in rows if x['sequence']=='cube04_indoor' and x['frame']==1)
assert first['iou']>.89 and first['corrected']['focal']<first['old']['focal']
healthy=[x for x in rows if x['iou']>=.5 and x['old']['label']==x['corrected']['label']=='centre_inside']
result=dict(status='same_frozen_outputs_supervision_comparison_complete',observed_utc=datetime.now(timezone.utc).isoformat(),
    sequences=3,predicted_frames=306,valid_gt_rows=len(rows),healthy_comparable_frames=len(healthy),
    first_example=first,healthy_mean_focal_old=float(np.mean([x['old']['focal'] for x in healthy])),
    healthy_mean_focal_corrected=float(np.mean([x['corrected']['focal'] for x in healthy])),
    native_heatmap_cell_checks=sum(x['native_heatmap_cell_checked'] for x in rows),
    old_source_sha256=sha(root/'causal_training.py'),corrected_source_sha256=sha(root/'causal_training_native_labels.py'),
    checker_sha256=sha(__file__),optimizer_steps=0,predictions_shared_between_loss_variants=True,
    elapsed_seconds=time.time()-started,rows=rows,
    scope='Fit-only label alignment diagnostic on shared zero-residual predictions; no learned performance or development selection.')
(root/'supervision_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
