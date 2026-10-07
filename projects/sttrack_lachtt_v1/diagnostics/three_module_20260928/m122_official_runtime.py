"""One final, one confirmed initialization bank, one own-history runtime for all datasets."""
import json
from pathlib import Path
from bind_m122_official_initializations import sha,initialization_key


def checked_plan(path):
    plan=json.loads(Path(path).read_text());bundle_path=Path(plan['bundle_path'])
    assert sha(bundle_path)==plan['bundle_sha256'];bundle=json.loads(bundle_path.read_text())
    assert bundle['schema']=='M122_full152_final_official_v1'
    for name,digest in bundle['source_sha256'].items():assert sha(name)==digest,name
    for key in ['final','training_result','native_checkpoint','clip_weight']:
        assert sha(bundle[key+'_path'])==bundle[key+'_sha256'],key
    trained=json.loads(Path(bundle['training_result_path']).read_text())
    assert trained['status']=='complete_M122_full152_training' and trained['track_calls']==659406 and trained['sequence_runs']==456
    assert trained['epochs']==3 and trained['seed']==2027 and trained['frozen_before_after_exact'] and trained['final_state_roundtrip_exact']
    assert trained['final_sha256']==bundle['final_sha256'] and trained['precision_weight']==bundle['precision_weight']
    assert sha(plan['bank_path'])==plan['bank_sha256'] and sha(plan['binding_path'])==plan['binding_sha256']
    binding=json.loads(Path(plan['binding_path']).read_text())
    assert binding['status']=='complete_M122_human_official_initialization_binding' and binding['RGB_bytes_and_exact_legal_bbox_bound']
    return plan,bundle


class OfficialDenseTracker:
    def __init__(self,plan,bundle):
        import torch
        from dense_target_decoder import DenseTargetDecoder
        from full_dense_tracker import FullDenseTracker
        bank=torch.load(plan['bank_path'],map_location='cpu')
        assert bank['format']=='M122_human_official_initialization_v1' and bank['human_confirmed']
        assert bank['binding_sha256']==plan['binding_sha256'] and bank['encoder_sha256']==bundle['clip_weight_sha256']
        assert len(bank['keys'])==len(set(bank['keys']))==1895
        assert bank['tokens'].shape==(1895,5,768) and bank['mask'].shape==(1895,5) and bank['mask'].dtype==torch.bool
        assert bool(bank['mask'][:,0].all()) and bool(torch.isfinite(bank['tokens']).all()) and bool(torch.isfinite(bank['empty']).all())
        assert plan['dataset'] in ['depthtrack_test','cdtb','vot']
        self.dataset=plan['dataset'];self.indices={key:index for index,key in enumerate(bank['keys'])}
        model=DenseTargetDecoder().cuda().eval();model.load_state_dict(torch.load(bundle['final_path'],map_location='cpu'),strict=True)
        self.actor=FullDenseTracker(Path(bundle['repository']),Path(bundle['native_checkpoint_path']),Path(bundle['clip_weight_path']),bank,model)

    def initialize(self,image,bbox,rgb_path):
        key=self.dataset+':'+initialization_key(sha(rgb_path),bbox)
        self.actor.initialize(image,list(bbox),self.indices[key])

    def track(self,image):return self.actor.track(image)
