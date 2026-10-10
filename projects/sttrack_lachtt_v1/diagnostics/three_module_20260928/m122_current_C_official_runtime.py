"""One fixed P1+Current-quality-C composite final for all official datasets."""
import json
from pathlib import Path

from bind_m122_official_initializations import sha, initialization_key
from m122_official_runtime import checked_plan as checked_base_plan


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    base_plan, base = checked_base_plan(plan['base_plan_path'])
    assert sha(plan['base_plan_path']) == plan['base_plan_sha256']
    assert plan['dataset'] == base_plan['dataset']
    assert sha(plan['bundle_path']) == plan['bundle_sha256']
    bundle = json.loads(Path(plan['bundle_path']).read_text())
    assert bundle['schema'] == 'M122_P1_Current_quality_C_Full152_official_v1'
    assert bundle['base_final_sha256'] == base['final_sha256']
    for name, digest in bundle['source_sha256'].items():
        assert sha(name) == digest, name
    assert sha(bundle['final_path']) == bundle['final_sha256']
    assert sha(bundle['training_result_path']) == bundle['training_result_sha256']
    trained = json.loads(Path(bundle['training_result_path']).read_text())
    assert trained['status'] == 'complete_M122_current_quality_C_full152'
    assert trained['arm'] == 'current' and trained['seed'] == 2027
    assert trained['epochs'] == 10 and trained['optimizer_steps'] == 400 and trained['common_events'] == 1257
    assert trained['base_final_sha256'] == base['final_sha256']
    assert trained['final_sha256'] == bundle['final_sha256'] and trained['frozen_base_roundtrip_exact']
    assert not trained['public_test_CDTB_VOT_used_for_optimization']
    assert trained['deployment_threshold'] == .5
    for key in ['bank_path', 'bank_sha256', 'binding_path', 'binding_sha256']:
        assert plan[key] == base_plan[key], key
    if plan['dataset'] != 'vot':
        for key in ['cases_path', 'cases_sha256', 'dataset_root', 'metric_source', 'metric_source_sha256']:
            assert plan[key] == base_plan[key], key
    bundle['base_bundle'] = base
    return plan, bundle


class OfficialDenseTracker:
    def __init__(self, plan, bundle):
        import torch
        from dense_target_decoder import DenseTargetDecoder
        from full_dense_tracker import FullDenseTracker
        from template_write_C import TemplateWriteC
        from trusted_template_tracker import TrustedTemplateTracker
        base = bundle['base_bundle']
        bank = torch.load(plan['bank_path'], map_location='cpu')
        assert bank['format'] == 'M122_human_official_initialization_v1' and bank['human_confirmed']
        assert bank['binding_sha256'] == plan['binding_sha256'] and bank['encoder_sha256'] == base['clip_weight_sha256']
        assert len(bank['keys']) == len(set(bank['keys'])) == 1895
        assert bank['tokens'].shape == (1895, 5, 768) and bank['mask'].shape == (1895, 5)
        assert bank['mask'].dtype == torch.bool and bool(bank['mask'][:, 0].all())
        assert bool(torch.isfinite(bank['tokens']).all()) and bool(torch.isfinite(bank['empty']).all())
        self.dataset = plan['dataset']
        self.indices = {key: index for index, key in enumerate(bank['keys'])}
        weights = torch.load(bundle['final_path'], map_location='cpu')
        original = torch.load(base['final_path'], map_location='cpu')
        assert weights['A_B'].keys() == original.keys()
        assert all(torch.equal(weights['A_B'][name], value) for name, value in original.items())
        model = DenseTargetDecoder().cuda().eval()
        model.load_state_dict(weights['A_B'], strict=True)
        writer = TemplateWriteC().cuda().eval()
        writer.load_state_dict(weights['C'], strict=True)
        for parameter in writer.parameters():
            parameter.requires_grad_(False)
        self.actor = FullDenseTracker(Path(base['repository']), Path(base['native_checkpoint_path']),
                                      Path(base['clip_weight_path']), bank, model)
        self.trusted = TrustedTemplateTracker(self.actor, writer, 'current')

    def initialize(self, image, bbox, rgb_path):
        key = self.dataset + ':' + initialization_key(sha(rgb_path), bbox)
        self.trusted.initialize(image, list(bbox), self.indices[key])

    def track(self, image):
        return self.trusted.track(image)
