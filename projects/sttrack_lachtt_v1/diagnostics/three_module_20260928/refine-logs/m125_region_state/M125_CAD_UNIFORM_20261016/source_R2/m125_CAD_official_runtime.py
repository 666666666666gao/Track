"""The same R0 final and one sealed template profile for all three datasets."""
from datetime import datetime, timezone
from pathlib import Path
import json

from bind_m122_official_initializations import sha, initialization_key
from m125_R0_fixed_official_runtime import checked_plan as checked_R0_plan, OfficialDenseTracker
from deploy_m125_region_fixed import verify_predecessor
from cad_template_profiles import profile_dict


SCHEMA = 'M125_CAD_same_R0_final_uniform_profile_v1'


def make_bundle(reference_bundle_path, source_gate_path, profile_id):
    assert datetime.now(timezone.utc) >= datetime.fromisoformat('2026-10-17T12:45:00+08:00')
    reference_bundle_path = Path(reference_bundle_path)
    source_gate_path = Path(source_gate_path)
    reference = json.loads(reference_bundle_path.read_bytes())
    gate = json.loads(source_gate_path.read_bytes())
    assert gate['status'] == 'reviewed_M125_CAD_uniform_inference_source_gate'
    assert gate['profiles'] == ['U0', 'U1', 'U2']
    assert profile_id in gate['profiles']
    assert gate['per_dataset_profile_selection'] is False
    for path, digest in gate['source_sha256'].items():assert sha(path) == digest, path
    predecessor = verify_predecessor(gate)
    assert reference_bundle_path == Path(predecessor['root']) / 'evaluation/bundle.json'
    assert reference['schema'] == 'M125_R0_fixed6_Train152_official_v1'
    assert reference['final_sha256'] == predecessor['final_sha256']
    return dict(reference, schema=SCHEMA, source_sha256=gate['source_sha256'],
        source_gate_path=str(source_gate_path), source_gate_sha256=sha(source_gate_path),
        reference_R0_bundle_path=str(reference_bundle_path), reference_R0_bundle_sha256=sha(reference_bundle_path),
        R0_completed_suite=predecessor, inference_profile_id=profile_id,
        inference_profile=profile_dict(profile_id), one_profile_for_all_datasets=True,
        per_dataset_profile_selection=False, new_training_or_weight_selection=False,
        inference_strategy_engineering_comparison=True)


def checked_plan(path):
    path = Path(path)
    plan = json.loads(path.read_bytes())
    reference_path = Path(plan['reference_R0_plan_path'])
    assert sha(reference_path) == plan['reference_R0_plan_sha256']
    reference_plan, reference_bundle = checked_R0_plan(reference_path)
    assert sha(plan['bundle_path']) == plan['bundle_sha256']
    bundle = json.loads(Path(plan['bundle_path']).read_bytes())
    expected_bundle = make_bundle(reference_plan['bundle_path'], bundle['source_gate_path'], bundle['inference_profile_id'])
    assert bundle == expected_bundle
    assert bundle['final_sha256'] == reference_bundle['final_sha256']
    expected_plan = dict(reference_plan, reference_R0_plan_path=str(reference_path),
        reference_R0_plan_sha256=sha(reference_path), bundle_path=plan['bundle_path'], bundle_sha256=plan['bundle_sha256'],
        inference_profile_id=bundle['inference_profile_id'], profile_events_root=str(path.parent/'profile_events'))
    if reference_plan['dataset'] != 'vot':expected_plan['output'] = str(path.parent/'predictions')
    assert plan == expected_plan
    bundle['base_bundle'] = reference_bundle['base_bundle']
    return plan, bundle


class OfficialProfileTracker(OfficialDenseTracker):
    def __init__(self, plan, bundle):
        from profiled_template_tracker import ProfiledTemplateTracker
        super().__init__(plan, bundle)
        self.trusted = ProfiledTemplateTracker(self.actor, self.trusted.writer, bundle['inference_profile_id'])
        self.profile_id = bundle['inference_profile_id']

    def initialize(self, image, bbox, rgb_path):
        self.initialization_key = self.dataset + ':' + initialization_key(sha(rgb_path), bbox)
        super().initialize(image, bbox, rgb_path)
