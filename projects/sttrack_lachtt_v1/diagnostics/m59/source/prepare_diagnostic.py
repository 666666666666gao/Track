"""Freeze a non-promoting, train-only content diagnosis of the completed negative M58 revision."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

OLD = Path('/root/autodl-tmp/sttrack_m58_content_controls_v1_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
ROOT = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
SHA_PARENT = '54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert sha(OLD/'spec.json') == 'be4a25212f9231963dcdb2acfcf446715910074eefa1265848850c1059b93096'
    assert sha(OLD/'run_controls.py') == '87902ed1a7e630fb922d3d7b3897cecd6b2dab718c62c60a65a10ee13749ddf4'
    assert sha(PARENT/'recursive_result.json') == SHA_PARENT
    parent=json.loads((PARENT/'recursive_result.json').read_text())
    assert not parent['primary_pass'] and parent['gates'] == dict(mean_vs_native=True,mean_vs_visual=True,low_frames=True,H10=False,native_success_protection=True)
    assert json.loads((OLD/'queue_result.json').read_text())['status']=='not_run_parent_gate_failed'
    assert not (OLD/'bundle.json').exists()
    ROOT.mkdir()
    original=(OLD/'run_controls.py').read_text()
    source=original
    replacements={
        '"""Fixed-text-head content interventions after the frozen M58 development gate."""': '"""M59: train-only content diagnosis of M58, whose promotion gate remains failed."""',
        "ROOT = Path('/root/autodl-tmp/sttrack_m58_content_controls_v1_20260906')": "ROOT = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')",
        "    assert result['primary_pass'] and all(result['gates'].values())": "    assert sha(PARENT / 'recursive_result.json') == spec['negative_parent_result_sha256']\n    assert not result['primary_pass'] and result['gates'] == spec['negative_parent_gates']\n    assert spec['purpose'] == 'train_only_content_diagnosis_of_failed_revision'\n    assert spec['public_evaluation_allowed'] is False",
        "status='candidate_bundle_sealed_after_parent_gate'": "status='diagnostic_bundle_sealed_with_parent_promotion_failed'",
        "status='completed_same_head_content_controls'": "status='completed_train_only_same_head_content_diagnosis'",
        "        gates=gates, primary_content_pass=all(gates.values()), attributes_descriptive_only=True,": "        descriptive_content_thresholds=gates, parent_promotion_pass=False, public_evaluation_allowed=False, attributes_descriptive_only=True,",
        "        next='Technical low22 entry validation before public evaluation' if all(gates.values()) else 'Stop this revision before public evaluation; diagnose fixed content effects')": "        next='Interpret all fixed-head content effects; M58 remains unpromoted; any model revision needs new Train training and recursive validation')"
    }
    for old,new in replacements.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    ast.parse(source)
    unchanged=['load_frame','prefix_parity','run','aggregate']
    def functions(src):
        return {n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(original),functions(source)
    assert all(before[n]==after[n] for n in unchanged)
    (ROOT/'run_controls.py').write_text(source)
    (ROOT/'runner_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='m58_frozen/run_controls.py',tofile='m59_diagnostic/run_controls.py')))
    shutil.copyfile(__file__,ROOT/'prepare_diagnostic.py')
    shutil.copyfile('/root/autodl-tmp/finalize_m59_diagnostic_20260906.py',ROOT/'finalize.py')
    spec=json.loads((OLD/'spec.json').read_text())
    spec.pop('prelaunch_revision')
    spec.update(status='frozen_after_negative_parent_before_diagnostic_inference',observed_utc=datetime.now(timezone.utc).isoformat(),
        purpose='train_only_content_diagnosis_of_failed_revision',negative_parent_result_sha256=SHA_PARENT,
        negative_parent_gates=parent['gates'],parent_promotion_pass=False,public_evaluation_allowed=False,
        inherited_content_spec_sha256=sha(OLD/'spec.json'),inherited_runner_sha256=sha(OLD/'run_controls.py'),
        inherited_queue_result_sha256=sha(OLD/'queue_result.json'),
        diagnostic_rationale='Text is +0.004510 pooled IoU over separately trained visual but -0.014912 macro IoU and +11 H10; fix the text weight to measure content effects before choosing a new language mechanism.',
        gate_definition='Inherited numeric margins are descriptive only. Parent M58 H10 gate stays failed regardless of these outcomes; no public promotion.',
        after_result='Read all content effects; no public evaluation or automatic new training. M58 remains unpromoted.',
        frozen_scope='Three controls use all development22; same-auto-caption-category attribute control uses the original prespecified14. No outcome-selected cases or new captions.',
        estimated_seconds_longest_worker=3550,optimizer_steps=0,new_caption_calls=0,
        independent_review_pass=False,independent_reviewer_status='REVIEW_UNAVAILABLE; existing Astra/max quota restriction; executor audit only',
        unchanged_inference_functions=unchanged,
        source_sha256={n:sha(ROOT/n) for n in ['prepare_diagnostic.py','run_controls.py','finalize.py']})
    for name,digest in spec['bank_sha256'].items():
        assert sha(OLD/(name+'.pt'))==digest
        (ROOT/(name+'.pt')).symlink_to(OLD/(name+'.pt'))
    assert sha(OLD/'mappings.json')==spec['mappings_sha256']
    (ROOT/'mappings.json').symlink_to(OLD/'mappings.json')
    assert sum(v['frames'] for v in spec['controls'].values())==119782
    write(ROOT/'spec.json',spec)
    sys.path.insert(0,str(ROOT))
    import run_controls
    run_controls.seal_bundle()
    sys.path.insert(0,str(run_controls.INTERFACE))
    from semantic_runtime import checked_plan
    for name in ['original']+list(spec['controls']):
        checked_plan(ROOT/(name+'_plan.json'))
    write(ROOT/'preparation_result.json',dict(status='diagnostic_plan_and_same_text_weight_bundle_sealed',
        spec_sha256=sha(ROOT/'spec.json'),bundle_sha256=sha(ROOT/'bundle.json'),parent_result_sha256=SHA_PARENT,
        source_sha256=spec['source_sha256'],inherited_content_spec_unchanged=sha(OLD/'spec.json')==spec['inherited_content_spec_sha256'],
        original_conditional_queue_status='not_run_parent_gate_failed',frozen_inference_functions_unchanged=unchanged,
        total_control_frames=119782,total_control_track_calls=119702,expected_prefix_frames=306,
        new_optimizer_steps=0,new_caption_calls=0,actual_model_tracking_calls=0,
        public_evaluation_allowed=False,parent_promotion_pass=False,independent_review_pass=False))
    print(json.dumps(json.loads((ROOT/'preparation_result.json').read_text()),indent=2))


if __name__=='__main__':
    main()
