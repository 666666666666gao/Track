"""Copy the sealed M84 experiment; only the frozen text protocol is replaced."""
from pathlib import Path
import hashlib,json,shutil,torch

R=Path(__file__).parent
P=Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

assert not (R/'training_spec.json').exists()
assert sha(P/'training_spec.json')=='0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5'
assert sha(P/'recursive_result.json')=='a9281e0833b8ba1c01220374f3d6a93dbc9d22e6e8648c3495bb009083f2d5f7'
parent=read(P/'training_spec.json');bank_result=read(R/'bank_result.json')
assert bank_result['status']=='crop_only_text_banks_complete'
assert bank_result['source_sha256']==sha(R/'prepare_banks.py')
assert bank_result['caption_result_sha256']==sha(R/'caption_result.json')
assert bank_result['caption_spec_sha256']==sha(R/'caption_spec.json')
assert read(R/'caption_spec.json')['plan_sha256']==sha(R/'EXPERIMENT_PLAN.md')
integration=read(P/'integration.json')
for name,digest in integration['source_sha256'].items():assert sha(P/'code'/name)==digest
shutil.copytree(P/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
copied=['causal_training.py','support_loss.py','window_competition.py','native_preservation.py','recursive_metric.py','data_inventory.json','integration.json']
for name in copied:shutil.copyfile(P/name,R/name)
expected=(P/'train_causal.py').read_text().replace(str(P),str(R))
assert (R/'train_causal.py').read_text()==expected
(R/'native_parity').mkdir()
shutil.copyfile(P/'native_parity/category_zero.pth',R/'native_parity/category_zero.pth')
assert sha(R/'native_parity/category_zero.pth')==parent['initial_checkpoint_sha256']['category']
zero=torch.load(R/'native_parity/category_zero.pth',map_location='cpu')
proof=[]
for split,arms in bank_result['banks'].items():
    for arm,item in arms.items():
        assert sha(item['path'])==item['sha256']
        bank=torch.load(item['path'],map_location='cpu')
        assert torch.equal(bank['empty'].float(),zero['model']['empty_text'])
        if arm!='empty':
            assert bank['mask'][:,0].all()
            difference=(bank['tokens'][:,0].float()-bank['empty'].float()).abs().amax(dim=1)
            assert (difference>0).all()
            proof.append(dict(split=split,arm=arm,minimum_max_abs_difference_from_empty=float(difference.min())))
spec=dict(parent)
spec.update(status='prepared_not_frozen',banks=bank_result['banks'],
    primary_control='M84 same architecture, initialization, loss, sequence order and training budget; old text protocol',
    parent_training_spec_sha256=sha(P/'training_spec.json'),parent_result_path=str(P/'recursive_result.json'),
    parent_result_sha256=sha(P/'recursive_result.json'),plan_sha256=sha(R/'EXPERIMENT_PLAN.md'),
    caption_protocol='M87 frozen crop-only category protocol; original five slots/mask/padding/Empty unchanged; new text is not semantic ground truth.',
    caption_spec_sha256=sha(R/'caption_spec.json'),caption_result_sha256=sha(R/'caption_result.json'),bank_result_sha256=sha(R/'bank_result.json'),
    training_script_sha256=sha(R/'train_causal.py'),run_queue_sha256=sha(R/'run_m87.sh'),
    preflight_source_sha256=sha(R/'preflight_m87.py'),
    acceptance_definition='4 M84 increment + 5 native including zero-H10 protection + 4 same-weight old content + 4 Swapped content + 1 Empty/native parity; all 18 required')
changed=set(k for k in parent if spec[k]!=parent[k])
assert changed<=set(['banks','primary_control','parent_training_spec_sha256','parent_result_path','parent_result_sha256',
    'plan_sha256','caption_protocol','training_script_sha256','run_queue_sha256','preflight_source_sha256','acceptance_definition'])
for key,name in [('inventory_sha256','data_inventory.json'),('integration_sha256','integration.json'),
    ('causal_script_sha256','causal_training.py'),('support_loss_sha256','support_loss.py'),
    ('window_loss_sha256','window_competition.py'),('preservation_loss_sha256','native_preservation.py')]:
    assert spec[key]==sha(R/name)
assert spec['seed']==2027 and spec['total_training_track_calls']==186694 and spec['expected_optimizer_steps']==5798
save(R/'training_spec.json',spec)
old_rec=read(P/'recursive_spec.json')
rec=dict(status='prepared_not_frozen',training_spec_sha256=sha(R/'training_spec.json'),runner_sha256=sha(R/'run_recursive.py'),
    metric_sha256=sha(R/'recursive_metric.py'),queue_sha256=sha(R/'run_m87.sh'),native_result_path=old_rec['native_result_path'],
    variants=['category','category_empty','category_old','category_swapped'],cases=old_rec['cases'],
    full_prediction_families_sealed_before_metric_GT=True,fixed_final_heads_only=True,poll_interval_seconds=240,public_evaluation=False)
save(R/'recursive_spec.json',rec)
report=dict(status='prepared',source_sha256=sha(__file__),nonempty_inputs=proof,
    identical_M84_initial_checkpoint=True,identical_M84_runtime_and_loss_files={n:sha(R/n) for n in copied},
    changed_existing_spec_fields=sorted(changed),training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'))
save(R/'preparation_receipt.json',report);print(json.dumps(report,indent=2))
