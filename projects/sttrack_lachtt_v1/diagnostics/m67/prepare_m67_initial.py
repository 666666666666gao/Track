from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
BASE=Path('/root/autodl-tmp');PARENT=BASE/'sttrack_m65_category_null_support_20260907'
ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert sha(PARENT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
ROOT.mkdir();shutil.copytree(PARENT/'code',ROOT/'code')
for n in ['integration.json','data_inventory.json','text_fit.pt','text_development.pt','recursive_metric.py']:
    shutil.copyfile(PARENT/n,ROOT/n)
shutil.copyfile(BASE/'m67_support_loss_20260907.py',ROOT/'support_loss.py')
source=(PARENT/'causal_training.py').read_text()
assert source.count('enhanced, _ = self.network.semantic_adapter')==1
source=source.replace('enhanced, _ = self.network.semantic_adapter','enhanced, semantic_aux = self.network.semantic_adapter')
source=source.replace('out = self.network.forward_head(enhanced)','out = self.network.forward_head(enhanced)\n        out[\'semantic_features\'] = semantic_aux')
source=source.replace('def supervision(network, out, groundtruth, previous, resize, search_size):','def base_supervision(network, out, groundtruth, previous, resize, search_size):')
source+='''

def supervision(network, out, groundtruth, previous, resize, search_size, support_weight=0.):
    loss, diagnostic=base_supervision(network,out,groundtruth,previous,resize,search_size)
    if loss is None:
        return loss, diagnostic
    from support_loss import support_supervision
    support, values=support_supervision(out,groundtruth,previous,resize,search_size)
    diagnostic.update(values)
    diagnostic['support_weight']=support_weight
    if support_weight==0.:
        return loss, diagnostic
    combined=loss+support_weight*support
    assert bool(torch.isfinite(combined))
    return combined, diagnostic
'''
(ROOT/'causal_training.py').write_text(source)
text=(PARENT/'train_causal.py').read_text().replace(str(PARENT),str(ROOT)).replace("['control', 'null']","['control', 'support']")
text=text.replace("tracker.network.semantic_adapter.null_support == (args.arm == 'null')","tracker.network.semantic_adapter.null_support")
text=text.replace("null_support=args.arm == 'null'","null_support=True, support_loss_weight=spec['support_loss_weights'][args.arm]")
text=text.replace("state['previous_bbox'], state['resize_factor'], 256)","state['previous_bbox'], state['resize_factor'], 256, support_weight=spec['support_loss_weights'][args.arm])")
text=text.replace("assert sha(root / 'causal_training.py') == spec['causal_script_sha256']","assert sha(root / 'causal_training.py') == spec['causal_script_sha256']\n    assert sha(root / 'support_loss.py') == spec['support_loss_sha256']")
(ROOT/'train_causal.py').write_text(text)
text=(PARENT/'run_recursive.py').read_text().replace(str(PARENT),str(ROOT)).replace("'null'","'support'")
text=text.replace("saved['null_support'] == (arm == 'support')","saved['null_support']")
text=text.replace('null_pooled_','support_pooled_')
text=text.replace("gates=dict(mean_vs_native=", "legacy_broken=[n for n in training['protected_prior_control_sequences'] if per['support'][n]['failure_episodes']>0]\n    gates=dict(prior_control_success_protection=not legacy_broken, mean_vs_native=")
text=text.replace("new_failure_sequences=broken, broken_control_success_sequences=control_broken,", "new_failure_sequences=broken, broken_control_success_sequences=control_broken, broken_prior_control_success_sequences=legacy_broken,")
(ROOT/'run_recursive.py').write_text(text)
queue=(PARENT/'run_m65.sh').read_text().replace(str(PARENT),str(ROOT)).replace('m65','m67').replace('null','support')
(ROOT/'run_m67.sh').write_text(queue)
initial=ROOT/'native_parity';initial.mkdir()
for arm in ['control','support']:shutil.copyfile(PARENT/'native_parity/null_zero.pth',initial/(arm+'_zero.pth'))
s=read(PARENT/'training_spec.json');parent_sha=sha(PARENT/'training_spec.json')
for key in ['causal_smoke_result_sha256','freeze_script_sha256','preparation_sha256']:s.pop(key)
old=read(PARENT/'recursive_result.json')
s.update(status='prepared_before_fit_smoke_not_training_authorization',observed_utc=datetime.now(timezone.utc).isoformat(),revision='m67_supervised_semantic_support_v1',
    parent_training_spec_sha256=parent_sha,primary_arm='support',control_arm='control',support_loss_weights={'control':0.,'support':.1},
    support_loss_sha256=sha(ROOT/'support_loss.py'),causal_script_sha256=sha(ROOT/'causal_training.py'),training_script_sha256=sha(ROOT/'train_causal.py'),
    initial_checkpoint_sha256={a:sha(initial/(a+'_zero.pth')) for a in ['control','support']},run_queue_sha256=sha(ROOT/'run_m67.sh'),
    hypothesis='Box-level target versus background support supervision makes the existing zero slot informative and improves recursive tracking over the identical unsupervised Null architecture.',
    architecture_control='Both arms are the exact M65 Null architecture and identical zero-residual initial tensors; 289154 learned parameters each. Only support-loss coefficient differs.',
    protected_prior_control_sequences=[n for n,v in old['per_sequence']['control'].items() if v['failure_episodes']==0],
    support_supervision='After causal prediction commit and original GT validity check: positive at native rounded GT centre cell when inside; negatives only outside GT rectangle expanded by one cell; other cells ignored. Equal averaging of available foreground/background group means. No labels from invalid GT; no caption correctness labels.',
    promotion_gates=dict(support_pooled_mean_vs_native_minimum=.002,support_pooled_mean_vs_control_minimum=.001,
        support_macro_mean_no_less_than_native_and_control=True,support_low_frames_no_more_than_native_and_control=True,
        support_H10_no_more_than_native_and_control=True,no_new_failure_on_native_or_control_zero_H10_sequences=True,
        protect_prior_M65_Control_zero_H10_sequences=True),
    scope_limitations=['Box-level support is weak supervision, not pixel foreground or attribute truth.',
        'No claim that this isolated loss solves search-outside-frame recovery.',
        'Content attribution remains necessary before low22. No automatic full-dataset evaluation.',
        'Control should reproduce M65 Null under the same seed, inputs, native loss and complete causal sequence order.',
        'This predeclared single-seed mechanism test is not statistical replication.'],
    after_training='Seal both final-head Train development22 predictions, apply all frozen gates including prior M65 Control protection; only then prepare fixed-head lexical controls.',
    historical_comparator='M65 Null is the stronger completed mechanistic comparator. Prior M65 Control success protection is retained rather than removed by changing the matched control.')
write(ROOT/'prepared_training_spec.json',s)
r=read(PARENT/'recursive_spec.json');r.update(status='prepared_before_fit_smoke',runner_sha256=sha(ROOT/'run_recursive.py'),queue_sha256=sha(ROOT/'run_m67.sh'),variants=['control','support'])
write(ROOT/'prepared_recursive_spec.json',r)
write(ROOT/'preparation.json',dict(status='isolated_M67_prepared_no_training',observed_utc=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
    code_source_files_unchanged=all(sha(ROOT/'code'/n)==h for n,h in read(ROOT/'integration.json')['source_sha256'].items()),
    prepared_training_spec_sha256=sha(ROOT/'prepared_training_spec.json'),prepared_recursive_spec_sha256=sha(ROOT/'prepared_recursive_spec.json'),
    no_formal_training_started=True,no_public_evaluation=True,new_learning_parameters=0))
print(json.dumps(dict(root=str(ROOT),preparation_sha256=sha(ROOT/'preparation.json'),prepared_spec_sha256=sha(ROOT/'prepared_training_spec.json')),indent=2))
