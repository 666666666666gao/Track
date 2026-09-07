"""Single-seed raw-score competition control; all inference sources remain sealed."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, difflib, hashlib, json, shutil, subprocess
B=Path('/root/autodl-tmp')
P=B/'sttrack_m77_window_competition_20260907'
R=B/'sttrack_m78_raw_competition_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def now():return datetime.now(timezone.utc).isoformat()
def prepare():
    assert not R.exists()
    assert sha(P/'training_spec.json')=='7162b4c6043d1895097a3cdec79f6373364031e6b100edbd08983f0f8a858f47'
    assert sha(P/'recursive_result.json')=='d3d33dac98c510f76711949f12b01d497157d25f2085fac9c83e0937018ff9c1'
    t=read(P/'training_spec.json');rs=read(P/'recursive_spec.json')
    assert sha(P/'train_causal.py')==t['training_script_sha256']
    assert sha(P/'run_recursive.py')==rs['runner_sha256']
    assert sha(P/'window_competition.py')==t['window_loss_sha256']
    R.mkdir();(R/'native_parity').mkdir()
    for n,h in read(P/'integration.json')['source_sha256'].items():
        src=P/'code'/n;assert sha(src)==h
        dst=R/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for n in ['integration.json','data_inventory.json','causal_training.py','support_loss.py','recursive_metric.py']:
        shutil.copyfile(P/n,R/n)
    for arm in ['category','empty']:
        shutil.copyfile(P/'native_parity'/(arm+'_zero.pth'),R/'native_parity'/(arm+'_zero.pth'))
        assert sha(R/'native_parity'/(arm+'_zero.pth'))==t['initial_checkpoint_sha256'][arm]
    for split in ['fit','development']:
        for arm in ['category','empty']:
            assert sha(t['banks'][split][arm]['path'])==t['banks'][split][arm]['sha256']
    old=(P/'window_competition.py').read_text()
    loss=replace(old,'Supervise the same Hann-weighted peak competition used by the public tracker.','Raw-response competition control; public inference still uses native Hann.')
    loss=replace(loss,'weighted = (score * output_window.reshape(16, 16)).flatten()','weighted = score.flatten()')
    loss=replace(loss,'; the native centered Hann is positive.','; raw competition does not use the Hann values.')
    loss=replace(loss,'competition_hann_top1_index','competition_raw_top1_index')
    (R/'window_competition.py').write_text(loss)
    (R/'loss_change.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),loss.splitlines(True),fromfile='M77/window_competition.py',tofile='M78/window_competition.py')))
    train=(P/'train_causal.py').read_text().replace(str(P),str(R)).replace('final-window hard-negative','raw-score hard-negative')
    (R/'train_causal.py').write_text(train)
    runner=(P/'run_recursive.py').read_text().replace(str(P),str(R))
    runner=replace(runner,'gates=dict(prior_control_success_protection=not legacy_broken, mean_vs_native=', 'gates=dict(mean_vs_native=')
    start=runner.index("    prior=parent['category']")
    end=runner.index("    result=dict(status='complete_recursive_development'",start)
    runner=runner[:start]+"    hann_path=Path(training['hann_recursive_result_path'])\n    assert sha(hann_path)==training['hann_recursive_result_sha256']\n    hann=json.loads(hann_path.read_text())['aggregates']\n"+runner[end:]
    runner=replace(runner,'parent_M73_aggregates=parent, parent_M73_result_sha256=sha(parent_path),','parent_M73_aggregates=parent, parent_M73_result_sha256=sha(parent_path),\n        reference_M77_Hann_aggregates=hann, reference_M77_result_sha256=sha(hann_path),')
    (R/'run_recursive.py').write_text(runner)
    (R/'run_pair.sh').write_text((P/'run_pair.sh').read_text().replace(str(P),str(R)))
    subprocess.run(['bash','-n',str(R/'run_pair.sh')],check=True)
    t.update(status='prepared_before_checks',revision='m78_raw_competition_v1',observed_utc=now(),seed=2027,
        hypothesis='Compare raw versus Hann-weighted competition training with identical Hann inference. Category contribution is separately tested against matched empty training and fixed-head content interventions.',
        architecture_control='M77 sources, initial tensors, banks, optimizer, data order and budget retained. Only training competition scores and their hard-negative ordering change from Hann*score to raw score.',
        loss='Native raw focal + 2 GIoU + 5 xyxy L1 + weight-1 competition at the rounded GT cell against up to 9 raw-score hard negatives with detached decoded center outside GT and IoU<=0.1.',
        window_loss_sha256=sha(R/'window_competition.py'),training_script_sha256=sha(R/'train_causal.py'),run_queue_sha256=sha(R/'run_pair.sh'),
        hann_recursive_result_path=str(P/'recursive_result.json'),hann_recursive_result_sha256=sha(P/'recursive_result.json'),
        parent_M77_training_spec_sha256=sha(P/'training_spec.json'),causal_check_sha256=None,
        scope_limitations=['Fixed seed2027 only; reused Train development22, no seed variance claims.',
            'Prospective ten promotion conditions compare native and same-budget Empty only. Prior M65/M73/M77 outcomes are descriptive references, not cumulative protection gates.',
            'The eligible negative geometry is identical; top-nine membership may change because raw rather than Hann-weighted scores determine hardness.',
            'No new inference parameters or policy, no automatic public evaluation. Past experiment conclusions stay frozen.'])
    # Remove historical mandatory-protection metadata while retaining its diagnostic sequence list.
    for k in list(t['promotion_gates']):
        if 'prior' in k or 'M73' in k:t['promotion_gates'].pop(k)
    t['promotion_gate_scope']='Ten native and paired Empty conditions only; historical protected sequences reported descriptively.'
    write(R/'prepared_training_spec.json',t)
    rs.update(status='prepared_before_checks',runner_sha256=sha(R/'run_recursive.py'),queue_sha256=sha(R/'run_pair.sh'))
    rs.pop('training_spec_sha256');write(R/'prepared_recursive_spec.json',rs)
    check=(P/'check_causal.py').read_text().replace(str(P),str(R)).replace('completed_M77_','completed_M78_')
    marker='rank.backward()'
    check=replace(check,marker,"alternate,_=competition_loss(synthetic,target,torch.ones_like(native.output_window))\nassert torch.equal(rank,alternate)\n"+marker)
    (R/'check_causal.py').write_text(check)
    for p in R.glob('*.py'):ast.parse(p.read_text())
    spec=dict(status='prepared_M78_before_checks',observed_utc=now(),source_sha256=sha(__file__),seed=2027,additional_seeds=[],
        training_prepared_sha256=sha(R/'prepared_training_spec.json'),recursive_prepared_sha256=sha(R/'prepared_recursive_spec.json'),check_source_sha256=sha(R/'check_causal.py'),
        expected_training_calls_per_arm=186694,expected_optimizer_steps_per_arm=5798,expected_development_calls_per_arm=33108,
        claims=['Raw versus final-window ranking can have different recursive effects.','Category-trained benefit and concrete word benefit require separate controls.'],
        promotion_conditions=10,public_evaluation_allowed=False,independent_model_review_pass=False,
        review_status='Previously requested gpt-6-astra/max reviewer unavailable; deterministic precheck is not independent model review.',
        fixed_head_controls='After both dev families, run same final Category head with Empty and existing Swapped contents regardless of gate; audit offline metrics using rows-based scalar interface.',
        next='Freeze only after causal and gradient contracts pass, then detached paired training and dev evaluation.')
    write(R/'spec.json',spec)
    (R/'EXPERIMENT_PLAN.md').write_text('''# M78: raw-response competition, fixed seed2027

Question: Does M77 degradation arise from added competition generally or its Hann-weighted formulation? Reuse sealed M73 (no added competition) and M77 (Hann competition); train only M78 RawCategory and RawEmpty. The same 289154-parameter adapter, initialization, 130 fit sequences, five-slot text banks, native frozen base, optimizer and 186694 tracking calls / 5798 updates per arm are retained. Both arms always use native Hann inference and default templates.

Only the added loss score changes: log(raw score), including raw-score ordering of at most nine eligible negatives. Rounded positive cell, detached outside-center / IoU<=0.1 negative mask and weight1 remain. Candidate membership can change with ranking; this is not an isolated change holding mined negatives fixed.

Ten prospective gates: Category pooled mean >= native by .002 and matched Empty by .001; macro mean >= both; low-IoU frames <= both; H10 episodes <= both; protect each reference's zero-H10 sequences. No accumulating historical successful-sequence union. Historical M65/M73/M77 results remain unchanged and are reported, not mandatory dominance criteria.

Final checkpoints only; no seed search or checkpoint selection. Run fixed-head Empty/Swapped diagnostics even after development failure. Concrete content claim requires original Category pooled/macro superiority, non-increased low frames and H10 versus both contents (eight conditions). Content success cannot override failed primary gates. No automatic public evaluations.

If both ranking versions degrade relative to M73, extra ranking is not supported under this budget. If raw improves over Hann, this supports a formulation-specific issue, not removing inference Hann. If original words underperform same-head alternatives, do not claim useful semantic content. Local ranking cannot repair absent-crop observations.

Budget: two GPUs, approximately 4.5 hours training plus 35 minutes development; content controls about 35 minutes. Poll near estimated completion, 240-second cadence when needed. Preserve both Qwen models. No new weights downloaded, smoke weights not saved. Independent requested reviewer is unavailable; deterministic checks are not a reviewer PASS.
''')
    (R/'EXPERIMENT_TRACKER.md').write_text('# M78\n\nPrepared; causal check pending. No training result and no promotion.\n')
    print(json.dumps(spec))
def freeze():
    s=read(R/'spec.json');assert s['source_sha256']==sha(__file__)
    assert (R/'causal_check.exit').read_text().strip()=='0'
    c=read(R/'causal_check.json')
    assert c['checker_sha256']==s['check_source_sha256']==sha(R/'check_causal.py')
    assert c['status']=='completed_M78_native_parity_and_causal_smoke'
    assert c['combined_loss_contract_verified'] and c['zero_residual_public_state_exact'] and c['base_frozen_all_arms']
    assert c['formal_optimizer_steps']==0 and not c['smoke_weights_saved']
    assert all(v['optimizer_steps']==3 for v in c['arms'].values())
    assert sha(R/'prepared_training_spec.json')==s['training_prepared_sha256']==c['prepared_spec_sha256']
    assert sha(R/'prepared_recursive_spec.json')==s['recursive_prepared_sha256']
    assert not (R/'frozen.json').exists() and not (R/'training').exists()
    t=read(R/'prepared_training_spec.json');rs=read(R/'prepared_recursive_spec.json')
    assert sha(R/'train_causal.py')==t['training_script_sha256'] and sha(R/'window_competition.py')==t['window_loss_sha256']
    t.update(status='frozen_before_training',causal_check_sha256=sha(R/'causal_check.json'))
    write(R/'training_spec.json',t);rs.update(status='frozen_before_training',training_spec_sha256=sha(R/'training_spec.json'))
    write(R/'recursive_spec.json',rs)
    f=dict(training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),spec_sha256=sha(R/'spec.json'),check_sha256=sha(R/'causal_check.json'),seed=2027,public_evaluation_allowed=False)
    write(R/'frozen.json',f);print(json.dumps(f))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze}[a.action]()
