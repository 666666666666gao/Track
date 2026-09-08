"""Prepare one fixed-seed, blockwise text-dropout training experiment."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, difflib, hashlib, json, random, shutil, subprocess

B=Path('/root/autodl-tmp')
P=B/'sttrack_m78_raw_competition_20260908'
R=B/'sttrack_m80_block_text_dropout_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

def prepare():
    assert not R.exists()
    assert shutil.disk_usage(B).free>700_000_000
    assert sha(P/'training_spec.json')=='57cdd314efd5359fa2e16be4f364c5568f1ca498b41df718517173610fe4d865'
    t=read(P/'training_spec.json')
    assert sha(P/'train_causal.py')==t['training_script_sha256']
    R.mkdir();(R/'native_parity').mkdir()
    for n,h in read(P/'integration.json')['source_sha256'].items():
        src=P/'code'/n; assert sha(src)==h
        dst=R/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for n in ['integration.json','data_inventory.json','causal_training.py','support_loss.py','recursive_metric.py','window_competition.py']:
        shutil.copyfile(P/n,R/n)
    for arm in ['category','empty']:
        shutil.copyfile(P/'native_parity'/(arm+'_zero.pth'),R/'native_parity'/(arm+'_zero.pth'))
    rng=random.Random(2027)
    schedule={r['sequence']:[rng.random()<.2 for _ in range((r['rgb_frames']-2)//32+1)] for r in t['sequence_order']}
    total_drop=sum(sum(min(32,r['rgb_frames']-1-i*32) for i,d in enumerate(schedule[r['sequence']]) if d) for r in t['sequence_order'])
    write(R/'text_schedule.json',dict(seed=2027,probability=.2,block_frames=32,sequences=schedule,dropout_track_calls=total_drop,total_track_calls=186694))
    helper='''"""Training-only content selection; keep slots, padding, initial RoI and mask."""
import torch

def make_contexts(tracker, empty):
    category = tracker.semantic_context['text']
    mask = tracker.semantic_context['mask'].unsqueeze(-1)
    blank = torch.where(mask, empty.to(category).reshape(1, 1, -1), category)
    return category, blank
'''
    (R/'text_dropout.py').write_text(helper)
    old=(P/'train_causal.py').read_text()
    train=old.replace(str(P),str(R)).replace('Single-seed category/empty causal training with raw-score hard-negative competition.','M80 single-seed causal training with fixed blockwise text dropout.')
    train=replace(train,"choices=['empty', 'category']","choices=['category']")
    train=replace(train,"    from causal_training import CausalTrainingTracker", "    from text_dropout import make_contexts\n    assert sha(root / 'text_dropout.py') == spec['text_dropout_source_sha256']\n    assert sha(root / 'text_schedule.json') == spec['text_schedule_sha256']\n    schedule = json.loads((root / 'text_schedule.json').read_text())['sequences']\n    from causal_training import CausalTrainingTracker")
    train=replace(train,'            tracker.initialize(frame(0), info)','            tracker.initialize(frame(0), info)\n            category_context, empty_context = make_contexts(tracker, bank[\'empty\'])\n            text_counts = Counter()')
    train=replace(train,'                out, state = tracker.step(frame(frame_index))',"                dropped = schedule[row['sequence']][(frame_index - 1) // 32]\n                tracker.semantic_context['text'] = empty_context if dropped else category_context\n                text_counts['empty' if dropped else 'category'] += 1\n                out, state = tracker.step(frame(frame_index))\n                state['text_dropped'] = dropped")
    train=replace(train,'frames=n, track_calls=n-1, supervised_frames=supervised, label_counts=dict(labels),','frames=n, track_calls=n-1, text_condition_calls=dict(text_counts), supervised_frames=supervised, label_counts=dict(labels),')
    train=replace(train,"status='complete' if complete else 'in_progress', seed=spec['seed'])","status='complete' if complete else 'in_progress', seed=spec['seed'], text_training_protocol='fixed_20_percent_32frame_block_dropout', text_schedule_sha256=spec['text_schedule_sha256'])")
    train=replace(train,"        evaluation_metrics_computed=False, observed_utc=", "        text_training_protocol='fixed_20_percent_32frame_block_dropout', text_schedule_sha256=spec['text_schedule_sha256'],\n        actual_text_condition_calls={key:sum(x['text_condition_calls'].get(key,0) for x in receipts) for key in ['category','empty']},\n        evaluation_metrics_computed=False, observed_utc=")
    (R/'train_causal.py').write_text(train)
    (R/'training_change.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),train.splitlines(True),fromfile='M78/train_causal.py',tofile='M80/train_causal.py')))
    check=(P/'check_causal.py').read_text().replace(str(P),str(R)).replace('completed_M78_','completed_M80_')
    check=replace(check,'from causal_training import CausalTrainingTracker, base_supervision','from text_dropout import make_contexts\nfrom causal_training import CausalTrainingTracker, base_supervision')
    check=replace(check,'    opt = torch.optim.AdamW',"    contexts=make_contexts(tracker,infos[arm]['empty_text'])\n    assert torch.equal(contexts[0][:,~infos[arm]['text_mask']],contexts[1][:,~infos[arm]['text_mask']])\n    mask_before=tracker.semantic_context['mask'].clone()\n    initial_before=tracker.semantic_context['initial'].clone()\n    opt = torch.optim.AdamW")
    check=replace(check,'        out, state = tracker.step(frame(i))',"        tracker.semantic_context['text']=contexts[int(arm=='category' and 33<=i<=64)]\n        assert torch.equal(tracker.semantic_context['mask'],mask_before)\n        assert torch.equal(tracker.semantic_context['initial'],initial_before)\n        out, state = tracker.step(frame(i))")
    check=replace(check,'formal_optimizer_steps=0, smoke_weights_saved=False','formal_optimizer_steps=0, smoke_weights_saved=False, training_text_switch_checked=True')
    (R/'check_causal.py').write_text(check)
    shutil.copyfile(B/'m80_evaluate_20260908.py',R/'evaluate.py')
    queue='''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$python" -u train_causal.py --arm category > training_category.log 2>&1
status=$?
printf '%s\\n' "$status" > training_category.exit
if [ "$status" -ne 0 ]; then printf '%s\\n' "$status" > controller.exit; exit "$status"; fi
run_eval() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u evaluate.py --condition "$1" > "eval_$1.log" 2>&1
    result=$?
    printf '%s\\n' "$result" > "eval_$1.exit"
    return "$result"
}
run_eval category 0 & first=$!
run_eval empty 1 & second=$!
wait "$first"; a=$?
wait "$second"; b=$?
if [ "$a" -ne 0 ] || [ "$b" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
run_eval swapped 0
status=$?
if [ "$status" -ne 0 ]; then printf '%s\\n' "$status" > controller.exit; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' "$python" -u evaluate.py --analyze > analysis.log 2>&1
status=$?
printf '%s\\n' "$status" > analysis.exit
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
    (R/'run.sh').write_text(queue);subprocess.run(['bash','-n',str(R/'run.sh')],check=True)
    t.update(status='prepared_before_checks',revision='m80_block_text_dropout_v1',primary_arm='category',control_arm='sealed_M78_empty',
        hypothesis='Fixed training text dropout may reduce brittle category conditioning under external domain change; language benefit must survive fixed-head content intervention.',
        architecture_control='Exactly M78 architecture, native runtime, loss, data order and optimizer. Only text content varies by fixed training block. No per-sequence GT reset.',
        text_control='Five slots/mask/padding unchanged; all valid slots become existing CLIP empty on scheduled blocks. No new caption. Full original category at deployment.',
        training_script_sha256=sha(R/'train_causal.py'),run_queue_sha256=sha(R/'run.sh'),text_dropout_source_sha256=sha(R/'text_dropout.py'),text_schedule_sha256=sha(R/'text_schedule.json'),
        parent_training_spec_sha256=sha(P/'training_spec.json'),causal_check_sha256=None,
        selection_protocol='Only seed2027 final M80 checkpoint. No probability sweep, no seed search. M78 category and independently trained Empty are reused controls.',
        after_training='Evaluate Category, Empty and existing Swapped on Train development22 regardless of gates. No automatic external evaluation.',
        scope_limitations=['Text dropout existed in historical SRTrack; this is a current-stack training intervention, not a novel module.',
            'Training block switches differ from static deployment; complete recursive results must test the consequence.',
            'Reused M78 controls are not a fresh independent replication. No random-seed variance claims.',
            'No early-checkpoint selection or historical union protection. Final public validation remains required.'],
        promotion_gate_scope='Native and sealed same-budget M78 Empty development gates; M78 Category is mandatory paired mechanism comparator, not a cumulative sequence protection union.',
        runtime_estimate_seconds_per_arm=14000)
    write(R/'prepared_training_spec.json',t)
    rs=read(P/'recursive_spec.json')
    write(R/'evaluation_spec.json',dict(cases=rs['cases'],native_result_path=rs['native_result_path'],native_result_sha256=t['native_result_sha256'],
        parent_result_path=str(P/'recursive_result.json'),parent_result_sha256=sha(P/'recursive_result.json'),
        evaluator_sha256=sha(R/'evaluate.py'),metric_sha256=sha(R/'recursive_metric.py'),
        swapped_bank=dict(path=str(B/'sttrack_m69_m65_content_diagnostic_20260907/swapped.pt'),sha256='955343f3e86d8e8d1bf15f29ee777de3b5098f37c80f2b2b5ea88aa44eda9125')))
    plan=dict(status='prepared',seed=2027,additional_seeds=[],training_arms=['category_with_block_dropout'],dropout_probability=.2,block_frames=32,
        scheduled_empty_calls=total_drop,training_calls=186694,expected_updates=5798,learned_parameters=289154,
        source_sha256=sha(__file__),checker_sha256=sha(R/'check_causal.py'),prepared_training_sha256=sha(R/'prepared_training_spec.json'),
        evaluation_sha256=sha(R/'evaluation_spec.json'),public_evaluation_allowed=False,independent_model_review_pass=False,
        reviewer_status='REVIEW_UNAVAILABLE: prior gpt-6-astra/max quota failure until September12; not retried or replaced.',
        prior_text_dropout='Historical SRTrack 15 percent text dropout is documented; current M78 trainer has no such condition.',
        observed_utc=datetime.now(timezone.utc).isoformat())
    write(R/'spec.json',plan)
    for p in R.glob('*.py'):ast.parse(p.read_text())
    print(json.dumps(plan))

def freeze():
    s=read(R/'spec.json');c=read(R/'causal_check.json')
    assert sha(__file__)==s['source_sha256']
    assert (R/'causal_check.exit').read_text().strip()=='0'
    assert c['status']=='completed_M80_native_parity_and_causal_smoke'
    assert c['checker_sha256']==s['checker_sha256']==sha(R/'check_causal.py')
    assert c['training_text_switch_checked'] and c['zero_residual_public_state_exact'] and c['combined_loss_contract_verified'] and c['base_frozen_all_arms']
    assert not c['smoke_weights_saved'] and c['formal_optimizer_steps']==0
    assert c['prepared_spec_sha256']==s['prepared_training_sha256']==sha(R/'prepared_training_spec.json')
    assert sha(R/'evaluation_spec.json')==s['evaluation_sha256']
    assert not (R/'frozen.json').exists() and not (R/'training').exists()
    t=read(R/'prepared_training_spec.json');t.update(status='frozen_before_training',causal_check_sha256=sha(R/'causal_check.json'))
    write(R/'training_spec.json',t)
    write(R/'frozen.json',dict(training_spec_sha256=sha(R/'training_spec.json'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),spec_sha256=sha(R/'spec.json'),check_sha256=sha(R/'causal_check.json'),public_evaluation_allowed=False))
    print(json.dumps(read(R/'frozen.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);a=p.parse_args();{'prepare':prepare,'freeze':freeze}[a.action]()
