"""Derive paired multistart training from sealed M78 without text dropout."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,difflib,hashlib,json,shutil,subprocess
import numpy as np
B=Path('/root/autodl-tmp');P=B/'sttrack_m78_raw_competition_20260908';R=B/'sttrack_m81_multistart_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
def prepare():
    assert not (R/'prepared_training_spec.json').exists()
    t=read(P/'training_spec.json');inv=read(R/'inventory_spec.json');bind=read(R/'banks/binding_result.json')
    assert sha(P/'training_spec.json')==inv['parent_training_spec_sha256']
    assert bind['status']=='complete_paired_multistart_text_binding' and bind['total_episode_bindings']==500
    assert bind['inventory_sha256']==sha(R/'inventory_spec.json')
    assert sha(P/'train_causal.py')==t['training_script_sha256']
    assert shutil.disk_usage(R).free>700000000
    for n,h in read(P/'integration.json')['source_sha256'].items():
        src=P/'code'/n;assert sha(src)==h
        dst=R/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for n in ['integration.json','data_inventory.json','causal_training.py','support_loss.py','recursive_metric.py','window_competition.py']:
        shutil.copyfile(P/n,R/n)
    (R/'native_parity').mkdir()
    for arm in ['category','empty']:
        shutil.copyfile(P/'native_parity'/(arm+'_zero.pth'),R/'native_parity'/(arm+'_zero.pth'))
    helper='''"""Initialize only a predeclared training/development episode."""
def initialize_episode(tracker, bank, episode, image):
    index = bank['ids'].index(episode['id'])
    tracker.initialize(image, dict(init_bbox=episode['init_bbox'], text_tokens=bank['tokens'][index],
        text_mask=bank['mask'][index], empty_text=bank['empty']))
    assert tracker.frame_id == 0
    return index
'''
    (R/'multistart.py').write_text(helper)
    original=(P/'train_causal.py').read_text();train=original.replace(str(P),str(R)).replace('Single-seed category/empty causal training with raw-score hard-negative competition.','Fixed-budget multistart category/empty causal training with M78 raw competition.')
    train=replace(train,'    from causal_training import CausalTrainingTracker',"    from multistart import initialize_episode\n    assert sha(root / 'fit_initializations.json') == spec['fit_initializations_sha256']\n    assert sha(root / 'multistart.py') == spec['multistart_source_sha256']\n    manifest = json.loads((root / 'fit_initializations.json').read_text())\n    from causal_training import CausalTrainingTracker")
    train=replace(train,'    receipts = []','    receipts = []\n    initialization_count = 0')
    a="            index = bank['sequences'].index(row['sequence'])\n            info = dict(init_bbox=row['first_box'], text_tokens=bank['tokens'][index],\n                        text_mask=bank['mask'][index], empty_text=bank['empty'])"
    b="            episodes = [e for e in manifest['episodes'] if e['sequence'] == row['sequence']]\n            by_start = {e['start_frame']: e for e in episodes}\n            initialization_records = []\n            for episode in episodes:\n                assert sha(episode['image']) == episode['image_sha256']\n                assert sha(episode['depth']) == episode['depth_sha256']\n            assert episodes[0]['start_frame'] == 0 and episodes[-1]['end_frame_inclusive'] == n-1"
    train=replace(train,a,b)
    train=replace(train,'            tracker.initialize(frame(0), info)',"            current_episode = by_start[0]\n            initialize_episode(tracker, bank, current_episode, frame(0))\n            initialization_count += 1\n            initialization_records.append(current_episode['id'])")
    train=replace(train,'                out, state = tracker.step(frame(frame_index))',"                if frame_index > 1 and frame_index-1 in by_start:\n                    assert pending == 0 and gradient_frames == 0\n                    current_episode = by_start[frame_index-1]\n                    initialize_episode(tracker, bank, current_episode, frame(frame_index-1))\n                    initialization_count += 1\n                    initialization_records.append(current_episode['id'])\n                out, state = tracker.step(frame(frame_index))\n                state['episode_start'] = current_episode['start_frame']\n                state['episode_frame_id'] = tracker.frame_id\n                assert tracker.frame_id == frame_index-current_episode['start_frame']")
    train=replace(train,'if frame_index % 50 == 0 or frame_index == 1 or frame_index == n - 1:', 'if frame_index % 50 == 0 or frame_index-1 in by_start or frame_index == n - 1:')
    train=replace(train,'frames=n, track_calls=n-1, supervised_frames=supervised, label_counts=dict(labels),','frames=n, track_calls=n-1, initializations=initialization_records, cumulative_initializations=initialization_count, supervised_frames=supervised, label_counts=dict(labels),')
    train=replace(train,"status='complete' if complete else 'in_progress', seed=spec['seed'])","status='complete' if complete else 'in_progress', seed=spec['seed'], initializations=initialization_count, initialization_manifest_sha256=spec['fit_initializations_sha256'])")
    train=replace(train,"    assert frame_count == spec['total_training_track_calls']","    assert frame_count == spec['total_training_track_calls']\n    assert initialization_count == 426 and steps == 5798")
    train=replace(train,'gt_after_prediction_for_loss_only=True, gt_reinitialization_after_first_frame=False,','gt_after_prediction_for_loss_only=True, gt_reinitialization_after_first_frame=True,\n        gt_reinitialization_policy=\'predeclared_episode_boundaries_only_never_failure_triggered\', initializations=initialization_count, initialization_manifest_sha256=spec[\'fit_initializations_sha256\'],')
    (R/'train_causal.py').write_text(train)
    (R/'training_change.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),train.splitlines(True),fromfile='M78/train_causal.py',tofile='M81/train_causal.py')))
    for arm in ['category','empty']:
        t['banks']['fit'][arm]=bind['banks']['fit_'+arm]
    t.update(status='prepared_before_checks',revision='m81_multistart_v1',
        hypothesis='Additional predeclared initialization observations may improve same-budget category-conditioned tracking across reference states.',
        architecture_control='M78 architecture, native inference, raw competition, initial weights, original tracking transitions and accumulation windows. Only fixed episode initialization protocol changes.',
        parent_training_spec_sha256=sha(P/'training_spec.json'),fit_initializations_sha256=sha(R/'fit_initializations.json'),development_initializations_sha256=sha(R/'development_initializations.json'),
        multistart_source_sha256=sha(R/'multistart.py'),training_script_sha256=sha(R/'train_causal.py'),binding_result_sha256=sha(R/'banks/binding_result.json'),causal_check_sha256=None,
        state_protocol='Initialize at426 predeclared fit episode boundaries using that frame box and initialization-only caption. No GT reset inside episodes. Every original chronological transition retained once; detach states.',
        initializations=426,additional_initializations=296,
        selection_protocol='Only seed2027 final Category; paired Empty and sealed M78 controls. No probability, seed or checkpoint selection.',
        promotion_gate_scope='Native and current same-budget Empty on full t0 development; additional matched-manifest multistart and original-content checks. No historical union.',
        after_training='Complete two trained heads, t0 four-family and multistart five-family evaluation regardless of gates; no automatic public evaluation.',
        scope_limitations=inv['limitations']+['No M80 dropout, consistency loss, reverse training, new memory or changed Hann/template policy.'],
        runtime_estimate_seconds_per_arm=15000)
    # Pick a fit-only legal later initialization with a fully valid 101-frame smoke span.
    smoke=None
    for ep in read(R/'fit_initializations.json')['episodes']:
        if ep['start_frame']==0 or ep['track_calls']<101:continue
        gt=np.loadtxt(Path(t['dataset_root'])/ep['sequence']/'groundtruth.txt',delimiter=',').reshape(-1,4)
        window=gt[ep['start_frame']:ep['start_frame']+102]
        if np.isfinite(window).all() and (window[:,2:]>0).all():smoke=ep;break
    assert smoke is not None
    write(R/'smoke_episode.json',smoke)
    check=(P/'check_causal.py').read_text().replace(str(P),str(R)).replace('completed_M78_','completed_M81_')
    check=replace(check,'from causal_training import CausalTrainingTracker, base_supervision','from multistart import initialize_episode\nfrom causal_training import CausalTrainingTracker, base_supervision')
    check=replace(check,"row = next(x for x in spec['sequence_order'] if x['sequence'] == 'chair01_indoor')", "episode=json.loads((ROOT/'smoke_episode.json').read_text())\nassert sha(ROOT/'smoke_episode.json')==spec['smoke_episode_sha256']\nrow = next(x for x in spec['sequence_order'] if x['sequence'] == episode['sequence'])\noffset=episode['start_frame'];assert offset>0")
    check=replace(check,"bank = torch.load(b['path'], map_location='cpu'); j = bank['sequences'].index(row['sequence'])", "bank = torch.load(b['path'], map_location='cpu'); j = bank['ids'].index(episode['id'])")
    check=check.replace("dict(init_bbox=row['first_box']", "dict(init_bbox=episode['init_bbox']")
    check=replace(check,'def frame(i):\n    return','def frame(i):\n    i += offset\n    return')
    check=replace(check,"for arm, tracker in trackers.items(): tracker.initialize(frame(0), dict(infos[arm]))", "for arm, tracker in trackers.items():\n    bank=torch.load(spec['banks']['fit'][arm]['path'],map_location='cpu')\n    initialize_episode(tracker,bank,episode,frame(0))")
    check=replace(check,"writes = {arm: [] for arm in trackers}", "writes = {arm: [] for arm in trackers}\nnative_writes=[]")
    check=replace(check,'        image = frame(i); baseline = native.track(image)',"        image = frame(i); baseline = native.track(image)\n        if i%50==0 and float(baseline['best_score'])>.75:native_writes.append(i)")
    check=replace(check,"assert writes['category'] == writes['empty'] == [100]", "assert writes['category'] == writes['empty'] == native_writes")
    check=replace(check,"gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)","gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)[offset:]")
    check=replace(check,'formal_optimizer_steps=0, smoke_weights_saved=False','formal_optimizer_steps=0, smoke_weights_saved=False, later_initialization_frame=offset, smoke_episode_sha256=sha(ROOT/\'smoke_episode.json\')')
    (R/'check_causal.py').write_text(check)
    t['smoke_episode_sha256']=sha(R/'smoke_episode.json')
    shutil.copyfile(B/'m81_evaluate_20260908.py',R/'evaluate.py')
    eval_spec=dict(parent_recursive_spec_sha256=sha(P/'recursive_spec.json'),cases=read(P/'recursive_spec.json')['cases'],
        evaluator_sha256=sha(R/'evaluate.py'),metric_sha256=sha(R/'recursive_metric.py'),
        native_result_path=read(P/'recursive_spec.json')['native_result_path'],native_result_sha256=t['native_result_sha256'],
        parent_result_path=str(P/'recursive_result.json'),parent_result_sha256=sha(P/'recursive_result.json'),
        parent_heads={a:dict(path=str(P/'training'/a/'final.pth'),sha256=sha(P/'training'/a/'final.pth')) for a in ['category','empty']},
        multi_banks={a:bind['banks']['development_'+a] for a in ['category','empty']},
        swapped_bank=dict(path=str(B/'sttrack_m69_m65_content_diagnostic_20260907/swapped.pt'),sha256='955343f3e86d8e8d1bf15f29ee777de3b5098f37c80f2b2b5ea88aa44eda9125'),
        families=['t0_category','t0_empty_trained','t0_empty_content','t0_swapped','multi_category','multi_empty_trained','multi_native','multi_M78_category','multi_M78_empty_trained'],
        public_evaluation_allowed=False)
    write(R/'evaluation_spec.json',eval_spec)
    shutil.copyfile(B/'m81_run_20260908.sh',R/'run.sh');subprocess.run(['bash','-n',str(R/'run.sh')],check=True)
    t['run_queue_sha256']=sha(R/'run.sh');write(R/'prepared_training_spec.json',t)
    p=dict(status='training_prepared_not_frozen',source_sha256=sha(__file__),seed=2027,additional_seeds=[],prepared_training_sha256=sha(R/'prepared_training_spec.json'),checker_sha256=sha(R/'check_causal.py'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),independent_model_review_pass=False,review_status='Previously requested gpt-6-astra/max unavailable; no replacement or invented PASS.',wait_for='M80 actual completion and idle GPUs before launch',formal_training_started=False,observed_utc=datetime.now(timezone.utc).isoformat())
    write(R/'training_preparation.json',p)
    for p in R.glob('*.py'):ast.parse(p.read_text())
    print(json.dumps(read(R/'training_preparation.json')))
def freeze():
    p=read(R/'training_preparation.json');c=read(R/'causal_check.json')
    assert sha(__file__)==p['source_sha256'] and sha(R/'prepared_training_spec.json')==p['prepared_training_sha256']==c['prepared_spec_sha256']
    assert sha(R/'check_causal.py')==p['checker_sha256']==c['checker_sha256']
    assert (R/'causal_check.exit').read_text().strip()=='0'
    assert c['status']=='completed_M81_native_parity_and_causal_smoke' and c['later_initialization_frame']>0
    assert c['zero_residual_public_state_exact'] and c['combined_loss_contract_verified'] and c['base_frozen_all_arms']
    assert not c['smoke_weights_saved'] and c['formal_optimizer_steps']==0
    assert not (R/'training').exists() and not (R/'frozen.json').exists()
    assert sha(R/'evaluation_spec.json')==p['evaluation_spec_sha256']
    t=read(R/'prepared_training_spec.json');t.update(status='frozen_before_training',causal_check_sha256=sha(R/'causal_check.json'))
    write(R/'training_spec.json',t)
    write(R/'frozen.json',dict(training_spec_sha256=sha(R/'training_spec.json'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),preparation_sha256=sha(R/'training_preparation.json'),causal_check_sha256=sha(R/'causal_check.json'),public_evaluation_allowed=False))
    print(json.dumps(read(R/'frozen.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);a=p.parse_args();{'prepare':prepare,'freeze':freeze}[a.action]()
