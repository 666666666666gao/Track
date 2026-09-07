"""Prepare deferred single-head controls without modifying the running pair."""
from pathlib import Path
import ast,hashlib,json,importlib.util
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';P=B/'sttrack_m77_window_competition_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
audit=B/'audit_m78_completed_20260908.py';source=B/'m78_content_followup_20260908.py'
assert not audit.exists() and not source.exists() and not (R/'content_followup').exists()
old=B/'audit_m77_completed_20260907.py';assert sha(old)=='4be53fa556a36d70630baba1b90a8052aab6f3615598fcd0badc902799a13bf2'
s=old.read_text().replace(P.name,R.name).replace('M77','M78').replace('m77_','m78_')
s=replace(s,"return dict(prior_control_success_protection=all(per['category'][s]['failure_episodes']==0 for s in t['protected_prior_control_sequences']),\n        mean_vs_native=","return dict(mean_vs_native=")
start=s.index("        mean_vs_M73_category=")
end=s.index('\n\ndef helpers()',start)
s=s[:start].rstrip().rstrip(',')+')\n'+s[end:]
s=s.replace("len(g)==15","len(g)==10").replace('competition_hann_top1_index','competition_raw_top1_index')
ast.parse(s);audit.write_text(s)
old=B/'m77_content_followup_20260907.py';assert sha(old)=='58e52b047f428857255e56ebd18a966f81f74fcb787217d5d52eebfdc6453a7e'
s=old.read_text().replace(P.name,R.name).replace('M77','M78').replace('m77_','m78_').replace('audit_m78_completed_20260907.py','audit_m78_completed_20260908.py')
s=replace(s,'iou,recomputed,spans=scalar(boxes,gt)','iou,recomputed,spans=scalar(rows,gt)')
start=s.index('def prepare():');end=s.index('\ndef checked():',start)
prep='''def prepare():
    import torch
    import numpy as np
    assert not O.exists()
    assert sha(R/'training_spec.json')=='57cdd314efd5359fa2e16be4f364c5568f1ca498b41df718517173610fe4d865'
    assert sha(R/'recursive_spec.json')=='fcef09d58ec79f690d6088e7f8fbc16a4ac42ef3c2a5b3c1c1cef389ed6a6f9f'
    t=read(R/'training_spec.json');rs=read(R/'recursive_spec.json')
    previous=read(B/'sttrack_m77_window_competition_20260907/content_followup/spec.json')
    banks=previous['banks'];values={n:torch.load(v['path'],map_location='cpu') for n,v in banks.items()}
    c=values['category'];assert c['tokens'].shape==(22,5,768)
    assert banks['category']==t['banks']['development']['category'] and banks['empty']==t['banks']['development']['empty']
    for n,v in values.items():
        assert sha(banks[n]['path'])==banks[n]['sha256']
        assert v['sequences']==c['sequences'] and torch.equal(v['mask'],c['mask']) and torch.equal(v['empty'],c['empty'])
        assert torch.equal(v['tokens'][~c['mask']],c['tokens'][~c['mask']])
    assert torch.equal(values['empty']['tokens'][c['mask']],c['empty'].expand_as(c['tokens'][c['mask']]))
    assert torch.equal(values['swapped']['tokens'][:,1:],c['tokens'][:,1:])
    assert bool((values['swapped']['tokens'][:,0]!=c['tokens'][:,0]).any(1).all())
    launch=read(R/'launch.json');pid=launch['controller_pid'];parent=Path('/proc')/str(pid)
    assert parent.exists() and (parent/'cwd').resolve()==R
    ticks=(parent/'stat').read_text().split()[21]
    argv=[x.decode() for x in (parent/'cmdline').read_bytes().split(b'\\0') if x]
    assert argv==['bash',str(R/'run_pair.sh')] and not (R/'controller.exit').exists()
    helper=module('m78_preparation_audit',AUDITOR);_,scalar=helper.helpers()
    old=B/'sttrack_m77_window_competition_20260907';prior=read(old/'recursive_result.json')
    g=helper.gates(prior['per_sequence'],prior['aggregates'],prior['parent_M73_aggregates'],t)
    assert len(g)==10 and g=={k:v for k,v in prior['gates'].items() if k in g}
    # Real sealed rows exercise the interface that failed in M77, before any M78 output exists.
    seq=rs['cases'][0]['sequence'];rows=read(old/'recursive/category'/ (seq+'.json'))['rows']
    gt=np.loadtxt(Path(t['dataset_root'])/seq/'groundtruth.txt',delimiter=',').reshape(-1,4)
    independent=scalar.scalar_metric();iou,metrics,spans=independent(rows,gt)
    sys.path.insert(0,str(R));from recursive_metric import statistics
    reference=statistics(np.asarray([r['bbox'] for r in rows]),gt)
    assert all(abs(metrics[k]-reference[k])<1e-8 for k in reference)
    O.mkdir()
    spec=dict(status='frozen_M78_followup_before_final_outputs',observed_utc=now(),source_sha256=sha(__file__),auditor_sha256=sha(AUDITOR),
        training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),seed=2027,additional_seeds=[],
        parent_pid=pid,parent_start_ticks=ticks,parent_argv=argv,poll_seconds=240,banks=banks,cases=rs['cases'],prefix_sequences=previous['prefix_sequences'],prefix_frames=102,
        head_selection='Fixed M78 Category final; no metric checkpoint selection, regardless of primary gate.',
        descriptive_lexical_criteria=previous['descriptive_lexical_criteria'],new_full_track_calls=66216,prefix_track_calls=303,new_optimizer_steps=0,
        public_evaluation_allowed=False,independent_model_review_pass=False,estimated_content_seconds=2100)
    write(O/'spec.json',spec)
    write(O/'preparation.json',dict(source_sha256=sha(__file__),spec_sha256=sha(O/'spec.json'),auditor_sha256=sha(AUDITOR),historical_gate_contract=g,
        rows_metric_interface_verified=True,reference_sequence=seq,reference_metrics=reference,prepared_before_current_final_outputs=True,current_GPU_calls=0))
    print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),preparation_sha256=sha(O/'preparation.json'))))
'''
s=s[:start]+prep+s[end:];ast.parse(s);source.write_text(s)
print(json.dumps(dict(auditor_sha256=sha(audit),source_sha256=sha(source))))
