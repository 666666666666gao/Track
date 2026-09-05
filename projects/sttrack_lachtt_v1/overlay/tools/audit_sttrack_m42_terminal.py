"""Verify the sealed M42 run and explain candidate capacity without new fitting."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    assert (root/'controller.exit').read_text().strip()=='0'
    spec=json.loads((root/'spec.json').read_text());launch=json.loads((root/'launch.json').read_text())
    result=json.loads((root/'training_result.json').read_text());assert result['status']=='complete'
    repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from lib.models.sttrack.lachtt_local_spatial_association import LocalSpatialAssociation
    torch.set_num_threads(1)
    assert sha(root/'spec.json')==launch['spec_sha256']==result['spec_sha256']
    for name,digest in {**spec['source_sha256'],**launch['source_sha256']}.items():assert sha(repo/name)==digest,name
    assert result['trainer_sha256']==launch['source_sha256']['tools/train_sttrack_m42.py']
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    assert sha(root/'training_labels.json')==spec['labels_sha256']
    assert sha(root/'inference_inputs.json')==spec['inference_inputs_sha256']
    labels=json.loads((root/'training_labels.json').read_text())
    receipts=[]
    for shard in [0,1]:
        receipt=json.loads((root/f'shard{shard}_receipt.json').read_text())
        assert receipt['status']=='complete' and receipt['labels_opened'] is False
        assert receipt['spec_sha256']==sha(root/'spec.json')
        receipts.extend(receipt['sequences'])
    assert len(receipts)==len({r['sequence'] for r in receipts})==85
    assert sum(r['events'] for r in receipts)==spec['event_count']
    assert sum(r['frames'] for r in receipts)+len(receipts)==sum(spec['shard_frames'])
    for r in receipts:assert sha(root/'features'/(r['sequence']+'.pt'))==r['feature_sha256']
    weights={};coverage={}
    for variant in ['spatial','pooled']:
        values=result['variants'][variant]
        path=root/(variant+'_final.pth');assert sha(path)==values['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu')
        assert saved['base_checkpoint_sha256']==spec['checkpoint_sha256'] and saved['epochs']==20
        assert saved['trainer_sha256']==result['trainer_sha256'] and saved['variant']==variant
        assert len(values['losses'])==20 and all(torch.isfinite(torch.tensor(values['losses'])))
        assert values['reload_exact']
        torch.manual_seed(spec['optimization']['seed'])
        initial=LocalSpatialAssociation(spatial=variant=='spatial').state_dict()
        changes={k:float((saved['model'][k]-v).abs().max()) for k,v in initial.items()}
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        assert changes['candidate.2.weight']>0 or changes['none.2.weight']>0
        weights[variant]=dict(sha256=sha(path),parameters=values['parameters'],
            optimizer_steps=20*((result['fit_events']+31)//32),
            changed_parameter_tensors=sum(v>0 for v in changes.values()),max_abs_parameter_change=max(changes.values()),
            first_loss=values['losses'][0],final_loss=values['losses'][-1])
        coverage[variant]={}
        for split in ['fit','development_holdout']:
            groups=defaultdict(lambda:dict(events=0,selected_correct=0,changes=0,default_iou_sum=0.,selected_iou_sum=0.))
            for row in values[split]['rows']:
                if not labels[row['key']]['visible']:category='unavailable'
                elif row['default_iou']>=.5:category='default_good'
                elif row['oracle_iou']>=.5:category='non_good_with_correct_candidate'
                else:category='non_good_without_correct_candidate'
                group=groups[category];group['events']+=1;group['selected_correct']+=int(row['selected_iou']>=.5)
                group['changes']+=int(row['chosen']!=0);group['default_iou_sum']+=row['default_iou'];group['selected_iou_sum']+=row['selected_iou']
            coverage[variant][split]=dict(groups)
    audit=dict(status='complete',integrity_pass=True,sequence_count=85,events=spec['event_count'],
        replay_frames=sum(r['frames'] for r in receipts),initialization_frames=85,
        feature_bytes=sum(r['bytes'] for r in receipts),
        maximum_bbox_error_px=max(r['max_bbox_error_px'] for r in receipts),
        maximum_score_error=max(r['max_score_error'] for r in receipts),default_template_writes=sum(r['template_updates'] for r in receipts),
        weights=weights,candidate_capacity_groups=coverage,information_gate=result['information_gate'],
        information_gate_pass=result['information_gate_pass'],training_result_sha256=sha(root/'training_result.json'),
        auditor_sha256=sha(__file__),
        quarantine_scope='Mixed full152 prediction JSONs were parsed and filtered to85 selected sequences. Folds0/1 supplied no GT supervision, native features, fitting examples or evaluation results to M42.',
        limitation='One fixed data/head/feature test on previously used development folds. No public metric or general impossibility claim.')
    (root/'terminal_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='candidate_capacity_groups'},indent=2),flush=True)


if __name__=='__main__':
    main()
