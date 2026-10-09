"""Train152 GT availability at template cadence points; no tracking or utility labels."""
import argparse,csv,hashlib,itertools,json
from datetime import datetime
from pathlib import Path
from analyze_m122_train_samples import valid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    dependency=Path(__file__).with_name('analyze_m122_train_samples.py')
    assert sha(dependency)==plan['analysis_source_sha256']
    spec_path=Path(plan['paths']['spec']);root=Path(plan['paths']['GT_root'])
    assert sha(spec_path)==plan['input_sha256'][str(spec_path)]
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    assert spec['seed']==2027 and len(spec['sequence_order'])==152
    inputs={'training_spec.json':sha(spec_path),'audited_validity_source.py':sha(dependency)}
    cadence=[];runs=[];sequences=[];direct_slice_checks=0
    for order,case in enumerate(spec['sequence_order']):
        name=case['sequence'];path=root/(name+'.txt')
        digest=sha(path)
        assert digest==case['groundtruth_sha256']==plan['input_sha256'][str(path)]
        inputs['groundtruth/'+name+'.txt']=digest
        truth=[[float(x) for x in line.split(',')] for line in path.read_text(encoding='utf-8').splitlines()]
        assert all(len(x)==4 for x in truth)
        raw_rows=len(truth)
        if name=='toy07_indoor_320':
            assert len(truth)==1406 and case['rgb_frames']==1367
            truth=truth[:1367]
        assert len(truth)==case['rgb_frames']==case['depth_frames']
        assert truth[0]==case['first_box'] and valid(truth[0])
        flags=[valid(box) for box in truth];n=len(flags)
        cumulative=[0]
        for flag in flags:
            cumulative.append(cumulative[-1]+int(flag))
        offset=0;sequence_runs=[]
        for is_valid,group in itertools.groupby(flags):
            length=sum(1 for _ in group)
            if not is_valid:
                row=dict(sequence=name,sequence_order=order,start=offset,end=offset+length,length=length,
                    next_valid_GT_frame=offset+length if offset+length<n else None)
                runs.append(row);sequence_runs.append(row)
            offset+=length
        assert offset==n
        for frame in range(50,n,50):
            for horizon in [32,64,128]:
                stop=min(n-1,frame+horizon)
                future_valid=cumulative[stop+1]-cumulative[frame+1]
                assert future_valid==sum(valid(box) for box in truth[frame+1:stop+1])
                direct_slice_checks+=1
                cadence.append(dict(sequence=name,sequence_order=order,frame=frame,horizon=horizon,current_GT_valid=flags[frame],
                    future_frames=stop-frame,future_valid_GT_frames=future_valid,
                    common_label_eligibility=flags[frame] and future_valid>0,
                    future_label_availability=future_valid>0,
                    unknown_current_with_future_label=not flags[frame] and future_valid>0))
        sequences.append(dict(sequence=name,sequence_order=order,raw_GT_rows=raw_rows,used_image_positions=n,
            initialization_positions=1,tracking_positions=n-1,valid_tracking_GT=sum(flags[1:]),invalid_tracking_GT=sum(not x for x in flags[1:]),
            cadence_points=len(range(50,n,50)),invalid_runs=len(sequence_runs),invalid_frames=sum(x['length'] for x in sequence_runs)))
    assert sum(x['raw_GT_rows'] for x in sequences)==219993
    assert sum(x['used_image_positions'] for x in sequences)==219954
    assert sum(x['tracking_positions'] for x in sequences)==219802
    assert sum(x['valid_tracking_GT'] for x in sequences)==203376
    assert sum(x['invalid_tracking_GT'] for x in sequences)==sum(x['length'] for x in runs)==16426
    summaries=[]
    for horizon in [32,64,128]:
        rows=[x for x in cadence if x['horizon']==horizon]
        summary=dict(horizon=horizon,cadence_points=len(rows),current_GT_known=sum(x['current_GT_valid'] for x in rows),
            current_GT_unknown=sum(not x['current_GT_valid'] for x in rows),future_GT_available=sum(x['future_label_availability'] for x in rows),
            common_label_eligible=sum(x['common_label_eligibility'] for x in rows),
            unknown_current_future_available=sum(x['unknown_current_with_future_label'] for x in rows),
            future_has_no_valid_GT=sum(x['future_frames']>0 and x['future_valid_GT_frames']==0 for x in rows),
            no_remaining_future_frames=sum(x['future_frames']==0 for x in rows))
        assert summary['future_GT_available']-summary['common_label_eligible']==summary['unknown_current_future_available']
        assert summary['future_GT_available']+summary['future_has_no_valid_GT']+summary['no_remaining_future_frames']==len(rows)
        summaries.append(summary)
    lengths=sorted(x['length'] for x in runs)
    result=dict(status='complete_Train152_GT_cadence_label_availability_diagnostic',observed_at=datetime.now().astimezone().isoformat(),
        source_sha256=sha(Path(__file__)),input_sha256=inputs,train_sequences=152,tracking_positions=219802,valid_tracking_GT=203376,invalid_tracking_GT=16426,
        invalid_run_count=len(runs),invalid_runs_longer_than_32=sum(x>32 for x in lengths),invalid_runs_longer_than_64=sum(x>64 for x in lengths),
        invalid_runs_longer_than_128=sum(x>128 for x in lengths),longest_invalid_run=max(lengths),
        invalid_frames_in_runs_longer_than_32=sum(x for x in lengths if x>32),summaries=summaries,direct_slice_verification_checks=direct_slice_checks,
        neural_executions=0,neural_progress_queries=0,optimizer_steps=0,existing_C_plan_changed=False,existing_R1_source_changed=False,
        scope='GT availability at every zero-based 50-frame cadence point, before native confidence eligibility; not actual write events or utility values.',
        limitations=['GT invalid is unknown, not evidence of target absence or wrong identity.',
            'Actual fixed-final native-qualified event counts require the future R1/R2 replay.',
            'Future GT availability alone does not supply W/K IoU differences or prove tracking improvements.',
            '64/128 are annotation coverage queries; they cross the next 50-frame write point and are not the existing 32-frame single-action teacher.',
            'This analysis changes no teacher horizon, training label mask, deployed policy or frozen evaluation.'])
    args.output.mkdir()
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,rows in [('cadence_rows.csv',cadence),('invalid_runs.csv',runs),('per_sequence.csv',sequences)]:
        with (args.output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False))


if __name__=='__main__':
    main()
