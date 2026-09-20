"""Wait on the existing controller, then collect sealed M87 evidence on CPU."""
import argparse
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path

R=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
M84=Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
ARMS=['category','category_empty','category_old','category_swapped']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def collect():
    for name in ['controller','training_category','recursive_analysis']+[a+'_recursive' for a in ARMS]:
        assert (R/(name+'.exit')).read_text().strip()=='0',name
    frozen=read(R/'frozen.json');training=read(R/'training_spec.json');spec=read(R/'recursive_spec.json')
    assert sha(R/'frozen.json')=='62dcab32fdca9baa34776ec7843f4fb2ed479f0bcf0b6b6a8a87b7cbc0663f0c'
    assert sha(R/'training_spec.json')==frozen['training_spec_sha256']
    assert sha(R/'recursive_spec.json')==frozen['recursive_spec_sha256']
    assert sha(R/'integration.json')==training['integration_sha256']
    trained=read(R/'training/category/result.json');result=read(R/'recursive_result.json')
    assert trained['status']=='one_full_causal_fit_pass_complete'
    assert trained['sequences']==130 and trained['total_track_calls']==186694 and trained['optimizer_steps']==5798
    assert trained['training_spec_sha256']==sha(R/'training_spec.json') and trained['base_parameters_and_buffers_unchanged']
    assert sha(R/'training/category/final.pth')==trained['final_checkpoint_sha256']==result['head_sha256']
    assert result['status']=='complete_recursive_development' and result['gate_count']==18
    assert result['training_spec_sha256']==sha(R/'training_spec.json')
    assert result['recursive_spec_sha256']==sha(R/'recursive_spec.json')
    assert sha(R/'training/category/sequence_log.jsonl')==trained['sequence_log_sha256']
    assert sha(R/'training/category/sampled_state_trace.jsonl')==trained['sampled_trace_sha256']
    files={}
    def add(path,name):
        path=Path(path);assert path.is_file(),str(path)
        assert name not in files,name
        files[name]=path
    root_names=['EXPERIMENT_PLAN.md','caption_spec.json','caption_result.json','caption_protocol.py','bank_result.json',
        'swapped_mapping.json','training_spec.json','recursive_spec.json','frozen.json','integration.json',
        'preflight_result.json','preparation_receipt.json','CODE_REVIEW.md','code_review_receipt.json','launch.json',
        'run_m87.sh','run_recursive.py','recursive_metric.py','train_causal.py','causal_training.py','support_loss.py',
        'window_competition.py','native_preservation.py','recursive_result.json','training_category.log','recursive_analysis.log',
        'controller.exit','training_category.exit','recursive_analysis.exit']
    for name in root_names:add(R/name,name)
    for name in ['final.pth','result.json','sequence_log.jsonl','sampled_state_trace.jsonl']:
        add(R/'training/category'/name,'training/category/'+name)
    for name,digest in read(R/'integration.json')['source_sha256'].items():
        assert sha(R/'code'/name)==digest,name
        add(R/'code'/name,'code/'+name)
    for split in ['fit','development']:
        for condition,entry in training['banks'][split].items():
            if not isinstance(entry,dict):continue
            assert sha(entry['path'])==entry['sha256']
            add(entry['path'],'banks/'+split+'_'+condition+'.pt')
    all_predictions={}
    for arm in ARMS:
        receipt=read(R/(arm+'_recursive_receipt.json'))
        assert sha(R/(arm+'_recursive_receipt.json'))==result['receipts'][arm]
        assert receipt['status']=='complete' and receipt['total_frames']==33130 and len(receipt['sequences'])==22
        assert receipt['head_sha256']==trained['final_checkpoint_sha256']
        assert receipt['training_result_sha256']==sha(R/'training/category/result.json')
        assert receipt['recursive_spec_sha256']==sha(R/'recursive_spec.json')
        assert [x['sequence'] for x in receipt['sequences']]==[c['sequence'] for c in spec['cases']]
        for suffix in ['_recursive_receipt.json','_recursive.exit','_recursive.log']:add(R/(arm+suffix),arm+suffix)
        all_predictions[arm]={}
        for case,item in zip(spec['cases'],receipt['sequences']):
            name=case['sequence'];p=R/'recursive'/arm/(name+'.json')
            assert sha(p)==item['sha256'] and item['frames']==case['frames']
            data=read(p);assert data['sequence']==name and data['arm']==arm
            assert [x['frame'] for x in data['rows']]==list(range(case['frames']))
            all_predictions[arm][name]=data['rows'];add(p,'recursive/'+arm+'/'+name+'.json')
    # No GT files are opened until every arm's sealed prediction set is verified.
    for case in spec['cases']:
        p=Path(training['dataset_root'])/case['sequence']/'groundtruth.txt'
        assert sha(p)==case['gt_sha256'];add(p,'development_gt/'+case['sequence']+'.txt')
    assert sha(training['parent_result_path'])==training['parent_result_sha256']
    assert sha(spec['native_result_path'])==training['native_result_sha256']
    add(training['parent_result_path'],'references/M84_recursive_result.json')
    add(spec['native_result_path'],'references/native_result.json')
    parent=read(training['parent_result_path'])
    previous=read(M84/'category_empty_recursive_receipt.json')
    assert sha(M84/'category_empty_recursive_receipt.json')==parent['receipts']['category_empty']
    assert previous['status']=='complete' and previous['total_frames']==33130 and len(previous['sequences'])==22
    add(M84/'category_empty_recursive_receipt.json','references/M84_empty_receipt.json')
    parity=[]
    for item in previous['sequences']:
        name=item['sequence'];p=M84/'recursive/category_empty'/(name+'.json')
        assert sha(p)==item['sha256']
        old=read(p)['rows'];new=all_predictions['category_empty'][name]
        assert len(old)==len(new)==item['frames']
        parity.append(dict(sequence=name,frames=len(new),bbox_equal=sum(a['bbox']==b['bbox'] for a,b in zip(new,old)),
            score_equal=sum(a['score']==b['score'] for a,b in zip(new,old)),reference_sha256=sha(p)))
        add(p,'references/M84_empty/'+name+'.json')
    parity_result=dict(reference='Sealed M84 Empty, previously audited against independent native trajectories; no new native inference',
        sequences=parity,all_bbox_score_equal=all(x['frames']==x['bbox_equal']==x['score_equal'] for x in parity),
        frozen_M87_gate_changed=False)
    write(OUT/'empty_reference_parity.json',parity_result)
    add(OUT/'empty_reference_parity.json','collection/empty_reference_parity.json')
    for name in ['COLLECTION_PLAN.md','collect_completed.py','run_collection.sh','launch.json']:add(OUT/name,'collection/'+name)
    manifest=[dict(path=name,sha256=sha(p),bytes=p.stat().st_size) for name,p in sorted(files.items())]
    write(OUT/'manifest.json',manifest)
    archive=OUT/'completed_evidence.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as t:
        for name,p in sorted(files.items()):t.add(p,arcname=name)
        t.add(OUT/'manifest.json',arcname='manifest.json')
    summary=dict(status='sealed_completed_evidence_collected',observed_utc=datetime.now(timezone.utc).isoformat(),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(manifest),
        final_checkpoint_sha256=trained['final_checkpoint_sha256'],recursive_result_sha256=sha(R/'recursive_result.json'),
        gates_passed=sum(sum(g.values()) for g in result['gates'].values()),gates_total=18,
        all_gates_pass=result['all_gates_pass'],empty_reference_exact=parity_result['all_bbox_score_equal'],
        free_disk_bytes=shutil.disk_usage('/root/autodl-tmp').free,new_training_steps=0,new_tracking_calls=0,
        independent_completed_audit=False,formal_evaluation_started=False)
    write(OUT/'collection_result.json',summary);print(json.dumps(summary),flush=True)

def wait_and_collect(pid):
    while True:
        state=subprocess.run(['ps','-p',str(pid),'-o','stat=,args='],stdout=subprocess.PIPE,text=True).stdout.strip()
        if not state or state.split()[0].startswith('Z'):break
        assert str(R/'run_m87.sh') in state,state
        print(json.dumps(dict(observed_utc=datetime.now(timezone.utc).isoformat(),status='controller_live_wait',pid=pid)),flush=True)
        time.sleep(240)
    collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller-pid',type=int,required=True);args=p.parse_args()
    wait_and_collect(args.controller_pid)
