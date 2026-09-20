"""Read current external-evaluation inputs; do not decode images or run models."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,torch
B=Path('/root/autodl-tmp');R=Path(__file__).parent
E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation'
N=B/'sttrack_default_rgbd_ope_v1_20260906'
M=B/'sttrack_default_full127_v1_20260905/run/shard_manifest.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
old=read(E/'full_preparation_readiness.json')
fingerprints={**old['source_sha256'],**old['full_source_sha256']}
checked={p:sha(Path(p))==h for p,h in fingerprints.items()}
assert all(checked.values())
spec=read(N/'spec.json');inputs=read(N/'inputs.json');manifest=read(M)
assert sha(Path(spec['metric_source']))==spec['metric_source_sha256']
datasets={}
for name,count,expected in [('depthtrack',50,76373),('cdtb',80,101956)]:
    rows=[]
    assert len(inputs[name])==count
    for row in inputs[name]:
        folder=Path(row['root'])/row['sequence']
        rgb=sum(p.suffix=='.jpg' for p in (folder/'color').iterdir())
        depth=sum(p.suffix=='.png' for p in (folder/'depth').iterdir())
        assert rgb==depth==row['frames'],(name,row['sequence'],rgb,depth,row['frames'])
        assert (folder/'color/00000001.jpg').is_file() and (folder/'depth/00000001.png').is_file()
        rows.append(dict(sequence=row['sequence'],rgb_frames=rgb,depth_frames=depth))
    assert sum(r['rgb_frames'] for r in rows)==expected
    datasets[name]=dict(root=spec['datasets'][name]['root'],sequences=count,frames=expected,rows=rows)
root=Path(manifest['source_sequences_root'])
assert len(manifest['sequences'])==127 and manifest['total_anchor_count']==1765
assert all((root/n).is_dir() for n in manifest['sequences'])
bank_path=E/'low22_category.pt';bank=torch.load(bank_path,map_location='cpu')
oldspec=read(E/'spec.json');assert sha(bank_path)==oldspec['banks']['low22']['sha256']
train=read(B/'sttrack_m84_centered_20260920/training_spec.json')
fitpath=Path(train['banks']['fit']['category']['path']);assert sha(fitpath)==train['banks']['fit']['category']['sha256']
fit=torch.load(fitpath,map_location='cpu')
assert len(bank['keys'])==len(set(bank['keys']))==303 and torch.equal(bank['empty'].float(),fit['empty'].float())
bundle=read(E/'bundle.json');interface={}
for n,h in bundle['interface_sha256'].items():
    p=E/'interface'/n;assert sha(p)==h
    interface[n]=h
protocol=Path(bundle['text_protocol_path']);assert sha(protocol)==bundle['text_protocol_sha256']
assert sha(Path(bundle['base_checkpoint']))==bundle['base_checkpoint_sha256']==train['native_checkpoint_sha256']
generator=B/'sttrack_m58_initialization_generator_20260906/initialization_captions.py'
assert sha(generator)==old['generator_sha256']
report=dict(status='read_only_inputs_and_source_inventory_complete_not_evaluation_ready',observed_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(Path(__file__)),parent_readiness_sha256=sha(E/'full_preparation_readiness.json'),
    checked_parent_files=checked,OPE=datasets,VOT=dict(root=str(root),sequences=127,anchors=1765,
    planned_tracking_positions=manifest['total_estimated_frames'],sequence_directories_exist=True,frame_files_not_counted=True),
    old_low22_bank=dict(path=str(bank_path),sha256=sha(bank_path),unique_initializations=303,empty_matches_M84_fit=True,
    protocol_sha256=bank['protocol_sha256'],requires_new_protocol_binding=True),
    old_full_evaluation_directory_exists=(E/'full_evaluation').exists(),
    remaining_expected_initialization_observations=dict(depthtrack=50,cdtb=80,vot=1462),
    interface_sources=interface,base_checkpoint_sha256=bundle['base_checkpoint_sha256'],generator_sha256=sha(generator),
    new_caption_calls=0,new_tracking_calls=0,new_optimizer_steps=0,images_decoded=0,
    subsequent_groundtruth_opened=False,free_disk_bytes=shutil.disk_usage(R).free,
    blockers_to_claim_full_ready=['M84 final and completed development decision pending','External runtime architecture assertion must target centered_v1',
    'Bind new protocol/bundle without changing preserved low22 text tensors','Generate missing initialization observations only after eligibility',
    'Run and review actual final-weight OPE/TraX entry parity before external evaluation'],
    scope='OPE complete frame counts; VOT manifest and directory existence only. No new benchmark results or promotion.')
(R/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['OPE','checked_parent_files','interface_sources']},indent=2))
