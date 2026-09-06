from pathlib import Path
import datetime,hashlib,json
root=Path(__file__).resolve().parent
old=Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v1_20260906')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((old/'training_spec.json').read_text());inventory=json.loads((old/'dataset_inventory.json').read_text())
assert not (root/'training_spec.json').exists()
counts={}
for row in inventory['sequences']:
 n=row['sequence'];count=row['rgb_count'];assert count==row['depth_count']
 assert not row['rgb_extra'] and not row['depth_extra']
 assert row['rgb_minmax']==row['depth_minmax']==[1,count]
 if n=='toy07_indoor_320':
  assert count==1367 and row['gt_rows']==1406
  assert row['missing_rgb']==row['missing_depth']==list(range(1368,1407))
  assert s['fit_gt_sha256'][n]=='683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
 else:
  assert row['gt_rows']==count and not row['missing_rgb'] and not row['missing_depth']
 counts[n]=count
assert set(counts)==set(s['fit_sequences'])
s.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=2,
 preparation_sha256=sha(root/'preparation.json'),trainer_sha256=sha(root/'train.py'),
 previous_training_spec_sha256=sha(old/'training_spec.json'),previous_training_abort_sha256=sha(old/'training_abort.json'),
 dataset_inventory_sha256=sha(old/'dataset_inventory.json'),training_frame_counts=counts,
 training_gt_tail_contract={'sequence':'toy07_indoor_320','rgb_depth_contiguous_frames':1367,'gt_rows':1406,'gt_sha256':'683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2','ignored_annotation_only_tail':39,'source':'Existing project master section 24.152.3 and tools/analyze_sttrack_lachtt_train152_gatea.py load_ground_truth. Original image and GT files unchanged.'},
 temporary_checkpoint_root='/dev/shm/sttrack_m55_training_checkpoints_v2_20260906')
(root/'training_spec.json').write_text(json.dumps(s,indent=2)+'\n')
for name in ['run_training.sh','run_sampler_contract.sh','wait_and_launch_clone.sh']:
 text=(old/name).read_text().replace(str(old),str(root)).replace('sttrack_m55_clone_train_20260906','sttrack_m55_clone_train_v2_20260906')
 (root/name).write_text(text)
print(json.dumps({'spec_sha256':sha(root/'training_spec.json'),'trainer_sha256':sha(root/'train.py'),'fit_frame_total':sum(counts.values()),'ignored_gt_tail':39}))
