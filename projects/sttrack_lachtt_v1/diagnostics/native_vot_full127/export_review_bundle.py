from pathlib import Path
import datetime,hashlib,json,tarfile
root=Path(__file__).resolve().parent
assert all((root/x).read_text()=='0\n' for x in ['evaluate.exit','finalize.exit','controller.exit'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=json.loads((root/'result.json').read_text());manifest=json.loads((root/'run/shard_manifest.json').read_text())
assert result['status']=='complete' and result['integrity_pass']
files={}
for name in ['spec.json','preseed_receipt.json','execution_binding.json','finalize_default_full127.py','run.sh','launch.json','evaluate.log','evaluate.exit','finalize.log','finalize.exit','controller.exit','analysis.log','result.json','export_review_bundle.py','run/shard_manifest.json','run/merge_result.json','run/master/config.yaml','run/master/trackers.ini','run/master/sequences/list.txt']:
 files[name]=root/name
analysis=Path(result['analysis']);files[analysis.relative_to(root).as_posix()]=analysis
for p in (root/'run/master/results'/result['tracker']/'baseline').rglob('*'):
 if p.is_file():files[p.relative_to(root).as_posix()]=p
for seq in manifest['sequences']:
 for p in (root/'run/master/sequences'/seq).iterdir():
  if p.is_file():files[p.relative_to(root).as_posix()]=p
package=Path('/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot')
for name in ['analysis/multistart.py','analysis/__init__.py','region/io.py','region/__init__.py','region/raster.py','region/shapes.py','experiment/multistart.py','dataset/common.py','dataset/__init__.py','stack/vot2022/rgbd.yaml']:
 p=package/name
 assert p.is_file(),name
 files['source_snapshot/vot/'+name]=p
files['source_snapshot/finalize_vot_transaction_low22.py']=Path('/home/SUTrack_RGBD_L/tools/finalize_vot_transaction_low22.py')
records={name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(files.items())}
assert sum(name.endswith('.bin') for name in records)==1765
assert sum(name.endswith('/groundtruth.txt') for name in records)==127
receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(root),'result_sha256':sha(root/'result.json'),'images_included':False,'predictions_sealed_before_GT_copy':True,'files':records}
(root/'review_bundle_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
archive=Path('/dev/shm/sttrack_default_full127_review_20260906.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
 tar.add(root/'review_bundle_manifest.json',arcname='review_bundle_manifest.json',recursive=False)
info={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'member_files':len(records)+1,'source_bytes':sum(x['bytes'] for x in records.values()),'manifest_sha256':sha(root/'review_bundle_manifest.json')}
(root/'review_archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
