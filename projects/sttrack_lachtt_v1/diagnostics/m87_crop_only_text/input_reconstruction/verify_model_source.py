from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
R=Path(__file__).resolve().parent
MODEL=Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct')
official=json.loads((R/'official_model_tree.json').read_text())
manifest=json.loads((MODEL/'PINNED_DOWNLOAD_MANIFEST.json').read_text())
revision='66285546d2b821cf421d4f5eb2576359d3770cd3'
assert manifest['repo_id']=='Qwen/Qwen2.5-VL-3B-Instruct' and manifest['revision']==revision
local={r['path']:r for r in manifest['files']}
assert set(local)=={r['path'] for r in official}
started=time.time();rows=[]
for entry in official:
    p=MODEL/entry['path'];size=p.stat().st_size
    assert entry['type']=='file' and size==entry['size']==local[entry['path']]['bytes']
    h256=hashlib.sha256();hgit=hashlib.sha1(('blob %d\0'%size).encode())
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h256.update(block)
            if 'lfs' not in entry:hgit.update(block)
    assert h256.hexdigest()==local[entry['path']]['sha256'],entry['path']
    remote_digest=entry['lfs']['oid'] if 'lfs' in entry else entry['oid']
    actual=h256.hexdigest() if 'lfs' in entry else hgit.hexdigest()
    assert actual==remote_digest,entry['path']
    rows.append(dict(path=entry['path'],bytes=size,sha256=h256.hexdigest(),official_digest_kind='lfs_sha256' if 'lfs' in entry else 'git_blob_sha1',official_digest=remote_digest,match=True))
result=dict(status='all_pinned_model_files_match_official_metadata',observed_utc=datetime.now(timezone.utc).isoformat(),
    repository=manifest['repo_id'],revision=revision,source_url='https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct/tree/'+revision,
    official_metadata_sha256=hashlib.sha256((R/'official_model_tree.json').read_bytes()).hexdigest(),
    local_manifest_sha256=hashlib.sha256((MODEL/'PINNED_DOWNLOAD_MANIFEST.json').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    files=rows,total_bytes=sum(r['bytes'] for r in rows),elapsed_seconds=time.time()-started,
    downloaded_weights=False,model_loaded=False,generation_calls=0,training_modified=False,
    limits=['Current file provenance only; not historical generation replay or semantic correctness'])
(R/'model_source_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],files=len(rows),bytes=result['total_bytes'],elapsed_seconds=result['elapsed_seconds'])))
