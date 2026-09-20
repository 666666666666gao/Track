import hashlib,json,tarfile
from pathlib import Path
R=Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
S=Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/recursive_spec.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
cases=read(R/'recursive_spec.json')['cases'];names={c['sequence'] for c in cases}
assert (R/'controller.exit').read_text().strip()=='0'
out=R/'native_reference';out.mkdir()
rows={s:[] for s in names};headers={}
for name,digest in read(S)['baseline_trace_sha256'].items():
    p=Path(name);assert sha(p)==digest
    data=read(p);assert data['complete'] and not data['ground_truth_used_after_initialization']
    assert data['checkpoint']['sha256']==read(R/'training_spec.json')['native_checkpoint_sha256']
    for row in data['rows']:
        if row['sequence'] in rows:rows[row['sequence']].append(row)
    headers[name]={k:v for k,v in data.items() if k!='rows'}
summary={}
for c in cases:
    s=c['sequence'];v=sorted(rows[s],key=lambda x:x['frame_index'])
    assert [x['frame_index'] for x in v]==list(range(c['frames']))
    clean=[dict(frame=x['frame_index'],bbox=x['public_bbox'],score=x['public_score']) for x in v]
    (out/(s+'.json')).write_text(json.dumps(dict(sequence=s,rows=clean))+'\n')
    e=read(R/'recursive/category_empty'/(s+'.json'))['rows']
    summary[s]=dict(frames=len(v),all_bbox_exact=all(x['bbox']==y['bbox'] for x,y in zip(clean,e)),
        noninitial_score_exact=all(x['score']==y['score'] for x,y in zip(clean[1:],e[1:])))
(out/'reference_provenance.json').write_text(json.dumps(dict(source_spec_path=str(S),source_spec_sha256=sha(S),source_trace_sha256=read(S)['baseline_trace_sha256'],headers=headers,parity=summary),indent=2)+'\n')
(out/'source_recursive_spec.json').write_bytes(S.read_bytes())
(out/'extract_native_reference.py').write_bytes(Path(__file__).read_bytes())
manifest={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'native_reference.tar.gz'
with tarfile.open(str(archive),'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(str(p),arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,bbox_all_exact=all(v['all_bbox_exact'] for v in summary.values()),score_all_exact=all(v['noninitial_score_exact'] for v in summary.values()))))
