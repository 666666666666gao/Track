from pathlib import Path
import hashlib,json,tarfile
R=Path(__file__).parent;M=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=json.loads((R/'started_publication_record.json').read_text())
assert sha(M)==record['previous_master_sha256']
assert sha(R/'started_master.md')==record['master_sha256'] and (R/'started_master.md').read_bytes().startswith(M.read_bytes())
assert sha(R/'started_publication.tar.gz')==record['archive_sha256']
T=R/'published_started';T.mkdir()
with tarfile.open(R/'started_publication.tar.gz') as archive:
    for member in archive.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    archive.extractall(T)
for row in json.loads((T/'publication_manifest.json').read_text()):assert sha(T/row['path'])==row['sha256']
M.write_bytes((R/'started_master.md').read_bytes())
print(json.dumps(dict(remote_master_sha256=sha(M),files=len(list(T.rglob('*'))))))
