"""Archive preparation and launch evidence without dataset images or model weights."""
from pathlib import Path
import hashlib,json,tarfile
R=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[]
for pattern in ['*.py','*.sh','*.md','*.json','*.exit','preflight.log','caption.log',
    'captions/*','caption_v1/*','preparation_attempt1/*.json','preparation_attempt1/*.exit','preparation_attempt1/*.log','code/**/*']:
    paths.extend(p for p in R.glob(pattern) if p.is_file())
paths=sorted(set(paths)-{R/'preparation_export_manifest.json'})
manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
mp=R/'preparation_export_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'preparation_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in paths+[mp]:tar.add(p,arcname=str(p.relative_to(R)))
print(json.dumps(dict(files=len(manifest),bytes=archive.stat().st_size,archive_sha256=sha(archive))))
