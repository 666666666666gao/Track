"""Export sealed diagnostic evidence only after the saved-output verifier succeeds."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,time

R=Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
while not (R/'verification.exit').exists():time.sleep(240)
assert (R/'verification.exit').read_text().strip()=='0'
for name in ['replay.exit','analysis.exit','controller.exit']:
    assert (R/name).read_text().strip()=='0'
receipt=json.loads((R/'predictions/receipt.json').read_text())
assert receipt['status']=='complete' and receipt['positions']==33108 and len(receipt['sequences'])==22
names=['spec.json','launch.json','m83_same_state.py','analyze_m83.py','run_m83.sh',
       'verify_m83_saved.py','watch_verification.py','export_m83_completed.py',
       'replay.log','analysis.log','verification.log','replay.exit','analysis.exit',
       'controller.exit','verification.exit','diagnostic_result.json','saved_diagnostic_verification.json',
       'predictions/receipt.json']
for row in receipt['sequences']:
    name='predictions/'+row['sequence']+'.json'
    assert sha(R/name)==row['sha256']
    names.append(name)
manifest=dict(status='complete_export_after_scalar_verification',observed_utc=datetime.now(timezone.utc).isoformat(),
    files=[dict(path=n,sha256=sha(R/n),bytes=(R/n).stat().st_size) for n in names])
(R/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'completed_review_evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    for n in names+['export_manifest.json']:t.add(R/n,arcname=n)
out=dict(status='complete',observed_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    manifest_sha256=sha(R/'export_manifest.json'))
(R/'export_receipt.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
