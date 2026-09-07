from pathlib import Path
import json,hashlib,shutil,tarfile
BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m66_same_state_diagnostic_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'result.json')=='a2715e3b3c34760b2955ce58d835e1290918cee7ec8d3369ac84f5c17910dbe3'
for n in ['control','null','controller','analysis']:assert (ROOT/(n+'.exit')).read_text().strip()=='0'
out=ROOT/'publication_evidence';out.mkdir()
for n in ['spec.json','launch_receipt.json','control_receipt.json','null_receipt.json','control.log','null.log',
          'control.exit','null.exit','controller.exit','analysis.log','analysis.exit','result.json','null_mass_diagnostic.json','run_m66.sh']:
    shutil.copyfile(ROOT/n,out/n)
for n in ['m66_same_state_diagnostic_20260907.py','analyze_m66_same_state_20260907.py','figure_m66_glass_20260907.py']:
    shutil.copyfile(BASE/n,out/n)
image=ROOT/'glass03_frames.png'
(out/'figure_receipt.json').write_text(json.dumps(dict(filename=image.name,sha256=sha(image),bytes=image.stat().st_size,
    scope='Derived diagnostic visualization kept on server and Desktop, not included in Git evidence archive.'),indent=2)+'\n')
shutil.copyfile(__file__,out/'export_m66_completed.py')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'publication_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(sha256=sha(archive),bytes=archive.stat().st_size)))
