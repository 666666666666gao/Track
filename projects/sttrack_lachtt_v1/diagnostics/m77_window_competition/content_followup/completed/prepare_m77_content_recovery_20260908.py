from pathlib import Path
import ast,difflib,hashlib,json

B=Path('/root/autodl-tmp');R=B/'sttrack_m77_window_competition_20260907';C=R/'content_followup';Q=C/'analysis_recovery_20260908'
source=B/'m77_content_followup_20260907.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest=='58e52b047f428857255e56ebd18a966f81f74fcb787217d5d52eebfdc6453a7e'
assert (C/'empty.exit').read_text().strip()==(C/'swapped.exit').read_text().strip()=='0'
assert (C/'content_analysis.exit').read_text().strip()==(C/'controller.exit').read_text().strip()=='1'
assert not (C/'result.json').exists() and not Q.exists()
text=source.read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='analyze'];assert len(nodes)==1
node=nodes[0];body='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
assert body.count('scalar(boxes,gt)')==1
corrected=body.replace('scalar(boxes,gt)','scalar(rows,gt)')
assert corrected.count("write(O/'result.json',result)")==1
corrected=corrected.replace("write(O/'result.json',result)","result.update(original_source_sha256=ORIGINAL_SHA, original_analysis_exit=1, recovery_scope='Only scalar input corrected from bbox array to frame records; no tracking, optimization or prediction changes', new_full_tracking_calls=0, reused_control_tracking_calls=66216, original_controller_exit_preserved=True)\n    write(Q/'result.json',result)")
header='''"""Offline M77 content analysis recovery; original failing controller remains sealed."""
from pathlib import Path
import sys,json,hashlib
from m77_content_followup_20260907 import R,O,AUDITOR,checked,read,sha,module,now,write
ORIGINAL_SHA='58e52b047f428857255e56ebd18a966f81f74fcb787217d5d52eebfdc6453a7e'
Q=O/'analysis_recovery_20260908'
assert sha('/root/autodl-tmp/m77_content_followup_20260907.py')==ORIGINAL_SHA
assert (O/'content_analysis.exit').read_text().strip()==(O/'controller.exit').read_text().strip()=='1'
assert not (Q/'result.json').exists()
'''
runner=B/'m77_content_analysis_recovery_20260908.py';assert not runner.exists();code=header+'\n'+corrected+'\nanalyze()\n';ast.parse(code)
Q.mkdir();runner.write_bytes(code.encode('utf-8'))
(Q/'analysis_function.diff').write_bytes(''.join(difflib.unified_diff(body.splitlines(True),corrected.splitlines(True),fromfile='original_analyze',tofile='recovered_analyze')).encode('utf-8'))
files=[C/'empty/receipt.json',C/'swapped/receipt.json',C/'activation.json',C/'spec.json',C/'completed_training_audit.json',C/'content_analysis.log',C/'controller.log',C/'controller.exit']
record=dict(original_source_sha256=digest,recovery_source_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),sealed_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},new_tracking_calls=0,new_optimizer_steps=0,seed=2027,additional_seeds=[])
(Q/'preparation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
