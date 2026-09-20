"""Stage the centered architecture assertion; no final bundle or execution."""
from pathlib import Path
import ast,hashlib,json,shutil
R=Path(__file__).parent
P=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\m67\evaluation_interface')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=json.loads((R/'readiness.json').read_text())['interface_sources']
T=R/'interface';T.mkdir()
for n,h in sources.items():
    assert sha(P/n)==h
    shutil.copyfile(P/n,T/n)
path=T/'semantic_runtime.py';old=path.read_bytes()
assert old.count(b"'semantic_spatial_support_v1'")==1
path.write_bytes(old.replace(b"'semantic_spatial_support_v1'",b"'semantic_spatial_centered_v1'"))
for n in sources:ast.parse((T/n).read_text())
result=dict(status='centered_external_interface_source_prepared_not_runtime_tested',source_sha256=sha(Path(__file__)),
    parent_sha256=sources,source_files={n:sha(T/n) for n in sources},
    only_changed_file='semantic_runtime.py',only_change='Require semantic_spatial_centered_v1 architecture instead of support_v1',
    unchanged_entry_points=['run_semantic_ope.py','run_semantic_vot.py','m39_vot_bridge.py','initialization_text.py'],
    final_bundle_created=False,live_entry_parity_run=False,new_tracking_calls=0,source_parsing_pass=True,
    remaining='After completed M84 decision, bind final/base/config/text protocol; verify actual OPE/TraX parity before any benchmark run.')
(R/'interface_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
