from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,subprocess

B=Path('/root/autodl-tmp');E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation';Q=E/'full_followup'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not Q.exists() and not (E/'low22_controller.exit').exists() and not (E/'full_evaluation').exists()
spec=read(E/'spec.json');entry=read(E/'entry_parity/result.json');low=read(E/'low22_execution.json')
assert entry['status']=='real_learned_OPE_and_TraX_entries_verified'
assert entry['bundle_sha256']==low['bundle_sha256']==sha(E/'bundle.json')
assert low['gate']==spec['full_evaluation_gate']
parent=Path('/proc/11656');st=(parent/'stat').read_text().split()
assert st[21]=='4585466805' and st[2]!='Z'
assert (parent/'cmdline').read_bytes().split(b'\x00')[:2]==[b'bash',b'/root/autodl-tmp/m78_candidate_low22_20260908.sh']
source=B/'m78_low22_full_followup_20260908.py';runner=B/'m78_low22_full_followup_20260908.sh';full=B/'m78_full_evaluation_20260908.sh'
ast.parse(source.read_text());subprocess.run(['bash','-n',str(runner)],check=True)
sources=E/'prepared_entry_and_full.json'
assert sha(full)==read(sources)['source_files'][full.name]
Q.mkdir()
s=dict(status='frozen_conditional_full_followup',observed_utc=datetime.now(timezone.utc).isoformat(),seed=2027,additional_seeds=[],poll_seconds=240,
    source_sha256=sha(source),runner_sha256=sha(runner),preparer_sha256=sha(__file__),parent_pid=11656,parent_start_ticks=st[21],parent_command_sha256=sha(parent/'cmdline'),
    bundle_sha256=sha(E/'bundle.json'),low22_execution_sha256=sha(E/'low22_execution.json'),candidate_spec_sha256=sha(E/'spec.json'),evaluation_sources_manifest_sha256=sha(sources),full_script=str(full),full_script_sha256=sha(full),
    policy='Wait for complete low22 and retain all frozen checks; run the already prepared same-bundle full evaluation only on pass. No threshold, caption, checkpoint or seed selection.',
    actual_full_tracking_started=False,goal_completion_claimed=False,independent_model_review_pass=False)
(Q/'spec.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
