"""Derive M78 evaluation entry from the sealed M77 interface, without running it."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json,subprocess
B=Path('/root/autodl-tmp');P=B/'sttrack_m77_window_competition_20260907/candidate_evaluation'
R=B/'sttrack_m78_raw_competition_20260908';E=R/'candidate_evaluation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
entry=['m77_candidate_evaluation_20260907.py','m77_learned_entry_parity_20260907.py','m77_vot_low22_20260907.py','m77_candidate_entry_20260907.sh','m77_candidate_low22_20260907.sh']
full=['m77_full_preparation_20260907.py','m77_full_vot_20260907.py','verify_m77_three_datasets_20260907.py','m77_full_evaluation_20260907.sh','check_m77_evaluation_preparation_20260907.py']
names={n:n.replace('m77','m78').replace('20260907','20260908') for n in entry+full}
def transform(s):
    s=s.replace('sttrack_m77_window_competition_20260907','sttrack_m78_raw_competition_20260908')
    for old,new in names.items():s=s.replace(old,new)
    s=s.replace('m77_content_followup_20260907.py','m78_content_followup_20260908.py')
    return s.replace('M77','M78').replace('m77','m78').replace('all fifteen','all ten').replace('All fifteen','All ten')
def write_source(n,s):
    p=B/names[n];assert not p.exists()
    if p.suffix=='.py':ast.parse(s)
    p.write_text(s)
    if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
def prepare_entry():
    assert not E.exists() and not (R/'training/category/final.pth').exists()
    old=read(P/'spec.json');content=read(R/'content_followup/spec.json')
    assert content['source_sha256']==sha(B/'m78_content_followup_20260908.py')
    for n in entry:
        assert sha(B/n)==old['source_sha256'][n]
        s=transform((B/n).read_text())
        if n==entry[0]:
            s=s.replace(old['content_source_sha256'],content['source_sha256']).replace(old['content_spec_sha256'],sha(R/'content_followup/spec.json'))
            s=s.replace("len(audit['gates'])==15","len(audit['gates'])==10")
            s=s.replace('same added Hann competition loss','same added raw-score competition loss')
            s=s.replace('Carry forward the M64/M67/M73 low22 gate unchanged before M78 final outcomes;',
                'Retain native low22 primary thresholds prospectively for M78; historical model dominance is not required;')
        write_source(n,s)
    subprocess.run([str(B/'envs/sttrack/bin/python'),str(B/names[entry[0]]),'prepare'],check=True)
    print(json.dumps(dict(candidate_spec_sha256=sha(E/'spec.json'),public_evaluation_started=False)))
def prepare_full():
    assert (E/'spec.json').exists() and not (E/'full_preparation_readiness.json').exists()
    old=read(P/'full_preparation_readiness.json')
    for n in full:
        p=B/n
        if n!=full[-1]:assert sha(p)==old['full_source_sha256'][str(p)]
        else:assert sha(p)==read(P/'evaluation_preparation_check.json')['checker_sha256']
        s=transform(p.read_text())
        if n==full[0]:s=s.replace(old['candidate_spec_sha256'],sha(E/'spec.json'))
        if n==full[-1]:s=s.replace(read(P/'spec.json')['training_spec_sha256'],sha(R/'training_spec.json'))
        write_source(n,s)
    envpy=str(B/'envs/sttrack/bin/python')
    subprocess.run([envpy,str(B/names[full[0]]),'check_sources'],check=True)
    subprocess.run([envpy,str(B/names[full[-1]])],check=True)
    proof=dict(status='M78_evaluation_sources_prepared_without_activation',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),entry_spec_sha256=sha(E/'spec.json'),readiness_sha256=sha(E/'full_preparation_readiness.json'),
        check_sha256=sha(E/'evaluation_preparation_check.json'),seed=2027,additional_seeds=[],new_tracking_calls=0,new_optimizer_steps=0,new_caption_calls=0,
        evaluation_started=False,source_files={names[n]:sha(B/names[n]) for n in entry+full})
    (E/'prepared_entry_and_full.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['entry','full']);a=p.parse_args()
    {'entry':prepare_entry,'full':prepare_full}[a.action]()
