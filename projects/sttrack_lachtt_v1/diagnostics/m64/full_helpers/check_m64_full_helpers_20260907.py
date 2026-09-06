"""Seal source-only checks without invoking conditional full evaluation."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import torch

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m64_category_candidate_20260907'
OUT=ROOT/'full_helpers_source_evidence'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir()
    prep=BASE/'m64_full_preparation_20260907.py'
    proof=json.loads((ROOT/'full_preparation_readiness.json').read_text())
    assert sha(prep)==proof['preparation_source_sha256']
    s=importlib.util.spec_from_file_location('prep_sources64',str(prep))
    helper=importlib.util.module_from_spec(s);s.loader.exec_module(helper)
    spec,bundle,_,_=helper.sources()
    programs=[BASE/'m64_full_vot_20260907.py',BASE/'verify_m64_three_datasets_20260907.py']
    functions={}
    for p in programs:
        tree=ast.parse(p.read_text(),filename=str(p),feature_version=(3,8))
        functions[p.name]=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)]
        shutil.copyfile(p,OUT/p.name)
    ckpt=torch.load(bundle['adapter_checkpoint'],map_location='cpu')
    assert ckpt['status']=='complete' and ckpt['completed_sequences']==130 and ckpt['use_text']
    assert ckpt['training_spec_sha256']==bundle['training_spec_sha256']
    region=ROOT/'full_vot_region_types.json';census=json.loads(region.read_text())
    assert census['manifest_sha256']==sha(helper.MANIFEST)
    assert census['region_types']=={'Rectangle':1765} and census['anchors']==1765
    assert all(census[k]==0 for k in ['model_image_decodes','new_caption_calls','new_tracker_calls','new_metrics'])
    assert not (ROOT/'full_evaluation').exists()
    preserved={}
    for n in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']:
        files=sorted((BASE/'qwen'/n).glob('*.safetensors'))
        preserved[n]=dict(shards=len(files),bytes=sum(p.stat().st_size for p in files))
    assert preserved['Qwen3_8B']['shards']==5 and preserved['Qwen2.5-VL-3B-Instruct']['shards']==2
    report=dict(status='source_syntax_and_existing_contract_checks_only',observed_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),source_sha256={str(p):sha(p) for p in programs},python38_AST_functions=functions,
        preparation_readiness_sha256=sha(ROOT/'full_preparation_readiness.json'),bundle_sha256=spec['bundle_sha256'],
        text_protocol_sha256=spec['text_protocol_sha256'],adapter_checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
        complete_text_checkpoint_and130_training_sequences_verified=True,region_census_sha256=sha(region),
        metric_source_sha256=sha('/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py'),
        shared_OPE_source_sha256=sha('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906/run_semantic_ope.py'),
        conditional_functions_executed=False,new_model_image_decodes=0,new_caption_calls=0,new_tracker_calls=0,new_metrics=0,
        full_evaluation_directory_exists=False,preserved_qwen=preserved,free_disk_bytes=shutil.disk_usage(BASE).free,
        independent_review_pass=False,scope='Executor source/metadata checks only; conditional full evaluation and final saved-output verifier have not run.')
    write(OUT/'source_contract_check.json',report)
    shutil.copyfile(region,OUT/region.name);shutil.copyfile(__file__,OUT/Path(__file__).name)
    write(OUT/'evidence_manifest.json',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir())])
    archive=ROOT/'full_helpers_source_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.iterdir()):t.add(p,arcname=p.name)
    print(json.dumps(dict(report=report,archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
