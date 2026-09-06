"""Snapshot real M63/M64 progress without publishing weights or image data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

BASE=Path('/root/autodl-tmp')
OUT=BASE/'sttrack_m63_m64_progress_export_20260907'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    OUT.mkdir()
    roots={'m63':BASE/'sttrack_m63_location_write_factorial_20260907','m64':BASE/'sttrack_m64_category_candidate_20260907'}
    for tag,root in roots.items():
        dest=OUT/tag;dest.mkdir()
        sources=['m63_location_write_factorial_20260907.py'] if tag=='m63' else ['m64_category_candidate_20260907.py','m64_vot_low22_20260907.py']
        spec=json.loads((root/'spec.json').read_text())
        assert sha(BASE/sources[0])==spec['source_sha256']
        for n in sources:shutil.copyfile(BASE/n,dest/n)
        shutil.copyfile(Path(__file__),dest/'export_progress.py')
        summary=dict(spec);summary.pop('cases',None)
        write(dest/'spec_summary.json',dict(source_spec_sha256=sha(root/'spec.json'),summary=summary))
        if tag=='m63':
            files=['preparation_result.json','run_queue.sh','active_stage.json','queue.log']
            for mode in ['CC','SS','CS','SC']:
                assert (root/('prefix_'+mode+'.exit')).read_text().strip()=='0'
                files += ['prefix_'+mode+'.log','prefix_'+mode+'.exit']
                shutil.copyfile(root/('prefix_'+mode)/'receipt.json',dest/('prefix_'+mode+'_receipt.json'))
        else:
            files=['bundle.json','text_protocol.json','development_plan.json','run_inputs.py','run_inputs.sh','input_stage.json','input_job.exit',
                   'generator_replay_result.json','train_generate.log','train_generate.exit','train_verify.log','train_verify.exit',
                   'low22_prepare.log','low22_prepare.exit','low22_generate.log','low22_generate.exit','low22_encode.log','low22_encode.exit',
                   'low22_bank.log','low22_bank.exit','low22_bank_result.json','low22_plan.json','low22_execution.json','m64_semantic_vot.py',
                   'run_vot_failure_family_shards.py','run_low22.sh','low22_launch.json',
                   'preparation_source_v1.py','spec_before_bank_variable_fix.json','bank_variable_fix.json','bank_variable_fix.diff',
                   'low22_bank_fixed.log','low22_bank_fixed.exit']
            for child in ['train_generator_replay','low22_captions']:
                for n in ['generation_result.json','encoding_result.json','records.jsonl']:
                    p=root/child/n
                    if p.exists():shutil.copyfile(p,dest/(child+'_'+n))
        for n in files:
            p=root/n
            if p.exists():shutil.copyfile(p,dest/n)
        write(dest/'snapshot.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),
            status='actual_progress_snapshot',disk_free_bytes=shutil.disk_usage(root).free,
            gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
            completed_exit_files={p.name:p.read_text().strip() for p in root.glob('*.exit')},
            formal_result_exists=(root/'low22_result.json').exists(),all_Qwen_preserved=True,
            new_weights_deleted=0,independent_review_pass=False))
        write(dest/'evidence_manifest.json',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.iterdir())])
    archive=OUT/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for tag in roots:
            for p in sorted((OUT/tag).iterdir()):tar.add(p,arcname=tag+'/'+p.name)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
