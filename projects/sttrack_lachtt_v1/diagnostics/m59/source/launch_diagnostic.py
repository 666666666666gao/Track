"""Launch the frozen M59 content diagnosis only after actual model prefix parity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
SPEC_SHA='2c190de16b186a3c9c97abcdabdeb76ce75428f4e32546c32d7d5e6226077cd6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(ROOT/'spec.json')==SPEC_SHA
    assert (ROOT/'parity.exit').read_text().strip()=='0'
    parity=json.loads((ROOT/'prefix_parity_result.json').read_text())
    assert parity['status']=='shared_runtime_prefix_parity_passed'
    assert parity['content_spec_sha256']==SPEC_SHA
    assert parity['bundle_sha256']==sha(ROOT/'bundle.json')
    assert not (ROOT/'launch.json').exists()
    sys.path.insert(0,str(ROOT))
    from run_controls import parent_ready
    spec,_,parent,_=parent_ready()
    assert not parent['primary_pass']
    memory=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(memory)==2 and all(v<500 for v in memory)
    runs=[]
    for index in [0,1]:
        script=ROOT/('run_worker%d.sh'%index)
        body='''#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906
export CUDA_VISIBLE_DEVICES={index}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/envs/sttrack/bin/python run_controls.py --worker {index} > worker{index}.log 2>&1
run_status=$?
printf '%s\\n' "$run_status" > worker{index}.exit
'''.format(index=index)
        if index==0:
            body+='''if [ "$run_status" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=''
    /root/autodl-tmp/envs/sttrack/bin/python finalize.py > finalization.log 2>&1
    run_status=$?
    printf '%s\\n' "$run_status" > finalization.exit
fi
'''
        body+='exit "$run_status"\n'
        script.write_text(body)
        script.chmod(0o755)
        screen='m59_content_gpu%d_20260906'%index
        subprocess.run(['screen','-dmS',screen,'bash',str(script)],check=True)
        runs.append(dict(gpu=index,screen=screen,controls=spec['workers'][str(index)],script_sha256=sha(script)))
    result=dict(started_utc=datetime.now(timezone.utc).isoformat(),spec_sha256=SPEC_SHA,
        launcher_sha256=sha(__file__),prefix_parity_result_sha256=sha(ROOT/'prefix_parity_result.json'),
        gpu_memory_mib_before=memory,runs=runs,poll_interval_seconds=240,
        expected_longest_worker_seconds=3550,parent_promotion_pass=False,public_evaluation_allowed=False)
    (ROOT/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
