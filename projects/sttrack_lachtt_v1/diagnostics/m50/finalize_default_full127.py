#!/usr/bin/env python3
"""Validate complete native STTrack output before official VOT analysis."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=ROOT;run=root/'run';spec=json.loads((root/'spec.json').read_text())
    binding=json.loads((root/'execution_binding.json').read_text())
    assert sha(root/'spec.json')==binding['spec_sha256']
    assert sha(Path(__file__))==binding['finalizer_sha256']
    assert (root/'evaluate.exit').read_text().strip()=='0'
    for path,digest in spec['source_sha256'].items():assert sha(Path(path))==digest,path
    assert sha(Path(spec['checkpoint']))==spec['checkpoint_sha256']
    assert sha(run/'shard_manifest.json')==spec['run_manifest_sha256']
    assert sha(root/'preseed_receipt.json')==spec['preseed_receipt_sha256']
    manifest=json.loads((run/'shard_manifest.json').read_text());tracker=manifest['tracker']
    merge=json.loads((run/'merge_result.json').read_text());master=run/'master'
    assert merge['status']=='complete' and merge['tracker']==tracker and merge['anchor_count']==1765
    assert merge['source_manifest_sha256']==sha(run/'shard_manifest.json')
    expected={}
    for shard in manifest['shards']:
        for anchor in shard['anchors']:
            seq=anchor['sequence'];stem=f'{seq}_{anchor["index"]:08d}'
            for suffix in ['.bin','_confidence.value','_time.value']:
                relative=f'results/{tracker}/baseline/{seq}/{stem}{suffix}'
                assert relative not in expected;expected[relative]=master/relative
    assert len(expected)==5295 and set(expected)==set(merge['result_sha256'])
    for path,actual in expected.items():assert sha(actual)==merge['result_sha256'][path],path
    seeded=json.loads((root/'preseed_receipt.json').read_text())
    for path,item in seeded['files'].items():
        source=root/path
        assert sha(source)==item['sha256']
        relative=source.relative_to(run/Path(path).parts[1])
        assert sha(master/relative)==item['sha256']
    name='sttrack_default_full127_analysis'
    environment=dict(os.environ);environment['PYTHONPATH']='/home/SUTrack_RGBD_L'
    with (root/'analysis.log').open('wb') as log:
        subprocess.run([sys.executable,'-m','vot','analysis','--workspace',str(master),
                        '--format','json','--name',name,tracker],env=environment,
                       cwd='/home/SUTrack_RGBD_L',stdout=log,stderr=subprocess.STDOUT,check=True)
    analysis_path=master/'analysis'/(name+'.json');analysis=json.loads(analysis_path.read_text())
    assert analysis['toolkit']=='0.7.1' and set(analysis['sequences'])==set(manifest['sequences'])
    assert len(analysis['sequences'])==127 and tracker in analysis['trackers']
    arrays=analysis['results']['baseline']['results']
    metrics=dict(eao=float(arrays[0][0][0]),acc=float(arrays[2][0][0]),rob=float(arrays[2][0][1]))
    assert all(math.isfinite(v) and 0<=v<=1 for v in metrics.values())
    sys.path.insert(0,'/home/SUTrack_RGBD_L')
    from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    helper=Path('/home/SUTrack_RGBD_L/tools/finalize_vot_transaction_low22.py')
    assert sha(helper)==binding['failure_helper_sha256']
    outcomes,failures,per_sequence,settings=collect_confirmed_failure_outcomes(master,tracker,expected_anchors=1765)
    assert len(outcomes)==1765 and len(per_sequence)==127
    for path,digest in spec['source_sha256'].items():assert sha(Path(path))==digest,path
    result=dict(schema='sttrack_native_default_full127_result_v1',status='complete',integrity_pass=True,
                tracker=tracker,sequences=127,anchors=1765,reused_m39_anchors=303,new_anchors=1462,
                checkpoint_sha256=spec['checkpoint_sha256'],spec_sha256=sha(root/'spec.json'),
                execution_binding_sha256=sha(root/'execution_binding.json'),merge_sha256=sha(run/'merge_result.json'),
                analysis_sha256=sha(analysis_path),analysis=str(analysis_path),metrics_fraction=metrics,
                metrics_percent={k:v*100 for k,v in metrics.items()},confirmed_failures=failures,
                per_sequence_failures=per_sequence,failure_outcomes=outcomes,failure_settings=settings,
                exceeds_requested_vot_thresholds={k:metrics[k]*100>v for k,v in spec['target_thresholds_percent'].items()},
                optimizer_steps=0,new_checkpoint=False,language_enabled=False,
                scope='Official full127 native STTrack reference. It is not a result for M48/M49/M50, nor final verification of a newly DepthTrack-trained three-dataset model bundle.')
    (root/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence_failures','failure_outcomes']},indent=2))


if __name__=='__main__':main()
