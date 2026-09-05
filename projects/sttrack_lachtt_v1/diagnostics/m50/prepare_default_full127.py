#!/usr/bin/env python3
"""Prepare the unchanged M39 native tracker's complete VOT reference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905')
M39=Path('/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902')
REPO=Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
TRACKER='sttrack_default_full127'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_manifest=Path('/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/shard_manifest.json')
    assert sha(old_manifest)=='8bf5271b3cdc0e0f4587657502f0aa4d873c6cfbc8716f88a1fabb55aa5334b3'
    frozen=json.loads(old_manifest.read_text())
    m39spec=json.loads((M39/'spec.json').read_text())
    assert sha(Path(m39spec['source']['checkpoint']))==m39spec['source']['checkpoint_sha256']
    wrapper=M39/'wrappers/m39_sttrack_default_vot22.py'
    assert sha(wrapper)==m39spec['arms']['default']['wrapper_sha256']
    assert sha(M39/'wrappers/m39_vot_bridge.py')==m39spec['runtime']['vot_bridge_sha256']
    root=ROOT;root.mkdir()
    tools=root/'tools';tools.mkdir()
    for name in ['create_vot_failure_family_shards.py','run_vot_failure_family_shards.py']:
        shutil.copy2(Path('/home/SUTrack_RGBD_L/tools')/name,tools/name)
    wrappers=root/'wrappers';wrappers.mkdir()
    for name in ['m39_sttrack_default_vot22.py','m39_vot_bridge.py']:
        shutil.copy2(M39/'wrappers'/name,wrappers/name)
    run=root/'run'
    subprocess.run([sys.executable,str(tools/'create_vot_failure_family_shards.py'),'--all-sequences',
                    '--output-root',str(run),'--shards','4','--gpus','1','--tracker',TRACKER],check=True)
    manifest_path=run/'shard_manifest.json';manifest=json.loads(manifest_path.read_text())
    flat=lambda x:{(a['sequence'],a['index'],a['direction'],a['value'],a['estimated_frames']) for s in x['shards'] for a in s['anchors']}
    assert flat(manifest)==flat(frozen) and len(flat(manifest))==1765
    assert manifest['sequences']==frozen['sequences'] and len(manifest['sequences'])==127
    ini=(f'[{TRACKER}]\nlabel = Native STTrack default full127\nprotocol = traxpython\n'
         f'command = m39_sttrack_default_vot22\npaths = {wrappers}\n'
         'python = /root/autodl-tmp/envs/sttrack/bin/python\n'
         f'env_CUDA_VISIBLE_DEVICES = 1\nenv_PYTHONPATH = {wrappers}\n'
         'env_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
    for shard in manifest['shards']:
        path=Path(shard['root'])/'trackers.ini';path.write_text(ini)
        shard['gpu']=1;shard['trackers_sha256']=sha(path)
    manifest['schema']='sttrack_native_default_full127_shards_v1'
    manifest['native_source']='Exact M39 default wrapper and checkpoint; no experimental head or text'
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    merge=json.loads((M39/'default/merge_result.json').read_text())
    old_tracker=merge['tracker'];master=M39/'default/master'
    seeded={};sequences=set();anchor_count=0;seeded_frames=0
    for shard in manifest['shards']:
        for anchor in shard['anchors']:
            sequence=anchor['sequence'];stem=f'{sequence}_{anchor["index"]:08d}'
            source=master/'results'/old_tracker/'baseline'/sequence
            if not (source/(stem+'.bin')).exists():continue
            dest=Path(shard['root'])/'results'/TRACKER/'baseline'/sequence;dest.mkdir(parents=True,exist_ok=True)
            for suffix in ['.bin','_confidence.value','_time.value']:
                path=source/(stem+suffix);expected=merge['result_sha256'][str(path.relative_to(master))]
                assert sha(path)==expected
                target=dest/path.name;shutil.copy2(path,target);assert sha(target)==expected
                seeded[str(target.relative_to(root))]=dict(source=str(path),sha256=expected)
            anchor_count+=1;seeded_frames+=anchor['estimated_frames'];sequences.add(sequence)
    assert anchor_count==303 and len(sequences)==22 and len(seeded)==909
    record=dict(status='complete',source_merge_sha256=sha(M39/'default/merge_result.json'),
                source_tracker=old_tracker,destination_tracker=TRACKER,anchors=anchor_count,
                estimated_frames_including_initialization=seeded_frames,files=seeded)
    (root/'preseed_receipt.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    sources=[REPO/'lib/test/tracker/sttrack.py',REPO/'lib/models/sttrack/sttrack.py',
             REPO/'lib/train/data/processing_utils.py',REPO/'lib/train/dataset/depth_utils.py',
             REPO/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
             wrappers/'m39_sttrack_default_vot22.py',wrappers/'m39_vot_bridge.py',
             tools/'create_vot_failure_family_shards.py',tools/'run_vot_failure_family_shards.py']
    current_inputs={str(p):sha(p) for p in sources}
    spec=dict(schema='sttrack_native_default_full127_reference_v1',status='frozen_before_launch',
              scope='Complete native reference after M39 low22 improvement; independent of experimental policy promotion',
              checkpoint=m39spec['source']['checkpoint'],checkpoint_sha256=m39spec['source']['checkpoint_sha256'],
              source_sha256=current_inputs,source_m39_spec_sha256=sha(M39/'spec.json'),
              full_anchor_manifest_sha256=sha(old_manifest),run_manifest_sha256=sha(manifest_path),
              preseed_receipt_sha256=sha(root/'preseed_receipt.json'),gpu=1,workers=4,poll_seconds=240,
              sequences=127,anchors=1765,reused_anchors=303,new_anchors=1462,
              new_estimated_frames=manifest['total_estimated_frames']-seeded_frames,
              optimizer_steps=0,new_weights=False,language_enabled=False,
              target_thresholds_percent=dict(eao=77.9,acc=82.1,rob=93.7),
              note='A baseline result alone does not fulfil the final new DepthTrack-trained model bundle requirement.')
    (root/'spec.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in spec.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
