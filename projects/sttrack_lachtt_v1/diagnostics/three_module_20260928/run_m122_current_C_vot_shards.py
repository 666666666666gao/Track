"""Run frozen VOT shards in two GPU waves, then seal all 5295 results.

The trajectory naming/merge layout follows the existing full127 shard runner.
"""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from bind_m122_official_initializations import sha
from prepare_m122_evaluation_suite import read,write

SUFFIXES=('.bin','_confidence.value','_time.value')


def trajectory_files(root,tracker,name):
    folder=Path(root)/'results'/tracker/'baseline'/name.rsplit('_',1)[0]
    return [folder/(name+suffix) for suffix in SUFFIXES]


def complete(shard,tracker):
    return sum(all(p.is_file() and p.stat().st_size>0 for p in trajectory_files(shard['root'],tracker,name)) for name in shard['expected_trajectories'])


def main(root):
    execution=read(root.parent/'execution.json')
    assert execution['status']=='frozen_before_M122_VOT_tracking'
    for path,digest in execution['source_sha256'].items():assert sha(path)==digest,path
    manifest=read(root/'shard_manifest.json');tracker=manifest['tracker']
    assert manifest['schema']=='M122_full127_frozen_shards_v1' and manifest['gpu_count']==2
    assert [s['index'] for s in manifest['shards']]==[0,1,2,3] and [s['gpu'] for s in manifest['shards']]==[0,1,0,1]
    logs=root/'controller_logs';logs.mkdir(exist_ok=False);launches=[]
    for offset in [0,2]:
        children=[]
        for shard in manifest['shards'][offset:offset+2]:
            folder=Path(shard['root']);assert not (folder/'results').exists()
            command=['/root/miniconda3/envs/mplt/bin/python','-m','vot','evaluate','--workspace','.',tracker]
            env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L',CUDA_VISIBLE_DEVICES=str(shard['gpu']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (logs/('shard-%02d.log'%shard['index'])).open('w') as log:
                child=subprocess.Popen(command,cwd=str(folder),env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(index=shard['index'],gpu=shard['gpu'],pid=child.pid,started=time.time(),args=command)
            children.append((child,shard,row));launches.append(row)
        write(root/'launch.json',launches)
        for child,shard,row in children:
            row.update(exit=child.wait(),finished=time.time(),completed_anchors=complete(shard,tracker))
            (root/('shard-%02d.exit'%shard['index'])).write_text(str(row['exit'])+'\n')
        write(root/'launch.json',launches)
        assert all(r['exit']==0 and r['completed_anchors']==s['anchor_count'] for _,s,r in children),'Original VOT shard failed; no automatic restart.'
    for path,digest in execution['source_sha256'].items():assert sha(path)==digest,path
    master=root/'master';master.mkdir(exist_ok=False);sequences=master/'sequences';sequences.mkdir()
    (master/'config.yaml').write_text('registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n')
    shutil.copy2(Path(manifest['shards'][0]['root'])/'trackers.ini',master/'trackers.ini')
    (sequences/'list.txt').write_text(''.join(name+'\n' for name in manifest['sequences']))
    for name in manifest['sequences']:(sequences/name).symlink_to((Path(manifest['source_sequences_root'])/name).resolve(),target_is_directory=True)
    copied={}
    for shard in manifest['shards']:
        for name in shard['expected_trajectories']:
            destination=master/'results'/tracker/'baseline'/name.rsplit('_',1)[0];destination.mkdir(parents=True,exist_ok=True)
            for source in trajectory_files(shard['root'],tracker,name):
                target=destination/source.name;assert not target.exists()
                shutil.copy2(source,target);assert sha(source)==sha(target);copied[str(target.relative_to(master))]=sha(target)
    assert len(copied)==5295 and sum(s['anchor_count'] for s in manifest['shards'])==1765
    write(root/'merge_result.json',dict(status='complete_M122_full127_merge',tracker=tracker,source_manifest_sha256=sha(root/'shard_manifest.json'),
        anchor_count=1765,result_file_count=5295,master_workspace=str(master),result_sha256=dict(sorted(copied.items())),launches=launches))
    print(json.dumps(dict(status='complete_M122_VOT_shards_and_merge',anchors=1765,result_files=5295)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
