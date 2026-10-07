"""Two native recursive Train replays; both sanities gate the full collection."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def phase(root, mode):
    launches = []; children = []
    for shard in [0, 1]:
        args = [sys.executable, '-u', str(Path(__file__).with_name('collect_m120_native_grids.py')),
            '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
            '--full152-spec', '/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json',
            '--repository', '/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1',
            '--checkpoint', '/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
            '--output', str(root/(mode+'_shard'+str(shard))), '--shard', str(shard), '--mode', mode]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_shard'+str(shard)+'.log')).open('w') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        launches.append(dict(pid=child.pid, args=args, gpu=shard, mode=mode, started=time.time()))
        children.append(child)
    path = root/(mode+'_launch.json'); path.write_text(json.dumps(launches, indent=2)+'\n')
    for child, row in zip(children, launches):
        row.update(exit=child.wait(), finished=time.time())
        (root/(mode+'_shard'+str(row['gpu'])+'.exit')).write_text(str(row['exit'])+'\n')
    path.write_text(json.dumps(launches, indent=2)+'\n')
    assert all(row['exit']==0 for row in launches), 'Inspect original logs; no automatic retry.'
    reports = [json.loads((root/(mode+'_shard'+str(shard))/'result.json').read_text()) for shard in [0, 1]]
    assert all(row['status']=='complete_M120_'+mode+'_native_grid_shard' and row['optimizer_steps']==0 for row in reports)
    assert all(row['frozen_state_exact'] and row['native_tracker_history_replayed'] for row in reports)
    assert all(not row['auxiliary_state_committed'] and not row['GT_loaded'] and not row['text_loaded'] for row in reports)
    assert all(row['channels']==768 for row in reports)
    assert all(row['checkpoint_sha256']==reports[0]['checkpoint_sha256'] for row in reports)
    assert all(row['candidate_regions_exact'] and row['candidate_boxes_exact'] and row['initial_region_exact']
        for report in reports for row in report['sequences'])
    if mode=='sanity':
        assert all(row['events']==3 and len(row['sequences'])==1 for row in reports)
    else:
        assert sum(row['events'] for row in reports)==3502
        assert sum(row['frames'] for row in reports)==219194
        assert sum(len(row['sequences']) for row in reports)==152
    return launches, reports


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False); started = time.time()
    sanity_launch, sanity = phase(args.output, 'sanity')
    full_launch, full = phase(args.output, 'full')
    result = dict(status='complete_M120_native_grid_pair', launches=sanity_launch+full_launch,
        sanity=sanity, full=full, elapsed_seconds=time.time()-started, optimizer_steps=0,
        checkpoints_created=False, no_public_evaluation=True, auxiliary_state_committed=False,
        native_tracker_history_replayed=True)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'])), flush=True)


if __name__=='__main__':
    main()
