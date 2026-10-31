"""A1: same R0 final + one CAD template profile + native-anchored selection, one profile for DT/CDTB/VOT.

The sealed CAD profile plans are only read; every output of this run lives under ROOT.
"""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

CAD = Path('/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016')
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'code'), str(CAD / 'code'), '/root/autodl-tmp/m125_R0_fixed6_20261015/code']
MARGIN = .25
RADIUS = 1
PROFILE = 'U2'
TRACKER = 'sttrack_m125_anchored_' + PROFILE
SUFFIXES = ('.bin', '_confidence.value', '_time.value')


def build(dataset):
    from m125_CAD_official_runtime import checked_plan
    from anchored_profile_tracker import OfficialAnchoredTracker
    plan, bundle = checked_plan(CAD / 'profiles' / PROFILE / dataset / 'plan.json')
    bundle.update(anchor_margin=MARGIN, anchor_radius=RADIUS)
    return plan, bundle, OfficialAnchoredTracker(plan, bundle)


def ope(dataset):
    import numpy as np
    plan, bundle, actor = build(dataset)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    cases = json.loads(Path(plan['cases_path']).read_text())
    output = ROOT / dataset
    output.mkdir()
    counts = {}
    for case in cases:
        folder = Path(plan['dataset_root']) / case['sequence']
        boxes = [case['init_bbox']]
        scores = [1.]
        for frame in range(case['frames']):
            stem = '%08d' % (frame + 1)
            rgb = folder / 'color' / (stem + '.jpg')
            image = get_rgbd_frame(str(rgb), str(folder / 'depth' / (stem + '.png')), dtype='rgbcolormap', depth_clip=True)
            if frame == 0:
                actor.initialize(image, case['init_bbox'], rgb)
            else:
                prediction = actor.track(image)
                boxes.append(prediction['target_bbox'])
                scores.append(prediction['best_score'])
        np.savetxt(output / (case['sequence'] + '.txt'), np.asarray(boxes), fmt='%.6f', delimiter=',')
        np.savetxt(output / (case['sequence'] + '_all_scores.txt'), np.asarray(scores), fmt='%.6f')
        counts[case['sequence']] = dict(actor.trusted.counts)
        print(case['sequence'], json.dumps(counts[case['sequence']]), flush=True)
    spec = importlib.util.spec_from_file_location('locked_metric', plan['metric_source'])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    values = metric.evaluate_depthtrack_results(plan['dataset_root'], output, resolution=100,
                                                sequence_names=[c['sequence'] for c in cases])
    (output / 'metrics.json').write_text(json.dumps(dict(metrics=values, counts=counts, margin=MARGIN, radius=RADIUS,
        profile=PROFILE, final_sha256=bundle['final_sha256']), indent=2) + '\n')
    print(json.dumps(values), flush=True)


def vot_entry():
    plan, bundle, actor = build('vot')
    spec = importlib.util.spec_from_file_location('locked_vot_bridge', bundle['vot_bridge_path'])
    vot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vot)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    handle = vot.VOT('rectangle', channels='rgbd')
    bbox = list(handle.region())
    files = handle.frame()
    actor.initialize(get_rgbd_frame(files[0], files[1], dtype='rgbcolormap', depth_clip=True), bbox, files[0])
    while True:
        files = handle.frame()
        if files is None:
            break
        out = actor.track(get_rgbd_frame(files[0], files[1], dtype='rgbcolormap', depth_clip=True))
        handle.report(vot.Rectangle(*out['target_bbox']), out['best_score'])
    name = hashlib.sha256(actor.initialization_key.encode()).hexdigest() + '.json'
    (ROOT / 'vot/events' / name).write_text(json.dumps(dict(initialization_key=actor.initialization_key,
        counts=dict(actor.trusted.counts)), indent=2) + '\n')


def trajectory_files(root, name):
    folder = Path(root) / 'results' / TRACKER / 'baseline' / name.rsplit('_', 1)[0]
    return [folder / (name + suffix) for suffix in SUFFIXES]


def vot_run(gpus):
    manifest = json.loads((CAD / 'profiles' / PROFILE / 'vot/run/shard_manifest.json').read_text())
    run = ROOT / 'vot'
    (run / 'events').mkdir(parents=True)
    entry = run / 'entry'
    entry.mkdir()
    (entry / 'selected_anchored_vot.py').write_text(
        "import sys\nsys.path.insert(0,'%s')\nfrom run_anchored import vot_entry\nvot_entry()\n" % (ROOT / 'code'))
    config = 'registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n'
    queue = []
    for shard in manifest['shards']:
        folder = run / ('shard-%02d' % shard['index'])
        folder.mkdir()
        (folder / 'sequences').symlink_to(Path(shard['root']) / 'sequences', target_is_directory=True)
        (folder / 'config.yaml').write_text(config)
        queue.append((shard, folder))
    while queue:
        wave, queue = queue[:len(gpus)], queue[len(gpus):]
        children = []
        for (shard, folder), gpu in zip(wave, gpus):
            (folder / 'trackers.ini').write_text(
                '[%s]\nlabel = M125 R0 final + %s + native-anchored selection m%.2f r%d\nprotocol = traxpython\n'
                'command = selected_anchored_vot\npaths = %s\npython = /root/autodl-tmp/envs/sttrack/bin/python\n'
                'env_CUDA_VISIBLE_DEVICES = %d\nenv_PYTHONPATH = %s\nenv_TOKENIZERS_PARALLELISM = false\n'
                'env_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n'
                % (TRACKER, PROFILE, MARGIN, RADIUS, entry, gpu, entry))
            env = dict(os.environ, PYTHONPATH='/home/SUTrack_RGBD_L', CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            log = (run / ('shard-%02d.log' % shard['index'])).open('w')
            child = subprocess.Popen(['/root/miniconda3/envs/mplt/bin/python', '-m', 'vot', 'evaluate', '--workspace', '.', TRACKER],
                                     cwd=str(folder), env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, shard, folder))
            print('started shard', shard['index'], 'gpu', gpu, 'pid', child.pid, flush=True)
        for child, shard, folder in children:
            code = child.wait()
            done = sum(all(p.is_file() and p.stat().st_size > 0 for p in trajectory_files(folder, n)) for n in shard['expected_trajectories'])
            print('shard', shard['index'], 'exit', code, 'anchors', done, '/', shard['anchor_count'], flush=True)
            assert code == 0 and done == shard['anchor_count']
    master = run / 'master'
    (master / 'sequences').mkdir(parents=True)
    (master / 'config.yaml').write_text(config)
    shutil.copy2(run / 'shard-00/trackers.ini', master / 'trackers.ini')
    (master / 'sequences/list.txt').write_text(''.join(n + '\n' for n in manifest['sequences']))
    for name in manifest['sequences']:
        (master / 'sequences' / name).symlink_to((Path(manifest['source_sequences_root']) / name).resolve(), target_is_directory=True)
    for shard in manifest['shards']:
        folder = run / ('shard-%02d' % shard['index'])
        for name in shard['expected_trajectories']:
            destination = master / 'results' / TRACKER / 'baseline' / name.rsplit('_', 1)[0]
            destination.mkdir(parents=True, exist_ok=True)
            for source in trajectory_files(folder, name):
                shutil.copy2(source, destination / source.name)
    subprocess.run(['/root/miniconda3/envs/mplt/bin/python', '-B', '-u', '-m', 'vot', 'analysis', '--workspace', str(master),
                    '--format', 'json', '--name', 'A1_anchored_' + PROFILE + '_full127', TRACKER], check=True)
    values = json.loads((master / ('analysis/A1_anchored_%s_full127.json' % PROFILE)).read_text())['results']['baseline']['results']
    print(json.dumps(values, indent=2)[:3000], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ope', 'vot_run'], required=True)
    parser.add_argument('--dataset', choices=['depthtrack_test', 'cdtb'])
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 1])
    args = parser.parse_args()
    if args.mode == 'ope':
        ope(args.dataset)
    else:
        vot_run(args.gpus)
