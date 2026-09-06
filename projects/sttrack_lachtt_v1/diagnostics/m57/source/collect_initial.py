"""Extract split-separated t0 references without reading subsequent frames or GT."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert sha(__file__) == spec['collector_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == spec['cuda_visible_devices']
    assert sha(spec['initialization_inputs']) == spec['initialization_inputs_sha256']
    assert sha(spec['base_decision']) == spec['base_decision_sha256']
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    code = Path(spec['code_root'])
    for name, digest in spec['source_sha256'].items():
        assert sha(code / name) == digest, name
    sys.path.insert(0, str(code))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_initial_instance_observation import extract_initial_instance_roi
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(code / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    cases = json.loads(Path(spec['initialization_inputs']).read_text())
    assert len(cases) == 85 and len({case['sequence'] for case in cases}) == 85
    groups = {split: dict(sequences=[], references=[], records=[]) for split in ['fit', 'development']}
    for case in cases:
        assert set(case) == {'sequence', 'split', 'frames', 'init_bbox'}
        folder = Path(spec['dataset_root']) / case['sequence']
        rgb, depth = folder / 'color/00000001.jpg', folder / 'depth/00000001.png'
        expected = spec['initial_image_sha256'][case['sequence']]
        assert sha(rgb) == expected['rgb'] and sha(depth) == expected['depth']
        image = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
        tracker.initialize(image, dict(init_bbox=list(case['init_bbox'])))
        templates = list(tracker.z_dict)
        reference = extract_initial_instance_roi(tracker, image, case['init_bbox'])
        assert reference.shape == (2, 16, 768) and torch.isfinite(reference).all()
        assert not reference.requires_grad and tracker.frame_id == 0 and tracker.track_query_before is None
        assert list(tracker.state) == case['init_bbox']
        assert all(left is right for left, right in zip(templates, tracker.z_dict))
        # STTrackCandidateSet rounds every ROI through float16 before head input.
        half = reference.half().cpu()
        record = dict(sequence=case['sequence'], split=case['split'], init_bbox=case['init_bbox'],
            initial_image_sha256=expected, reference_fp16_sha256=hashlib.sha256(half.numpy().tobytes()).hexdigest())
        group = groups[case['split']]
        group['sequences'].append(case['sequence'])
        group['references'].append(half)
        group['records'].append(record)
        print(json.dumps(record), flush=True)
    output = Path(spec['output'])
    output.mkdir()
    files = {}
    for split, count in [('fit', 63), ('development', 22)]:
        group = groups[split]
        assert len(group['sequences']) == count
        path = output / ('initial_' + split + '.pt')
        torch.save(dict(schema='m57_t0_references_v1', split=split, sequences=group['sequences'],
            references=torch.stack(group['references']), records=group['records'],
            collection_spec_sha256=sha(args.spec), checkpoint_sha256=spec['checkpoint_sha256'],
            precision='float16 storage; exactly the half-to-float head-input convention of the native candidate runtime'), path)
        files[split] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, sequences=count)
    result = dict(status='complete_initial_reference_collection', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_sha256=sha(args.spec), files=files, initial_frames=85, rgb_depth_files=170,
        subsequent_images_read=False, subsequent_gt_read=False, fitting_and_development_storage_separate=True,
        checkpoint_sha256=spec['checkpoint_sha256'], source_sha256=spec['source_sha256'])
    for name, digest in spec['source_sha256'].items():
        assert sha(code / name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
