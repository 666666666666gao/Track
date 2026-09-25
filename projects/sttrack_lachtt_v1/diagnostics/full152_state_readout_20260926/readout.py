"""Read adapted/native score and geometry at sealed M82 Full152 states."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


EVAL = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = Path('/root/autodl-tmp/sttrack_full152_state_readout_20260926')
sys.path.insert(0, str(EVAL / 'interface'))
from semantic_runtime import checked_plan, make_tracker, text_bank


EVENTS = {
    'two_mugs': [(395, 405)],
    'bottle_room_occ_1': [(478, 512), (568, 580)],
    'robot_corridor_occ_1': [(433, 460), (544, 560)],
    'jug': [(270, 318)],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def in_event(sequence, frame):
    return any(start <= frame < end for start, end in EVENTS[sequence])


def main():
    plan_path = EVAL / 'M82/cdtb/plan.json'
    plan, bundle = checked_plan(plan_path)
    receipt_path = EVAL / 'M82/cdtb/predictions/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['plan_sha256'] == sha(plan_path)
    cases = {row['sequence']: row for row in json.loads(Path(plan['cases_path']).read_text())}
    refs = {row['sequence']: row for row in receipt['sequences']}
    bank = text_bank(plan, bundle)
    tracker = make_tracker(bundle)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.utils.box_ops import clip_box
    capture = {}
    tracker.network.semantic_adapter.register_forward_pre_hook(
        lambda module, inputs: capture.update(adapter_inputs=inputs))
    tracker.network.box_head.register_forward_hook(
        lambda module, inputs, output: capture.update(head_output=output))
    torch.set_grad_enabled(False)
    ROOT.mkdir()
    reports = []
    for sequence, windows in EVENTS.items():
        case = cases[sequence]
        ref = refs[sequence]
        folder = Path(plan['dataset_root']) / sequence
        box_path = Path(plan['output']) / (sequence + '.txt')
        score_path = Path(plan['output']) / (sequence + '_all_scores.txt')
        assert sha(box_path) == ref['bbox_sha256'] and sha(score_path) == ref['confidence_sha256']
        boxes = np.loadtxt(box_path, delimiter=',').reshape(-1, 4)
        scores = np.loadtxt(score_path).reshape(-1)
        assert boxes.shape == (case['frames'], 4) and scores.shape == (case['frames'],)
        rows = []
        for frame in range(max(end for _, end in windows)):
            stem = '%08d' % (frame + 1)
            rgb_path = folder / 'color' / (stem + '.jpg')
            depth_path = folder / 'depth' / (stem + '.png')
            image = get_rgbd_frame(str(rgb_path), str(depth_path), dtype='rgbcolormap', depth_clip=True)
            if frame == 0:
                tracker.initialize(image, bank.info(rgb_path, case['init_bbox']))
                continue
            previous = list(tracker.state)
            actual = tracker.track(image)
            assert np.max(np.abs(np.asarray(actual['target_bbox']) - boxes[frame])) <= 5.01e-7
            assert abs(float(actual['best_score']) - scores[frame]) <= 5.01e-7
            if not in_event(sequence, frame):
                continue
            adapted_score, _, adapted_size, adapted_offset = capture['head_output']
            rgb, depth, fused, initial, text, mask = capture['adapter_inputs']
            native = tracker.network.forward_head(fused)
            side = math.ceil(math.sqrt(previous[2] * previous[3]) * 4.)
            resize = 256. / side
            height, width = image.shape[:2]

            def decode(score_map, size_map, offset_map):
                response = score_map * tracker.output_window
                box = tracker.network.box_head.cal_bbox(response, size_map, offset_map).view(-1, 4)
                cx, cy, bw, bh = (box.mean(dim=0) * 256. / resize).tolist()
                cx += previous[0] + .5 * previous[2] - .5 * 256. / resize
                cy += previous[1] + .5 * previous[3] - .5 * 256. / resize
                return clip_box([cx - .5 * bw, cy - .5 * bh, bw, bh], height, width, margin=10)

            variants = {}
            for name, score_map, size_map, offset_map in [
                ('adapted_score_adapted_geometry', adapted_score, adapted_size, adapted_offset),
                ('adapted_score_native_geometry', adapted_score, native['size_map'], native['offset_map']),
                ('native_score_adapted_geometry', native['score_map'], adapted_size, adapted_offset),
                ('native_score_native_geometry', native['score_map'], native['size_map'], native['offset_map']),
            ]:
                variants[name] = dict(bbox=decode(score_map, size_map, offset_map),
                                      raw_peak=int(score_map.argmax()),
                                      hann_peak=int((score_map * tracker.output_window).argmax()),
                                      raw_max=float(score_map.max()),
                                      hann_max=float((score_map * tracker.output_window).max()))
            assert variants['adapted_score_adapted_geometry']['bbox'] == actual['target_bbox']
            rows.append(dict(frame=frame, previous_bbox=previous, search_side=side, variants=variants))
        path = ROOT / (sequence + '.json')
        path.write_text(json.dumps(dict(sequence=sequence, windows=windows, rows=rows),
                                   separators=(',', ':'), allow_nan=False) + '\n')
        reports.append(dict(sequence=sequence, positions=len(rows), output_sha256=sha(path),
                            sealed_bbox_sha256=ref['bbox_sha256'],
                            sealed_score_sha256=ref['confidence_sha256']))
        print(json.dumps(reports[-1]), flush=True)
    result = dict(status='complete', source_sha256=sha(__file__), plan_sha256=sha(plan_path),
                  checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
                  receipt_sha256=sha(receipt_path), cases=reports,
                  category_replay_exact=True, counterfactual_state_committed=False,
                  groundtruth_loaded=False)
    (ROOT / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
