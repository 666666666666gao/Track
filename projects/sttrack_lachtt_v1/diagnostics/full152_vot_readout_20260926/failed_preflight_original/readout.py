"""Replay sealed VOT prefixes and read Category, Empty and unadapted dense maps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path('/root/autodl-tmp/sttrack_full152_vot_readout_20260926')
EVAL = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
CONTROLS = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
sys.path.insert(0, str(EVAL / 'interface'))
from initialization_text import vot_wire_bbox
from semantic_runtime import checked_plan, make_tracker, text_bank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    # Existing GPU evaluations must finish before this independent replay.
    assert (CONTROLS / 'controls.exit').read_text().strip() == '0'
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert spec['source_sha256']['readout.py'] == sha(__file__)
    assert sha(spec['plan_path']) == spec['plan_sha256']
    plan, bundle = checked_plan(spec['plan_path'])
    bank = text_bank(plan, bundle)
    tracker = make_tracker(bundle)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.utils.box_ops import clip_box
    capture = {}
    tracker.network.semantic_adapter.register_forward_pre_hook(lambda module, inputs: capture.update(inputs=inputs))
    tracker.network.box_head.register_forward_hook(lambda module, inputs, output: capture.update(head=output))
    cases = [case for case in spec['cases'] if case['shard'] == args.shard]
    if args.preflight:
        cases = cases[:1]
    mode = 'preflight' if args.preflight else 'predictions'
    output = ROOT / mode / str(args.shard)
    output.mkdir(parents=True)
    receipts, started = [], time.time()
    with torch.no_grad():
        for case in cases:
            assert sha(case['reference']) == case['reference_sha256']
            with np.load(case['reference'], allow_pickle=False) as archive:
                saved_boxes, saved_scores = archive['boxes'], archive['scores']
            assert saved_boxes.shape == (case['prefix_length'], 4)
            folder = Path(case['folder'])
            observations = set(case['observe_indices'])
            dense = {arm: [] for arm in ('category', 'empty', 'native')}
            rows = []
            for index in range(case['prefix_length']):
                source_index = case['anchor'] + case['direction_step'] * index
                rgb = folder / 'color' / f'{source_index + 1:08d}.jpg'
                depth = folder / 'depth' / f'{source_index + 1:08d}.png'
                image = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
                if index == 0:
                    tracker.initialize(image, bank.info(rgb, case['init_bbox']))
                    continue
                previous = list(tracker.state)
                actual = tracker.track(image)
                # Use the known TraX rectangle serialization, not a guessed tolerance.
                assert np.array_equal(np.asarray(vot_wire_bbox(actual['target_bbox'])), saved_boxes[index]), (case['anchor_key'], index, 'sealed bbox')
                assert np.float32(actual['best_score']) == np.float32(saved_scores[index]), (case['anchor_key'], index, 'sealed score')
                if index not in observations:
                    continue
                rgb_features, depth_features, fused, initial, text, mask = capture['inputs']
                score, _, size, offset = capture['head']
                outputs = dict(category=dict(score_map=score, size_map=size, offset_map=offset))
                state = list(tracker.state)
                queries = [q.clone() for q in tracker.track_query_before]
                templates = list(tracker.z_dict)
                empty_text = text.clone()
                empty_text[:, 0] = bank.bank['empty'].to(text)
                enhanced, _ = tracker.network.semantic_adapter(rgb_features, depth_features, fused, initial, empty_text, mask)
                outputs['empty'] = tracker.network.forward_head(enhanced)
                outputs['native'] = tracker.network.forward_head(fused)
                assert tracker.state == state and len(tracker.z_dict) == len(templates)
                assert all(a is b for a, b in zip(tracker.z_dict, templates))
                assert all(torch.equal(a, b) for a, b in zip(tracker.track_query_before, queries))
                side = math.ceil(math.sqrt(previous[2] * previous[3]) * 4.)
                resize, (height, width) = 256. / side, image.shape[:2]

                def decode(out, response):
                    predicted = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4)
                    cx, cy, bw, bh = (predicted.mean(dim=0) * 256. / resize).tolist()
                    cx += previous[0] + .5 * previous[2] - .5 * side
                    cy += previous[1] + .5 * previous[3] - .5 * side
                    return clip_box([cx - .5 * bw, cy - .5 * bh, bw, bh], height, width, margin=10)

                variants = {}
                for arm, maps in outputs.items():
                    tensor = torch.cat([maps[name] for name in ('score_map', 'size_map', 'offset_map')], dim=1)[0]
                    assert tensor.shape == (5, 16, 16) and torch.isfinite(tensor).all()
                    dense[arm].append(tensor.cpu().numpy())
                    raw, hann = maps['score_map'], maps['score_map'] * tracker.output_window
                    variants[arm] = dict(raw_peak=int(raw.argmax()), hann_peak=int(hann.argmax()),
                                         raw_bbox=decode(maps, raw), hann_bbox=decode(maps, hann),
                                         raw_max=float(raw.max()), hann_max=float(hann.max()))
                assert variants['category']['hann_bbox'] == actual['target_bbox']
                rows.append(dict(run_index=index, source_frame=source_index, previous_bbox=previous,
                                 search_side=side, image_hw=[height, width],
                                 crop_origin=[round(previous[0] + .5 * previous[2] - .5 * side),
                                              round(previous[1] + .5 * previous[3] - .5 * side)],
                                 variants=variants))
            assert [row['run_index'] for row in rows] == case['observe_indices']
            stem = output / case['anchor_key']
            json_path, dense_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
            json_path.write_text(json.dumps(dict(anchor_key=case['anchor_key'], rows=rows), separators=(',', ':'), allow_nan=False) + '\n')
            np.savez_compressed(dense_path, window=tracker.output_window.cpu().numpy(),
                                **{arm: np.stack(values) for arm, values in dense.items()})
            item = dict(anchor_key=case['anchor_key'], replay_calls=case['prefix_length'] - 1,
                        positions=len(rows), json_sha256=sha(json_path), dense_sha256=sha(dense_path),
                        elapsed_seconds=time.time() - started)
            receipts.append(item)
            print(json.dumps(item), flush=True)
    result = dict(status='complete', mode=mode, shard=args.shard, spec_sha256=sha(ROOT / 'spec.json'),
                  source_sha256=sha(__file__), checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
                  category_bbox_parity='exact after existing TraX float32/four-decimal serialization',
                  category_score_parity='exact float32', counterfactual_state_committed=False,
                  post_initialization_groundtruth_loaded=False, cases=receipts)
    (output / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', anchors=len(receipts), shard=args.shard, elapsed_seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
