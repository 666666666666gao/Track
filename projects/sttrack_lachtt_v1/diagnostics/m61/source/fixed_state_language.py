"""Measure direct head effects of language on sealed recursive reference states."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/sttrack_m61_fixed_state_language_20260907')
M59 = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
M60 = Path('/root/autodl-tmp/sttrack_m60_category_isolation_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
CONDITIONS = ['unadapted_same_state', 'original', 'empty', 'category', 'swapped_category']
REFERENCES = ['original', 'category']
BANK_PATHS = {name: M59 / (name + '.pt') for name in ['original', 'empty', 'category']}
BANK_PATHS['swapped_category'] = M60 / 'swapped_category.pt'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parent():
    for name in ['tracking.exit', 'analysis.exit', 'job.exit']:
        assert (M60 / name).read_text().strip() == '0'
    assert sha(M60 / 'result.json') == 'febddd93eda9f25f3530377948751c1999030db9a08656f9bb24ae151ad64287'
    path = Path('/root/autodl-tmp/m60_category_isolation_20260906.py')
    assert sha(path) == 'f8c1f789a7f927121e9f50cefc1d2995774d6bd9bf73d047ccb11e7ea920303b'
    spec = importlib.util.spec_from_file_location('category_isolation', str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frozen, training, result = module.plans()
    return module.previous, frozen, training, result


def prepare():
    _, frozen, _, _ = parent()
    banks = {name: torch.load(path, map_location='cpu') for name, path in BANK_PATHS.items()}
    reference = banks['original']
    for bank in banks.values():
        assert bank['keys'] == reference['keys'] and bank['sequences'] == reference['sequences']
        assert torch.equal(bank['mask'], reference['mask'])
    checks = sum((case['frames'] - 1) // 50 for case in frozen['cases'])
    assert checks == 654
    ROOT.mkdir()
    spec = dict(status='frozen_before_fixed_state_probe', observed_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=sha(__file__), m60_result_sha256=sha(M60 / 'result.json'),
                m59_result_sha256=sha(M59 / 'result.json'), m59_source_sha256=sha(M59 / 'run_controls.py'),
                head_sha256=frozen['head_sha256'], bank_sha256={name: sha(path) for name, path in BANK_PATHS.items()},
                references=REFERENCES, conditions=CONDITIONS, cases=frozen['cases'],
                checks_per_reference=654, image_frames_per_reference=33130, track_calls_per_reference=33108,
                prefix_sequences=['bag05_indoor', 'container01_indoor', 'mobilephone02_indoor'], prefix_frames=202,
                event_rule='Every frame_id divisible by 50; no GT or failure selection.',
                reference_behavior='Unchanged native semantic runtime, compared on every frame against the sealed M58/M59 trajectory.',
                intervention='After the reference frame commits, reuse cloned pre-adapter RGB, Depth, fused tokens and initial RoI. Change only lexical token values; do not commit any diagnostic output.',
                outputs='Hann peak, own-peak and fixed-reference-peak decoded boxes, raw score/size/offset maps, write eligibility, and full-reference parity.',
                score_only_transforms=[0.5, 2.0],
                score_transform_scope='Replay eligibility on sealed scores with boxes/features unchanged. No recursive benefit or independent quality controller is tested.',
                new_optimizer_steps=0, new_network_parameters=0, new_caption_calls=0,
                public_evaluation_allowed=False, independent_review_pass=False,
                interpretation='Matched-state direct effects and update-score coupling only. Not causal future template utility, semantic ground truth, or full public tracking performance.',
                next='Use all paired results to choose a minimal rejectable-semantic or independent-memory-quality experiment; any new learned parameters must train on DepthTrack Train.')
    write(ROOT / 'spec.json', spec)
    write(ROOT / 'preparation_result.json', dict(status='frozen', spec_sha256=sha(ROOT / 'spec.json'),
          source_sha256=sha(__file__), references=REFERENCES, events=1308, new_weight_files=0,
          new_caption_calls=0, subsequent_GT_opened=False))
    print(json.dumps(json.loads((ROOT / 'preparation_result.json').read_text()), indent=2))


def checked():
    previous, _, training, m59 = parent()
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert sha(__file__) == spec['source_sha256']
    assert sha(M59 / 'run_controls.py') == spec['m59_source_sha256']
    assert sha(M59 / 'result.json') == spec['m59_result_sha256']
    for name, path in BANK_PATHS.items():
        assert sha(path) == spec['bank_sha256'][name]
    return previous, spec, training, m59


def mapped_box(tracker, head, response, previous, resize, shape):
    from lib.utils.box_ops import clip_box
    box = tracker.network.box_head.cal_bbox(response, head['size_map'], head['offset_map']).view(-1, 4)
    cx, cy, w, h = (box.mean(0) * tracker.params.search_size / resize).tolist()
    cx += previous[0] + .5 * previous[2] - .5 * tracker.params.search_size / resize
    cy += previous[1] + .5 * previous[3] - .5 * tracker.params.search_size / resize
    return clip_box([cx - .5 * w, cy - .5 * h, w, h], shape[0], shape[1], margin=10)


def collect(reference, prefix):
    previous, spec, training, m59 = checked()
    if not prefix:
        for ref in REFERENCES:
            assert (ROOT / ('prefix_' + ref + '.exit')).read_text().strip() == '0'
            assert json.loads((ROOT / ('prefix_' + ref) / 'receipt.json').read_text())['status'] == 'verified'
    _, _, tracker, bank = previous.runtime(reference)
    from lib.train.data.processing_utils import sample_target
    adapter = tracker.network.semantic_adapter
    assert not any(module.training for module in tracker.network.modules())
    banks = {name: torch.load(path, map_location='cpu') for name, path in BANK_PATHS.items()}
    reference_receipt = (PARENT / 'text_recursive_receipt.json') if reference == 'original' else M59 / 'category_receipt.json'
    assert sha(reference_receipt) == m59['receipt_sha256'][reference]
    files = {row['sequence']: row for row in json.loads(reference_receipt.read_text())['sequences']}
    destination = ROOT / (('prefix_' if prefix else 'collect_') + reference); destination.mkdir()
    held = {'enabled': False, 'inputs': None}

    def capture(module, args, output):
        if held['enabled']:
            assert held['inputs'] is None
            held['inputs'] = tuple(value.detach().clone() for value in args)

    handle = adapter.register_forward_hook(capture)
    selected = [case for case in spec['cases'] if not prefix or case['sequence'] in spec['prefix_sequences']]
    receipts = []; started = time.time()
    for case in selected:
        name = case['sequence']; index = banks['original']['sequences'].index(name)
        tokens = {arm: data['tokens'][index].float().cuda().unsqueeze(0) for arm, data in banks.items()}
        reference_path = (PARENT / 'recursive/text' if reference == 'original' else M59 / 'predictions/category') / (name + '.json')
        assert sha(reference_path) == files[name]['sha256']
        sealed = json.loads(reference_path.read_text())['rows']
        folder = Path(training['dataset_root']) / name
        rgb, image = previous.load_frame(folder, 0)
        tracker.initialize(image, bank.info(rgb, case['init_bbox']))
        count = spec['prefix_frames'] if prefix else case['frames']
        events = []; maps = []; max_box = max_score = 0.; writes = 0
        for frame in range(1, count):
            prev_box = list(tracker.state)
            old_template = tracker.z_dict[1]
            _, image = previous.load_frame(folder, frame)
            held['enabled'] = frame % 50 == 0
            output = tracker.track(image)
            max_box = max(max_box, float(np.max(np.abs(np.asarray(output['target_bbox']) - sealed[frame]['bbox']))))
            max_score = max(max_score, abs(float(output['best_score']) - sealed[frame]['score']))
            assert max_box <= 1e-4 and max_score <= 1e-6, (reference, name, frame, max_box, max_score)
            actual_write = tracker.z_dict[1] is not old_template
            assert actual_write == (frame % 50 == 0 and sealed[frame]['score'] > .75)
            writes += actual_write
            if not held['enabled']:
                continue
            args = held['inputs']; assert args is not None
            originals = tuple(value.clone() for value in args)
            _, resize, _ = sample_target(image, prev_box, tracker.params.search_factor, output_sz=tracker.params.search_size)
            heads = {}
            with torch.no_grad():
                for arm in CONDITIONS:
                    if arm == 'unadapted_same_state':
                        enhanced = args[2]
                    else:
                        assert torch.equal(args[5], banks[arm]['mask'][index].cuda().unsqueeze(0))
                        enhanced, _ = adapter.forward(args[0], args[1], args[2], args[3], tokens[arm], args[5])
                    heads[arm] = tracker.network.forward_head(enhanced)
                reference_response = tracker.output_window * heads[reference]['score_map']
                reference_peak = int(reference_response.flatten().argmax())
                fixed_response = torch.zeros_like(reference_response); fixed_response.view(-1)[reference_peak] = 1.
                observations = {}
                for arm in CONDITIONS:
                    head = heads[arm]; response = tracker.output_window * head['score_map']
                    peak = int(response.flatten().argmax()); score = float(response.flatten()[peak])
                    bbox = mapped_box(tracker, head, response, prev_box, resize, image.shape)
                    fixed_bbox = mapped_box(tracker, head, fixed_response, prev_box, resize, image.shape)
                    observations[arm] = dict(peak=peak, score=score, bbox=bbox, fixed_reference_peak_bbox=fixed_bbox,
                         score_at_reference_peak=float(response.flatten()[reference_peak]), write_eligible=score > .75)
                    assert np.isfinite(bbox).all() and np.isfinite(score)
                own = observations[reference]
                assert np.max(np.abs(np.asarray(own['bbox']) - output['target_bbox'])) <= 1e-4
                assert abs(own['score'] - float(output['best_score'])) <= 1e-6
                maps.append({key: np.stack([heads[arm][key].cpu().numpy()[0] for arm in CONDITIONS])
                             for key in ['score_map', 'size_map', 'offset_map']})
            assert all(torch.equal(a, b) for a, b in zip(args, originals))
            events.append(dict(frame=frame, previous_bbox=prev_box, resize_factor=resize,
                               reference_peak=reference_peak, actual_template_write=actual_write, conditions=observations))
            held['inputs'] = None
        event_path = destination / (name + '.json'); map_path = destination / (name + '.npz')
        write(event_path, dict(reference=reference, sequence=name, events=events))
        np.savez_compressed(map_path, **{key: np.stack([m[key] for m in maps]) for key in ['score_map', 'size_map', 'offset_map']})
        row = dict(sequence=name, frames=count, events=len(events), max_bbox_error=max_box,
                   max_score_error=max_score, actual_template_writes=writes, event_sha256=sha(event_path), maps_sha256=sha(map_path))
        receipts.append(row); print(json.dumps(row), flush=True)
    handle.remove(); checked()
    assert sum(r['events'] for r in receipts) == (12 if prefix else 654)
    assert sum(r['actual_template_writes'] for r in receipts) > 0
    receipt = dict(status='verified', spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__),
         reference=reference, prefix_only=prefix, rows=receipts, elapsed_seconds=time.time()-started,
         frames=sum(r['frames'] for r in receipts), events=sum(r['events'] for r in receipts),
         parameters_updated=0, subsequent_GT_opened=False, diagnostic_outputs_committed=False,
         all_cloned_inputs_unchanged=True, entire_internal_state_parity_claimed=False, public_evaluation_allowed=False)
    write(destination / 'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'prefix', 'collect'])
    parser.add_argument('--reference', choices=REFERENCES)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        assert args.reference is not None
        collect(args.reference, args.mode == 'prefix')
