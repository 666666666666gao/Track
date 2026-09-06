"""Separate automatic-category content in localization and template-write scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/sttrack_m63_location_write_factorial_20260907')
M59 = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
M60 = Path('/root/autodl-tmp/sttrack_m60_category_isolation_20260906')
M61 = Path('/root/autodl-tmp/sttrack_m61_fixed_state_language_20260907')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
PLANS = {'C': M59 / 'category_plan.json', 'S': M60 / 'plan.json'}
MODES = ['CC', 'SS', 'CS', 'SC']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parent():
    assert (M61 / 'job.exit').read_text().strip() == '0'
    assert sha(M61 / 'result.json') == 'dafe3d3fd3d1667e2702fff179345025efe5c741c586b93c520b818174bfaa6a'
    assert sha(M60 / 'result.json') == 'febddd93eda9f25f3530377948751c1999030db9a08656f9bb24ae151ad64287'
    sys.path.insert(0, str(M59))
    import run_controls as previous
    frozen, training, _, _ = previous.parent_ready()
    return previous, frozen, training


def prepare():
    _, frozen, training = parent()
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, text_bank
    banks = {}
    bundles = {}
    for role, path in PLANS.items():
        plan, bundle = checked_plan(path); banks[role] = text_bank(plan, bundle).bank; bundles[role] = plan['bundle_sha256']
    assert bundles['C'] == bundles['S']
    assert banks['C']['keys'] == banks['S']['keys'] and torch.equal(banks['C']['mask'], banks['S']['mask'])
    assert torch.equal(banks['C']['tokens'][:, 1:], banks['S']['tokens'][:, 1:])
    assert int((banks['C']['tokens'][:, 0] != banks['S']['tokens'][:, 0]).any(1).sum()) == 22
    ROOT.mkdir()
    spec = dict(status='frozen_before_location_write_factorial', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), m61_result_sha256=sha(M61 / 'result.json'), m60_result_sha256=sha(M60 / 'result.json'),
        role_plan_sha256={role: sha(path) for role, path in PLANS.items()}, bundle_sha256=bundles['C'],
        head_sha256=json.loads((M60 / 'result.json').read_text())['head_sha256'], cases=frozen['cases'],
        dataset_root=training['dataset_root'], modes=MODES,
        roles={'C': 'Original automatic category; all attributes empty.', 'S': 'Previously frozen swapped automatic category; identical empty attributes and masks.'},
        mode_order='First letter selects localization text, second selects template-write scoring text.',
        localization='Normal semantic response, size and offset choose the box and reported confidence on every frame.',
        write_score='At each default 50-frame check, apply the same frozen adapter/head using write-role text to cloned pre-adapter features. Read its Hann score at the localization-selected peak; threshold remains strictly greater than 0.75.',
        template_action='Only the write decision changes. A write crops the localization box and follows the native dynamic-template append/pop rule. The quality-head box and query are never committed.',
        controls='CC must reproduce M59 category and SS must reproduce M60 swapped category, every frame of all22, before either full crossed arm starts.',
        prefix_sequences=['bag05_indoor', 'container01_indoor', 'mobilephone02_indoor'], prefix_frames=202,
        full_image_frames=132520, full_track_calls=132432, prefix_image_frames=2424, prefix_track_calls=2412,
        stages=[['CC', 'SS'], ['CS', 'SC']], execution_device=0,
        execution_order='Sequential on GPU0; GPU1 remains available for candidate validation.',
        new_optimizer_steps=0, new_learned_parameters=0, new_captions=0,
        metric_sha256=sha(PARENT / 'recursive_metric.py'), reference_receipt_sha256={'C': sha(M59 / 'category_receipt.json'), 'S': sha(M60 / 'receipt.json')},
        outcomes='Full pooled/macro IoU, low frames, H10, native/category success protection, write counts, first write-decision divergence and factorial contrasts.',
        GT_policy='Initialization boxes only in inference. Subsequent GT opens after all four complete prediction families and receipts verify.',
        public_evaluation_allowed=False, independent_review_pass=False,
        interpretation='Fixed trained network; a causal runtime input-channel ablation. Not a new trained quality head, a future-memory-utility predictor, or semantic ground truth.',
        next='Use the four complete results to choose a deployable localization/write-control design. Swapped donor words remain diagnostic inputs. Independently freeze any public candidate protocol; new learned modules still require DepthTrack Train training.')
    write(ROOT / 'spec.json', spec)
    write(ROOT / 'preparation_result.json', dict(status='four_cell_factorial_frozen', spec_sha256=sha(ROOT / 'spec.json'),
          source_sha256=sha(__file__), changed_learned_parameters=0, changed_category_slots=22,
          all_other_text_tokens_and_masks_unchanged=True, new_GPU_calls=0, public_evaluation_allowed=False))
    print(json.dumps(json.loads((ROOT / 'preparation_result.json').read_text()), indent=2))


def checked():
    previous, _, training = parent()
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert sha(__file__) == spec['source_sha256']
    assert sha(PARENT / 'recursive_metric.py') == spec['metric_sha256']
    for role, path in PLANS.items(): assert sha(path) == spec['role_plan_sha256'][role]
    return previous, spec, training


@torch.no_grad()
def step(tracker, image, write_tokens, held, crossed):
    from lib.train.data.processing_utils import sample_target
    from lib.utils.box_ops import clip_box
    height, width, _ = image.shape
    tracker.frame_id += 1
    scheduled = tracker.frame_id % tracker.update_intervals == 0
    held['enabled'] = scheduled and crossed
    patch, resize, _ = sample_target(image, tracker.state, tracker.params.search_factor, output_sz=tracker.params.search_size)
    search = tracker.preprocessor.process(patch)
    output = tracker.network.forward(template=tracker.z_dict, search=[search], ce_template_mask=tracker.box_mask_z,
        track_query_before=tracker.track_query_before, keep_rate=tracker.keep_rate, semantic_context=tracker.semantic_context)[0]
    tracker.track_query_before = output['track_query_before']
    response = tracker.output_window * output['score_map']
    boxes = tracker.network.box_head.cal_bbox(response, output['size_map'], output['offset_map']).view(-1, 4)
    bbox = (boxes.mean(0) * tracker.params.search_size / resize).tolist()
    tracker.state = clip_box(tracker.map_box_back(bbox, resize), height, width, margin=10)
    score, peak = torch.max(response.flatten(1), dim=1, keepdim=True)
    write_score = None; wrote = False
    if scheduled:
        if crossed:
            args = held['inputs']; assert args is not None
            enhanced, _ = tracker.network.semantic_adapter.forward(args[0], args[1], args[2], args[3], write_tokens, args[5])
            quality = tracker.network.forward_head(enhanced)
            quality_response = tracker.output_window * quality['score_map']
            write_score = float(quality_response.flatten(1).gather(1, peak).item())
            held['inputs'] = None
        else:
            write_score = float(score.item())
        if write_score > tracker.update_threshold:
            template, _, _ = sample_target(image, tracker.state, tracker.params.template_factor, output_sz=tracker.params.template_size)
            tracker.z_patch_arr = template
            tracker.z_dict.append(tracker.preprocessor.process(template))
            if len(tracker.z_dict) > tracker.num_template:
                tracker.z_dict.pop(1)
            wrote = True
    return dict(bbox=list(tracker.state), score=float(score.item()), write_score=write_score, template_write=wrote)


def run(mode, prefix):
    previous, spec, training = checked()
    if not prefix:
        for name in MODES: assert (ROOT / ('prefix_' + name + '.exit')).read_text().strip() == '0'
        if mode in ['CS', 'SC']:
            for name in ['CC', 'SS']:
                assert (ROOT / ('full_' + name + '.exit')).read_text().strip() == '0'
                assert json.loads((ROOT / ('full_' + name) / 'receipt.json').read_text())['all_frame_reference_parity']
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, make_tracker, text_bank
    banks = {}
    for role, path in PLANS.items():
        plan, bundle = checked_plan(path); banks[role] = text_bank(plan, bundle)
    tracker = make_tracker(bundle)
    assert tracker.num_template == 2 and tracker.update_intervals == 50 and tracker.update_threshold == .75
    assert not tracker.debug and not tracker.save_all_boxes
    assert not any(module.training for module in tracker.network.modules())
    held = {'enabled': False, 'inputs': None}

    def capture(module, args, output):
        if held['enabled']:
            assert held['inputs'] is None
            held['inputs'] = tuple(value.detach().clone() for value in args)

    handle = tracker.network.semantic_adapter.register_forward_hook(capture)
    loc, write_role = mode; crossed = loc != write_role
    receipt_path = M59 / 'category_receipt.json' if loc == 'C' else M60 / 'receipt.json'
    assert sha(receipt_path) == spec['reference_receipt_sha256'][loc]
    reference_files = {r['sequence']: r for r in json.loads(receipt_path.read_text())['sequences']}
    destination = ROOT / (('prefix_' if prefix else 'full_') + mode); destination.mkdir()
    cases = [case for case in spec['cases'] if not prefix or case['sequence'] in spec['prefix_sequences']]
    receipts = []; started = time.time()
    for case in cases:
        name = case['sequence']; folder = Path(training['dataset_root']) / name
        reference_path = (M59 / 'predictions/category' if loc == 'C' else M60 / 'predictions') / (name + '.json')
        assert sha(reference_path) == reference_files[name]['sha256']
        reference = json.loads(reference_path.read_text())['rows']
        rgb, image = previous.load_frame(folder, 0)
        info = banks[loc].info(rgb, case['init_bbox']); write_info = banks[write_role].info(rgb, case['init_bbox'])
        assert torch.equal(info['text_mask'], write_info['text_mask'])
        tracker.initialize(image, info)
        write_tokens = write_info['text_tokens'].float().cuda().unsqueeze(0)
        rows = [dict(frame=0, bbox=list(tracker.state), score=None, write_score=None, template_write=False)]
        count = spec['prefix_frames'] if prefix else case['frames']
        branched = False; first_branch = None; max_box = max_score = 0.; writes = 0
        for frame in range(1, count):
            _, image = previous.load_frame(folder, frame)
            old_template = tracker.z_dict[1]
            observation = step(tracker, image, write_tokens, held, crossed)
            assert observation['template_write'] == (tracker.z_dict[1] is not old_template)
            if not branched:
                max_box = max(max_box, float(np.max(np.abs(np.asarray(observation['bbox']) - reference[frame]['bbox']))))
                max_score = max(max_score, abs(observation['score'] - reference[frame]['score']))
                assert max_box <= 1e-4 and max_score <= 1e-6, (mode, name, frame, max_box, max_score)
                default_write = frame % 50 == 0 and reference[frame]['score'] > .75
                if observation['template_write'] != default_write:
                    assert crossed
                    branched = True; first_branch = frame
            rows.append(dict(frame=frame, **observation)); writes += observation['template_write']
        assert crossed or not branched
        path = destination / (name + '.json'); write(path, dict(mode=mode, sequence=name, rows=rows))
        item = dict(sequence=name, frames=count, sha256=sha(path), template_writes=writes,
                    max_pre_branch_bbox_error=max_box, max_pre_branch_score_error=max_score, first_write_branch_frame=first_branch)
        receipts.append(item); print(json.dumps(item), flush=True)
    handle.remove(); checked()
    write(destination / 'receipt.json', dict(status='complete', spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__),
          mode=mode, prefix_only=prefix, sequences=receipts, frames=sum(r['frames'] for r in receipts),
          all_frame_reference_parity=not crossed, pre_write_divergence_reference_parity=True,
          subsequent_GT_opened=False, optimizer_steps=0, new_captions=0, elapsed_seconds=time.time()-started))


def analyze():
    previous, spec, training = checked()
    sys.path.insert(0, str(PARENT))
    from recursive_metric import statistics
    receipts = {}; boxes = {}; scores = {}; writes = {}
    for mode in MODES:
        assert (ROOT / ('full_' + mode + '.exit')).read_text().strip() == '0'
        folder = ROOT / ('full_' + mode); receipt = json.loads((folder / 'receipt.json').read_text())
        assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(ROOT / 'spec.json')
        assert not receipt['prefix_only'] and receipt['frames'] == 33130
        assert [r['sequence'] for r in receipt['sequences']] == [c['sequence'] for c in spec['cases']]
        boxes[mode] = {}; scores[mode] = {}; writes[mode] = {}
        for case, item in zip(spec['cases'], receipt['sequences']):
            path = folder / (case['sequence'] + '.json'); assert sha(path) == item['sha256']
            data = json.loads(path.read_text()); rows = data['rows']
            assert data['mode'] == mode and data['sequence'] == case['sequence']
            assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames']))
            assert rows[0]['bbox'] == case['init_bbox']
            values = np.asarray([r['bbox'] for r in rows]); score = np.asarray([r['score'] for r in rows[1:]])
            assert np.isfinite(values).all() and (values[:, 2:] > 0).all() and np.isfinite(score).all()
            for row in rows[1:]:
                if row['frame'] % 50 == 0:
                    assert np.isfinite(row['write_score']) and row['template_write'] == (row['write_score'] > .75)
                else: assert row['write_score'] is None and not row['template_write']
            boxes[mode][case['sequence']] = values; scores[mode][case['sequence']] = score
            writes[mode][case['sequence']] = sum(r['template_write'] for r in rows)
        receipts[mode] = sha(folder / 'receipt.json')
    per = {mode: {} for mode in MODES}
    for case in spec['cases']:
        path = Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt'; assert sha(path) == case['gt_sha256']
        gt = np.loadtxt(path, delimiter=',').reshape(-1, 4); assert len(gt) == case['frames']
        for mode in MODES: per[mode][case['sequence']] = statistics(boxes[mode][case['sequence']], gt)
    aggregates = {mode: previous.aggregate(per[mode]) for mode in MODES}
    m60 = json.loads((M60 / 'result.json').read_text())
    for mode, arm in [('CC', 'original_category'), ('SS', 'swapped_category')]:
        for key, value in aggregates[mode].items(): assert abs(value - m60['aggregates'][arm][key]) < 1e-8
    contrasts = {}
    for name, a, b in [('localization_given_C_write', 'CC', 'SC'), ('localization_given_S_write', 'CS', 'SS'),
                       ('write_given_C_localization', 'CC', 'CS'), ('write_given_S_localization', 'SC', 'SS')]:
        contrasts[name] = {key: aggregates[a][key] - aggregates[b][key] for key in ['mean_iou', 'macro_sequence_mean_iou', 'low_iou_frames', 'failure_episodes']}
    protected = [name for name, row in per['CC'].items() if row['failure_episodes'] == 0]
    result = dict(status='completed_fixed_weight_location_write_factorial', observed_utc=datetime.now(timezone.utc).isoformat(),
                  spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__), head_sha256=spec['head_sha256'],
                  receipts=receipts, aggregates=aggregates, per_sequence=per, contrasts=contrasts,
                  template_writes={mode: sum(values.values()) for mode, values in writes.items()},
                  per_sequence_template_writes=writes, category_zero_H10_sequences=protected,
                  new_failure_sequences={mode: [name for name in protected if per[mode][name]['failure_episodes'] > 0] for mode in MODES},
                  public_evaluation_allowed=False, independent_review_pass=False,
                  scope='Same learned weight and empty attributes; original versus swapped automatic category assigned to localization and write-score channels. Full Train development recursion; no new official metric or semantic ground-truth claim.')
    write(ROOT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['per_sequence', 'per_sequence_template_writes']}, indent=2), flush=True)


def queue():
    checked()
    stages = [('prefix', [mode]) for mode in MODES] + [('full', [mode]) for mode in MODES]
    for stage, modes in stages:
        usage = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split()]
        assert len(usage) == 2 and usage[0] < 500
        children = []
        for gpu, mode in enumerate(modes):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            log = (ROOT / (stage + '_' + mode + '.log')).open('w')
            process = subprocess.Popen([PYTHON, str(Path(__file__)), stage, '--mode', mode], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close(); children.append((mode, process))
        write(ROOT / 'active_stage.json', dict(stage=stage, modes=modes, pids=[p.pid for _, p in children],
              observed_utc=datetime.now(timezone.utc).isoformat(), gpu_memory_before_mib=usage))
        statuses = []
        for mode, process in children:
            code = process.wait(); statuses.append(code); (ROOT / (stage + '_' + mode + '.exit')).write_text(str(code) + '\n')
        if any(code != 0 for code in statuses):
            (ROOT / 'job.exit').write_text('1\n'); return 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with (ROOT / 'analysis.log').open('w') as log:
        code = subprocess.run([PYTHON, str(Path(__file__)), 'analyze'], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (ROOT / 'analysis.exit').write_text(str(code) + '\n'); (ROOT / 'job.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'prefix', 'full', 'analyze', 'queue'])
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'analyze': analyze()
    elif args.action == 'queue': raise SystemExit(queue())
    else:
        assert args.mode is not None
        run(args.mode, args.action == 'prefix')
