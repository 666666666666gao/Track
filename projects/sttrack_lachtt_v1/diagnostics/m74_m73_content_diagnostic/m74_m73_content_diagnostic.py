"""Fixed M73 Category final lexical diagnosis; single seed, no promotion route."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

BASE = Path('/root/autodl-tmp')
M73 = BASE / 'sttrack_m73_paired_lexical_replication_20260907'
ROOT = M73 / 'seed2027'
OUT = BASE / 'sttrack_m74_m73_content_diagnostic_20260907'
M69 = BASE / 'sttrack_m69_m65_content_diagnostic_20260907'
TRAIN_SHA = '0f90ad42fc4084271e8ad6153714d0252531b1c08f71e2646cb94697c8f506be'
RECURSIVE_SHA = '0dcb0dfec7da7c0bc0398cd99dcaf56b6747b176294a35a109fc9c72218280c9'
AUDIT_SHA = 'cbf3e05886c4ec23a0e19e4f4416a3b5da3063d3fc6f52e8da94ce904684663c'
AUDITOR_SHA = '6ea04715e50733a0dd41ca7b458f80c6afbd6af826b01e8865ff4674aa427747'
HEAD_SHA = '8f912e8b8ced42e67ce39b4643cb46aeb835f672309261967c354f2150e82e72'
M69_SPEC_SHA = '1edc827b9f5aaf9dffee70b4f85e456b6f66c9a2e0dfab5c189bb4132ceeab8e'
AMENDMENT_SHA = 'a3c03c8036e9f40bee4e9c24e7eade1dbd04e81e6c23d44b9343007b29b9756f'
CONTROLLER_SHA = '8bc0d893dcdf4af21a1a736d32bb960ef1e34bb48afcd24d4d18d443076b5796'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def identity(pid):
    folder = Path('/proc') / str(pid)
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z'
    return dict(pid=pid, start_ticks=fields[19], cwd=str((folder / 'cwd').resolve()),
        argv=[part.decode() for part in (folder / 'cmdline').read_bytes().split(b'\0') if part])


def parents():
    assert sha(ROOT / 'training_spec.json') == TRAIN_SHA and sha(ROOT / 'recursive_spec.json') == RECURSIVE_SHA
    training = read(ROOT / 'training_spec.json'); recursive = read(ROOT / 'recursive_spec.json')
    assert training['seed'] == 2027
    assert sha(ROOT / 'run_recursive.py') == recursive['runner_sha256']
    assert sha(ROOT / 'integration.json') == training['integration_sha256']
    for name, expected in read(ROOT / 'integration.json')['source_sha256'].items(): assert sha(ROOT / 'code' / name) == expected
    assert sha(training['native_checkpoint']) == training['native_checkpoint_sha256']
    for name in ['category', 'empty']:
        bank = training['banks']['development'][name]; assert sha(bank['path']) == bank['sha256']
    assert sha(BASE / 'audit_m73_completed_20260907.py') == AUDITOR_SHA
    assert sha(M73 / 'completion_tools/completed.json') == AUDIT_SHA
    audit = read(M73 / 'completion_tools/completed.json')
    assert audit['status'] == 'completed_M73_single_seed_checkpoint_and_scalar_audit'
    assert not audit['single_seed_development_gates_pass'] and not audit['content_counterfactuals_allowed']
    assert set(audit['seeds']) == {'2027'} and audit['cancelled_seeds'] == [2028]
    assert sha(M73 / 'single_seed_amendment/plan.json') == AMENDMENT_SHA == audit['amendment_sha256']
    assert sha(M73 / 'single_seed_amendment/controller/result.json') == CONTROLLER_SHA
    assert read(M73 / 'single_seed_amendment/controller/result.json')['status'] == 'completed_M73_audit_content_skipped_by_frozen_gates'
    for path in [M73 / 'controller.exit', M73 / 'single_seed_amendment/controller/controller.exit', ROOT / 'controller.exit']:
        assert path.read_text().strip() == '0'
    assert not (M73 / 'seed2028/training').exists() and not (M73 / 'content_counterfactuals/activation.json').exists()
    seed = audit['seeds']['2027']
    assert sum(seed['gates'].values()) == 10 and not seed['gates']['prior_control_success_protection']
    assert sha(ROOT / 'recursive_result.json') == seed['result_sha256']
    assert sha(ROOT / 'training/category/final.pth') == HEAD_SHA == seed['training']['category']['head_sha256']
    assert sha(ROOT / 'category_recursive_receipt.json') == seed['families']['category']['receipt_sha256']
    assert sha(M69 / 'spec.json') == M69_SPEC_SHA
    return training, recursive, audit


def prepare():
    import torch
    training, recursive, audit = parents()
    previous = read(M69 / 'spec.json')
    for bank in previous['banks'].values(): assert sha(bank['path']) == bank['sha256']
    banks = dict(previous['banks'])
    banks['category'] = dict(training['banks']['development']['category'])
    banks['empty'] = dict(training['banks']['development']['empty'])
    assert banks['category']['sha256'] == previous['banks']['category']['sha256']
    original = torch.load(banks['category']['path'], map_location='cpu')
    empty = torch.load(banks['empty']['path'], map_location='cpu')
    swapped = torch.load(banks['swapped']['path'], map_location='cpu')
    assert original['tokens'].shape == (22, 5, 768)
    assert original['sequences'] == empty['sequences'] == swapped['sequences']
    assert set(original['sequences']) == {case['sequence'] for case in recursive['cases']}
    for bank in [empty, swapped]:
        assert torch.equal(bank['mask'], original['mask']) and torch.equal(bank['empty'], original['empty'])
        assert torch.equal(bank['tokens'][~bank['mask']], original['tokens'][~original['mask']])
    assert torch.equal(empty['tokens'][empty['mask']], original['empty'].expand_as(empty['tokens'][empty['mask']]))
    assert torch.equal(swapped['tokens'][:, 1:], original['tokens'][:, 1:])
    assert bool((swapped['tokens'][:, 0] != original['tokens'][:, 0]).any(1).all())
    for case, old in zip(recursive['cases'], previous['cases']):
        assert all(case[key] == old[key] for key in ['sequence', 'frames', 'init_bbox', 'gt_sha256'])
    head = torch.load(ROOT / 'training/category/final.pth', map_location='cpu')
    assert head['architecture'] == 'semantic_spatial_support_v1' and head['status'] == 'complete'
    assert head['completed_sequences'] == 130 and head['frame_count'] == 186694 and head['optimizer_steps'] == 5798
    assert head['null_support'] and head['use_text'] and head['support_loss_weight'] == 0.0
    assert sum(value.numel() for value in head['model'].values()) == 289154
    assert all(torch.isfinite(value).all() for value in head['model'].values())
    OUT.mkdir()
    queue = OUT / 'run_controls.sh'
    queue.write_text('''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m74_m73_content_diagnostic_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m74_m73_content_diagnostic_20260907.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" eligible > eligibility.log 2>&1
status=$?; printf '%s\\n' "$status" > eligibility.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" idle > device_check.log 2>&1
status=$?; printf '%s\\n' "$status" > device_check.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES=0 "$python" -u "$script" prefix > prefix.log 2>&1
status=$?; printf '%s\\n' "$status" > prefix.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
run_arm() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u "$script" "$1" > "$1.log" 2>&1
    status=$?; printf '%s\\n' "$status" > "$1.exit"
    return "$status"
}
run_arm empty 0 & first=$!
run_arm swapped 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" analyze > analysis.log 2>&1
status=$?; printf '%s\\n' "$status" > analysis.exit
printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    subprocess.run(['bash', '-n', str(queue)], check=True)
    spec = dict(status='frozen_M73_Category_content_diagnostic_before_new_outputs', observed_utc=now(),
        source_sha256=sha(__file__), queue_sha256=sha(queue),
        training_spec_sha256=TRAIN_SHA, recursive_spec_sha256=RECURSIVE_SHA, M73_audit_sha256=AUDIT_SHA,
        fixed_category_head_sha256=HEAD_SHA, single_seed_amendment_sha256=AMENDMENT_SHA,
        original_M73_primary_gate_pass=False, original_M73_content_permission=False,
        authorization_scope='Continue the authorized text-centered tracking research: fixed-head content attribution only. The user prohibits multiseed experiments; no training or additional seed is used. M73 remains failed and its content/public stages remain skipped.',
        head_selection='M73 prospectively fixed seed2027 Category final, retained unchanged after its development gate failed. This diagnostic is newly specified after development; no checkpoint or seed selection.',
        banks=banks, M69_spec_sha256=M69_SPEC_SHA, cases=recursive['cases'],
        prefix_sequences=previous['prefix_sequences'], prefix_frames=previous['prefix_frames'],
        category_vectors_changed=22, masks_padding_and_noncategory_slots_fixed=True,
        primary='category', controls=['empty', 'swapped'], reuse_complete_category_predictions=True,
        new_full_track_calls=66216, prefix_track_calls=303,
        inference='Frozen final Category head and original base; same five slots, mask, t0 box, crop, query and native template code. Only word content changes. No online caption, optimizer or later GT input.',
        descriptive_lexical_criteria=dict(category_pooled_margin_vs_each_control=.001, category_macro_no_less_than_each_control=True,
            category_low_frames_no_more_than_each_control=True, category_H10_no_more_than_each_control=True),
        criteria_scope='Descriptive comparisons, not a promotion gate, caption truth or same-class identity proof.',
        seed=2027, additional_training_seeds=[], poll_seconds=240, estimated_runtime_seconds=1800,
        run_after='M73 single-seed training, development and artifact audit completed with exit0; original failed gates remain unchanged.',
        new_captions=0, new_embeddings=0, new_learned_parameters=0, new_optimizer_steps=0,
        low22_candidate_preparation_allowed=False, full_three_dataset_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'spec.json', spec)
    prepared = dict(status='M74_CPU_inputs_and_fixed_Category_checkpoint_verified', source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'),
        head_sha256=HEAD_SHA, learned_head_loaded_on_CPU=True, finite_learned_parameters=289154,
        category_vectors_changed=22, all_masks_padding_and_noncategory_slots_exact=True,
        bank_source='Reuse M73 Category/Empty banks and frozen M69 swapped-category bank; masks, padding and noncategory slots verified, no new embedding generation.',
        GPU_forward_calls=0, optimizer_steps=0, parent_queues_modified=False, public_evaluation_allowed=False)
    write(OUT / 'preparation_result.json', prepared)
    print(json.dumps(prepared, indent=2))


def checked():
    training, recursive, audit = parents(); spec = read(OUT / 'spec.json')
    assert sha(__file__) == spec['source_sha256'] and sha(OUT / 'run_controls.sh') == spec['queue_sha256']
    for bank in spec['banks'].values(): assert sha(bank['path']) == bank['sha256']
    assert not spec['low22_candidate_preparation_allowed'] and not spec['full_three_dataset_evaluation_allowed']
    return spec, training, audit


def eligible():
    return checked()


def idle():
    used = [int(value.strip()) for value in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
    assert len(used) == 2 and max(used) < 500
    free = shutil.disk_usage(BASE).free; assert free > 1_000_000_000
    print(json.dumps(dict(gpu_memory_MiB=used, disk_free_bytes=free)))


def track(name, prefix=False):
    spec, training, audit = eligible()
    if not prefix:
        assert (OUT / 'prefix.exit').read_text().strip() == '0'
        receipt = read(OUT / 'prefix/receipt.json')
        assert receipt['exact_original_prefix_parity'] and receipt['head_sha256'] == HEAD_SHA
        assert receipt['spec_sha256'] == sha(OUT / 'spec.json')
    import torch
    sys.path.insert(0, str(ROOT / 'code'))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1); torch.manual_seed(training['seed']); torch.cuda.manual_seed_all(training['seed'])
    update_config_from_file(str(ROOT / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=training['native_checkpoint'], base_checkpoint_sha256=training['native_checkpoint_sha256'],
        template_factor=2., template_size=128, search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrackSemantic(params, str(ROOT / 'training/category/final.pth'))
    assert tracker.use_text and tracker.network.semantic_adapter.null_support and not tracker.network.training
    bank = torch.load(spec['banks'][name]['path'], map_location='cpu')
    folder = OUT / ('prefix' if prefix else name); folder.mkdir()
    refs = {r['sequence']: r for r in read(ROOT / 'category_recursive_receipt.json')['sequences']}
    started = time.time(); records = []
    for case in spec['cases']:
        seq = case['sequence']
        if prefix and seq not in spec['prefix_sequences']: continue
        sequence_root = Path(training['dataset_root']) / seq
        def frame(index): return get_rgbd_frame(str(sequence_root / 'color' / ('%08d.jpg' % (index + 1))),
            str(sequence_root / 'depth' / ('%08d.png' % (index + 1))), dtype='rgbcolormap', depth_clip=True)
        index = bank['sequences'].index(seq)
        tracker.initialize(frame(0), dict(init_bbox=case['init_bbox'], text_tokens=bank['tokens'][index], text_mask=bank['mask'][index], empty_text=bank['empty']))
        rows = [dict(frame=0, bbox=list(tracker.state), score=None)]
        count = spec['prefix_frames'] if prefix else case['frames']
        for index in range(1, count):
            prediction = tracker.track(frame(index))
            rows.append(dict(frame=index, bbox=list(prediction['target_bbox']), score=float(prediction['best_score'])))
        if prefix:
            reference = ROOT / 'recursive/category' / (seq + '.json')
            assert sha(reference) == refs[seq]['sha256'] and rows == read(reference)['rows'][:count], seq
        path = folder / (seq + '.json'); write(path, dict(sequence=seq, arm=name, rows=rows))
        record = dict(sequence=seq, frames=len(rows), sha256=sha(path), elapsed_seconds=time.time() - started)
        records.append(record); print(json.dumps(record), flush=True)
    checked()
    receipt = dict(status='complete', variant=name, prefix_only=prefix, source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'),
        head_sha256=HEAD_SHA, bank_sha256=spec['banks'][name]['sha256'], M73_audit_sha256=AUDIT_SHA,
        sequences=records, total_frames=sum(r['frames'] for r in records), subsequent_GT_opened=False,
        optimizer_steps=0, exact_original_prefix_parity=prefix, elapsed_seconds=time.time() - started)
    write(folder / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'sequences'}, indent=2), flush=True)


def analyze():
    import numpy as np
    spec, training, audit = eligible(); data = {}; receipts = {}
    for name in ['category', 'empty', 'swapped']:
        if name == 'category':
            directory = ROOT / 'recursive/category'; receipt_path = ROOT / 'category_recursive_receipt.json'; expected_arm = 'category'
        else:
            assert (OUT / (name + '.exit')).read_text().strip() == '0'
            directory = OUT / name; receipt_path = directory / 'receipt.json'; expected_arm = name
        receipt = read(receipt_path)
        if name != 'category':
            assert receipt['spec_sha256'] == sha(OUT / 'spec.json') and receipt['variant'] == name and not receipt['prefix_only']
            assert receipt['bank_sha256'] == spec['banks'][name]['sha256'] and receipt['M73_audit_sha256'] == AUDIT_SHA
        assert receipt['status'] == 'complete' and receipt['head_sha256'] == HEAD_SHA
        assert receipt['total_frames'] == 33130 and [row['sequence'] for row in receipt['sequences']] == [case['sequence'] for case in spec['cases']]
        data[name] = {}
        for item, case in zip(receipt['sequences'], spec['cases']):
            path = directory / (case['sequence'] + '.json'); assert sha(path) == item['sha256']
            value = read(path); rows = value['rows']
            assert value['sequence'] == case['sequence'] and value['arm'] == expected_arm
            assert len(rows) == case['frames'] and [row['frame'] for row in rows] == list(range(case['frames']))
            assert rows[0] == dict(frame=0, bbox=case['init_bbox'], score=None)
            boxes = np.asarray([row['bbox'] for row in rows])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            data[name][case['sequence']] = rows
        receipts[name] = sha(receipt_path)
    # All three complete prediction families are sealed before opening later-frame GT.
    assert sha(ROOT / 'recursive_metric.py') == read(ROOT / 'recursive_spec.json')['metric_sha256']
    sys.path.insert(0, str(ROOT)); from recursive_metric import statistics
    per = {name: {} for name in data}; writes = {name: 0 for name in data}
    for case in spec['cases']:
        path = Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt'; assert sha(path) == case['gt_sha256']
        gt = np.loadtxt(path, delimiter=',').reshape(-1, 4); assert len(gt) == case['frames']
        for name in data:
            rows = data[name][case['sequence']]
            per[name][case['sequence']] = statistics([row['bbox'] for row in rows], gt)
            writes[name] += sum(row['frame'] % 50 == 0 and row['score'] > .75 for row in rows[1:])
    aggregates = {}
    for name, values in per.items():
        row = {key: sum(value[key] for value in values.values()) for key in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
        row.update(mean_iou=row['iou_sum'] / row['valid_frames'], macro_sequence_mean_iou=float(np.mean([value['mean_iou'] for value in values.values()])))
        assert row['valid_frames'] == 28897; aggregates[name] = row
    for key, value in aggregates['category'].items(): assert abs(value - audit['seeds']['2027']['aggregates']['category'][key]) < 1e-8
    primary = aggregates['category']; comparisons = {}
    for name in ['empty', 'swapped']:
        other = aggregates[name]
        comparisons[name] = dict(pooled_margin=primary['mean_iou'] >= other['mean_iou'] + spec['descriptive_lexical_criteria']['category_pooled_margin_vs_each_control'],
            macro=primary['macro_sequence_mean_iou'] >= other['macro_sequence_mean_iou'],
            low_frames=primary['low_iou_frames'] <= other['low_iou_frames'], H10=primary['failure_episodes'] <= other['failure_episodes'])
    result = dict(status='completed_diagnostic_only_M73_Category_fixed_head_content', observed_utc=now(),
        source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'), head_sha256=HEAD_SHA, M73_audit_sha256=AUDIT_SHA,
        receipts=receipts, aggregates=aggregates, per_sequence=per, reconstructed_template_writes=writes,
        descriptive_lexical_criteria=comparisons, descriptive_criteria_pass=all(flag for item in comparisons.values() for flag in item.values()),
        original_M73_primary_gate_pass=False, original_M73_content_permission=False, new_full_track_calls=66216,
        scope='Prospectively fixed M73 Category final, diagnostic specified after its development outcome, one fixed seed on reused Train development22; automatic original and donor categories are not semantic ground truth. No same-class identity or unbiased generalization proof.',
        low22_candidate_preparation_allowed=False, full_three_dataset_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'eligible', 'idle', 'prefix', 'empty', 'swapped', 'analyze'])
    args = parser.parse_args()
    if args.action == 'prefix': track('category', prefix=True)
    elif args.action in ['empty', 'swapped']: track(args.action)
    else: {'prepare': prepare, 'check': checked, 'eligible': eligible, 'idle': idle, 'analyze': analyze}[args.action]()
