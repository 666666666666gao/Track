"""Fixed M67 Control lexical diagnosis after the existing queue; no promotion route."""
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
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
OUT = BASE / 'sttrack_m72_m67_control_content_20260907'
M69 = BASE / 'sttrack_m69_m65_content_diagnostic_20260907'
M71 = BASE / 'sttrack_m71_equal_budget_routing_20260907'
PARENT = M71 / 'post_queue_runner'
TRAIN_SHA = '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
RECURSIVE_SHA = 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
AUDIT_SHA = 'f85a7105c87a68e86943eeeb40ad55c53e3c368d36c649592b5e23e699c6f5bb'
AUDITOR_SHA = '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
HEAD_SHA = '7a8907a45cec672f11563ad7cfaee21cb64acfee3fa13e854621c146cb8d058e'
M69_SPEC_SHA = '1edc827b9f5aaf9dffee70b4f85e456b6f66c9a2e0dfab5c189bb4132ceeab8e'
ORIGINAL_CONTENT_SHA = '40b765b22c398bc18ed4ac3f5ef33edc7b9611cf1c8d3172062f518d435f8285'
PARENT_SPEC_SHA = '35ff57b50096ec28a3e51bfc78efd2605faa853eaecda0644f3568c2bd6acd10'
PARENT_SOURCE_SHA = 'ee8376769bbc827813e60324baae1d688241c1818cc3a063094585a09ace5023'


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
    assert sha(ROOT / 'run_recursive.py') == recursive['runner_sha256']
    assert sha(ROOT / 'integration.json') == training['integration_sha256']
    for name, expected in read(ROOT / 'integration.json')['source_sha256'].items(): assert sha(ROOT / 'code' / name) == expected
    assert sha(training['native_checkpoint']) == training['native_checkpoint_sha256']
    assert sha(ROOT / 'text_development.pt') == training['text_development_sha256']
    assert sha(BASE / 'audit_m67_completed_20260907.py') == AUDITOR_SHA
    assert sha(ROOT / 'completed_evidence_audit.json') == AUDIT_SHA
    audit = read(ROOT / 'completed_evidence_audit.json')
    assert audit['status'] == 'completed_M67_artifacts_and_development_audited'
    assert not audit['paired_development_gate_pass'] and not audit['content_counterfactuals_allowed']
    assert sum(audit['recomputed_frozen_gates'].values()) == 4
    assert sha(ROOT / 'recursive_result.json') == audit['result_sha256']
    assert sha(ROOT / 'training/control/final.pth') == HEAD_SHA == audit['training']['control']['final_checkpoint_sha256']
    assert sha(ROOT / 'control_recursive_receipt.json') == audit['families']['control']['receipt_sha256']
    assert sha(BASE / 'm67_content_counterfactuals_20260907.py') == ORIGINAL_CONTENT_SHA
    assert sha(M69 / 'spec.json') == M69_SPEC_SHA
    assert sha(PARENT / 'spec.json') == PARENT_SPEC_SHA and sha(BASE / 'm71_after_diagnostics_20260907.py') == PARENT_SOURCE_SHA
    return training, recursive, audit


def prepare():
    import torch
    training, recursive, audit = parents()
    parent = read(PARENT / 'running_identity.json')['identity']
    assert parent == identity(474465) and parent['start_ticks'] == '4578576825'
    assert parent['cwd'] == str(PARENT)
    assert not (PARENT / 'controller.exit').exists()
    previous = read(M69 / 'spec.json')
    for bank in previous['banks'].values(): assert sha(bank['path']) == bank['sha256']
    banks = dict(previous['banks'])
    banks['category'] = dict(path=str(ROOT / 'text_development.pt'), sha256=sha(ROOT / 'text_development.pt'))
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
    head = torch.load(ROOT / 'training/control/final.pth', map_location='cpu')
    assert head['architecture'] == 'semantic_spatial_support_v1' and head['status'] == 'complete'
    assert head['completed_sequences'] == 130 and head['frame_count'] == 186694 and head['optimizer_steps'] == 5798
    assert head['null_support'] and head['use_text'] and head['support_loss_weight'] == 0.0
    assert sum(value.numel() for value in head['model'].values()) == 289154
    assert all(torch.isfinite(value).all() for value in head['model'].values())
    OUT.mkdir()
    queue = OUT / 'run_controls.sh'
    queue.write_text('''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m72_m67_control_content_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m72_m67_control_content_20260907.py
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
    launch = OUT / 'launch_after_M71.sh'
    launch.write_text('''#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m72_m67_control_content_20260907 || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m72_m67_control_content_20260907.py wait_parent > wait.log 2>&1
status=$?; printf '%s\\n' "$status" > wait.exit
if [ "$status" -eq 0 ]; then bash run_controls.sh > controller.log 2>&1; status=$?; fi
printf '%s\\n' "$status" > after_parent.exit
exit "$status"
''')
    for path in [queue, launch]: subprocess.run(['bash', '-n', str(path)], check=True)
    spec = dict(status='frozen_M67_Control_content_diagnostic_before_new_outputs', observed_utc=now(),
        source_sha256=sha(__file__), queue_sha256=sha(queue), launch_sha256=sha(launch),
        training_spec_sha256=TRAIN_SHA, recursive_spec_sha256=RECURSIVE_SHA, M67_audit_sha256=AUDIT_SHA,
        fixed_control_head_sha256=HEAD_SHA, original_M67_support_content_source_sha256=ORIGINAL_CONTENT_SHA,
        original_M67_primary_gate_pass=False, original_M67_content_permission=False,
        authorization_scope='Continue the authorized text-centered tracking research: diagnose the current stronger Control without changing the failed M67 primary study or public evaluation permission.',
        head_selection='Post-development choice of the already completed M67 Control final, fixed by SHA; not an unbiased new test or checkpoint sweep.',
        banks=banks, M69_spec_sha256=M69_SPEC_SHA, cases=recursive['cases'],
        prefix_sequences=previous['prefix_sequences'], prefix_frames=previous['prefix_frames'],
        category_vectors_changed=22, masks_padding_and_noncategory_slots_fixed=True,
        primary='category', controls=['empty', 'swapped'], reuse_complete_category_predictions=True,
        new_full_track_calls=66216, prefix_track_calls=303,
        inference='Frozen final Control and original base; same five slots, mask, t0 box, crop, query and native template code. Only word content changes. No online caption, optimizer or later GT input.',
        descriptive_lexical_criteria=dict(category_pooled_margin_vs_each_control=.001, category_macro_no_less_than_each_control=True,
            category_low_frames_no_more_than_each_control=True, category_H10_no_more_than_each_control=True),
        criteria_scope='Descriptive comparisons, not a promotion gate, caption truth or same-class identity proof.',
        parent_identity=parent, parent_spec_sha256=PARENT_SPEC_SHA, parent_source_sha256=PARENT_SOURCE_SHA, poll_seconds=240,
        run_after='Existing M71 CPU after-queue runner succeeds; original M67/M69/M68/M70/M71 order stays unchanged.',
        new_captions=0, new_embeddings=0, new_learned_parameters=0, new_optimizer_steps=0,
        low22_candidate_preparation_allowed=False, full_three_dataset_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'spec.json', spec)
    prepared = dict(status='M72_CPU_inputs_and_fixed_Control_checkpoint_verified', source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'),
        head_sha256=HEAD_SHA, learned_head_loaded_on_CPU=True, finite_learned_parameters=289154,
        category_vectors_changed=22, all_masks_padding_and_noncategory_slots_exact=True,
        bank_source='Reuse frozen M69 banks, verified against M67 category tensors; no new embedding generation.',
        GPU_forward_calls=0, optimizer_steps=0, parent_queues_modified=False, public_evaluation_allowed=False)
    write(OUT / 'preparation_result.json', prepared)
    print(json.dumps(prepared, indent=2))


def checked():
    training, recursive, audit = parents(); spec = read(OUT / 'spec.json')
    assert sha(__file__) == spec['source_sha256'] and sha(OUT / 'run_controls.sh') == spec['queue_sha256']
    assert sha(OUT / 'launch_after_M71.sh') == spec['launch_sha256']
    for bank in spec['banks'].values(): assert sha(bank['path']) == bank['sha256']
    assert not spec['low22_candidate_preparation_allowed'] and not spec['full_three_dataset_evaluation_allowed']
    return spec, training, audit


def wait_parent():
    spec, _, _ = checked()
    assert not (OUT / 'waiting_identity.json').exists()
    write(OUT / 'waiting_identity.json', dict(observed_utc=now(), identity=identity(os.getpid()),
        source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json')))
    while not (PARENT / 'controller.exit').exists():
        current = identity(spec['parent_identity']['pid']); assert current == spec['parent_identity']
        event = dict(time=now(), event='waiting_for_registered_M71_runner', parent_identity=current, poll_seconds=spec['poll_seconds'])
        write(OUT / 'latest_wait.json', event); print(json.dumps(event), flush=True)
        time.sleep(spec['poll_seconds'])
    assert (PARENT / 'controller.exit').read_text().strip() == '0'
    result = read(PARENT / 'result.json')
    assert result['status'] == 'completed_frozen_M71_after_diagnostic_queue' and result['spec_sha256'] == PARENT_SPEC_SHA
    assert result['M71_result_sha256'] == sha(M71 / 'result.json')
    checked()
    write(OUT / 'parent_completed.json', dict(status='registered_M71_runner_completed', observed_utc=now(),
        parent_result_sha256=sha(PARENT / 'result.json'), M71_result_sha256=sha(M71 / 'result.json')))


def eligible():
    spec, training, audit = checked()
    assert (OUT / 'wait.exit').read_text().strip() == '0'
    assert (PARENT / 'controller.exit').read_text().strip() == '0'
    ready = read(OUT / 'parent_completed.json')
    assert ready['parent_result_sha256'] == sha(PARENT / 'result.json')
    assert ready['M71_result_sha256'] == sha(M71 / 'result.json')
    return spec, training, audit


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
    tracker = STTrackSemantic(params, str(ROOT / 'training/control/final.pth'))
    assert tracker.use_text and tracker.network.semantic_adapter.null_support and not tracker.network.training
    bank = torch.load(spec['banks'][name]['path'], map_location='cpu')
    folder = OUT / ('prefix' if prefix else name); folder.mkdir()
    refs = {r['sequence']: r for r in read(ROOT / 'control_recursive_receipt.json')['sequences']}
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
            reference = ROOT / 'recursive/control' / (seq + '.json')
            assert sha(reference) == refs[seq]['sha256'] and rows == read(reference)['rows'][:count], seq
        path = folder / (seq + '.json'); write(path, dict(sequence=seq, arm=name, rows=rows))
        record = dict(sequence=seq, frames=len(rows), sha256=sha(path), elapsed_seconds=time.time() - started)
        records.append(record); print(json.dumps(record), flush=True)
    checked()
    receipt = dict(status='complete', variant=name, prefix_only=prefix, source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'),
        head_sha256=HEAD_SHA, bank_sha256=spec['banks'][name]['sha256'], M67_audit_sha256=AUDIT_SHA,
        sequences=records, total_frames=sum(r['frames'] for r in records), subsequent_GT_opened=False,
        optimizer_steps=0, exact_original_prefix_parity=prefix, elapsed_seconds=time.time() - started)
    write(folder / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'sequences'}, indent=2), flush=True)


def analyze():
    import numpy as np
    spec, training, audit = eligible(); data = {}; receipts = {}
    for name in ['category', 'empty', 'swapped']:
        if name == 'category':
            directory = ROOT / 'recursive/control'; receipt_path = ROOT / 'control_recursive_receipt.json'; expected_arm = 'control'
        else:
            assert (OUT / (name + '.exit')).read_text().strip() == '0'
            directory = OUT / name; receipt_path = directory / 'receipt.json'; expected_arm = name
        receipt = read(receipt_path)
        if name != 'category':
            assert receipt['spec_sha256'] == sha(OUT / 'spec.json') and receipt['variant'] == name and not receipt['prefix_only']
            assert receipt['bank_sha256'] == spec['banks'][name]['sha256'] and receipt['M67_audit_sha256'] == AUDIT_SHA
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
    for key, value in aggregates['category'].items(): assert abs(value - audit['recomputed_aggregates']['control'][key]) < 1e-8
    primary = aggregates['category']; comparisons = {}
    for name in ['empty', 'swapped']:
        other = aggregates[name]
        comparisons[name] = dict(pooled_margin=primary['mean_iou'] >= other['mean_iou'] + spec['descriptive_lexical_criteria']['category_pooled_margin_vs_each_control'],
            macro=primary['macro_sequence_mean_iou'] >= other['macro_sequence_mean_iou'],
            low_frames=primary['low_iou_frames'] <= other['low_iou_frames'], H10=primary['failure_episodes'] <= other['failure_episodes'])
    result = dict(status='completed_diagnostic_only_M67_Control_fixed_head_content', observed_utc=now(),
        source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'), head_sha256=HEAD_SHA, M67_audit_sha256=AUDIT_SHA,
        receipts=receipts, aggregates=aggregates, per_sequence=per, reconstructed_template_writes=writes,
        descriptive_lexical_criteria=comparisons, descriptive_criteria_pass=all(flag for item in comparisons.values() for flag in item.values()),
        original_M67_primary_gate_pass=False, original_M67_support_content_permission=False, new_full_track_calls=66216,
        scope='Post-development-selected fixed Control, one run on reused Train development22; automatic original and donor categories are not semantic ground truth. No same-class identity or unbiased generalization proof.',
        low22_candidate_preparation_allowed=False, full_three_dataset_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'wait_parent', 'eligible', 'idle', 'prefix', 'empty', 'swapped', 'analyze'])
    args = parser.parse_args()
    if args.action == 'prefix': track('category', prefix=True)
    elif args.action in ['empty', 'swapped']: track(args.action)
    else: {'prepare': prepare, 'check': checked, 'wait_parent': wait_parent, 'eligible': eligible, 'idle': idle, 'analyze': analyze}[args.action]()
