"""Prospectively fixed seed2027 content controls, conditional on both M73 seeds."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m73_paired_lexical_replication_20260907'
ROOT = PARENT / 'seed2027'
OUT = PARENT / 'content_counterfactuals'
M69 = BASE / 'sttrack_m69_m65_content_diagnostic_20260907'
AUDITOR = BASE / 'audit_m73_completed_20260907.py'
AUDITOR_SHA = '4fcaf10e73225ea7bf9f57dbd031c1a6d9f58ebed29fa04341277d0d69fb3ef8'
REFERENCE_SHA = 'fb387791f1de063990ff2b9f105b7387616893c382dce44642e5b591375c45fa'
M69_SPEC_SHA = '1edc827b9f5aaf9dffee70b4f85e456b6f66c9a2e0dfab5c189bb4132ceeab8e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def parents():
    assert sha(AUDITOR) == AUDITOR_SHA
    assert sha(PARENT / 'completion_tools/reference.json') == REFERENCE_SHA
    loader = importlib.util.spec_from_file_location('m73_completion_binding', str(AUDITOR))
    auditor = importlib.util.module_from_spec(loader); loader.loader.exec_module(auditor)
    spec, frozen, _ = auditor.checked()
    assert spec['candidate_seed'] == 2027 and spec['candidate_arm'] == 'category'
    assert sha(M69 / 'spec.json') == M69_SPEC_SHA
    return read(ROOT / 'training_spec.json'), read(ROOT / 'recursive_spec.json')


def prepare():
    import torch
    training, recursive = parents(); previous = read(M69 / 'spec.json')
    assert not (PARENT / 'controller.exit').exists()
    assert not (PARENT / 'completion_tools/completed.json').exists()
    assert not (ROOT / 'training/category/final.pth').exists()
    banks = {a: training['banks']['development'][a] for a in ['category', 'empty']}
    banks['swapped'] = previous['banks']['swapped']
    for b in banks.values(): assert sha(b['path']) == b['sha256']
    category, empty, swapped = [torch.load(banks[n]['path'], map_location='cpu') for n in ['category', 'empty', 'swapped']]
    assert category['tokens'].shape == (22, 5, 768)
    assert category['sequences'] == empty['sequences'] == swapped['sequences']
    assert set(category['sequences']) == {c['sequence'] for c in recursive['cases']}
    for b in [empty, swapped]:
        assert torch.equal(b['mask'], category['mask']) and torch.equal(b['empty'], category['empty'])
        assert torch.equal(b['tokens'][~b['mask']], category['tokens'][~category['mask']])
    assert torch.equal(empty['tokens'][empty['mask']], category['empty'].expand_as(empty['tokens'][empty['mask']]))
    assert torch.equal(swapped['tokens'][:, 1:], category['tokens'][:, 1:])
    assert bool((swapped['tokens'][:, 0] != category['tokens'][:, 0]).any(1).all())
    assert len(recursive['cases']) == len(previous['cases']) == 22
    for c, p in zip(recursive['cases'], previous['cases']):
        assert all(c[k] == p[k] for k in ['sequence', 'frames', 'init_bbox', 'gt_sha256'])
    OUT.mkdir()
    queue = OUT / 'run_controls.sh'
    queue.write_text("""#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/content_counterfactuals || exit 1
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m73_content_counterfactuals_20260907.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_step() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u "$script" "$1" > "$1.log" 2>&1
    status=$?; printf '%s\\n' "$status" > "$1.exit"
    return "$status"
}
for action in activate idle; do
    run_step "$action" '' || { printf '1\\n' > controller.exit; exit 1; }
done
run_step prefix 0 || { printf '1\\n' > controller.exit; exit 1; }
run_step empty 0 & first=$!
run_step swapped 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
run_step analyze ''; status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
""")
    subprocess.run(['bash', '-n', str(queue)], check=True)
    spec = dict(status='frozen_M73_content_protocol_before_seed_results', observed_utc=now(),
        source_sha256=sha(__file__), queue_sha256=sha(queue), auditor_sha256=AUDITOR_SHA, reference_sha256=REFERENCE_SHA,
        M73_spec_sha256=sha(PARENT / 'spec.json'), M73_frozen_sha256=sha(PARENT / 'frozen.json'),
        training_spec_sha256=sha(ROOT / 'training_spec.json'), recursive_spec_sha256=sha(ROOT / 'recursive_spec.json'),
        candidate_seed=2027, candidate_arm='category', candidate_path=str(ROOT / 'training/category/final.pth'),
        head_selection='Prospective seed2027 Category final only; seed2028 verifies replication and cannot replace the candidate.',
        prerequisite='Both seeds individually pass all inherited eleven development requirements, verified by completed-checkpoint/scalar auditor.',
        banks=banks, M69_spec_sha256=M69_SPEC_SHA, cases=recursive['cases'],
        prefix_sequences=previous['prefix_sequences'], prefix_frames=previous['prefix_frames'],
        primary='category', controls=['empty', 'swapped'], reuse_complete_category_predictions=True,
        new_full_track_calls=66216, prefix_track_calls=303, masks_padding_and_noncategory_slots_fixed=True,
        lexical_criteria=dict(category_pooled_margin_vs_each_control=.001, category_macro_no_less_than_each_control=True,
            category_low_frames_no_more_than_each_control=True, category_H10_no_more_than_each_control=True),
        inference='Same seed2027 Category final head, five slots/masks, base, native crop/query/template code; only valid-slot word content differs.',
        fixed_head_empty_is_separate_from_trained_empty_head=True,
        interpretation='Reused Train development22 and automatic captions; not same-class identity or unbiased generalization evidence.',
        after_pass='Only candidate OPE/TraX entry parity preparation; this script never launches public evaluation.',
        new_parameters=0, optimizer_steps=0, new_captions=0, public_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'spec.json', spec)
    result = dict(status='M73_content_CPU_banks_and_prospective_selection_verified', source_sha256=sha(__file__),
        spec_sha256=sha(OUT / 'spec.json'), bank_shape=[22, 5, 768], changed_category_slots=22,
        all_masks_padding_and_noncategory_slots_exact=True, candidate_final_does_not_yet_exist=True,
        GPU_forward_calls=0, new_optimizer_steps=0, parent_queue_modified=False, public_evaluation_allowed=False)
    write(OUT / 'preparation_result.json', result); print(json.dumps(result))


def checked():
    training, recursive = parents(); spec = read(OUT / 'spec.json')
    assert sha(__file__) == spec['source_sha256'] and sha(OUT / 'run_controls.sh') == spec['queue_sha256']
    assert sha(ROOT / 'training_spec.json') == spec['training_spec_sha256']
    assert sha(ROOT / 'recursive_spec.json') == spec['recursive_spec_sha256']
    for b in spec['banks'].values(): assert sha(b['path']) == b['sha256']
    assert spec['candidate_seed'] == 2027 and spec['candidate_arm'] == 'category'
    assert spec['candidate_path'] == str(ROOT / 'training/category/final.pth')
    assert not spec['public_evaluation_allowed']
    return spec, training


def eligible():
    spec, training = checked()
    assert (PARENT / 'completion_tools/completed.json').exists(), 'M73 completion audit is not yet available'
    audit = read(PARENT / 'completion_tools/completed.json')
    assert audit['status'] == 'completed_M73_two_seed_checkpoint_and_scalar_audit' and audit['auditor_sha256'] == AUDITOR_SHA
    assert (PARENT / 'controller.exit').read_text().strip() == '0'
    assert audit['parent_result_sha256'] == sha(PARENT / 'result.json')
    assert audit['spec_sha256'] == spec['M73_spec_sha256'] and audit['frozen_sha256'] == spec['M73_frozen_sha256']
    assert audit['both_seed_development_gates_pass'] and audit['content_counterfactuals_allowed']
    assert audit['candidate_seed'] == 2027
    for seed in ['2027', '2028']:
        assert audit['seeds'][seed]['gate_pass'] and all(audit['seeds'][seed]['gates'].values())
        assert audit['seeds'][seed]['result_sha256'] == sha(PARENT / ('seed'+seed) / 'recursive_result.json')
    assert sha(spec['candidate_path']) == audit['candidate_head_sha256']
    assert sha(ROOT / 'category_recursive_receipt.json') == audit['seeds']['2027']['families']['category']['receipt_sha256']
    return spec, training, audit


def activate():
    import torch
    spec, training, audit = eligible(); assert not (OUT / 'activation.json').exists()
    head = torch.load(spec['candidate_path'], map_location='cpu')
    assert head['status'] == 'complete' and head['seed'] == 2027
    assert head['architecture'] == 'semantic_spatial_support_v1' and head['use_text'] and head['null_support']
    assert head['support_loss_weight'] == 0. and head['completed_sequences'] == 130
    assert head['frame_count'] == 186694 and head['optimizer_steps'] == 5798
    assert sum(v.numel() for v in head['model'].values()) == 289154 and all(torch.isfinite(v).all() for v in head['model'].values())
    r = dict(status='activated_predetermined_M73_seed2027_Category_final', observed_utc=now(),
        spec_sha256=sha(OUT / 'spec.json'), audit_sha256=sha(PARENT / 'completion_tools/completed.json'),
        head_sha256=audit['candidate_head_sha256'], seed=2027, public_evaluation_allowed=False)
    write(OUT / 'activation.json', r); print(json.dumps(r))


def activated():
    spec, training, audit = eligible(); a = read(OUT / 'activation.json')
    assert a['spec_sha256'] == sha(OUT / 'spec.json') and a['audit_sha256'] == sha(PARENT / 'completion_tools/completed.json')
    assert a['head_sha256'] == audit['candidate_head_sha256'] and a['seed'] == 2027
    return spec, training, audit, a['head_sha256'], a['audit_sha256']


def idle():
    used = [int(value.strip()) for value in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
    assert len(used) == 2 and max(used) < 500
    free = shutil.disk_usage(BASE).free; assert free > 1_000_000_000
    print(json.dumps(dict(gpu_memory_MiB=used, disk_free_bytes=free)))


def track(name, prefix=False):
    spec, training, audit, head_sha, audit_sha = activated()
    if not prefix:
        assert (OUT / 'prefix.exit').read_text().strip() == '0'
        receipt = read(OUT / 'prefix/receipt.json')
        assert receipt['exact_original_prefix_parity'] and receipt['head_sha256'] == head_sha
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
        head_sha256=head_sha, bank_sha256=spec['banks'][name]['sha256'], M73_audit_sha256=audit_sha,
        sequences=records, total_frames=sum(r['frames'] for r in records), subsequent_GT_opened=False,
        optimizer_steps=0, exact_original_prefix_parity=prefix, elapsed_seconds=time.time() - started)
    write(folder / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'sequences'}, indent=2), flush=True)


def analyze():
    import numpy as np
    spec, training, audit, head_sha, audit_sha = activated(); data = {}; receipts = {}
    for name in ['category', 'empty', 'swapped']:
        if name == 'category':
            directory = ROOT / 'recursive/category'; receipt_path = ROOT / 'category_recursive_receipt.json'; expected_arm = 'category'
        else:
            assert (OUT / (name + '.exit')).read_text().strip() == '0'
            directory = OUT / name; receipt_path = directory / 'receipt.json'; expected_arm = name
        receipt = read(receipt_path)
        if name != 'category':
            assert receipt['spec_sha256'] == sha(OUT / 'spec.json') and receipt['variant'] == name and not receipt['prefix_only']
            assert receipt['bank_sha256'] == spec['banks'][name]['sha256'] and receipt['M73_audit_sha256'] == audit_sha
        assert receipt['status'] == 'complete' and receipt['head_sha256'] == head_sha
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
        comparisons[name] = dict(pooled_margin=primary['mean_iou'] >= other['mean_iou'] + spec['lexical_criteria']['category_pooled_margin_vs_each_control'],
            macro=primary['macro_sequence_mean_iou'] >= other['macro_sequence_mean_iou'],
            low_frames=primary['low_iou_frames'] <= other['low_iou_frames'], H10=primary['failure_episodes'] <= other['failure_episodes'])
    result = dict(status='completed_M73_predetermined_final_fixed_head_content', observed_utc=now(),
        source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'), head_sha256=head_sha, M73_audit_sha256=audit_sha,
        receipts=receipts, aggregates=aggregates, per_sequence=per, reconstructed_template_writes=writes,
        lexical_criteria=comparisons, lexical_criteria_pass=all(flag for item in comparisons.values() for flag in item.values()),
        both_seed_development_gates_pass=True, candidate_seed=2027, new_full_track_calls=66216,
        scope=spec['interpretation'],
        candidate_entry_parity_preparation_allowed=all(flag for item in comparisons.values() for flag in item.values()), public_evaluation_allowed=False, independent_model_review_pass=False)
    write(OUT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'eligible', 'activate', 'idle', 'prefix', 'empty', 'swapped', 'analyze'])
    args = parser.parse_args()
    if args.action == 'prefix': track('category', prefix=True)
    elif args.action in ['empty', 'swapped']: track(args.action)
    else: {'prepare': prepare, 'check': checked, 'eligible': eligible, 'activate': activate, 'idle': idle, 'analyze': analyze}[args.action]()
