"""Independent, CPU-only verification of sealed M87 files. No tracker imports.

IoU/H10 are implemented here with scalar arithmetic, not imported from the
experiment metric. Torch zip files use a restricted metadata/storage decoder;
no installed Torch and no checkpoint-defined code is executed.
"""
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import pickle
import struct
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COLLECTION = ROOT / 'completion_collection'
E = COLLECTION / 'evidence'
M84 = ROOT.parent / 'sttrack_m84_centered_20260920' / 'completed'
ARMS = ['category', 'category_empty', 'category_old', 'category_swapped']
HASHES = {}
CHECKS = []


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(name, passed, detail=None):
    CHECKS.append(dict(name=name, passed=bool(passed), detail=detail))


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=0., abs_tol=1e-9)
    return a == b


def intervals(values):
    starts = []
    run = 0
    for i, value in enumerate(values + [False]):
        if value:
            run += 1
        else:
            if run >= 10:
                starts.append([i - run, i])
            run = 0
    return starts


def stats(rows, gt):
    values = []
    for i, (row, target) in enumerate(zip(rows, gt)):
        if i == 0 or not all(math.isfinite(v) for v in target) or target[2] <= 0 or target[3] <= 0:
            values.append(None)
            continue
        box = row['bbox']
        width = max(0., min(box[0] + box[2], target[0] + target[2]) - max(box[0], target[0]))
        height = max(0., min(box[1] + box[3], target[1] + target[3]) - max(box[1], target[1]))
        intersection = width * height
        values.append(intersection / (box[2] * box[3] + target[2] * target[3] - intersection))
    good = [v for v in values if v is not None]
    total = math.fsum(good)
    return dict(valid_frames=len(good), iou_sum=total, mean_iou=total / len(good),
                low_iou_frames=sum(v <= .1 for v in good),
                failure_episodes=len(intervals([v is not None and v <= .1 for v in values])),
                invalid_gt_frames=sum(v is None for v in values[1:])), values


def aggregate(per):
    out = {k: sum(v[k] for v in per.values()) for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
    out['mean_iou'] = out['iou_sum'] / out['valid_frames']
    out['macro_sequence_mean_iou'] = sum(v['mean_iou'] for v in per.values()) / len(per)
    return out


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Tensor:
    def __init__(self, storage, offset, shape, stride, requires_grad, hooks, metadata=None):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride

    def numel(self):
        return math.prod(self.shape)

    def scalar(self):
        assert self.numel() == 1
        return self.storage['values'][self.offset]

    def at(self, *indices):
        assert len(indices) == len(self.shape)
        return self.storage['values'][self.offset + sum(i * s for i, s in zip(indices, self.stride))]

    def vector(self, *prefix):
        return [self.at(*prefix, i) for i in range(self.shape[-1])]


class RestrictedTorchZip(pickle.Unpickler):
    def __init__(self, archive, prefix):
        self.archive, self.prefix, self.storages = archive, prefix, {}
        super().__init__(io.BytesIO(archive.read(prefix + 'data.pkl')))

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return Tensor
        if module == 'torch' and name in ['FloatStorage', 'BoolStorage']:
            return name
        raise ValueError('Unapproved pickle global: ' + module + '.' + name)

    def persistent_load(self, pid):
        kind, dtype, key, location, count = pid
        assert kind == 'storage'
        if key not in self.storages:
            content = self.archive.read(self.prefix + 'data/' + key)
            code = {'FloatStorage': 'f', 'BoolStorage': '?'}[dtype]
            assert len(content) == count * struct.calcsize(code)
            values = [v[0] for v in struct.iter_unpack('<' + code, content)]
            self.storages[key] = dict(dtype=dtype, count=count, values=values,
                finite=all(math.isfinite(v) for v in values), sha256=hashlib.sha256(content).hexdigest())
        return self.storages[key]


def load_tensor_zip(path):
    sha(path)
    with zipfile.ZipFile(path) as archive:
        prefix = next(n for n in archive.namelist() if n.endswith('data.pkl'))[:-8]
        decoder = RestrictedTorchZip(archive, prefix)
        obj = decoder.load()
        check('tensor_storage_finite:' + str(path), all(s['finite'] for s in decoder.storages.values()))
        return obj


def main():
    manifest = read(E / 'manifest.json')
    check('manifest_count_unique', len(manifest) == len({x['path'] for x in manifest}) == 352)
    for item in manifest:
        path = E / item['path']
        check('manifest:' + item['path'], path.is_file() and sha(path) == item['sha256'] and path.stat().st_size == item['bytes'])
    collection = read(COLLECTION / 'collection_result.json')
    transfer = read(COLLECTION / 'transfer_verification.json')
    archive = COLLECTION / 'completed_evidence.tar.gz'
    check('archive_sha_and_bytes', sha(archive) == collection['archive_sha256'] == transfer['archive_sha256'] and archive.stat().st_size == collection['archive_bytes'])
    spec = read(E / 'training_spec.json')
    recursive = read(E / 'recursive_spec.json')
    frozen = read(E / 'frozen.json')
    trained = read(E / 'training/category/result.json')
    result = read(E / 'recursive_result.json')
    integration = read(E / 'integration.json')
    check('frozen_training_recursive_bindings', sha(E / 'training_spec.json') == frozen['training_spec_sha256'] == recursive['training_spec_sha256'] == trained['training_spec_sha256'] == result['training_spec_sha256'] and sha(E / 'recursive_spec.json') == frozen['recursive_spec_sha256'] == result['recursive_spec_sha256'])
    for name, key in [('preflight_result.json', 'preflight_sha256'), ('code_review_receipt.json', 'review_receipt_sha256')]:
        check('frozen:' + name, sha(E / name) == frozen[key])
    for name, key in [('EXPERIMENT_PLAN.md', 'plan_sha256'), ('integration.json', 'integration_sha256'), ('train_causal.py', 'training_script_sha256'), ('causal_training.py', 'causal_script_sha256'), ('support_loss.py', 'support_loss_sha256'), ('window_competition.py', 'window_loss_sha256'), ('native_preservation.py', 'preservation_loss_sha256'), ('run_m87.sh', 'run_queue_sha256'), ('caption_spec.json', 'caption_spec_sha256'), ('caption_result.json', 'caption_result_sha256'), ('bank_result.json', 'bank_result_sha256')]:
        check('training_binding:' + name, sha(E / name) == spec[key])
    check('runner_metric_queue_bindings', sha(E / 'run_recursive.py') == recursive['runner_sha256'] and sha(E / 'recursive_metric.py') == recursive['metric_sha256'] and sha(E / 'run_m87.sh') == recursive['queue_sha256'])
    for name, digest in integration['source_sha256'].items():
        check('runtime_source:' + name, sha(E / 'code' / name) == digest)
    for name in ['controller', 'training_category', 'recursive_analysis'] + [a + '_recursive' for a in ARMS]:
        path = E / (name + '.exit')
        check('exit:' + name, path.read_text().strip() == '0')
    native_result = read(E / 'references/native_result.json')
    parent_result = read(E / 'references/M84_recursive_result.json')
    check('native_parent_result_binding', sha(E / 'references/native_result.json') == spec['native_result_sha256'] and sha(E / 'references/M84_recursive_result.json') == spec['parent_result_sha256'])
    parent_spec = read(M84 / 'training_spec.json')
    check('parent_training_spec_binding', sha(M84 / 'training_spec.json') == spec['parent_training_spec_sha256'])
    preserve = ['seed', 'architecture', 'sequence_order', 'epochs', 'learned_parameters', 'visual_base_frozen_eval', 'native_checkpoint_sha256', 'total_training_image_frames', 'total_training_track_calls', 'expected_optimizer_steps', 'optimizer', 'learning_rate', 'weight_decay', 'gradient_accumulation_frames', 'gradient_clip', 'loss', 'state_protocol', 'backprop_through_time_or_discrete_crops', 'native_update_interval', 'native_update_threshold', 'initial_checkpoint_sha256', 'causal_script_sha256', 'support_loss_sha256', 'window_loss_sha256', 'preservation_loss_sha256', 'integration_sha256']
    for key in preserve:
        check('M84_preserved:' + key, spec[key] == parent_spec[key])
    write_json('spec_comparison.json', {k: dict(M87=spec.get(k), M84=parent_spec.get(k)) for k in spec.keys() | parent_spec.keys() if spec.get(k) != parent_spec.get(k)})
    check('single_seed_one_head_fit_dev_disjoint', spec['seed'] == frozen['seed'] == 2027 and spec['trained_arms'] == frozen['trained_arms'] == ['category'] and len(spec['sequence_order']) == 130 and len(spec['development_sequences']) == 22 and not {s['sequence'] for s in spec['sequence_order']} & set(spec['development_sequences']))
    checkpoint = load_tensor_zip(E / 'training/category/final.pth')
    checkpoint_info = {k: v for k, v in checkpoint.items() if k not in ['model', 'optimizer']}
    checkpoint_info['tensor_shapes'] = {k: list(v.shape) for k, v in checkpoint['model'].items()}
    checkpoint_info['parameter_count_excluding_empty_buffer'] = sum(v.numel() for k, v in checkpoint['model'].items() if k != 'empty_text')
    checkpoint_info['all_optimizer_step_values'] = [v['step'].scalar() for v in checkpoint['optimizer']['state'].values()]
    checkpoint_info['optimizer_parameter_groups'] = checkpoint['optimizer']['param_groups']
    check('final_checkpoint_hash', sha(E / 'training/category/final.pth') == trained['final_checkpoint_sha256'] == result['head_sha256'] == collection['final_checkpoint_sha256'])
    check('final_checkpoint_completion_and_budget', checkpoint_info['completed_sequences'] == 130 and checkpoint_info['optimizer_steps'] == checkpoint_info['actual_dataset_optimizer_steps'] == 5798 and checkpoint_info['frame_count'] == 186694 and checkpoint_info['training_spec_sha256'] == sha(E / 'training_spec.json') and checkpoint_info['status'] == 'complete' and checkpoint_info['seed'] == 2027 and checkpoint_info['architecture'] == spec['architecture'] and checkpoint_info['base_checkpoint_sha256'] == spec['native_checkpoint_sha256'])
    check('final_parameter_count_and_optimizer_steps', checkpoint_info['parameter_count_excluding_empty_buffer'] == 289154 and len(checkpoint_info['all_optimizer_step_values']) == 23 and all(s == 5798 for s in checkpoint_info['all_optimizer_step_values']))
    write_json('checkpoint_verification.json', checkpoint_info)
    logs = [json.loads(s) for s in (E / 'training/category/sequence_log.jsonl').read_text().splitlines()]
    traces = [json.loads(s) for s in (E / 'training/category/sampled_state_trace.jsonl').read_text().splitlines()]
    check('training_log_hashes', sha(E / 'training/category/sequence_log.jsonl') == trained['sequence_log_sha256'] and sha(E / 'training/category/sampled_state_trace.jsonl') == trained['sampled_trace_sha256'])
    check('training_log_full_order', len(logs) == 130 and [x['sequence'] for x in logs] == [x['sequence'] for x in spec['sequence_order']])
    labels = Counter()
    tracks = previous_steps = 0
    for index, (log, sequence) in enumerate(zip(logs, spec['sequence_order'])):
        labels.update(log['label_counts'])
        tracks += log['track_calls']
        check('training_row:' + log['sequence'], log['sequence_index'] == index and log['frames'] == sequence['rgb_frames'] and log['track_calls'] == log['frames'] - 1 and sum(log['label_counts'].values()) == log['track_calls'] and log['supervised_frames'] == log['label_counts'].get('centre_inside', 0) + log['label_counts'].get('centre_outside', 0) and log['total_track_calls'] == tracks and previous_steps <= log['total_optimizer_steps'] <= previous_steps + math.ceil(log['track_calls'] / 32) and all(math.isfinite(log[k]) for k in ['mean_training_loss', 'maximum_preclip_gradient_norm', 'seconds', 'elapsed_seconds']))
        previous_steps = log['total_optimizer_steps']
    check('training_totals', tracks == trained['total_track_calls'] == spec['total_training_track_calls'] == 186694 and previous_steps == trained['optimizer_steps'] == 5798 and dict(labels) == trained['training_label_counts'])
    for key, logkey in [('competition_frames', 'cumulative_competition_frames'), ('competition_loss_sum', 'cumulative_competition_loss_sum'), ('competition_negatives', 'cumulative_competition_negatives'), ('native_eligible_frames', 'cumulative_native_eligible_frames'), ('preservation_kl_sum', 'cumulative_preservation_kl_sum'), ('maximum_preservation_kl', 'maximum_preservation_kl')]:
        check('training_terminal:' + key, trained[key] == logs[-1][logkey])
    expected_trace = [(s['sequence'], i) for s in spec['sequence_order'] for i in range(1, s['rgb_frames']) if i == 1 or i % 50 == 0 or i == s['rgb_frames'] - 1]
    check('sampled_trace_exact_schedule', [(t['sequence'], t['frame_index']) for t in traces] == expected_trace and len(traces) == 3917)
    first_boxes = {s['sequence']: s['first_box'] for s in spec['sequence_order']}
    check('sampled_trace_initial_previous_box', all(t['previous_bbox'] == first_boxes[t['sequence']] for t in traces if t['frame_index'] == 1))
    check('sampled_trace_template_rule', all(t['template_write'] == (t['frame_index'] % 50 == 0 and t['best_score'] > .75) for t in traces))
    check('sampled_trace_finite', all(all(math.isfinite(x) for x in t['previous_bbox'] + t['bbox'] + [t['best_score'], t['resize_factor']]) for t in traces))
    check('sampled_trace_native_eligibility', all(t['native_eligible'] == (t['label'] == 'centre_inside' and t['native_iou'] >= .5) for t in traces))
    write_json('training_log_verification.json', dict(sequences=len(logs), track_calls=tracks, optimizer_steps=previous_steps, labels=dict(labels), sampled_rows=len(traces), expected_sampled_rows=len(expected_trace), full_state_stream_recomputed=False, full_base_rerun_performed=False))
    banks = {}
    for split, entries in spec['banks'].items():
        for content, entry in entries.items():
            path = E / 'banks' / (split + '_' + content + '.pt')
            check('bank_hash:' + split + '_' + content, sha(path) == entry['sha256'])
            banks[split + '_' + content] = load_tensor_zip(path)
    check('checkpoint_empty_buffer', checkpoint['model']['empty_text'].vector() == banks['development_empty']['empty'].vector())
    dev = banks['development_category']
    for content in ['empty', 'old', 'swapped']:
        b = banks['development_' + content]
        check('bank_structure:' + content, b['sequences'] == dev['sequences'] and b['mask'].shape == dev['mask'].shape and b['tokens'].shape == dev['tokens'].shape and b['empty'].vector() == dev['empty'].vector() and all(b['mask'].at(i, j) == dev['mask'].at(i, j) for i in range(22) for j in range(5)))
    check('bank_only_slot0_changed_vs_old', all(dev['tokens'].vector(i, j) == banks['development_old']['tokens'].vector(i, j) for i in range(22) for j in range(1, 5)))
    empty = banks['development_empty']
    check('empty_active_slots_equal_empty_buffer', all(empty['tokens'].vector(i, j) == empty['empty'].vector() for i in range(22) for j in range(5) if empty['mask'].at(i, j)))
    data = {a: {} for a in ARMS}
    for arm in ARMS:
        receipt = read(E / (arm + '_recursive_receipt.json'))
        check('receipt_binding:' + arm, sha(E / (arm + '_recursive_receipt.json')) == result['receipts'][arm] and receipt['status'] == 'complete' and receipt['arm'] == arm and receipt['head_sha256'] == trained['final_checkpoint_sha256'] and receipt['training_result_sha256'] == sha(E / 'training/category/result.json') and receipt['recursive_spec_sha256'] == sha(E / 'recursive_spec.json') and receipt['total_frames'] == 33130 and len(receipt['sequences']) == 22)
        for case, row in zip(recursive['cases'], receipt['sequences']):
            path = E / 'recursive' / arm / (case['sequence'] + '.json')
            obj = read(path)
            rows = obj['rows']
            check('prediction:' + arm + ':' + case['sequence'], sha(path) == row['sha256'] and row['sequence'] == obj['sequence'] == case['sequence'] and obj['arm'] == arm and len(rows) == row['frames'] == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames'])) and rows[0]['bbox'] == case['init_bbox'] and rows[0]['score'] is None and all(all(math.isfinite(v) for v in r['bbox']) and r['bbox'][2] > 0 and r['bbox'][3] > 0 for r in rows) and all(math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in rows[1:]))
            data[arm][case['sequence']] = rows
        log = [json.loads(line) for line in (E / (arm + '_recursive.log')).read_text().splitlines() if line.startswith('{')]
        check('inference_stdout_matches_receipt:' + arm, log[:-1] == receipt['sequences'] and log[-1] == {k: v for k, v in receipt.items() if k != 'sequences'})
    # This verifier also waits until all four sealed prediction families are read.
    gts = {}
    for case in recursive['cases']:
        path = E / 'development_gt' / (case['sequence'] + '.txt')
        check('GT_hash:' + case['sequence'], sha(path) == case['gt_sha256'])
        gts[case['sequence']] = [[float(v) for v in row] for row in csv.reader(path.read_text().splitlines())]
        check('GT_length_init:' + case['sequence'], len(gts[case['sequence']]) == case['frames'] and gts[case['sequence']][0] == case['init_bbox'])
    per, curves = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for arm in ARMS:
        for sequence, rows in data[arm].items():
            per[arm][sequence], curves[arm][sequence] = stats(rows, gts[sequence])
            check('raw_metrics:' + arm + ':' + sequence, same(per[arm][sequence], result['per_sequence'][arm][sequence]))
    m84empty_receipt = read(E / 'references/M84_empty_receipt.json')
    check('M84_empty_receipt_binding', sha(E / 'references/M84_empty_receipt.json') == parent_result['receipts']['category_empty'])
    native_manifest = read(M84 / 'native_reference/manifest.json')
    for name, item in native_manifest.items():
        path = M84 / 'native_reference' / name
        check('native_extraction_manifest:' + name, sha(path) == item['sha256'] and path.stat().st_size == item['bytes'])
    provenance = read(M84 / 'native_reference/reference_provenance.json')
    check('native_source_spec_hash', sha(M84 / 'native_reference/source_recursive_spec.json') == provenance['source_spec_sha256'])
    old_native_spec = read(M84 / 'native_reference/source_recursive_spec.json')
    check('GT_identical_to_prior_frozen_dataset_identity', all(old_native_spec['development_gt_sha256'][c['sequence']] == c['gt_sha256'] for c in recursive['cases']))
    oldreceipt = read(M84 / 'category_recursive_receipt.json')
    check('M84_Category_receipt_binding', sha(M84 / 'category_recursive_receipt.json') == parent_result['receipts']['category'])
    per['native'], per['M84_protocol_control'] = {}, {}
    parity = []
    for case, erec, crec in zip(recursive['cases'], m84empty_receipt['sequences'], oldreceipt['sequences']):
        sequence = case['sequence']
        previous_path = E / 'references/M84_empty' / (sequence + '.json')
        previous = read(previous_path)['rows']
        check('M84_empty_prediction_binding:' + sequence, sha(previous_path) == erec['sha256'])
        nrows = read(M84 / 'native_reference' / (sequence + '.json'))['rows']
        cpath = M84 / 'recursive/category' / (sequence + '.json')
        crows = read(cpath)['rows']
        check('M84_Category_prediction_binding:' + sequence, sha(cpath) == crec['sha256'])
        per['native'][sequence], _ = stats(nrows, gts[sequence])
        per['M84_protocol_control'][sequence], _ = stats(crows, gts[sequence])
        for arm in ['native', 'M84_protocol_control']:
            expected = result['per_sequence'][arm][sequence]
            check('reference_raw_metrics:' + arm + ':' + sequence, same({k: per[arm][sequence][k] for k in expected}, expected))
        parity.append(dict(sequence=sequence, frames=case['frames'],
            M87_M84_bbox_equal=sum(a['bbox'] == b['bbox'] for a, b in zip(data['category_empty'][sequence], previous)),
            M87_M84_score_equal=sum(a['score'] == b['score'] for a, b in zip(data['category_empty'][sequence], previous)),
            M87_native_bbox_equal=sum(a['bbox'] == b['bbox'] for a, b in zip(data['category_empty'][sequence], nrows)),
            M87_native_score_equal=sum(a['score'] == b['score'] for a, b in zip(data['category_empty'][sequence], nrows))))
    check('Empty_exact_M84_and_native_extracted_reference', all(all(v == row['frames'] for k, v in row.items() if k.endswith('_equal')) for row in parity))
    per['M82_historical'] = parent_result['per_sequence']['M82_mask_control']
    ag = {a: aggregate(v) for a, v in per.items()}
    for arm, numbers in ag.items():
        check('aggregate:' + arm, same(numbers, result['aggregates'][arm]))
    current = ag['category']
    def comparison(other):
        other = ag[other]
        return dict(mean=current['mean_iou'] > other['mean_iou'], macro=current['macro_sequence_mean_iou'] >= other['macro_sequence_mean_iou'], low=current['low_iou_frames'] <= other['low_iou_frames'], H10=current['failure_episodes'] <= other['failure_episodes'])
    broken = [s for s in per['native'] if per['native'][s]['failure_episodes'] == 0 and per['category'][s]['failure_episodes'] > 0]
    parity_metric = {s: all(per['category_empty'][s][k] == per['native'][s][k] for k in ['valid_frames', 'low_iou_frames', 'failure_episodes']) and abs(per['category_empty'][s]['iou_sum'] - per['native'][s]['iou_sum']) <= 1e-8 for s in per['native']}
    gates = dict(M84_increment=comparison('M84_protocol_control'), native={**comparison('native'), 'native_zero_H10_protection': not broken}, old_content=comparison('category_old'), swapped_content=comparison('category_swapped'), empty_native_parity={'all_sequences': all(parity_metric.values())})
    check('all_18_gates', gates == result['gates'] and sum(len(v) for v in gates.values()) == result['gate_count'] == 18 and all(all(v.values()) for v in gates.values()) == result['all_gates_pass'])
    check('zero_H10_damage', broken == result['native_success_damage'])
    loo = {}
    for other in ARMS[1:]:
        loo[other] = {s: (current['iou_sum'] - per['category'][s]['iou_sum']) / (current['valid_frames'] - per['category'][s]['valid_frames']) - (ag[other]['iou_sum'] - per[other][s]['iou_sum']) / (ag[other]['valid_frames'] - per[other][s]['valid_frames']) for s in per['category']}
    check('all_66_LOO', same(loo, result['content_leave_one_out']))
    write_json('recomputed_metrics.json', dict(aggregates=ag, per_sequence=per, gates=gates, gates_passed=sum(sum(v.values()) for v in gates.values()), native_success_damage=broken, empty_native_parity=parity_metric, content_leave_one_out=loo, M82_source='Reaggregated from hash-bound historical per-sequence values; no raw M82 trajectories in the M87 evidence bundle.'))
    write_csv('per_sequence_recomputed.csv', [dict(arm=arm, sequence=sequence, **numbers) for arm, values in per.items() for sequence, numbers in values.items()])
    write_json('empty_exact_reference_verification.json', parity)
    # Independent posthoc check, including all intervals and reconstructed writes.
    events, writes = [], []
    for sequence in gts:
        for arm in ARMS:
            for i, row in enumerate(data[arm][sequence]):
                if i > 0 and i % 50 == 0 and row['score'] > .75:
                    value = curves[arm][sequence][i]
                    quality = 'invalid_gt' if value is None else 'correct' if value >= .5 else 'severe' if value <= .1 else 'intermediate'
                    writes.append(dict(sequence=sequence, arm=arm, frame=i, score=row['score'], iou=value, quality=quality))
        for other in ARMS[1:]:
            for direction in ['damage', 'improvement']:
                bad, good = ('category', other) if direction == 'damage' else (other, 'category')
                mask = [a is not None and b is not None and a <= .1 and b >= .5 for a, b in zip(curves[bad][sequence], curves[good][sequence])]
                for start, end in intervals(mask):
                    events.append(dict(sequence=sequence, reference=other, direction=direction, start=start, end_exclusive=end, frames=end-start))
    analysis = read(COLLECTION / 'analysis/analysis.json')
    strict_summary = {other: {direction: dict(segments=sum(e['reference'] == other and e['direction'] == direction for e in events), frames=sum(e['frames'] for e in events if e['reference'] == other and e['direction'] == direction)) for direction in ['damage', 'improvement']} for other in ARMS[1:]}
    write_summary = {arm: {quality: sum(w['arm'] == arm and w['quality'] == quality for w in writes) for quality in ['correct', 'severe', 'intermediate', 'invalid_gt']} for arm in ARMS}
    check('posthoc_strict_intervals_and_write_counts', strict_summary == analysis['strict_intervals'] and write_summary == analysis['template_writes'])
    for name, digest in analysis['artifacts_sha256'].items():
        check('posthoc_artifact_hash:' + name, sha(COLLECTION / 'analysis' / name) == digest)
    register_path = ROOT / 'semantic_screening/paired_register.csv'
    check('posthoc_register_binding', sha(register_path) == analysis['input_register_sha256'])
    register = {r['sequence']: r for r in csv.DictReader(register_path.open(encoding='utf-8-sig'))}
    groups = {}
    for status in ['supported', 'conflicting', 'uncertain']:
        sequences = [s for s in gts if register[s]['new_screening'] == status]
        n = sum(per['category'][s]['valid_frames'] for s in sequences)
        groups[status] = dict(sequences=len(sequences), valid_frames=n, delta_category_old=sum(per['category'][s]['iou_sum']-per['category_old'][s]['iou_sum'] for s in sequences)/n, delta_category_empty=sum(per['category'][s]['iou_sum']-per['category_empty'][s]['iou_sum'] for s in sequences)/n)
    check('posthoc_proxy_subgroup_arithmetic', same(groups, analysis['semantic_subgroups']))
    report_path = COLLECTION / 'COMPLETED_REPORT.md'
    report_hash = sha(report_path)
    sha(COLLECTION / 'analyze_completed.py')
    sha(Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\docs\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md'))
    check('posthoc_analysis_source_binding', sha(COLLECTION / 'analyze_completed.py') == analysis['source_sha256'] and sha(E / 'recursive_result.json') == analysis['input_result_sha256'])
    write_json('posthoc_verification.json', dict(strict_intervals=strict_summary, template_writes=write_summary, semantic_subgroups=groups, semantic_scope='Assistant proxy labels, not independent human or semantic ground truth.', update_scope='Reconstructed write events from actual deterministic condition; no causal attribution across diverged histories.'))
    write_csv('strict_intervals_recomputed.csv', events)
    write_csv('template_writes_recomputed.csv', writes)
    write_json('audited_input_hashes.json', HASHES)
    output = dict(status='PASS' if all(c['passed'] for c in CHECKS) else 'FAIL', generated_at=datetime.now(timezone.utc).isoformat(), checks=len(CHECKS), passed=sum(c['passed'] for c in CHECKS), failed=[c for c in CHECKS if not c['passed']], verification=CHECKS, performance_gate_pass=False, new_training_steps=0, new_tracking_calls=0, reviewed_narrative_sha256=report_hash, reviewer_family='openai', verification_type='deterministic CPU arithmetic and restricted storage inspection')
    write_json('deterministic_verification.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'verification'}, indent=2))
    print(json.dumps(dict(aggregates=ag, gates=gates, broken=broken, loo_ranges={a: dict(min=min(v.values()), max=max(v.values()), positive=sum(x > 0 for x in v.values())) for a, v in loo.items()}, strict_intervals=strict_summary, template_writes=write_summary), indent=2))


if __name__ == '__main__':
    main()
