"""Train-only, fixed-final-head confidence readout diagnostic; no state intervention."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = BASE / 'sttrack_m68_reported_confidence_20260907'
METRIC = Path('/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py')
METRIC_SHA = '05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
TRAIN_SHA = '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
RECURSIVE_SHA = 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
AUDITOR = BASE / 'audit_m67_completed_20260907.py'
AUDITOR_SHA = '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
PYTHON = BASE / 'envs/sttrack/bin/python'
METRIC_PYTHON = Path('/root/miniconda3/envs/mplt/bin/python')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_plans():
    assert sha(PARENT / 'training_spec.json') == TRAIN_SHA
    assert sha(PARENT / 'recursive_spec.json') == RECURSIVE_SHA
    assert sha(AUDITOR) == AUDITOR_SHA and sha(METRIC) == METRIC_SHA
    return load_module('m68_parent_recursive', PARENT / 'run_recursive.py').plans()


def prepare():
    recursion, training = parent_plans()
    assert not (PARENT / 'recursive_result.json').exists()
    assert all(not (PARENT / 'training' / a / 'final.pth').exists() for a in ['control', 'support'])
    ROOT.mkdir()
    queue = ROOT / 'run_diagnostic.sh'
    queue.write_text('''#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/sttrack_m68_reported_confidence_20260907
trap 'rc=$?; printf "%s\\n" "$rc" > controller.exit' EXIT
/root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py eligible > eligibility.log 2>&1 || exit $?
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
while read -r value; do [ "$value" -lt 500 ] || exit 70; done <<< "$used"
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py control > control.log 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py support > support.log 2>&1 &
p1=$!
wait "$p0"; r0=$?; printf "%s\\n" "$r0" > control.exit
wait "$p1"; r1=$?; printf "%s\\n" "$r1" > support.exit
[ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] || exit 71
/root/miniconda3/envs/mplt/bin/python /root/autodl-tmp/m68_reported_confidence_20260907.py analyze > analysis.log 2>&1
rc=$?; printf "%s\\n" "$rc" > analysis.exit
exit "$rc"
''')
    spec = dict(status='frozen_before_M67_final_heads_and_development_results', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), queue_sha256=sha(queue), parent_training_spec_sha256=TRAIN_SHA,
        parent_recursive_spec_sha256=RECURSIVE_SHA, parent_auditor_sha256=AUDITOR_SHA,
        metric_source=str(METRIC), metric_source_sha256=METRIC_SHA, cases=recursion['cases'],
        dataset_root=training['dataset_root'], arms=['control', 'support'], head_selection='Both fixed M67 final heads after completion audit; no checkpoint selection.',
        readouts=['raw', 'semantic_product'], formula='reported_confidence = raw_Hann_peak * (1 - null_mass_at_actual_Hann_argmax)',
        arithmetic='Null mass uses the existing masked attribute logits plus fixed zero logit, float32 softmax; product uses Python floats; export 6 decimals.',
        initialization_confidence=1.0, pr_resolution=100, output_decimals=6,
        unchanged=['all predicted boxes', 'raw Hann peak and its argmax', 'crop', 'query', 'template contents and write predicate', 'text'],
        instrumentation='Read-only detached clones in adapter and Center Head forward hooks; no additional network forward or parameter update.',
        exact_original_public_box_and_raw_score_replay_required=True,
        prediction_sealing='Both complete replay families sealed and verified before any subsequent GT evaluation.',
        metrics='Existing bounded VOT-overlap long-term PR code, including initialization and invalid-GT convention; Train development22 only.',
        primary_comparison='Support semantic_product minus Support raw P/R/F; matched Control readout difference also reported.',
        interpretation='Descriptive fixed-formula confidence diagnostic, not a promotion gate; never changes original M67 gates or starts public evaluation.',
        prediction_calls=66216, sequences_per_arm=22, image_frames_per_arm=33130, optimizer_steps=0, fresh_captions=0,
        public_evaluation_allowed=False, independent_model_review_pass=False,
        limitations=['Frame support is box-supervised, not verified caption truth or a calibrated presence probability.',
            'Invalid GT follows the frozen long-term evaluator convention; it is not a visual diagnosis of absence or occlusion.',
            'Train PR is not DepthTrack Test or CDTB performance. Fixed boxes cannot improve VOT region metrics.',
            'No selection of a deployment threshold from the evaluation best-F threshold.',
            'One seed, repeatedly used development sequences; lexical attribution remains a separate requirement.'])
    write(ROOT / 'spec.json', spec)
    subprocess.run(['bash', '-n', str(queue)], check=True)
    result = dict(status='conditional_train_confidence_diagnostic_prepared', source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'),
        parent_training_or_inference_modified=False, final_head_loaded=False, new_GT_opened=False, new_tracking_calls=0,
        new_optimizer_steps=0, execution_started=False, queue_syntax_check=True)
    write(ROOT / 'preparation_result.json', result)
    print(json.dumps(result, indent=2))


def checked():
    spec = read(ROOT / 'spec.json')
    assert spec['source_sha256'] == sha(__file__)
    assert spec['queue_sha256'] == sha(ROOT / 'run_diagnostic.sh')
    recursion, training = parent_plans()
    assert spec['cases'] == recursion['cases']
    assert spec['dataset_root'] == training['dataset_root']
    assert len(spec['cases']) == 22 and sum(x['frames'] for x in spec['cases']) == 33130
    return spec, training


def eligible():
    spec, training = checked()
    audit = read(PARENT / 'completed_evidence_audit.json')
    assert audit['status'] == 'completed_M67_artifacts_and_development_audited'
    assert audit['auditor_sha256'] == AUDITOR_SHA and audit['result_sha256'] == sha(PARENT / 'recursive_result.json')
    assert audit['training_spec_sha256'] == TRAIN_SHA and audit['recursive_spec_sha256'] == RECURSIVE_SHA
    for arm in spec['arms']:
        head = PARENT / 'training' / arm / 'final.pth'
        assert sha(head) == audit['families'][arm]['head_sha256']
        assert sha(PARENT / (arm + '_recursive_receipt.json')) == audit['families'][arm]['receipt_sha256']
    return spec, training, audit


def track(arm):
    spec, training, audit = eligible()
    import torch
    sys.path.insert(0, str(PARENT / 'code'))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    torch.manual_seed(training['seed']); torch.cuda.manual_seed_all(training['seed'])
    update_config_from_file(str(PARENT / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=training['native_checkpoint'], base_checkpoint_sha256=training['native_checkpoint_sha256'],
        template_factor=2., template_size=128, search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    head = PARENT / 'training' / arm / 'final.pth'
    tracker = STTrackSemantic(params, str(head))
    assert tracker.network.semantic_adapter.null_support and tracker.use_text
    bank = torch.load(PARENT / 'text_development.pt', map_location='cpu')
    evidence, centers = [], []

    def read_adapter(module, inputs, output):
        evidence.append(output[1]['attribute_scores'].detach().clone())

    def read_head(module, inputs, output):
        centers.append(output[0].detach().clone())

    hook_a = tracker.network.semantic_adapter.register_forward_hook(read_adapter)
    hook_h = tracker.network.box_head.register_forward_hook(read_head)
    parent_receipt = read(PARENT / (arm + '_recursive_receipt.json'))
    folder = ROOT / arm; folder.mkdir()
    started = time.time(); records = []
    for case, parent_item in zip(spec['cases'], parent_receipt['sequences']):
        seq = case['sequence']; assert seq == parent_item['sequence']
        reference_path = PARENT / 'recursive' / arm / (seq + '.json')
        assert sha(reference_path) == parent_item['sha256']
        reference = read(reference_path)['rows']
        assert len(reference) == case['frames']
        sequence_root = Path(training['dataset_root']) / seq

        def frame(i):
            return get_rgbd_frame(str(sequence_root / 'color' / ('%08d.jpg' % (i + 1))),
                str(sequence_root / 'depth' / ('%08d.png' % (i + 1))), dtype='rgbcolormap', depth_clip=True)

        index = bank['sequences'].index(seq)
        tracker.initialize(frame(0), dict(init_bbox=case['init_bbox'], text_tokens=bank['tokens'][index],
            text_mask=bank['mask'][index], empty_text=bank['empty']))
        assert list(tracker.state) == reference[0]['bbox']
        rows = [dict(frame=0, bbox=list(tracker.state), score=None, null_mass=None, semantic_product=1.0)]
        for i in range(1, case['frames']):
            evidence.clear(); centers.clear()
            prediction = tracker.track(frame(i))
            assert len(evidence) == len(centers) == 1
            assert evidence[0].shape == (1, 256, 5) and centers[0].shape == (1, 1, 16, 16)
            response = tracker.output_window * centers[0]
            score, peak = torch.max(response.flatten(1), dim=1)
            raw = float(prediction['best_score']); assert float(score.item()) == raw
            row = dict(frame=i, bbox=list(prediction['target_bbox']), score=raw)
            assert row == reference[i], (arm, seq, i)
            e = evidence[0]
            mass = float(torch.cat((e, torch.zeros_like(e[..., :1])), dim=-1).softmax(dim=-1)[0, int(peak.item()), -1].item())
            assert 0.0 <= mass <= 1.0 and 0.0 <= raw <= 1.0
            row.update(null_mass=mass, semantic_product=raw * (1.0 - mass))
            rows.append(row)
        path = folder / (seq + '.json')
        write(path, dict(sequence=seq, arm=arm, rows=rows))
        item = dict(sequence=seq, frames=len(rows), sha256=sha(path), parent_prediction_sha256=parent_item['sha256'],
            exact_public_replay=True, template_writes=sum(r['frame'] % 50 == 0 and r['score'] > .75 for r in rows[1:]),
            elapsed_seconds=time.time() - started)
        records.append(item); print(json.dumps(item), flush=True)
    hook_a.remove(); hook_h.remove()
    checked()
    receipt = dict(status='complete_read_only_confidence_replay', arm=arm, spec_sha256=sha(ROOT / 'spec.json'),
        source_sha256=sha(__file__), head_sha256=sha(head), parent_audit_sha256=sha(PARENT / 'completed_evidence_audit.json'),
        parent_receipt_sha256=sha(PARENT / (arm + '_recursive_receipt.json')), sequences=records,
        total_frames=sum(x['frames'] for x in records), exact_public_boxes_and_raw_scores=True,
        subsequent_GT_opened=False, optimizer_steps=0, reported_confidence_used_for_tracking_or_templates=False,
        elapsed_seconds=time.time() - started)
    write(folder / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'sequences'}, indent=2), flush=True)


def analyze():
    spec, training, audit = eligible()
    import numpy as np
    data = {}; receipts = {}
    for arm in spec['arms']:
        assert (ROOT / (arm + '.exit')).read_text().strip() == '0'
        receipt = read(ROOT / arm / 'receipt.json')
        assert receipt['status'] == 'complete_read_only_confidence_replay' and receipt['spec_sha256'] == sha(ROOT / 'spec.json')
        assert receipt['head_sha256'] == audit['families'][arm]['head_sha256']
        assert receipt['parent_audit_sha256'] == sha(PARENT / 'completed_evidence_audit.json')
        assert receipt['total_frames'] == 33130 and len(receipt['sequences']) == 22
        data[arm] = {}
        for case, item in zip(spec['cases'], receipt['sequences']):
            seq = case['sequence']; assert item['sequence'] == seq
            path = ROOT / arm / (seq + '.json'); assert sha(path) == item['sha256']
            family = read(path); rows = family['rows']; assert family['arm'] == arm and family['sequence'] == seq
            assert len(rows) == case['frames'] and [x['frame'] for x in rows] == list(range(case['frames']))
            assert rows[0] == dict(frame=0, bbox=case['init_bbox'], score=None, null_mass=None, semantic_product=1.0)
            reference = PARENT / 'recursive' / arm / (seq + '.json'); assert sha(reference) == item['parent_prediction_sha256']
            assert [dict(frame=x['frame'], bbox=x['bbox'], score=x['score']) for x in rows] == read(reference)['rows']
            boxes = np.asarray([x['bbox'] for x in rows]); assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            for row in rows[1:]:
                assert 0.0 <= row['null_mass'] <= 1.0 and 0.0 <= row['score'] <= 1.0
                assert row['semantic_product'] == row['score'] * (1.0 - row['null_mass'])
            data[arm][seq] = rows
        receipts[arm] = sha(ROOT / arm / 'receipt.json')
    # Both full families are sealed and verified before export and subsequent GT evaluation.
    exports = ROOT / 'ope'; exports.mkdir()
    files = {}; evaluated = {}; distributions = {}
    evaluator = load_module('m68_frozen_long_term_pr', METRIC)
    for arm in spec['arms']:
        evaluated[arm] = {}; files[arm] = {}
        for readout in spec['readouts']:
            destination = exports / (arm + '_' + readout); destination.mkdir()
            files[arm][readout] = {}
            for case in spec['cases']:
                seq = case['sequence']; rows = data[arm][seq]
                boxes = np.asarray([x['bbox'] for x in rows], dtype=np.float64)
                scores = np.asarray([1.0] + [x['score'] if readout == 'raw' else x['semantic_product'] for x in rows[1:]])
                box_path = destination / (seq + '.txt'); score_path = destination / (seq + '_all_scores.txt')
                np.savetxt(box_path, boxes, delimiter=',', fmt='%.6f')
                np.savetxt(score_path, scores, fmt='%.6f')
                files[arm][readout][seq] = dict(bbox_sha256=sha(box_path), confidence_sha256=sha(score_path))
        assert all(files[arm]['raw'][c['sequence']]['bbox_sha256'] == files[arm]['semantic_product'][c['sequence']]['bbox_sha256'] for c in spec['cases'])
    for case in spec['cases']:
        assert sha(Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt') == case['gt_sha256']
    for arm in spec['arms']:
        samples = {'valid_correct': [], 'valid_low_overlap': [], 'valid_intermediate': [], 'invalid_GT': []}
        for readout in spec['readouts']:
            evaluated[arm][readout] = evaluator.evaluate_depthtrack_results(training['dataset_root'], exports / (arm + '_' + readout),
                resolution=spec['pr_resolution'], sequence_names=[x['sequence'] for x in spec['cases']])
        for case in spec['cases']:
            seq = case['sequence']; rows = data[arm][seq]; sequence_root = Path(training['dataset_root']) / seq
            gt = evaluator._load_rows(sequence_root / 'groundtruth.txt', 4)
            boxes = evaluator._load_rows(exports / (arm + '_raw') / (seq + '.txt'), 4)
            image = evaluator.cv2.imread(str(sequence_root / 'color/00000001.jpg')); height, width = image.shape[:2]
            overlap, visible = evaluator._vot_overlaps(boxes, gt, width, height)
            for i in range(1, case['frames']):
                group = 'invalid_GT' if not visible[i] else ('valid_correct' if overlap[i] >= .5 else ('valid_low_overlap' if overlap[i] <= .1 else 'valid_intermediate'))
                samples[group].append([rows[i]['null_mass'], rows[i]['score'], rows[i]['semantic_product']])
        distributions[arm] = {}
        for group, values in samples.items():
            values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
            distributions[arm][group] = dict(frames=len(values), columns=['null_mass', 'raw', 'semantic_product'],
                mean=values.mean(axis=0).tolist() if len(values) else None,
                median=np.median(values, axis=0).tolist() if len(values) else None)
    delta = {arm: {k: evaluated[arm]['semantic_product'][k] - evaluated[arm]['raw'][k]
        for k in ['precision_percent', 'recall_percent', 'f_score_percent']} for arm in spec['arms']}
    result = dict(status='completed_Train_confidence_readout_diagnostic', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'), metric_source_sha256=METRIC_SHA,
        parent_completed_audit_sha256=sha(PARENT / 'completed_evidence_audit.json'), receipts=receipts,
        metrics=evaluated, semantic_product_minus_raw_percentage_points=delta, null_mass_and_confidence_by_GT_group=distributions,
        exports=files, all_exported_boxes_identical_between_readouts=True,
        parent_M67_development_gate_pass=audit['paired_development_gate_pass'], original_M67_gates_modified=False,
        no_deployment_threshold_selected=True, public_evaluation_allowed=False, independent_model_review_pass=False,
        scope='Only reused DepthTrack Train development22. No new official Test/CDTB/VOT results; no direct lexical-effect claim.')
    write(ROOT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'exports'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'eligible', 'control', 'support', 'analyze'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        checked(); print('FROZEN_TRAIN_CONFIDENCE_SOURCE_AND_INPUTS_CHECKED_NO_EXECUTION')
    elif args.action == 'eligible':
        eligible(); print('M67_COMPLETED_AUDIT_AND_FINAL_HEADS_VERIFIED_DIAGNOSTIC_ONLY')
    elif args.action in ['control', 'support']:
        track(args.action)
    else:
        analyze()
