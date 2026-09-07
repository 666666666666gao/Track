"""Validate the frozen learned OPE/TraX entries on already sealed Train prefixes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907/candidate_evaluation/entry_parity')
M67 = Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907')
CANDIDATE = Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907/candidate_evaluation')
INTERFACE = Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907/evaluation_interface')
GENERATOR = Path('/root/autodl-tmp/sttrack_m58_initialization_generator_20260906')
MODEL_PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wire_function():
    path = GENERATOR / 'vot_rectangle_transport.py'
    assert sha(path) == '20878b6ada788d609605c6328570f19e66d6b3a93e735ca86b1c6ec0646402cc'
    spec = importlib.util.spec_from_file_location('full_wire_rectangle', str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.rectangle_wire_bbox


def candidate():
    path = Path('/root/autodl-tmp/m67_candidate_evaluation_20260907.py')
    spec = importlib.util.spec_from_file_location('m67_evaluation_candidate', str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.checked()


def prepare():
    frozen = candidate()
    parent = json.loads((M67 / 'recursive_spec.json').read_text())
    training = json.loads((M67 / 'training_spec.json').read_text())
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, text_bank
    plan, bundle = checked_plan(CANDIDATE / 'development_plan.json')
    bank = text_bank(plan, bundle)
    wire = wire_function()
    names = ['bag05_indoor', 'container01_indoor', 'mobilephone02_indoor']
    cases = [dict(case, frames=202) for case in parent['cases'] if case['sequence'] in names]
    assert len(cases) == 3
    reference_receipt = json.loads((M67 / 'support_recursive_receipt.json').read_text())
    reference_files = {r['sequence']: r for r in reference_receipt['sequences']}
    for case in cases:
        assert wire(case['init_bbox']) == case['init_bbox']
        folder = Path(training['dataset_root']) / case['sequence']
        bank.info(folder / 'color/00000001.jpg', case['init_bbox'])
        for frame in range(202):
            stem = '%08d' % (frame + 1)
            assert (folder / 'color' / (stem + '.jpg')).is_file()
            assert (folder / 'depth' / (stem + '.png')).is_file()
        assert sha(M67 / 'recursive/support' / (case['sequence'] + '.json')) == reference_files[case['sequence']]['sha256']
    ROOT.mkdir()
    write(ROOT / 'cases.json', cases)
    ope_plan = dict(plan, dataset_root=training['dataset_root'], cases_path=str(ROOT / 'cases.json'),
                   cases_sha256=sha(ROOT / 'cases.json'), output=str(ROOT / 'ope'))
    write(ROOT / 'ope_plan.json', ope_plan)
    write(ROOT / 'trax_plan.json', plan)
    spec = dict(status='frozen_before_real_model_entry_validation', observed_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=sha(__file__), candidate_binding_sha256=frozen['binding_result_sha256'],
                bundle_path=plan['bundle_path'], bundle_sha256=plan['bundle_sha256'], head_sha256=bundle['adapter_checkpoint_sha256'],
                cases_sha256=sha(ROOT / 'cases.json'), ope_plan_sha256=sha(ROOT / 'ope_plan.json'),
                trax_plan_sha256=sha(ROOT / 'trax_plan.json'), text_bank_sha256=plan['text_bank_sha256'],
                reference_receipt_sha256=sha(M67 / 'support_recursive_receipt.json'),
                reference_prediction_sha256={case['sequence']: reference_files[case['sequence']]['sha256'] for case in cases},
                dataset_root=training['dataset_root'], sequences=names, prefix_frames=202,
                ope_expected_frames=606, trax_expected_sessions=3, trax_expected_reports=603,
                OPE_box_and_score_tolerance=5.01e-7, trax_box_comparison='Exact full float32 -> four-decimal text -> float32 conversion of the sealed raw prediction.',
                trax_confidence_tolerance=1e-6, trax_process_timeout_seconds=60,
                full_wire_initializations_equal_original=True, new_captions=0, new_optimizer_steps=0,
                purpose='Train-only real learned entry/transport verification; no public benchmark or automatic promotion.',
                public_evaluation_allowed=False, independent_review_pass=False)
    write(ROOT / 'spec.json', spec)
    write(ROOT / 'preparation_result.json', dict(status='ready_for_GPU_entry_checks', source_sha256=sha(__file__),
          spec_sha256=sha(ROOT / 'spec.json'), head_sha256=spec['head_sha256'], cases=3, frames_per_entry=606,
          new_model_calls=0, subsequent_GT_opened=False, public_evaluation_allowed=False))
    print(json.dumps(json.loads((ROOT / 'preparation_result.json').read_text()), indent=2))


def checked():
    frozen = candidate()
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert sha(__file__) == spec['source_sha256']
    assert spec['candidate_binding_sha256'] == frozen['binding_result_sha256']
    assert sha(M67 / 'support_recursive_receipt.json') == spec['reference_receipt_sha256']
    for name, key in [('cases.json', 'cases_sha256'), ('ope_plan.json', 'ope_plan_sha256'), ('trax_plan.json', 'trax_plan_sha256')]:
        assert sha(ROOT / name) == spec[key]
    assert sha(Path(spec['bundle_path'])) == spec['bundle_sha256']
    bundle = json.loads(Path(spec['bundle_path']).read_text())
    for path, key in [(bundle['base_checkpoint'], 'base_checkpoint_sha256'), (bundle['adapter_checkpoint'], 'adapter_checkpoint_sha256')]:
        assert sha(Path(path)) == bundle[key]
    for name, digest in bundle['source_sha256'].items():
        assert sha(Path(bundle['repository']) / name) == digest
    for name, digest in bundle['interface_sha256'].items():
        assert sha(INTERFACE / name) == digest
    assert sha(Path(bundle['text_protocol_path'])) == bundle['text_protocol_sha256']
    plan = json.loads((ROOT / 'trax_plan.json').read_text())
    assert sha(Path(plan['text_bank_path'])) == spec['text_bank_sha256']
    for name, digest in spec['reference_prediction_sha256'].items():
        assert sha(M67 / 'recursive/support' / (name + '.json')) == digest
    return spec, json.loads((ROOT / 'cases.json').read_text())


class Frame:
    def __init__(self, folder, index):
        stem = '%08d' % (index + 1)
        self.paths = dict(color=str(folder / 'color' / (stem + '.jpg')), depth=str(folder / 'depth' / (stem + '.png')))

    def filename(self, channel):
        return self.paths[channel]


def region_box(region):
    return [float(region.x), float(region.y), float(region.width), float(region.height)]


def client():
    import importlib.metadata as metadata
    import inspect
    import shlex
    import trax.client
    from vot.region import Rectangle
    from vot.tracker import ObjectStatus
    import vot.tracker.trax as toolkit
    spec, cases = checked()
    assert metadata.version('vot-toolkit') == '0.7.1'
    assert metadata.version('vot-trax') == '4.0.2'
    wire = wire_function()
    destination = ROOT / 'trax'; destination.mkdir()
    reports = []; started = time.time()
    for case in cases:
        name = case['sequence']; folder = Path(spec['dataset_root']) / name
        expected = json.loads((M67 / 'recursive/support' / (name + '.json')).read_text())['rows']
        command = shlex.join([MODEL_PYTHON, str(INTERFACE / 'run_semantic_vot.py'), '--plan', str(ROOT / 'trax_plan.json')])
        log = destination / (name + '.trax.log')
        process = toolkit.TrackerProcess(command, envvars={'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '4'},
                                         timeout=spec['trax_process_timeout_seconds'], log=str(log))
        child = process._process
        rows = []; max_score = 0.
        try:
            assert process.has_vot_wrapper
            statuses, _ = process.initialize(Frame(folder, 0), ObjectStatus(Rectangle(*case['init_bbox']), {}))
            assert len(statuses) == 1 and region_box(statuses[0].region) == wire(case['init_bbox'])
            assert 'confidence' not in statuses[0].properties
            for frame in range(1, 202):
                statuses, _ = process.update(Frame(folder, frame))
                assert len(statuses) == 1
                box = region_box(statuses[0].region); score = float(statuses[0].properties['confidence'])
                assert box == wire(expected[frame]['bbox']), (name, frame, box, wire(expected[frame]['bbox']))
                max_score = max(max_score, abs(score - expected[frame]['score']))
                assert max_score <= spec['trax_confidence_tolerance']
                rows.append(dict(frame=frame, bbox=box, confidence=score))
        finally:
            process.terminate()
        code = child.wait(timeout=5); assert code == 0
        path = destination / (name + '.json')
        write(path, dict(sequence=name, rows=rows))
        item = dict(sequence=name, pid=child.pid, exit_code=code, tracking_reports=len(rows),
                    max_confidence_error=max_score, exact_wire_boxes=True, response_sha256=sha(path), protocol_log_sha256=sha(log))
        reports.append(item); print(json.dumps(item), flush=True)
    checked()
    receipt = dict(status='real_learned_TraX_entry_verified', spec_sha256=sha(ROOT / 'spec.json'),
                   bundle_sha256=spec['bundle_sha256'], sessions=reports, reports=sum(r['tracking_reports'] for r in reports),
                   toolkit_version=metadata.version('vot-toolkit'), trax_version=metadata.version('vot-trax'),
                   toolkit_source_sha256=sha(Path(inspect.getfile(toolkit))), trax_client_source_sha256=sha(Path(inspect.getfile(trax.client))),
                   elapsed_seconds=time.time()-started, new_optimizer_steps=0, subsequent_GT_opened=False,
                   public_evaluation_allowed=False, independent_review_pass=False)
    write(destination / 'receipt.json', receipt)


def verify():
    spec, cases = checked()
    for name in ['ope.exit', 'trax.exit']:
        assert (ROOT / name).read_text().strip() == '0'
    ope = json.loads((ROOT / 'ope/receipt.json').read_text())
    trax = json.loads((ROOT / 'trax/receipt.json').read_text())
    assert ope['status'] == 'complete' and ope['plan_sha256'] == spec['ope_plan_sha256']
    assert ope['bundle_sha256'] == trax['bundle_sha256'] == spec['bundle_sha256']
    assert ope['frames'] == 606 and trax['reports'] == 603 and len(trax['sessions']) == 3
    assert trax['spec_sha256'] == sha(ROOT / 'spec.json')
    for item in trax['sessions']:
        assert sha(ROOT / 'trax' / (item['sequence'] + '.json')) == item['response_sha256']
        assert sha(ROOT / 'trax' / (item['sequence'] + '.trax.log')) == item['protocol_log_sha256']
        assert item['exit_code'] == 0 and item['exact_wire_boxes']
    assert [r['sequence'] for r in ope['sequences']] == [c['sequence'] for c in cases]
    rows = []
    for case, item in zip(cases, ope['sequences']):
        name = case['sequence']; folder = ROOT / 'ope'
        box_path = folder / (name + '.txt'); score_path = folder / (name + '_all_scores.txt')
        assert sha(box_path) == item['bbox_sha256'] and sha(score_path) == item['confidence_sha256']
        boxes = np.loadtxt(box_path, delimiter=',').reshape(-1, 4); scores = np.loadtxt(score_path).reshape(-1)
        reference = json.loads((M67 / 'recursive/support' / (name + '.json')).read_text())['rows'][:202]
        expected_boxes = np.asarray([r['bbox'] for r in reference])
        expected_scores = np.asarray([1.] + [r['score'] for r in reference[1:]])
        assert boxes.shape == (202, 4) and scores.shape == (202,)
        box_error = float(np.abs(boxes - expected_boxes).max()); score_error = float(np.abs(scores - expected_scores).max())
        assert max(box_error, score_error) <= spec['OPE_box_and_score_tolerance']
        rows.append(dict(sequence=name, frames=202, max_OPE_bbox_rounding_error=box_error, max_OPE_score_rounding_error=score_error))
    result = dict(status='real_learned_OPE_and_TraX_entries_verified', observed_utc=datetime.now(timezone.utc).isoformat(),
                  spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__), head_sha256=spec['head_sha256'],
                  bundle_sha256=spec['bundle_sha256'], OPE_receipt_sha256=sha(ROOT / 'ope/receipt.json'),
                  TraX_receipt_sha256=sha(ROOT / 'trax/receipt.json'), OPE_comparisons=rows,
                  TraX_sessions=trax['sessions'], real_model_tracking_calls=1206,
                  initializations=6, public_evaluation_allowed=False, independent_review_pass=False,
                  scope='Fixed newly trained M67 Support final and category input bank on three sealed Train prefixes. OPE six-decimal files and actual toolkit/TraX wire outputs agree with direct predictions. No official metric, new caption generation, or public benchmark.')
    write(ROOT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'client', 'verify'])
    args = parser.parse_args()
    {'prepare': prepare, 'client': client, 'verify': verify}[args.mode]()
