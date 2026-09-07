"""Recompute completed diagnostic evidence without changing sealed results or tracking."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path

B = Path('/root/autodl-tmp')
OUT = B / 'sttrack_m68_m72_completed_evidence_20260907'
JOBS = {
    'M68': ('sttrack_m68_reported_confidence_20260907', 'm68_reported_confidence_20260907.py'),
    'M69': ('sttrack_m69_m65_content_diagnostic_20260907', 'm69_m65_content_diagnostic_20260907.py'),
    'M70': ('sttrack_m70_recovery_window_inventory_20260907/candidate_capacity', 'm70_candidate_capacity_20260907.py'),
    'M71': ('sttrack_m71_equal_budget_routing_20260907', 'm71_equal_budget_routing_20260907.py'),
    'M72': ('sttrack_m72_m67_control_content_20260907', 'm72_m67_control_content_20260907.py'),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def load(name, p):
    s = importlib.util.spec_from_file_location(name, str(p))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            if k != 'observed_utc': compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(a, float): assert abs(a - b) < 1e-10
    else: assert a == b, (a, b)


def audit_confidence(app, result):
    import numpy as np
    spec, training, parent = app.eligible()
    sealed = {}
    for arm in spec['arms']:
        folder = app.ROOT / arm
        assert sha(folder / 'receipt.json') == result['receipts'][arm]
        receipt = read(folder / 'receipt.json')
        assert receipt['source_sha256'] == sha(app.__file__)
        assert receipt['head_sha256'] == parent['families'][arm]['head_sha256']
        assert receipt['parent_audit_sha256'] == sha(app.PARENT / 'completed_evidence_audit.json')
        assert receipt['spec_sha256'] == sha(app.ROOT / 'spec.json')
        assert receipt['total_frames'] == 33130 and len(receipt['sequences']) == 22
        sealed[arm] = {}
        for case, item in zip(spec['cases'], receipt['sequences']):
            seq = case['sequence']; assert item['sequence'] == seq
            path = folder / (seq + '.json'); assert sha(path) == item['sha256']
            rows = read(path)['rows']; reference = app.PARENT / 'recursive' / arm / (seq + '.json')
            assert sha(reference) == item['parent_prediction_sha256']
            assert [dict(frame=r['frame'], bbox=r['bbox'], score=r['score']) for r in rows] == read(reference)['rows']
            assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames']))
            for r in rows[1:]:
                assert 0 <= r['null_mass'] <= 1 and 0 <= r['score'] <= 1
                assert r['semantic_product'] == r['score'] * (1 - r['null_mass'])
            sealed[arm][seq] = rows
            for kind in spec['readouts']:
                dest = app.ROOT / 'ope' / (arm + '_' + kind)
                files = result['exports'][arm][kind][seq]
                box = dest / (seq + '.txt'); score = dest / (seq + '_all_scores.txt')
                assert sha(box) == files['bbox_sha256'] and sha(score) == files['confidence_sha256']
                bb = io.StringIO(); ss = io.StringIO()
                np.savetxt(bb, [r['bbox'] for r in rows], delimiter=',', fmt='%.6f')
                np.savetxt(ss, [1.] + [r['score'] if kind == 'raw' else r['semantic_product'] for r in rows[1:]], fmt='%.6f')
                assert box.read_text() == bb.getvalue() and score.read_text() == ss.getvalue()
            assert result['exports'][arm]['raw'][seq]['bbox_sha256'] == result['exports'][arm]['semantic_product'][seq]['bbox_sha256']
    # All prediction families and readouts verified before reading later GT.
    evaluator = app.load_module('completed_metric', app.METRIC)
    for case in spec['cases']:
        assert sha(Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt') == case['gt_sha256']
    for arm in spec['arms']:
        for kind in spec['readouts']:
            actual = evaluator.evaluate_depthtrack_results(training['dataset_root'], app.ROOT / 'ope' / (arm + '_' + kind),
                resolution=spec['pr_resolution'], sequence_names=[c['sequence'] for c in spec['cases']])
            compare(actual, result['metrics'][arm][kind])
        samples = {k: [] for k in result['null_mass_and_confidence_by_GT_group'][arm]}
        for case in spec['cases']:
            seq = case['sequence']; root = Path(training['dataset_root']) / seq
            gt = evaluator._load_rows(root / 'groundtruth.txt', 4)
            boxes = evaluator._load_rows(app.ROOT / 'ope' / (arm + '_raw') / (seq + '.txt'), 4)
            height, width = evaluator.cv2.imread(str(root / 'color/00000001.jpg')).shape[:2]
            overlap, visible = evaluator._vot_overlaps(boxes, gt, width, height)
            for i in range(1, case['frames']):
                key = 'invalid_GT' if not visible[i] else ('valid_correct' if overlap[i] >= .5 else ('valid_low_overlap' if overlap[i] <= .1 else 'valid_intermediate'))
                row = sealed[arm][seq][i]; samples[key].append([row['null_mass'], row['score'], row['semantic_product']])
        for key, vals in samples.items():
            arr = np.asarray(vals); actual = dict(frames=len(vals), columns=['null_mass', 'raw', 'semantic_product'],
                mean=arr.mean(axis=0).tolist(), median=np.median(arr, axis=0).tolist())
            compare(actual, result['null_mass_and_confidence_by_GT_group'][arm][key])
    return 1


def main(name):
    root_name, source_name = JOBS[name]; root = B / root_name; source = B / source_name
    report_path = OUT / (name + '_audit.json'); assert not report_path.exists()
    result = read(root / 'result.json'); original_sha = sha(root / 'result.json')
    assert result['source_sha256'] == sha(source) and result['spec_sha256'] == sha(root / 'spec.json')
    codes = ['analysis', 'controller'] if name != 'M71' else []
    for code in codes: assert (root / (code + '.exit')).read_text().strip() == '0'
    app = load('completed_' + name, source)
    captured = {}
    with contextlib.redirect_stdout(io.StringIO()):
        if name == 'M68': outputs = audit_confidence(app, result)
        else:
            # Frozen analysis validates source/bank/head/receipt/GT bindings and recomputes
            # from sealed traces/arrays. Redirect only its JSON writer into memory.
            app.write = lambda p, v: captured.__setitem__(str(p), v)
            app.analyze()
            for path, value in captured.items(): compare(value, read(path))
            outputs = len(captured)
    assert sha(root / 'result.json') == original_sha
    OUT.mkdir(exist_ok=True)
    report = dict(status='completed_sealed_diagnostic_recomputation', experiment=name,
        observed_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha(__file__),
        source_sha256=sha(source), spec_sha256=sha(root / 'spec.json'), result_sha256=original_sha,
        recomputed_outputs=outputs, new_tracking_calls=0, new_optimizer_steps=0,
        sealed_outputs_modified=False, public_evaluation_allowed=False,
        independent_model_review_pass=False,
        scope='Integrity/recomputation audit. Reuses frozen metric implementation; not independent algorithm validation or a new learned-model review.')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('experiment', choices=JOBS); main(parser.parse_args().experiment)
