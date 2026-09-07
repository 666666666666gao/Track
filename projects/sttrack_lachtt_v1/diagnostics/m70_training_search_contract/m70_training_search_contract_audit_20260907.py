"""Compare sealed M65 adaptation inputs with M70 crop geometry; no new inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp')
TRAIN = BASE / 'sttrack_m65_category_null_support_20260907'
M70 = BASE / 'sttrack_m70_recovery_window_inventory_20260907'
ROOT = M70 / 'training_search_contract'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scaled_geometry(gt, side, rectangle, width, height):
    x, y, w, h = rectangle
    intersection = max(0, min(x + w, width) - max(x, 0)) * max(0, min(y + h, height) - max(y, 0))
    size = np.asarray(gt[2:], dtype=np.float64) * 256 / side
    return dict(short=float(size.min()), long=float(size.max()), linear=float(np.sqrt(size.prod())),
        padding_fraction=1 - intersection / (side * side))


def summary(rows):
    assert rows
    return dict(samples=len(rows), sequences=len({r['sequence'] for r in rows}),
        quantiles={key: dict(zip(['p05', 'p50', 'p95'], np.quantile([r[key] for r in rows], [.05, .5, .95]).tolist()))
            for key in ['short', 'long', 'linear', 'padding_fraction']},
        nominal_short_side_below={str(p): sum(r['short'] < p for r in rows) for p in [4, 8, 16]},
        nominal_padding_positive=sum(r['padding_fraction'] > 0 for r in rows))


def main():
    assert sha(TRAIN / 'completed_evidence_audit.json') == '26a79a424ca8b0dec01ff3c12f8cda6653b4a156c12e5c3b297112f52cf69f88'
    audit = read(TRAIN / 'completed_evidence_audit.json')
    assert sha(TRAIN / 'training_spec.json') == audit['training_spec_sha256']
    training = read(TRAIN / 'training_spec.json')
    assert sha(TRAIN / 'training/null/result.json') == audit['training']['null']['result_sha256']
    completed = read(TRAIN / 'training/null/result.json')
    trace_path = TRAIN / 'training/null/sampled_state_trace.jsonl'
    assert sha(trace_path) == completed['sampled_trace_sha256']
    assert completed['status'] == 'one_full_causal_fit_pass_complete' and completed['sequences'] == 130 and completed['total_track_calls'] == 186694
    assert completed['final_checkpoint_sha256'] == audit['training']['null']['final_checkpoint_sha256']
    assert sha(TRAIN / 'integration.json') == training['integration_sha256']
    integration = read(TRAIN / 'integration.json')
    paths = ['lib/models/sttrack/sttrack.py', 'lib/models/sttrack/base_backbone.py', 'lib/models/sttrack/vit_care.py',
        'lib/test/tracker/sttrack.py', 'lib/train/data/processing_utils.py', 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml']
    for name in paths: assert sha(TRAIN / 'code' / name) == integration['source_sha256'][name]
    model = (TRAIN / 'code/lib/models/sttrack/sttrack.py').read_text()
    backbone = (TRAIN / 'code/lib/models/sttrack/vit_care.py').read_text()
    assert model.index('self.backbone(z=template') < model.index('self.MambaFusion(temp_x,temp_r)') < model.index('self.semantic_adapter(')
    assert 'x_r += self.pos_embed_x' in backbone and 'x_x += self.pos_embed_x' in backbone
    assert 'track_query_before[0] = torch.cat([track_query_before[0],track_query_now_x],dim=1)' in model
    assert sha(BASE / 'm70_candidate_capacity_20260907.py') == 'e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977'
    assert sha(M70 / 'candidate_capacity/spec.json') == '796a2756db4c2be0b5282d567294a76d494ad031cc4e186ba96deb3228c00fa5'
    assert not (M70 / 'candidate_capacity/result.json').exists()
    assert sha(M70 / 'result.json') == '8822782394e73154a034e71d783b8eb4e8b27c3302b37ad94e06d9f029b9f17a'
    inventory = read(M70 / 'result.json'); assert sha(M70 / 'geometry_events.json') == inventory['events_sha256']
    geometry_path = BASE / 'm70_recovery_window_inventory_20260907.py'
    assert sha(geometry_path) == '26a1be55df55b83fb4028bb7f86faa33788dd3434debd82f87f4950a514e05d9'
    module_spec = importlib.util.spec_from_file_location('m70_contract_geometry', str(geometry_path))
    g = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(g)
    traces = [json.loads(x) for x in trace_path.read_text().splitlines()]
    assert len(traces) == audit['training']['null']['sampled_state_rows'] == 3917
    assert len({(r['sequence'], r['frame_index']) for r in traces}) == 3917
    cases = {r['sequence']: r for r in training['sequence_order']}; assert len(cases) == 130
    assert set(r['sequence'] for r in traces) == set(cases)
    ROOT.mkdir()
    write(ROOT / 'spec.json', dict(status='frozen_before_contract_statistics', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), M65_completed_audit_sha256=sha(TRAIN / 'completed_evidence_audit.json'),
        training_result_sha256=sha(TRAIN / 'training/null/result.json'), training_trace_sha256=sha(trace_path),
        model_sources={name: sha(TRAIN / 'code' / name) for name in paths},
        M70_capacity_spec_sha256=sha(M70 / 'candidate_capacity/spec.json'), M70_geometry_sha256=sha(M70 / 'geometry_events.json'),
        training_groups=['all valid sampled inputs', 'native supervision centre_inside', 'current prediction continuous IoU>=0.5'],
        event_groups=['all', 'H10', 'H10_local_outside', 'healthy'], quantiles=[.05, .5, .95],
        comparison='Nominal GT size and nominal padding in the256 search tensor. Actual recorded training resize; M70 whole-image square and its fixed-size grid.',
        scope='M65 adapter training samples only. Does not describe every training frame, native backbone pretraining, visibility or semantic truth. Event selection is reused Train development.',
        no_model_outputs_for_M70_read=True, new_tracking_calls=0, new_optimizer_steps=0, no_threshold_selected=True, public_evaluation_allowed=False))
    gt = {}; sizes = {}
    for name, case in cases.items():
        path = Path(training['dataset_root']) / name / 'groundtruth.txt'
        assert sha(path) == case['groundtruth_sha256']; gt[name] = np.loadtxt(path, delimiter=',').reshape(-1, 4)
        with Image.open(case['initial_rgb']) as image: sizes[name] = image.size
    labels = Counter(); rows = []
    for row in traces:
        name = row['sequence']; index = row['frame_index']; target = gt[name][index]; labels[row['label']] += 1
        previous = row['previous_bbox']; rectangle = g.crop(previous, 4); side = rectangle[2]
        assert side == math.ceil(math.sqrt(previous[2] * previous[3]) * 4)
        assert row['resize_factor'] == 256 / side
        valid = bool(np.isfinite(target).all() and (target[2:] > 0).all())
        assert valid == (row['label'] != 'invalid')
        if not valid: continue
        value = scaled_geometry(target, side, rectangle, *sizes[name])
        a = np.asarray(row['bbox']); left = np.maximum(a[:2], target[:2]); right = np.minimum(a[:2] + a[2:], target[:2] + target[2:])
        intersection = np.maximum(0, right - left).prod(); overlap = float(intersection / (a[2:].prod() + target[2:].prod() - intersection))
        rows.append(dict(value, sequence=name, frame=index, label=row['label'], current_iou=overlap))
    summaries = {'fit_valid_sampled': summary(rows), 'fit_centre_inside_sampled': summary([r for r in rows if r['label'] == 'centre_inside']),
        'fit_current_correct_sampled': summary([r for r in rows if r['current_iou'] >= .5])}
    threshold = summaries['fit_centre_inside_sampled']['quantiles']['short']['p05']
    events = read(M70 / 'geometry_events.json')['events']; event_rows = []
    for event in events:
        target = event['GT_bbox']; width = event['image_width']; height = event['image_height']; side = max(width, height)
        coarse = [round((width - side) / 2), round((height - side) / 2), side, side]
        local = event['arms']['null']['local']['rectangle']
        # All grid windows retain this exact local side; their positions change, their nominal target size does not.
        for kind, rectangle in [('coarse', coarse), ('local_grid_scale', local)]:
            value = scaled_geometry(target, rectangle[2], rectangle, width, height)
            if kind == 'local_grid_scale': value.pop('padding_fraction')
            event_rows.append(dict(value, sequence=event['sequence'], frame=event['frame'], kind=kind, tags=event['tags'],
                outside=not event['arms']['null']['local']['centre_inside']))
    comparisons = {}
    for group in ['all', 'H10', 'H10_local_outside', 'healthy']:
        points = [r for r in event_rows if group == 'all' or (group == 'H10' and 'H10' in r['tags'])
            or (group == 'H10_local_outside' and 'H10' in r['tags'] and r['outside']) or (group == 'healthy' and 'healthy' in r['tags'])]
        comparisons[group] = {}
        for kind in ['coarse', 'local_grid_scale']:
            selected = [r for r in points if r['kind'] == kind]
            comparisons[group][kind] = dict(events=len(selected),
                quantiles={key: dict(zip(['p05', 'p50', 'p95'], np.quantile([r[key] for r in selected], [.05, .5, .95]).tolist())) for key in ['short', 'long', 'linear']},
                short_below_fit_centre_inside_p05=sum(r['short'] < threshold for r in selected),
                short_below_16=sum(r['short'] < 16 for r in selected))
            if kind == 'coarse': comparisons[group][kind]['nominal_padding_fraction_values'] = sorted({r['padding_fraction'] for r in selected})
    write(ROOT / 'sampled_geometry.json', dict(spec_sha256=sha(ROOT / 'spec.json'), training=rows, development=event_rows))
    result = dict(status='completed_M65_adaptation_and_M70_search_contract_audit', observed_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__), sampled_geometry_sha256=sha(ROOT / 'sampled_geometry.json'),
        complete_fit_sequences=130, sampled_training_inputs=3917, trace_labels=dict(labels), actual_recorded_resize_checks=3917,
        training_statistics=summaries, development_comparisons=comparisons, reference_p05_not_a_deployment_threshold=threshold,
        source_findings=['All search tensors remain256; learned search positions stay on16x16 grid.',
            'Network receives resized search tensor, templates and history queries, not original image crop origin.',
            'M70 probes copy current templates/query/semantic context and discard returned state; this is a local tracker on new observations, not a separately trained global detector.',
            'Current semantic adapter runs after TSG/Mamba and does not directly regenerate the current TSG query; later bbox/template effects can still change future queries.'],
        limits=['Fit records are sampled along changing training weights, not a final-head heldout evaluation.',
            'Current-backbone pretraining coverage is not established by this audit.',
            'Nominal target scale and padding do not prove the cause of any candidate failure.',
            'The local-grid-scale entry describes target size only; grid padding depends on the selected window and is intentionally not borrowed from the old local window.'],
        new_tracking_calls=0, new_optimizer_steps=0, candidate_recognition_or_recovery_measured=False, public_evaluation_allowed=False, independent_model_review_pass=False)
    write(ROOT / 'result.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
