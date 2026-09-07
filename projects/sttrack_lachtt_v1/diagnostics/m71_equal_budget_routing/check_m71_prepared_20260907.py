"""Check real M70 window geometry and M71 source readiness without model replay."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import numpy as np

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m71_equal_budget_routing_20260907'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = BASE / 'm71_equal_budget_routing_20260907.py'; py_compile.compile(str(source), doraise=True)
    spec = importlib.util.spec_from_file_location('m71_cpu_readiness', str(source))
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    frozen, capacity, geometry, training, _ = app.checked()
    labels = app.read(app.INVENTORY / 'geometry_events.json')['events']
    saved = app.read(ROOT / 'geometry_accounting.json')
    assert saved['spec_sha256'] == sha(ROOT / 'spec.json')
    assert len(saved['events']) == len(labels) == 216
    sys.path.insert(0, str(BASE / 'sttrack_m65_category_null_support_20260907/code'))
    import cv2
    import torch
    from lib.train.data.processing_utils import sample_target
    torch.set_num_threads(1)
    total_grid = 0; target_short = []
    for event, row in zip(labels, saved['events']):
        assert row == app.geometry_row(event, capacity, geometry)
        previous = event['arms']['null']['previous_bbox']; target = np.asarray(event['GT_bbox'], dtype=np.float64)
        image = cv2.imread(str(Path(training['dataset_root']) / event['sequence'] / 'color' / ('%08d.jpg' % (event['frame'] + 1))))
        assert image.shape[:2] == (event['image_height'], event['image_width'])
        windows = capacity.windows(previous, image.shape[1], image.shape[0], geometry)
        centres = np.asarray([[w['rectangle'][0] + w['rectangle'][2] / 2, w['rectangle'][1] + w['rectangle'][3] / 2] for w in windows[3:]])
        indices = np.arange(3, len(windows)); before = np.asarray(previous)
        distance = ((centres - (before[:2] + before[2:] / 2)) ** 2).sum(axis=1)
        ranked = indices[np.lexsort((indices, distance))].tolist()
        assert ranked == app.prior_order(windows, previous)
        for k in [1, 2, 3]:
            selected = ranked[:k]; assert selected == row['prior_windows'][str(k)]['indices']
            rects = np.asarray([windows[i]['rectangle'] for i in selected])
            centre = target[:2] + target[2:] / 2
            inside = ((centre >= rects[:, :2]) & (centre < rects[:, :2] + rects[:, 2:])).all(axis=1).any()
            full = ((target[:2] >= rects[:, :2]) & (target[:2] + target[2:] <= rects[:, :2] + rects[:, 2:])).all(axis=1).any()
            assert bool(inside) == row['prior_windows'][str(k)]['centre_covered']
            assert bool(full) == row['prior_windows'][str(k)]['full_box_covered']
        patch, resize, _ = sample_target(image, windows[2]['prior'], 1., output_sz=256)
        assert patch.shape[:2] == (256, 256) and resize == 256 / max(image.shape[:2])
        # The native resize uses256/side; the report multiplies the integer width before division.
        assert abs(row['coarse_nominal_target_width'] - float(target[2]) * resize) <= 1e-12
        assert abs(row['coarse_nominal_target_height'] - float(target[3]) * resize) <= 1e-12
        total_grid += len(ranked); target_short.append(min(row['coarse_nominal_target_width'], row['coarse_nominal_target_height']))
    assert total_grid == 10907
    assert saved['summaries'] == app.geometry_summary(saved['events'])
    assert not (app.PARENT / 'controller.exit').exists()
    result = subprocess.run([sys.executable, str(source), 'analyze'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    assert result.returncode == 1 and 'FileNotFoundError' in result.stderr and 'controller.exit' in result.stderr
    assert not result.stdout.strip() and not (ROOT / 'result.json').exists() and not (ROOT / 'matched_event_metrics.json').exists()
    assert not torch.cuda.is_initialized()
    receipt = dict(status='cpu_M71_actual_geometry_and_completion_guard_checked', observed_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), source_sha256=sha(source), spec_sha256=sha(ROOT / 'spec.json'), geometry_sha256=sha(ROOT / 'geometry_accounting.json'),
        actual_Train_RGB_frames_checked=216, actual_native_full_image_crop_calls=216, ranked_grid_windows_checked=total_grid,
        independent_vectorized_coverage_comparisons=648, unchanged_geometry_summaries=True,
        before_parent_completion_analysis_exit=result.returncode, missing_parent_completion_rejected=True,
        fabricated_candidate_or_score_inputs=0, model_outputs_read=0, new_tracking_calls=0, optimizer_steps=0,
        cuda_initialized=False, candidate_routing_attribution_completed=False, independent_model_review_pass=False)
    (ROOT / 'cpu_check_result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
