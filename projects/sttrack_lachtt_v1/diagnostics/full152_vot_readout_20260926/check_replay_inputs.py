"""CPU check of every replay reference and initialization-bank observation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_full152_vot_readout_20260926')
EVAL = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
sys.path.insert(0, str(EVAL / 'interface'))
from initialization_text import InitializationTextBank, vot_wire_bbox


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert sha(spec['plan_path']) == spec['plan_sha256']
    plan = json.loads(Path(spec['plan_path']).read_text())
    assert sha(plan['bundle_path']) == plan['bundle_sha256']
    bundle = json.loads(Path(plan['bundle_path']).read_text())
    bank = InitializationTextBank(plan['text_bank_path'], plan['text_bank_sha256'], bundle['text_protocol_sha256'])
    rows, calls = [], 0
    for case in spec['cases']:
        assert sha(case['reference']) == case['reference_sha256']
        with np.load(case['reference'], allow_pickle=False) as archive:
            boxes, scores = archive['boxes'], archive['scores']
        assert boxes.shape == (case['prefix_length'], 4)
        assert scores.shape == (case['prefix_length'],)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        serialized = np.asarray([vot_wire_bbox(list(box)) for box in boxes])
        assert np.array_equal(boxes, serialized), (case['anchor_key'], 'wire fixed point')
        rgb = Path(case['folder']) / 'color' / f"{case['anchor'] + 1:08d}.jpg"
        info = bank.info(rgb, case['init_bbox'])
        assert info['init_bbox'] == boxes[0].tolist()
        assert tuple(info['text_tokens'].shape) == (5, 768)
        assert tuple(info['text_mask'].shape) == (5,)
        calls += len(boxes) - 1
        rows.append(dict(anchor_key=case['anchor_key'], direction=case['direction'],
                         initialization_rgb_sha256=sha(rgb), reference_sha256=case['reference_sha256']))
    assert len(rows) == 124 and calls == spec['replay_calls'] == 52262
    report = dict(status='complete_cpu_input_check', source_sha256=sha(__file__),
                  spec_sha256=sha(ROOT / 'spec.json'), plan_sha256=spec['plan_sha256'],
                  text_bank_sha256=plan['text_bank_sha256'], anchors=124, replay_calls=calls,
                  initialization_bank_binding_all_pass=True, saved_box_wire_fixed_point_all_pass=True,
                  gpu_inference_performed=False, actual_replay_parity_proven=False, cases=rows)
    (ROOT / 'input_check.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
