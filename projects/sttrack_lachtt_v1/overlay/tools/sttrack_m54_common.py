"""Frozen experiment bindings shared by M54 collection, fitting and recursion."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(root):
    plan = json.loads((root / 'spec.json').read_text())
    parent = Path(plan['source_root'])
    spec = json.loads((parent / 'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.train_sttrack_m44 import check_binding
    check_binding(parent, spec)
    assert sha(parent / 'spec.json') == plan['source_spec_sha256']
    assert sha(parent / 'inference_inputs.json') == spec['inference_inputs_sha256']
    assert sha(root / 'EXPERIMENT_PLAN.md') == plan['experiment_plan_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(repo / name) == digest, name
    for name, digest in plan['run_files_sha256'].items():
        assert sha(root / name) == digest, name
    assert sha(Path(plan['m53_root']) / 'capacity_result.json') == plan['m53_capacity_sha256']
    return plan, parent, spec


def parameters(spec):
    from lib.config.sttrack.config import cfg, update_config_from_file
    update_config_from_file(str(Path(spec['repository']) / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    return SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                           search_factor=4., search_size=256, save_all_boxes=False, debug=0)


def event_frames(case):
    return sorted(set(case['event_frames']) | set(range(10, max(case['event_frames']) + 1, 10)))
