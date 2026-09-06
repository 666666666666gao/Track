"""One frozen STTrack semantic runtime for OPE and VOT entry points."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from initialization_text import InitializationTextBank, sha


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    assert sha(plan['bundle_path']) == plan['bundle_sha256']
    bundle = json.loads(Path(plan['bundle_path']).read_text())
    assert bundle['architecture'] == 'semantic_spatial_v1'
    assert sha(bundle['base_checkpoint']) == bundle['base_checkpoint_sha256']
    assert sha(bundle['adapter_checkpoint']) == bundle['adapter_checkpoint_sha256']
    for name, digest in bundle['source_sha256'].items():
        assert sha(Path(bundle['repository']) / name) == digest, name
    for name, digest in bundle['interface_sha256'].items():
        assert sha(Path(__file__).parent / name) == digest, name
    assert sha(bundle['text_protocol_path']) == bundle['text_protocol_sha256']
    assert sha(plan['text_bank_path']) == plan['text_bank_sha256']
    return plan, bundle


def text_bank(plan, bundle):
    return InitializationTextBank(plan['text_bank_path'], plan['text_bank_sha256'],
                                  bundle['text_protocol_sha256'])


def make_tracker(bundle):
    import torch
    sys.path.insert(0, bundle['repository'])
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    torch.set_num_threads(1)
    torch.manual_seed(bundle['seed'])
    torch.cuda.manual_seed_all(bundle['seed'])
    update_config_from_file(str(Path(bundle['repository']) / bundle['configuration']))
    assert cfg.MODEL.TSG.FIX_QUERY_WINDOW
    assert cfg.DATA.TEMPLATE.NUMBER == 2
    assert cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
    assert cfg.TEST.TEMPLATE_FACTOR == 2. and cfg.TEST.TEMPLATE_SIZE == 128
    assert cfg.TEST.SEARCH_FACTOR == 4. and cfg.TEST.SEARCH_SIZE == 256
    saved = torch.load(bundle['adapter_checkpoint'], map_location='cpu')
    assert saved['status'] == 'complete' and saved['completed_sequences'] == 130
    assert saved['training_spec_sha256'] == bundle['training_spec_sha256']
    assert saved['use_text'] == bundle['use_text']
    params = SimpleNamespace(cfg=cfg, checkpoint=bundle['base_checkpoint'],
        base_checkpoint_sha256=bundle['base_checkpoint_sha256'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    return STTrackSemantic(params, bundle['adapter_checkpoint'])
