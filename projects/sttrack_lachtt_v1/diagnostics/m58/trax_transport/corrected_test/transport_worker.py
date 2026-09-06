"""Exercise the unchanged semantic VOT loop with a deterministic CPU recorder."""
import argparse
import atexit
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(name):
    fixture = json.loads((ROOT / 'fixture.json').read_text())
    assert sha(__file__) == fixture['source_sha256']['transport_worker.py']
    for path, digest in fixture['interface_sha256'].items():
        assert sha(INTERFACE / path) == digest
    assert sha(PARENT / 'code/lib/train/dataset/depth_utils.py') == fixture['depth_utility_sha256']
    sys.path.insert(0, str(PARENT / 'code'))
    sys.path.insert(0, str(INTERFACE))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from initialization_text import InitializationTextBank
    import run_semantic_vot as entry
    import trax
    case = next(c for c in fixture['cases'] if c['name'] == name)
    assert metadata.version('vot-trax') == '4.0.2'
    bank = InitializationTextBank(case['bank_path'], case['bank_sha256'], fixture['protocol_sha256'])
    expected = []
    for pair in fixture['images']:
        assert sha(pair['color']) == pair['color_sha256'] and sha(pair['depth']) == pair['depth_sha256']
        expected.append(get_rgbd_frame(pair['color'], pair['depth'], dtype='rgbcolormap', depth_clip=True))
    assert all(image.shape == (64, 96, 6) for image in expected)
    assert len({hashlib.sha256(image[:, :, :3].tobytes()).hexdigest() for image in expected}) == 3
    assert len({hashlib.sha256(image[:, :, 3:].tobytes()).hexdigest() for image in expected}) == 3
    record = dict(status='entry_not_returned', case=name, initialize_calls=0, track_calls=0,
        rgbd_frame_indices=[], fixture_sha256=sha(ROOT / 'fixture.json'),
        model_factory_mocked=True, bundle_validation_mocked=True, text_bank_factory_mocked=True,
        initialization_text_bank_real=True, rgbd_decoder_real=True, vot_entry_and_bridge_real=True,
        model_weights_loaded=False, trax_version=metadata.version('vot-trax'),
        trax_server_source_sha256=sha(Path(trax.__file__).parent / 'server.py'))

    def save():
        record['cuda_initialized'] = torch.cuda.is_initialized()
        (ROOT / (name + '_worker.json')).write_text(json.dumps(record, indent=2) + '\n')

    atexit.register(save)

    class Recorder:
        def initialize(self, image, info):
            record['initialize_calls'] += 1
            assert case['expected_key_index'] is not None
            assert np.array_equal(image, expected[0])
            assert info['init_bbox'] == case['wire_bbox']
            index = case['expected_key_index']
            assert torch.equal(info['text_tokens'], bank.bank['tokens'][index])
            assert torch.equal(info['text_mask'], bank.bank['mask'][index])
            record['rgbd_frame_indices'].append(0)
            record['received_bbox'] = info['init_bbox']
            record['text_key_index'] = index
            record['text_tokens_sha256'] = hashlib.sha256(info['text_tokens'].numpy().tobytes()).hexdigest()

        def track(self, image):
            record['track_calls'] += 1
            step = record['track_calls']
            assert np.array_equal(image, expected[step])
            record['rgbd_frame_indices'].append(step)
            prediction = case['predictions'][step - 1]
            return dict(target_bbox=prediction['bbox'], best_score=prediction['score'])

    recorder = Recorder()
    # Only construction/plan boundaries are mocked. TraX, the full VOT entry
    # loop, RGB-D decoding, actual bank lookup and report serialization execute.
    with patch.object(entry, 'checked_plan', return_value=({}, {})), \
         patch.object(entry, 'make_tracker', return_value=recorder), \
         patch.object(entry, 'text_bank', return_value=bank):
        entry.run('synthetic_fixture_plan')
    assert record['initialize_calls'] == 1 and record['track_calls'] == 2
    assert not torch.cuda.is_initialized()
    for path, digest in fixture['interface_sha256'].items():
        assert sha(INTERFACE / path) == digest
    record['status'] = 'vot_entry_completed_with_cpu_recorder'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    main(parser.parse_args().case)
