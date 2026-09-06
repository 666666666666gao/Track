"""Use the actual VOT toolkit TrackerProcess client against the CPU worker."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import shlex
import time
import unittest

import numpy as np
from trax import TraxException
import trax.client
from vot.region import Rectangle
from vot.tracker import ObjectStatus
import vot.tracker.trax as toolkit_trax


ROOT = Path(__file__).resolve().parent
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SyntheticFrame:
    def __init__(self, pair):
        self.pair = pair

    def filename(self, channel):
        return self.pair[channel]


def box(region):
    return [float(region.x), float(region.y), float(region.width), float(region.height)]


def main():
    started = time.time()
    fixture = json.loads((ROOT / 'fixture.json').read_text())
    assert sha(__file__) == fixture['source_sha256']['check_transport.py']
    assert metadata.version('vot-trax') == '4.0.2'
    sessions = []
    test = unittest.TestCase()
    for case in fixture['cases']:
        command = shlex.join([PYTHON, str(ROOT / 'transport_worker.py'), '--case', case['name']])
        process = toolkit_trax.TrackerProcess(command, envvars={'CUDA_VISIBLE_DEVICES':''},
            timeout=20, log=str(ROOT / (case['name'] + '.trax.log')))
        # This Popen handle is owned by the toolkit process just created here.
        child = process._process
        observed = []
        expected_failure = case['expected_key_index'] is None
        failure_message = None
        try:
            assert process.has_vot_wrapper
            status, _ = process.initialize(SyntheticFrame(fixture['images'][0]),
                ObjectStatus(Rectangle(*case['raw_bbox']), {}))
            assert len(status) == 1 and box(status[0].region) == case['wire_bbox']
            assert 'confidence' not in status[0].properties
            if expected_failure:
                with test.assertRaises(TraxException) as caught:
                    process.update(SyntheticFrame(fixture['images'][1]))
                failure_message = str(caught.exception)
            else:
                for i, expected in enumerate(case['predictions'], start=1):
                    status, _ = process.update(SyntheticFrame(fixture['images'][i]))
                    assert len(status) == 1 and box(status[0].region) == expected['wire_bbox']
                    confidence = float(status[0].properties['confidence'])
                    assert abs(confidence - expected['score']) <= 1e-12
                    observed.append(dict(frame=i, bbox=box(status[0].region), confidence=confidence))
        finally:
            process.terminate()
        exit_code = child.wait(timeout=5)
        assert exit_code == (1 if expected_failure else 0)
        worker = json.loads((ROOT / (case['name'] + '_worker.json')).read_text())
        assert not worker['cuda_initialized'] and not worker['model_weights_loaded']
        if expected_failure:
            assert worker['initialize_calls'] == worker['track_calls'] == 0
            assert worker['status'] == 'entry_not_returned'
            assert 'KeyError' in (ROOT / (case['name'] + '.trax.log')).read_text()
        else:
            assert worker['status'] == 'vot_entry_completed_with_cpu_recorder'
            assert worker['rgbd_frame_indices'] == [0, 1, 2]
            assert worker['initialize_calls'] == 1 and worker['track_calls'] == 2
            assert worker['text_key_index'] == case['expected_key_index']
        sessions.append(dict(case=case['name'], pid=child.pid, exit_code=exit_code,
            expected_unknown_key_failure=expected_failure, failure_message=failure_message,
            initialization_ack_without_confidence=True, responses=observed,
            worker_receipt_sha256=sha(ROOT / (case['name'] + '_worker.json')),
            protocol_log_sha256=sha(ROOT / (case['name'] + '.trax.log'))))
    a = json.loads((ROOT / 'first_object_worker.json').read_text())
    b = json.loads((ROOT / 'second_object_same_image_worker.json').read_text())
    assert a['text_tokens_sha256'] != b['text_tokens_sha256']
    result = dict(status='real_toolkit_trax_transport_with_cpu_recorder_passed',
        observed_utc=datetime.now(timezone.utc).isoformat(), fixture_sha256=sha(ROOT / 'fixture.json'),
        source_sha256=fixture['source_sha256'], interface_sha256=fixture['interface_sha256'],
        sessions=sessions, successful_sessions=2, intentional_unknown_key_rejections=1,
        intentional_constructor_only_index_rejections=1, precision_example=fixture['precision_example'],
        completed_tracking_reports=4, same_image_distinct_initialized_targets=True,
        rgbd_channel_order_and_frame_order_verified=True, application_confidence_remapping=False,
        toolkit_version=metadata.version('vot-toolkit'), client_trax_version=metadata.version('vot-trax'),
        toolkit_client_source_sha256=sha(inspect.getfile(toolkit_trax)),
        trax_client_source_sha256=sha(inspect.getfile(trax.client)),
        elapsed_seconds=time.time() - started, actual_model_tracking_calls=0, dataset_images_used=0,
        real_model_constructor_and_bundle_validation_tested=False, independent_review_pass=False,
        scope='Real VOT 0.7.1 client, TraX 4.0.2, unchanged semantic VOT loop and bank lookup; deterministic synthetic CPU predictor. No model performance or real-weight GPU entry claim.')
    (ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
