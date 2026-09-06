"""Exercise real multi-start scheduling with a CPU recorder and no result files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import vot.experiment.multistart as multistart
from vot.region import Rectangle, Special
from vot.tracker import ObjectStatus
from vot.workspace import Workspace

from export_initializations import add_frame, sha, write, xywh


ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / 'low22_inputs'


def main():
    started = time.time()
    export = json.loads((EXPORT / 'export_result.json').read_text())
    assert export['exporter_sha256'] == sha(ROOT / 'export_initializations.py')
    for path, digest in export['toolkit_source_sha256'].items():
        assert sha(path) == digest
    for name, digest in export['files_sha256'].items():
        assert sha(EXPORT / name) == digest
    for path, digest in json.loads((EXPORT / 'metadata_sha256.json').read_text()).items():
        assert sha(path) == digest
    manifest = json.loads(Path(export['manifest_path']).read_text())
    rows = {r['id']:r for r in json.loads((EXPORT / 'anchors.json').read_text())}
    order = json.loads((EXPORT / 'execution_order.json').read_text())
    receipts = []

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def begin(self, name):
            self.name = name
            self.row = rows[name]
            self.digest = hashlib.sha256()
            self.frames = 0

        def initialize(self, frame, objects):
            assert xywh(objects) == self.row['bbox']
            add_frame(self.digest, frame)
            self.frames += 1
            return None, 0.

        def update(self, frame):
            add_frame(self.digest, frame)
            self.frames += 1
            return ObjectStatus(Rectangle(*self.row['bbox']), {}), 0.

    recorder = Recorder()
    initialization_code = multistart.Trajectory.INITIALIZATION

    class AuditTrajectory:
        INITIALIZATION = initialization_code

        @staticmethod
        def exists(results, name):
            recorder.begin(name)
            return False

        def __init__(self, length):
            self.length = length
            self.count = 0
            assert length == recorder.row['frames']

        def set(self, index, region, properties):
            assert index == self.count
            assert isinstance(region, Special if index == 0 else Rectangle)
            self.count += 1

        def write(self, results, name):
            assert name == recorder.name and self.count == recorder.frames == self.length
            assert recorder.digest.hexdigest() == recorder.row['frame_paths_sha256']
            receipts.append(dict(id=name, frames=self.length, frame_paths_sha256=recorder.digest.hexdigest(),
                                 initialization_region_equal=True))

    with patch.object(multistart, 'Trajectory', AuditTrajectory), \
         patch.object(multistart.MultiStartExperiment, 'results', return_value=None), \
         patch.object(multistart.MultiStartExperiment, '_get_runtime', return_value=recorder):
        for shard in manifest['shards']:
            workspace = Workspace.load(shard['root'])
            experiment = list(workspace.stack)[0]
            start = len(receipts)
            for sequence in workspace.dataset:
                for transformed in experiment.transform(sequence):
                    experiment.execute(None, transformed)
            expected = next(o['cases'] for o in order if o['shard_root'] == shard['root'])
            assert [r['id'] for r in receipts[start:]] == expected
    assert len(receipts) == export['anchors'] == 303
    assert {r['id'] for r in receipts} == set(rows)
    assert sum(r['frames'] for r in receipts) == export['estimated_frames']
    write(ROOT / 'scheduler_receipts.json', receipts)
    result = dict(status='real_multistart_scheduler_with_cpu_recorder_passed',
        observed_utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        export_result_sha256=sha(EXPORT / 'export_result.json'), receipts_sha256=sha(ROOT / 'scheduler_receipts.json'),
        initialized_anchors=len(receipts), checked_frame_paths=sum(r['frames'] for r in receipts),
        simulated_runtime_updates=sum(r['frames']-1 for r in receipts),
        all_initialization_regions_equal=True, all_frame_orders_equal=True,
        all_workspace_case_orders_equal=True, tracking_model_calls=0, qwen_generate_calls=0,
        model_image_decodes=0, actual_trax_sessions=0, benchmark_prediction_files_written=0,
        runtime_and_trajectory_storage_replaced_by_recorder=True,
        elapsed_seconds=time.time()-started, independent_review_pass=False,
        scope='Real toolkit experiment.execute and transforms; CPU metadata recorder. No model tracking, TraX transport, caption generation or benchmark metrics.')
    write(ROOT / 'scheduler_result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
