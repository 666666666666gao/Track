"""Export the actual rectangle initializations of a frozen VOT shard manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path

from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.region import Rectangle
from vot.workspace import Workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def paths(frame):
    return [str(Path(frame.filename(c)).resolve()) for c in ['color', 'depth']]


def add_frame(digest, frame):
    digest.update((json.dumps(paths(frame), separators=(',', ':')) + '\n').encode())


def xywh(region):
    assert isinstance(region, Rectangle), type(region).__name__
    return [region.x, region.y, region.width, region.height]


def collect(manifest_path, expected_sha, output):
    assert metadata.version('vot-toolkit') == '0.7.1'
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    assert sha(manifest_path) == expected_sha
    manifest = json.loads(manifest_path.read_text())
    expected = {a['sequence'] + '_%08d' % a['index']:a for s in manifest['shards'] for a in s['anchors']}
    assert len(expected) == sum(len(s['anchors']) for s in manifest['shards']) == manifest['total_anchor_count']
    rows, metadata_hashes, execution_order = [], {}, []
    for shard in manifest['shards']:
        base = Path(shard['root'])
        for name in ['config.yaml', 'sequences/list.txt']:
            metadata_hashes[str(base / name)] = sha(base / name)
        workspace = Workspace.load(str(base))
        experiments = list(workspace.stack)
        assert len(experiments) == 1 and isinstance(experiments[0], MultiStartExperiment)
        experiment = experiments[0]
        assert experiment.identifier == 'baseline' and experiment.noise is None and experiment.inject is None
        order = []
        for sequence in workspace.dataset:
            for p in (base / 'sequences' / sequence.name).iterdir():
                if p.is_file():
                    metadata_hashes[str(p)] = sha(p)
            for transformed in experiment.transform(sequence):
                forward, backward = find_anchors(transformed, experiment.anchor)
                for index, reverse in [(i, False) for i in forward] + [(i, True) for i in backward]:
                    name = '%s_%08d' % (transformed.name, index)
                    old = expected[name]
                    direction = 'backward' if reverse else 'forward'
                    frame_indices = list(reversed(range(0, index + 1))) if reverse else list(range(index, len(transformed)))
                    assert old['direction'] == direction and old['estimated_frames'] == len(frame_indices)
                    assert transformed.values(index)[experiment.anchor] == old['value']
                    initial_frame = transformed.frame(index)
                    region = experiment._get_initialization(transformed, index)
                    box = xywh(region)
                    assert box[2] > 0 and box[3] > 0
                    digest = hashlib.sha256()
                    for i in frame_indices:
                        add_frame(digest, transformed.frame(i))
                    pair = paths(initial_frame)
                    rows.append(dict(id=name, sequence=transformed.name, index=index, direction=direction,
                        shard_root=str(base), image=pair[0], depth=pair[1], bbox=box,
                        region_type='Rectangle', frames=len(frame_indices), frame_paths_sha256=digest.hexdigest()))
                    order.append(name)
        assert set(order) == {a['sequence'] + '_%08d' % a['index'] for a in shard['anchors']}
        execution_order.append(dict(shard_root=str(base), cases=order))
    assert {r['id'] for r in rows} == set(expected) and len(rows) == len(expected)
    assert sum(r['frames'] for r in rows) == manifest['total_estimated_frames']
    rows.sort(key=lambda r:r['id'])
    output.mkdir()
    inputs = dict(coordinate_convention='vot_toolkit_xywh',
        cases=[{k:r[k] for k in ['id', 'image', 'bbox']} for r in rows])
    write(output / 'caption_inputs.json', inputs)
    write(output / 'anchors.json', rows)
    write(output / 'metadata_sha256.json', metadata_hashes)
    write(output / 'execution_order.json', execution_order)
    record = dict(status='frozen_toolkit_rectangle_initializations_exported',
        observed_utc=datetime.now(timezone.utc).isoformat(), manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path), exporter_sha256=sha(__file__),
        toolkit_version=metadata.version('vot-toolkit'),
        toolkit_source_sha256={inspect.getfile(cls):sha(inspect.getfile(cls)) for cls in [Workspace, MultiStartExperiment, Rectangle]},
        files_sha256={n:sha(output / n) for n in ['caption_inputs.json', 'anchors.json', 'metadata_sha256.json', 'execution_order.json']},
        anchors=len(rows), sequences=len({r['sequence'] for r in rows}),
        forward=sum(r['direction'] == 'forward' for r in rows), backward=sum(r['direction'] == 'backward' for r in rows),
        estimated_frames=sum(r['frames'] for r in rows), coordinate_convention='vot_toolkit_xywh',
        caption_list_order='Sorted official trajectory ID; tracking keeps each workspace scheduler order.',
        raw_gt_strings_manually_parsed=False, polygon_or_mask_conversion_used=False,
        tracking_model_calls=0, qwen_generate_calls=0, model_image_decodes=0,
        scope='Existing benchmark metadata and initialization paths only; no pixels passed to tracking or caption models.',
        independent_review_pass=False)
    write(output / 'export_result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.manifest, args.expected_sha, args.output), indent=2))
