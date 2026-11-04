"""Read sealed VOT trajectories and verify identity, frame coverage and hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def declared_anchors(manifest):
    assert manifest["schema"] == "M122_full127_frozen_shards_v1"
    assert len(manifest["sequences"]) == len(set(manifest["sequences"])) == 127
    assert set(manifest["source"]) == set(manifest["sequences"])
    assert manifest["total_anchor_count"] == 1765
    assert [row["index"] for row in manifest["shards"]] == [0, 1, 2, 3]
    assert [row["anchor_count"] for row in manifest["shards"]] == [441, 441, 441, 442]
    rows = {}
    names = set()
    for shard in manifest["shards"]:
        assert len(shard["anchors"]) == shard["anchor_count"]
        expected_names = []
        for anchor in shard["anchors"]:
            assert anchor["sequence"] in manifest["source"]
            assert anchor["direction"] in ("forward", "backward")
            key = (anchor["sequence"], anchor["index"], anchor["direction"])
            name = "%s_%08d" % (anchor["sequence"], anchor["index"])
            assert key not in rows and name not in names
            names.add(name)
            expected_names.append(name)
            rows[key] = dict(anchor, shard=shard["index"], name=name)
        assert expected_names == shard["expected_trajectories"]
    assert len(rows) == len(names) == 1765
    return rows


def checked_length(row, frame_count, length, properties):
    anchor = row["index"]
    assert 0 <= anchor < frame_count
    expected = frame_count - anchor if row["direction"] == "forward" else anchor + 1
    assert row["estimated_frames"] == length == expected, row["name"]
    assert set(properties) == {"confidence", "time"}, row["name"]
    assert all(len(values) == expected for values in properties.values()), row["name"]
    return expected


def read_properties(results, name, trajectory):
    names = trajectory.properties()
    assert set(names) == {"confidence", "time"}, name
    properties = {}
    for key in names:
        with results.read(name + "_" + key + ".value") as stream:
            properties[key] = stream.readlines()
    return properties


def audit(args):
    import vot
    from vot.dataset.proxy import FrameMapSequence
    from vot.experiment.multistart import find_anchors
    from vot.tracker import Trajectory
    from vot.workspace import Workspace

    manifest = read(args.manifest)
    declared = declared_anchors(manifest)
    merge = read(args.merge)
    assert merge["tracker"] == manifest["tracker"]
    assert merge["source_manifest_sha256"] == sha(args.manifest)
    assert merge["anchor_count"] == 1765 and merge["result_file_count"] == 5295
    master = Path(merge["master_workspace"]).resolve()
    assert master == args.workspace.resolve()
    assert sha(args.execution) == args.execution_sha256
    execution = read(args.execution)
    assert execution["anchors"] == 1765 and execution["sequences"] == 127
    for path, digest in execution["source_sha256"].items():
        assert sha(path) == digest, path
    assert [row["index"] for row in merge["launches"]] == [0, 1, 2, 3]
    assert [row["completed_anchors"] for row in merge["launches"]] == [441, 441, 441, 442]
    assert all(row["exit"] == 0 for row in merge["launches"])
    for shard in manifest["shards"]:
        exit_path = args.manifest.parent / ("shard-%02d.exit" % shard["index"])
        assert exit_path.read_text().strip() == "0"

    analysis = read(args.analysis)
    tracker_id = manifest["tracker"]
    assert vot.__version__ == analysis["toolkit"] == "0.7.1"
    assert set(analysis["trackers"]) == {tracker_id}
    assert set(analysis["sequences"]) == set(manifest["sequences"])
    workspace = Workspace.load(str(master))
    trackers = workspace.registry.resolve(tracker_id,
        storage=workspace.storage.substorage("results"), skip_unknown=False)
    assert len(trackers) == 1
    experiment = workspace.stack.experiments["baseline"]
    assert experiment.dump() == analysis["results"]["baseline"]["parameters"]
    sequences = list(experiment.transform(workspace.dataset))
    assert len(sequences) == 127
    assert {sequence.name for sequence in sequences} == set(manifest["sequences"])
    records, files = [], {}
    seen = set()
    for sequence in sequences:
        source = manifest["source"][sequence.name]
        assert (master / "sequences" / sequence.name).resolve() == Path(source["root"]).resolve()
        assert len(sequence) == source["frame_count"]
        assert sha(Path(source["root"]) / "anchor.value") == source["anchor_sha256"]
        results = experiment.results(trackers[0], sequence)
        forward, backward = find_anchors(sequence, experiment.anchor)
        anchors = [(index, False) for index in forward]
        anchors.extend((index, True) for index in backward)
        for anchor, reverse in anchors:
            key = (sequence.name, anchor, "backward" if reverse else "forward")
            assert key in declared and key not in seen
            seen.add(key)
            row = declared[key]
            frame_map = list(reversed(range(anchor + 1))) if reverse else list(range(anchor, len(sequence)))
            proxy = FrameMapSequence(sequence, frame_map)
            assert Trajectory.exists(results, row["name"])
            trajectory = Trajectory.read(results, row["name"])
            properties = read_properties(results, row["name"], trajectory)
            length = checked_length(row, len(sequence), len(trajectory), properties)
            assert length == len(proxy)
            folder = Path("results") / tracker_id / "baseline" / sequence.name
            for suffix in (".bin", "_confidence.value", "_time.value"):
                relative = folder / (row["name"] + suffix)
                digest = sha(master / relative)
                assert digest == merge["result_sha256"][relative.as_posix()]
                shard_root = Path(manifest["shards"][row["shard"]]["root"])
                assert sha(shard_root / relative) == digest
                files[relative.as_posix()] = digest
            records.append(dict(sequence=sequence.name, anchor=int(anchor),
                direction=key[2], shard=row["shard"], expected_frames=length,
                actual_frames=len(trajectory), properties_lengths_exact=True))
    assert seen == set(declared) and len(records) == 1765
    assert len(files) == 5295 and files == merge["result_sha256"]
    actual_files = {path.relative_to(master).as_posix()
        for path in (master / "results" / tracker_id / "baseline").rglob("*") if path.is_file()}
    assert actual_files == set(files)
    return dict(status="sealed_VOT_trajectory_coverage_verified", tracker=tracker_id,
        sequences=127, anchors=1765, result_files=5295,
        total_trajectory_frames=sum(row["actual_frames"] for row in records),
        execution_source_files_checked=len(execution["source_sha256"]),
        manifest_sha256=sha(args.manifest), merge_sha256=sha(args.merge),
        execution_sha256=sha(args.execution), analysis_sha256=sha(args.analysis),
        trajectory_coverage_checked=True, merged_to_shard_hashes_exact=True,
        checkpoint_source_binding_checked=False, independent_completed_audit=False,
        metrics_recomputed=False, neural_inference=False,
        trajectories=records, auditor_source_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser()
    for name in ("workspace", "manifest", "merge", "execution", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = audit(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trajectories"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
