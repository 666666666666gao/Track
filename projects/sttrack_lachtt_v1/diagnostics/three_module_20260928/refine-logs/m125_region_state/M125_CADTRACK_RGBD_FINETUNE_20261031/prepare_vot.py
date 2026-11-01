import hashlib
import json
import shutil
import argparse
from pathlib import Path


ROOT = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe")
SOURCE_MANIFEST = ROOT / "vot_shard_manifest.json"
DATA_ROOT = Path("/root/autodl-tmp/VOT-RGBD2022/sequences")
TRACKER = "cadtrack_rgbd_jet"
DEFAULT_CHECKPOINT = ROOT / "CADTrack_ep0020.pth.tar"
DEFAULT_RUN_NAME = "CADTrack_RGBD_JET_v1"
PYTHON = "/root/miniconda3/envs/mplt/bin/python"
VOT = "/root/miniconda3/envs/mplt/bin/vot"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def workspace_files(workspace, sequence_root, names, gpu, results_root, checkpoint):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    sequences = workspace / "sequences"
    sequences.mkdir()
    (sequences / "list.txt").write_text(
        "".join(name + "\n" for name in names), encoding="utf-8"
    )
    for name in names:
        (sequences / name).symlink_to(
            (Path(sequence_root) / name).resolve(), target_is_directory=True
        )
    (workspace / "config.yaml").write_text(
        "registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n",
        encoding="utf-8",
    )
    (workspace / "trackers.ini").write_text(
        "[{}]\n".format(TRACKER)
        + "label = CADTrack RGB-D JET\n"
        + "protocol = traxpython\n"
        + "command = cadtrack_vot\n"
        + "paths = {}\n".format(ROOT)
        + "python = {}\n".format(PYTHON)
        + "env_CADTRACK_CHECKPOINT = {}\n".format(checkpoint)
        + "env_CUDA_VISIBLE_DEVICES = {}\n".format(gpu)
        + "env_PYTHONDONTWRITEBYTECODE = 1\n"
        + "timeout = 300\nrestart = false\n",
        encoding="utf-8",
    )
    (workspace / "results").symlink_to(
        Path(results_root).resolve(), target_is_directory=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()

    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        source["total_anchor_count"] != 1765
        or len(source["sequences"]) != 127
        or len(source["shards"]) != 4
    ):
        raise ValueError("Frozen VOT manifest count mismatch")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = sha256(checkpoint)

    base = ROOT / "results" / args.run_name / "vot"
    if base.exists():
        raise FileExistsError(str(base))
    base.mkdir(parents=True)
    shared_results = base / "shared_results"
    shared_results.mkdir()
    run = base / "run"
    run.mkdir()

    shard_rows = []
    all_anchors = []
    for shard in source["shards"]:
        anchors = shard["anchors"]
        expected = shard["expected_trajectories"]
        if len(anchors) != shard["anchor_count"] or len(expected) != len(anchors):
            raise ValueError("Shard anchor manifest is internally inconsistent")
        names = sorted({row["sequence"] for row in anchors})
        workspace = run / "shard-{:02d}".format(shard["index"])
        workspace_files(
            workspace, Path(shard["root"]) / "sequences", names,
            shard["index"] % 2, shared_results, checkpoint,
        )
        shard_rows.append({
            "index": shard["index"],
            "gpu": shard["index"] % 2,
            "workspace": str(workspace),
            "sequences": len(names),
            "anchors": len(anchors),
            "expected_trajectories": expected,
            "source_anchor_manifest_sha256": sha256(
                Path(shard["root"]) / "sequences/list.txt"
            ),
        })
        all_anchors.extend(
            (row["sequence"], row["index"]) for row in anchors
        )

    if len(all_anchors) != 1765 or len(set(all_anchors)) != 1765:
        raise ValueError("Full VOT anchor coverage is incomplete or overlapping")
    master = base / "master"
    workspace_files(
        master, DATA_ROOT, source["sequences"], 0, shared_results, checkpoint
    )
    manifest = {
        "status": "prepared_CADTrack_RGBD_VOT2022_official_multistart",
        "protocol": "official_vot2022_rgbd_multistart",
        "tracker": TRACKER,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "run_name": args.run_name,
        "bridge": str(ROOT / "cadtrack_vot.py"),
        "bridge_sha256": sha256(ROOT / "cadtrack_vot.py"),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "master_workspace": str(master),
        "shared_results": str(shared_results),
        "sequences": 127,
        "anchors": 1765,
        "shard_result_files": 5295,
        "shards": shard_rows,
    }
    write(base / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
