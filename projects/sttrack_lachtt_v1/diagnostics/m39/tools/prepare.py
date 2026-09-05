#!/usr/bin/env python3
"""Materialize exact copies of the frozen low22 VOT shards for M39."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


FROZEN_MANIFEST = Path(
    "/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/"
    "shard_manifest.json")
EXPECTED_MANIFEST_SHA256 = (
    "600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed")
WRAPPER_ROOT = Path(
    "/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_"
    "v1_20260902/wrappers")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracker_text(tracker, module, gpu):
    return (
        "[{0}]\n"
        "label = {0}\n"
        "protocol = traxpython\n"
        "command = {1}\n"
        "paths = {2}\n"
        "python = /root/autodl-tmp/envs/sttrack/bin/python\n"
        "env_CUDA_VISIBLE_DEVICES = {3}\n"
        "env_PYTHONPATH = {2}\n"
        "env_TOKENIZERS_PARALLELISM = false\n"
        "env_PYTHONDONTWRITEBYTECODE = 1\n"
        "timeout = 600\n"
        "restart = false\n"
    ).format(tracker, module, WRAPPER_ROOT, gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if sha256_file(FROZEN_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen low22 manifest SHA differs")
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(str(root))
    root.mkdir(parents=True)

    source_shards = frozen["shards"]
    if args.smoke:
        shortest = min(
            (anchor["estimated_frames"], anchor["sequence"],
             anchor["index"], shard, anchor)
            for shard in source_shards for anchor in shard["anchors"])
        source_shards = [shortest[3]]
        selected_names = {
            "{}_{:08d}".format(shortest[4]["sequence"], shortest[4]["index"])}
    else:
        selected_names = None

    records = []
    for new_index, source in enumerate(source_shards):
        source_root = Path(source["root"])
        destination = root / "shard-{:02d}".format(new_index)
        shutil.copytree(source_root / "sequences", destination / "sequences",
                        symlinks=True)
        shutil.copy2(source_root / "config.yaml", destination / "config.yaml")
        (destination / "trackers.ini").write_text(
            tracker_text(args.tracker, args.module, args.gpu), encoding="utf-8")

        anchors = source["anchors"]
        trajectories = source["expected_trajectories"]
        if selected_names is not None:
            anchors = [
                item for item in anchors
                if "{}_{:08d}".format(item["sequence"], item["index"])
                in selected_names]
            trajectories = sorted(selected_names)
            keep = {item["sequence"] for item in anchors}
            for sequence_path in list((destination / "sequences").iterdir()):
                if sequence_path.name != "list.txt" and sequence_path.name not in keep:
                    if sequence_path.is_symlink():
                        sequence_path.unlink()
                    else:
                        shutil.rmtree(sequence_path)
            (destination / "sequences/list.txt").write_text(
                "".join(name + "\n" for name in frozen["sequences"] if name in keep),
                encoding="utf-8")
            for item in anchors:
                anchor_path = destination / "sequences" / item["sequence"] / "anchor.value"
                values = anchor_path.read_text(encoding="utf-8").splitlines()
                values = ["0"] * len(values)
                values[item["index"]] = str(item["value"])
                anchor_path.write_text("\n".join(values) + "\n", encoding="utf-8")

        records.append({
            **source,
            "index": new_index,
            "gpu": args.gpu,
            "root": str(destination),
            "anchors": anchors,
            "expected_trajectories": trajectories,
            "anchor_count": len(anchors),
            "estimated_frames": sum(item["estimated_frames"] for item in anchors),
            "config_sha256": sha256_file(destination / "config.yaml"),
            "trackers_sha256": sha256_file(destination / "trackers.ini"),
            "list_sha256": sha256_file(destination / "sequences/list.txt"),
        })

    sequences = []
    for name in frozen["sequences"]:
        if any(name == item["sequence"] for shard in records for item in shard["anchors"]):
            sequences.append(name)
    manifest = {
        **frozen,
        "schema": "sttrack_m39_vot_low22_template_ablation_shards_v1",
        "tracker": args.tracker,
        "sequences": sequences,
        "shard_count": len(records),
        "gpu_count": 1,
        "total_anchor_count": sum(item["anchor_count"] for item in records),
        "total_estimated_frames": sum(item["estimated_frames"] for item in records),
        "shards": records,
        "m39": {
            "arm": "no_update" if "no_update" in args.tracker else "default",
            "module": args.module,
            "source_manifest": str(FROZEN_MANIFEST),
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "low22_only": True,
            "automatic_full127_launch": False,
        },
    }
    output = root / "shard_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "root": str(root),
        "tracker": args.tracker,
        "anchor_count": manifest["total_anchor_count"],
        "estimated_frames": manifest["total_estimated_frames"],
        "manifest_sha256": sha256_file(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
