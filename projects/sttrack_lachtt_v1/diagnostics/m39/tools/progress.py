#!/usr/bin/env python3
"""Report M39 progress weighted by frozen per-anchor frame estimates."""

import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902")


def main():
    output = {}
    for arm in ("default", "no_update"):
        arm_root = ROOT / arm
        manifest = json.loads(
            (arm_root / "shard_manifest.json").read_text(encoding="utf-8"))
        arm_rows = []
        for shard in manifest["shards"]:
            result_root = (
                Path(shard["root"]) / "results" / manifest["tracker"] /
                "baseline")
            completed = []
            for item in shard["anchors"]:
                trajectory = "{}_{:08d}".format(
                    item["sequence"], item["index"])
                path = result_root / item["sequence"] / (trajectory + ".bin")
                if path.is_file() and path.stat().st_size > 0:
                    completed.append(item)
            arm_rows.append({
                "shard": shard["index"],
                "anchors_complete": len(completed),
                "anchors_total": shard["anchor_count"],
                "estimated_frames_complete": sum(
                    item["estimated_frames"] for item in completed),
                "estimated_frames_total": shard["estimated_frames"],
            })
        complete_frames = sum(
            row["estimated_frames_complete"] for row in arm_rows)
        total_frames = manifest["total_estimated_frames"]
        output[arm] = {
            "anchors_complete": sum(row["anchors_complete"] for row in arm_rows),
            "anchors_total": manifest["total_anchor_count"],
            "estimated_frames_complete": complete_frames,
            "estimated_frames_total": total_frames,
            "estimated_frame_fraction": complete_frames / total_frames,
            "shards": arm_rows,
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
