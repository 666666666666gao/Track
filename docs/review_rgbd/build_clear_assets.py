"""Rebuild clearer three-dataset review media from original frames, preserving case IDs."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from build_assets import CAPTION_ROOTS, make_board, make_video, records_for, sha256


VERSION = "20261003"


def build(dataset, output, baseline_path, source_manifest=None, source_text_manifest=None):
    started = time.time()
    manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
    if dataset == "depthtrack":
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        assert manifest["source_manifest_sha256"] == sha256(source_manifest)
        text_rows = json.loads(source_text_manifest.read_text(encoding="utf-8"))
        text_by_sequence = {row["sequence"]: row for row in text_rows}
        plan = {"rows": []}
        records = {}
        for item in source["initializations"]:
            image = Path(item["source"])
            x, y, width, height = item["protocol_bbox"]
            image_hash = text_by_sequence[item["sequence"]]["first_image_sha256"]
            plan["rows"].append({"key": item["audit_id"], "image": str(image),
                                 "image_size": list(Image.open(image).size),
                                 "target_xyxy": [x, y, x + width, y + height],
                                 "image_sha256": image_hash})
            records[item["audit_id"]] = {"image_sha256": image_hash,
                                         "category": item["category"], "attributes": item["attributes"]}
        source_hashes = {"source_manifest_sha256": sha256(source_manifest),
                         "source_text_manifest_sha256": sha256(source_text_manifest)}
    else:
        plan, records, plan_hash, records_hash = records_for(CAPTION_ROOTS[dataset])
        assert manifest["caption_plan_sha256"] == plan_hash
        assert manifest["caption_records_sha256"] == records_hash
        source_hashes = {"caption_plan_sha256": plan_hash, "caption_records_sha256": records_hash}
    assert [row["id"] for row in manifest["rows"]] == [row["key"] for row in plan["rows"]]
    sequences = {}
    for number, (case, row) in enumerate(zip(manifest["rows"], plan["rows"]), 1):
        image = Path(row["image"])
        assert sha256(image) == row["image_sha256"] == records[row["key"]]["image_sha256"]
        assert case["category"] == records[row["key"]]["category"]
        assert case["attributes"] == records[row["key"]]["attributes"]
        sequence = image.parent.parent.name
        if sequence not in sequences:
            sequences[sequence] = sorted(image.parent.glob("*.jpg"))
        frames = sequences[sequence]
        index = frames.index(image)
        assert case["sequence"] == sequence and case["frame_index"] == index
        assert case["sequence_frames"] == len(frames)
        case["board_detail"] = f"media/{dataset}/details_{VERSION}/{case['id']}.jpg"
        case["video"] = f"media/{dataset}/videos_{VERSION}/{sequence}.mp4"
        make_board(row, frames, index, output / case["board_detail"], scale=2, quality=88)
        if number % 100 == 0 or number == manifest["count"]:
            print(dataset, "clear_boards", number, "/", manifest["count"],
                  "seconds", round(time.time() - started, 1), flush=True)
    for number, (sequence, frames) in enumerate(sequences.items(), 1):
        make_video(frames[0].parent,
                   output / f"media/{dataset}/videos_{VERSION}/{sequence}.mp4",
                   width=640, sampling_fps=8, crf=28)
        print(dataset, "clear_videos", number, "/", len(sequences), sequence,
              "seconds", round(time.time() - started, 1), flush=True)
    manifest.update({"clear_media_version": VERSION, "board_detail_size": [1920, 1130],
                     "board_detail_jpeg_quality": 88, "video_width": 640,
                     "video_sampling_fps": 8, "video_crf": 28})
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "data" / f"{dataset}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    paths = [output / case["board_detail"] for case in manifest["rows"]]
    paths += list((output / f"media/{dataset}/videos_{VERSION}").glob("*.mp4"))
    receipt = {"dataset": dataset, "status": "complete", "cases": len(manifest["rows"]),
               "videos": len(sequences), "source_initialization_hashes_verified": len(plan["rows"]),
               "source_hashes": source_hashes,
               "baseline_manifest_sha256": sha256(baseline_path),
               "clear_manifest_sha256": sha256(output / "data" / f"{dataset}.json"),
               "media_bytes": sum(path.stat().st_size for path in paths),
               "elapsed_seconds": round(time.time() - started, 2)}
    (output / f"{dataset}_build_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("depthtrack", *CAPTION_ROOTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-text-manifest", type=Path)
    args = parser.parse_args()
    build(args.dataset, args.output, args.baseline_manifest, args.source_manifest, args.source_text_manifest)
