"""Build matching public Train initialization review assets without future GT overlays."""

import argparse
import json
from pathlib import Path

from PIL import Image

from build_assets import make_board, make_video, sha256


def main(manifest_path, out):
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for number, item in enumerate(source["initializations"], 1):
        image = Path(item["source"])
        frames = sorted(image.parent.glob("*.jpg"))
        index = frames.index(image)
        x, y, w, h = item["protocol_bbox"]
        row = {
            "image": str(image), "image_size": list(Image.open(image).size),
            "target_xyxy": [x, y, x + w, y + h],
        }
        board = out / "media" / "depthtrack" / "boards" / f"{item['audit_id']}.jpg"
        make_board(row, frames, index, board)
        rows.append({
            "number": number, "id": item["audit_id"], "sequence": item["sequence"],
            "frame": image.stem, "frame_index": index, "sequence_frames": len(frames),
            "board": f"media/depthtrack/boards/{item['audit_id']}.jpg",
            "video": f"media/depthtrack/videos/{item['sequence']}.mp4",
            "video_start_seconds": round(index / 25, 2),
            "category": item["category"], "attributes": item["attributes"],
        })
        if number % 25 == 0:
            print("depthtrack boards", number, "/", len(source["initializations"]), flush=True)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "depthtrack.json").write_text(json.dumps({
        "dataset": "depthtrack", "count": len(rows), "sequence_count": len(rows),
        "source_manifest_sha256": sha256(manifest_path), "initialization_only": True,
        "review_fps": 25, "review_fps_is_a_playback_setting_not_measured_capture_rate": True,
        "rows": rows,
    }, ensure_ascii=False, separators=(",", ":")))
    for number, item in enumerate(source["initializations"], 1):
        folder = Path(item["source"]).parent
        make_video(folder, out / "media" / "depthtrack" / "videos" / f"{item['sequence']}.mp4")
        if number % 20 == 0:
            print("depthtrack videos", number, "/", len(source["initializations"]), flush=True)
    print("depthtrack complete", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.manifest, args.output)
