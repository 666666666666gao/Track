"""Build public, initialization-only RGB-D review evidence from frozen caption plans."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


CAPTION_ROOTS = {
    "cdtb": Path("/root/autodl-tmp/sttrack_full152_evaluation_20260925/inputs_bfloat16/cdtb/captions"),
    "vot": Path("/root/autodl-tmp/sttrack_full152_evaluation_20260925/vot_inputs/all_initializations"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_for(root):
    plan_path, record_path = root / "plan.json", root / "records.jsonl"
    plan = json.loads(plan_path.read_text())
    generation = json.loads((root / "generation_result.json").read_text())
    assert generation["records_sha256"] == sha256(record_path)
    records = {r["key"]: r for r in map(json.loads, record_path.read_text().splitlines())}
    assert len(records) == len(plan["rows"])
    return plan, records, sha256(plan_path), sha256(record_path)


def tile(canvas, source, box, title):
    x, y, w, h = box
    fitted = ImageOps.contain(source, (w, h - 20))
    canvas.paste(fitted, (x + (w - fitted.width) // 2, y + 20 + (h - 20 - fitted.height) // 2))
    ImageDraw.Draw(canvas).text((x + 5, y + 3), title, fill="white")


def make_board(row, frames, index, output):
    original = Image.open(row["image"]).convert("RGB")
    assert len(original.size) == 2 and list(original.size) == row["image_size"]
    x1, y1, x2, y2 = row["target_xyxy"]
    marked = original.copy()
    ImageDraw.Draw(marked).rectangle((x1, y1, x2, y2), outline="red", width=5)
    margin = max(x2 - x1, y2 - y1) * 0.65
    crop = original.crop((max(0, x1 - margin), max(0, y1 - margin),
                          min(original.width, x2 + margin), min(original.height, y2 + margin)))
    board = Image.new("RGB", (960, 565), "#151923")
    tile(board, marked, (0, 0, 640, 380), "INITIAL FRAME - RED BOX IS THE TARGET")
    tile(board, crop, (640, 0, 320, 380), "INITIAL TARGET ENLARGED")
    offsets = (-20, -5, 5, 20)
    for k, offset in enumerate(offsets):
        j = min(len(frames) - 1, max(0, index + offset))
        image = Image.open(frames[j]).convert("RGB")
        tile(board, image, (k * 240, 380, 240, 185), f"CONTEXT FRAME {frames[j].stem} (NO BOX)")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, "JPEG", quality=72, optimize=True)


def make_video(folder, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-framerate", "25", "-pattern_type", "glob", "-i", str(folder / "*.jpg"),
               "-vf", "fps=5,scale=320:-2", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "36", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-threads", "2", str(output)]
    subprocess.run(command, check=True)


def build(dataset, out, videos):
    plan, records, plan_hash, records_hash = records_for(CAPTION_ROOTS[dataset])
    media = out / "media" / dataset
    rows = []
    sequences = {}
    for position, row in enumerate(plan["rows"], 1):
        image = Path(row["image"])
        sequence = image.parent.parent.name
        frames = sequences.get(sequence)
        if frames is None:
            frames = sorted(image.parent.glob("*.jpg"))
            sequences[sequence] = frames
        index = frames.index(image)
        record = records[row["key"]]
        assert record["image_sha256"] == row["image_sha256"]
        board = media / "boards" / f"{row['key']}.jpg"
        make_board(row, frames, index, board)
        rows.append({
            "number": position, "id": row["key"], "sequence": sequence,
            "frame": image.stem, "frame_index": index, "sequence_frames": len(frames),
            "board": f"media/{dataset}/boards/{row['key']}.jpg",
            "video": f"media/{dataset}/videos/{sequence}.mp4",
            "video_start_seconds": round(index / 25, 2),
            "category": record["category"], "attributes": record["attributes"],
        })
        if position % 100 == 0:
            print(dataset, "boards", position, "/", len(plan["rows"]), flush=True)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{dataset}.json").write_text(json.dumps({
        "dataset": dataset, "count": len(rows), "sequence_count": len(sequences),
        "caption_plan_sha256": plan_hash, "caption_records_sha256": records_hash,
        "initialization_only": True, "review_fps": 25,
        "review_fps_is_a_playback_setting_not_measured_capture_rate": True,
        "rows": rows,
    }, ensure_ascii=False, separators=(",", ":")))
    if videos:
        for i, (sequence, frames) in enumerate(sequences.items(), 1):
            make_video(frames[0].parent, media / "videos" / f"{sequence}.mp4")
            print(dataset, "videos", i, "/", len(sequences), sequence, flush=True)
    print(dataset, "complete", len(rows), "anchors", len(sequences), "videos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(CAPTION_ROOTS), required=True)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    build(args.dataset, args.output, args.videos)
