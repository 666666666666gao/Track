"""Verify the published review manifest and every linked board/video."""

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPECTED = {"depthtrack": (152, 152), "cdtb": (80, 80), "vot": (1765, 127)}


def main():
    for dataset, (cases, sequences) in EXPECTED.items():
        manifest = json.loads((ROOT / "data" / f"{dataset}.json").read_text(encoding="utf-8"))
        rows = manifest["rows"]
        assert manifest["dataset"] == dataset
        assert manifest["initialization_only"] is True
        assert manifest["count"] == len(rows) == cases
        assert manifest["sequence_count"] == sequences
        assert [r["number"] for r in rows] == list(range(1, cases + 1))
        assert len({r["id"] for r in rows}) == cases
        boards, details, videos = set(), set(), set()
        assert manifest["board_detail_size"] == [1920, 1130]
        assert manifest["video_width"] == 640 and manifest["video_sampling_fps"] == 8
        for row in rows:
            for kind, links in (("board", boards), ("board_detail", details), ("video", videos)):
                rel = Path(row[kind])
                assert not rel.is_absolute() and ".." not in rel.parts
                path = ROOT / rel
                assert path.is_file() and path.stat().st_size > 100
                links.add(rel.as_posix())
            with Image.open(ROOT / row["board_detail"]) as image:
                assert image.size == (1920, 1130)
                image.verify()
        assert len(boards) == len(details) == cases and len(videos) == sequences
        print(dataset, "cases", cases, "videos", sequences,
              "clear_boards", len(details),
              "bytes", sum((ROOT / p).stat().st_size for p in boards | details | videos))


if __name__ == "__main__":
    main()
