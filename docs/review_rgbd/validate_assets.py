"""Verify the published review manifest and every linked board/video."""

import json
from pathlib import Path


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
        boards, videos = set(), set()
        for row in rows:
            for kind, links in (("board", boards), ("video", videos)):
                rel = Path(row[kind])
                assert not rel.is_absolute() and ".." not in rel.parts
                path = ROOT / rel
                assert path.is_file() and path.stat().st_size > 100
                links.add(rel.as_posix())
        assert len(boards) == cases and len(videos) == sequences
        print(dataset, "cases", cases, "videos", sequences,
              "bytes", sum((ROOT / p).stat().st_size for p in boards | videos))


if __name__ == "__main__":
    main()
