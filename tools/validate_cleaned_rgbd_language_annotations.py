#!/usr/bin/env python3
"""Validate standardized RGB-D-L language annotation JSONL files."""
import argparse
import json
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

ALLOWED_REL = {"closer_than_background", "farther_than_background", "similar_to_background", "depth_uncertain", "unknown"}
ALLOWED_OCC = {"none", "partial", "heavy", "unknown"}
ALLOWED_QUALITY = {"high", "medium", "low", "unknown"}
BBOX_RE = re.compile(r"\b(bounding box|bbox|red box|red bounding box|annotation box|annotation rectangle|red rectangle|selected object|marked object|target inside the box|object inside the box|coordinates|pixel location|pixel position)\b", re.I)
ABS_RE = re.compile(r"(^|[\s\"'])(([a-zA-Z]:\\)|(/home/)|(/mnt/)|(/data/))")


def validate_file(path):
    c = Counter()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            c["items"] += 1
            try:
                row = json.loads(line)
            except Exception:
                c["parse_error"] += 1
                continue
            lang = row.get("language", "")
            tt = row.get("target_tokens") or {}
            ct = row.get("context_tokens") or {}
            aq = row.get("annotation_quality") or {}
            blob = json.dumps(row, ensure_ascii=False)
            if not lang:
                c["empty_language"] += 1
            if not tt.get("category"):
                c["empty_category"] += 1
            if ct.get("depth_relation") not in ALLOWED_REL:
                c["bad_depth_relation"] += 1
            if ct.get("occlusion") not in ALLOWED_OCC:
                c["bad_occlusion"] += 1
            if row.get("depth_quality") not in ALLOWED_QUALITY:
                c["bad_depth_quality"] += 1
            if BBOX_RE.search(blob):
                c["bbox_leak"] += 1
            if ABS_RE.search(blob):
                c["absolute_path"] += 1
            if aq.get("is_valid") is not True:
                c["invalid_flag"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_dir", default="annotations_cleaned")
    ap.add_argument("--report", default="reports/rgbd_language_annotation_report.txt")
    args = ap.parse_args()
    root = Path.cwd()
    ann_dir = root / args.annotation_dir
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    total = Counter()
    for path in sorted(ann_dir.glob("*.jsonl")):
        c = validate_file(path)
        total.update(c)
        lines.append(f"{path.name}: {dict(c)}")
    lines.append(f"TOTAL: {dict(total)}")
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    return 1 if any(total.get(k, 0) for k in ["parse_error", "empty_language", "empty_category", "bad_depth_relation", "bad_occlusion", "bad_depth_quality", "bbox_leak", "absolute_path"]) else 0

if __name__ == "__main__":
    raise SystemExit(main())
