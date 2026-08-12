#!/usr/bin/env python3
"""Convert existing Qwen3 RGB-D annotations into clean RGB-D-L training JSONL.

This script does not call any online API. It only standardizes existing annotations.
"""
import argparse
import json
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

DATASET_FILES = OrderedDict({
    "depthtrack_train": "depthtrack_train_first_qwen3_corrected.jsonl",
    "depthtrack_test": "depthtrack_test_first_qwen3_corrected.jsonl",
    "cdtb": "cdtb_first_qwen3_corrected.jsonl",
    "votrgbd2022": "vot_rgbd2022_first_qwen3_corrected.jsonl",
})

BBOX_LEAK_PATTERNS = [
    r"\bbounding\s+box\b", r"\bbbox\b", r"\bred\s+box\b", r"\bred\s+bounding\s+box\b",
    r"\bannotation\s+box\b", r"\bannotation\s+rectangle\b", r"\bred\s+rectangle\b",
    r"\bgreen\s+box\b", r"\byellow\s+box\b", r"\bselected\s+object\b", r"\bmarked\s+object\b",
    r"\btarget\s+inside\s+the\s+box\b", r"\bobject\s+inside\s+the\s+box\b",
    r"\binside\s+the\s+box\b", r"\binside\s+the\s+rectangle\b", r"\bframed\s+target\b",
    r"\bcoordinates?\b", r"\bpixel\s+location\b", r"\bpixel\s+position\b",
]

ALLOWED_REL = {"closer_than_background", "farther_than_background", "similar_to_background", "depth_uncertain", "unknown"}
ALLOWED_OCC = {"none", "partial", "heavy", "unknown"}
ALLOWED_QUALITY = {"high", "medium", "low", "unknown"}

REL_MAP = {
    "closer_than_background": "closer_than_background",
    "closer than background": "closer_than_background",
    "closer": "closer_than_background",
    "close": "closer_than_background",
    "farther_than_background": "farther_than_background",
    "farther than background": "farther_than_background",
    "farther": "farther_than_background",
    "far": "farther_than_background",
    "similar_to_background": "similar_to_background",
    "similar to background": "similar_to_background",
    "similar_depth": "similar_to_background",
    "similar depth": "similar_to_background",
    "same_depth": "similar_to_background",
    "same depth": "similar_to_background",
    "uncertain": "depth_uncertain",
    "depth_uncertain": "depth_uncertain",
    "unknown": "unknown",
}

OCC_MAP = {
    "none": "none", "no occlusion": "none", "not_occluded": "none", "not occluded": "none", "no obvious occlusion": "none",
    "partial": "partial", "partly occluded": "partial", "partial occlusion": "partial", "occluded": "partial",
    "heavy": "heavy", "severe occlusion": "heavy", "heavily occluded": "heavy",
    "unknown": "unknown",
}

QUALITY_MAP = {
    "high": "high", "good": "high", "reliable": "high",
    "medium": "medium", "moderate": "medium", "normal": "medium",
    "low": "low", "poor": "low", "bad": "low", "noisy": "low", "unreliable": "low",
    "unknown": "unknown",
}

ABS_PATH_RE = re.compile(r"(^|[\s\"'])(([a-zA-Z]:\\)|(/home/)|(/mnt/)|(/data/))")


def clean_text(text):
    text = str(text or "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    replacements = [
        (r"\s*in\s+the\s+bounding\s+box\s*", " "),
        (r"\s*inside\s+the\s+bounding\s+box\s*", " "),
        (r"\s*inside\s+the\s+box\s*", " "),
        (r"\s*within\s+the\s+box\s*", " "),
        (r"\s*red\s+bounding\s+box\s*", " "),
        (r"\s*bounding\s+box\s*", " "),
        (r"\s*bbox\s*", " "),
        (r"\s*annotation\s+rectangle\s*", " "),
        (r"\s*annotation\s+box\s*", " "),
        (r"\s*selected\s+object\s*", "target object"),
        (r"\s*marked\s+object\s*", "target object"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.I)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text


def has_bbox_leak(text):
    low = str(text or "").lower()
    return any(re.search(p, low, flags=re.I) for p in BBOX_LEAK_PATTERNS)


def canonical(value, mapping, allowed, default="unknown"):
    key = str(value or "").strip().lower().replace("-", "_")
    key = key.replace("_", " ") if key not in mapping else key
    out = mapping.get(key, mapping.get(key.replace(" ", "_"), default))
    return out if out in allowed else default


def split_attributes(appearance, category):
    text = clean_text(appearance)
    text = re.sub(r"\([^)]*\)", "", text).strip()
    attrs = []
    for part in re.split(r",|\band\b|\bwith\b", text):
        part = clean_text(part)
        if not part:
            continue
        if category and part.lower() == category.lower():
            continue
        if len(part.split()) <= 6:
            attrs.append(part)
    dedup = []
    for a in attrs:
        if a and a.lower() not in {x.lower() for x in dedup}:
            dedup.append(a)
    return dedup[:6]


def relation_phrase(rel):
    return {
        "closer_than_background": "closer than the surrounding background",
        "farther_than_background": "farther than the surrounding background",
        "similar_to_background": "at a similar depth to the surrounding background",
        "depth_uncertain": "with uncertain depth relation",
        "unknown": "with unknown depth relation",
    }[rel]


def quality_phrase(q):
    return {
        "high": "high-quality depth",
        "medium": "medium-quality depth",
        "low": "low-quality depth",
        "unknown": "unknown depth quality",
    }[q]


def occ_phrase(occ):
    return {
        "none": "no obvious occlusion",
        "partial": "partial occlusion",
        "heavy": "heavy occlusion",
        "unknown": "unknown occlusion state",
    }[occ]


def normalize_record(row, dataset_slug):
    ann = row.get("annotation") or {}
    stats = row.get("depth_stats") or {}
    seq = row.get("sequence") or row.get("sequence_name") or "unknown_sequence"
    category = clean_text(ann.get("category") or row.get("category_hint") or "object")
    if not category or category.lower() in {"unknown", "unknown object"}:
        category = clean_text(row.get("category_hint") or "object")
    appearance = clean_text(ann.get("appearance") or row.get("final_description") or f"visually identifiable {category}")
    appearance = re.sub(r"\s*\([^)]*\)\s*", " ", appearance).strip(" ,.;")
    if not appearance:
        appearance = f"visually identifiable {category}"

    rel = canonical(ann.get("depth_relation") or stats.get("foreground_relation"), REL_MAP, ALLOWED_REL)
    occ = canonical(ann.get("occlusion_state") or ann.get("occlusion"), OCC_MAP, ALLOWED_OCC)
    quality = canonical(ann.get("depth_quality") or stats.get("depth_quality"), QUALITY_MAP, ALLOWED_QUALITY)

    distractor_raw = clean_text(ann.get("distractor_relation") or "")
    distractors = []
    if distractor_raw and distractor_raw.lower() not in {"none", "no significant distractors", "unknown", "no significant distractors in"}:
        distractors = [distractor_raw]

    # Compose a stable, leakage-free sequence-level language sentence.
    # Use a simple grammar pattern to avoid awkward phrases like "A adapter".
    cat_for_sentence = category if category != "object" else "target object"
    language = (
        f"The target category is {cat_for_sentence}. "
        f"It appears as {appearance}; it is {relation_phrase(rel)}, "
        f"with {quality_phrase(quality)} and {occ_phrase(occ)}."
    )
    language = clean_text(language)
    language = re.sub(r"\bwith with\b", "with", language, flags=re.I)
    language = re.sub(r"\s+", " ", language).strip()
    if language and language[-1] not in ".!?":
        language += "."

    text_blob = " ".join([language, category, appearance, distractor_raw])
    warnings = []
    if has_bbox_leak(text_blob):
        warnings.append("bbox_leak_removed_or_flagged")
    if ABS_PATH_RE.search(json.dumps(row)):
        warnings.append("source_record_contains_absolute_path_but_clean_record_omits_it")

    clean_row = OrderedDict({
        "dataset": dataset_slug,
        "sequence_name": seq,
        "language": language,
        "target_tokens": OrderedDict({
            "category": category,
            "appearance": appearance,
            "attributes": split_attributes(appearance, category),
        }),
        "context_tokens": OrderedDict({
            "background": ["surrounding background"],
            "distractors": distractors,
            "depth_relation": rel,
            "occlusion": occ,
        }),
        "depth_quality": quality,
        "annotation_quality": OrderedDict({
            "has_bbox_leak": has_bbox_leak(text_blob),
            "has_absolute_path": False,
            "is_valid": bool(language and category and rel in ALLOWED_REL and occ in ALLOWED_OCC and quality in ALLOWED_QUALITY),
            "warnings": warnings,
        }),
    })
    return clean_row


def process_file(src, dst, dataset_slug):
    stats = Counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            stats["input"] += 1
            try:
                row = json.loads(line)
            except Exception:
                stats["parse_error"] += 1
                continue
            old_text = json.dumps(row, ensure_ascii=False)
            if has_bbox_leak(old_text):
                stats["source_bbox_leak"] += 1
            if ABS_PATH_RE.search(old_text):
                stats["source_absolute_path"] += 1
            clean_row = normalize_record(row, dataset_slug)
            if clean_row["annotation_quality"]["has_bbox_leak"]:
                stats["remaining_bbox_leak"] += 1
            fout.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            stats["output"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="annotations")
    ap.add_argument("--output_dir", default="annotations_cleaned")
    ap.add_argument("--dataset", default="all", choices=list(DATASET_FILES) + ["all"])
    args = ap.parse_args()

    root = Path.cwd()
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    wanted = DATASET_FILES.keys() if args.dataset == "all" else [args.dataset]
    total = Counter()
    for ds in wanted:
        src = input_dir / DATASET_FILES[ds]
        if not src.is_file():
            print(f"skip missing: {src}")
            continue
        dst = output_dir / f"{ds}_language.jsonl"
        stats = process_file(src, dst, ds)
        total.update(stats)
        print(f"{ds}: {dict(stats)} -> {dst}")
    print("total:", dict(total))

if __name__ == "__main__":
    main()
