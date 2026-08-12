#!/usr/bin/env python3
"""Sanity check for RGB-D language annotations.

Usage:
  python tools/check_rgbd_language_annotations.py annotations/depthtrack_train_first_qwen3_corrected.jsonl
  python tools/check_rgbd_language_annotations.py annotations/CDTBLang_Qwen3_RAGStyle
"""
import json
import os
import sys
from collections import Counter

BAD_TERMS = ['bounding box', 'red bounding box', 'thermal', 'infrared', 'rgb-t', 'rgbt']
OK_REL = {'closer_than_background', 'farther_than_background', 'similar_to_background', 'uncertain', 'unknown'}
OK_QUALITY = {'reliable', 'medium', 'low', 'unknown'}


def check_jsonl(path):
    issues = Counter(); n = 0
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                row = json.loads(line)
            except Exception:
                issues['json_parse_error'] += 1
                continue
            ann = row.get('annotation') or {}
            text = (row.get('final_description') or ann.get('final_description') or '').lower()
            if not text:
                issues['empty_description'] += 1
            for term in BAD_TERMS:
                if term in text:
                    issues['bad_term:' + term] += 1
            if ann.get('depth_relation') not in OK_REL:
                issues['noncanonical_depth_relation'] += 1
            if ann.get('depth_quality') not in OK_QUALITY:
                issues['noncanonical_depth_quality'] += 1
    return n, issues


def check_rag_dir(root):
    issues = Counter(); n = 0
    for seq in sorted(os.listdir(root)):
        d = os.path.join(root, seq)
        if not os.path.isdir(d):
            continue
        n += 1
        for fn in ['class.txt', 'visible_description.txt', 'depth_description.txt', 'sequence_description.txt', 'metadata.json']:
            if not os.path.isfile(os.path.join(d, fn)):
                issues['missing:' + fn] += 1
        p = os.path.join(d, 'sequence_description.txt')
        if os.path.isfile(p):
            text = open(p, encoding='utf-8').read().lower()
            for term in BAD_TERMS:
                if term in text:
                    issues['bad_term:' + term] += 1
    return n, issues


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    if os.path.isdir(path):
        n, issues = check_rag_dir(path)
    else:
        n, issues = check_jsonl(path)
    print('checked:', path)
    print('items:', n)
    print('issues:', dict(issues))
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
