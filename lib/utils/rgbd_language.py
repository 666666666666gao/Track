"""Utilities for RGB-D language annotations.

The project can use either the compact JSONL files generated for DepthTrack/CDTB/
VOT-RGBD2022 or the RAG-style per-sequence directory layout:

  annotations_cleaned/<dataset>_language.jsonl
  annotations/<DatasetLang_Qwen3_RAGStyle>/<sequence>/{class,visible,depth,sequence}_description.txt

The loaders use sequence names as keys and never depend on absolute Windows paths
stored in the annotation metadata.
"""

import json
import os
import re
from collections import OrderedDict


RELATION_MAP = {
    'closer than background': 'closer_than_background',
    'closer_than_background': 'closer_than_background',
    'farther than background': 'farther_than_background',
    'farther_than_background': 'farther_than_background',
    'similar depth': 'similar_to_background',
    'similar_depth': 'similar_to_background',
    'similar to background': 'similar_to_background',
    'similar_to_background': 'similar_to_background',
    'uncertain': 'depth_uncertain',
    'depth_uncertain': 'depth_uncertain',
    'unknown': 'unknown',
}

QUALITY_MAP = {
    'high': 'high',
    'good': 'high',
    'reliable': 'high',
    'medium': 'medium',
    'moderate': 'medium',
    'normal': 'medium',
    'poor': 'low',
    'bad': 'low',
    'low': 'low',
    'unreliable': 'low',
    'unknown': 'unknown',
}

OCCLUSION_MAP = {
    'none': 'none',
    'no occlusion': 'none',
    'not_occluded': 'none',
    'not occluded': 'none',
    'no obvious occlusion': 'none',
    'partial': 'partial',
    'partly occluded': 'partial',
    'partial occlusion': 'partial',
    'heavy': 'heavy',
    'heavily occluded': 'heavy',
    'severe occlusion': 'heavy',
    'unknown': 'unknown',
}


def canonical_relation(value):
    key = str(value or '').strip().lower().replace('-', '_')
    return RELATION_MAP.get(key, key.replace(' ', '_') if key else 'unknown')


def canonical_quality(value):
    key = str(value or '').strip().lower().replace('-', '_')
    return QUALITY_MAP.get(key, QUALITY_MAP.get(key.replace('_', ' '), 'unknown'))


def canonical_occlusion(value):
    key = str(value or '').strip().lower().replace('-', '_')
    return OCCLUSION_MAP.get(key, OCCLUSION_MAP.get(key.replace('_', ' '), 'unknown'))


def clean_description(text):
    """Remove annotation artifacts that should not become target language."""
    text = str(text or '').strip()
    patterns = [
        r'\s*with a red bounding box around it\s*',
        r'\s*with a red bounding box\s*',
        r'\s*red bounding box around it\s*',
        r'\s*red bounding box\s*',
        r'\s*located in the center of the bounding box\s*',
        r'\s*in the center of the bounding box\s*',
        r'\s*within the bounding box\s*',
        r'\s*in the bounding box\s*',
    ]
    for pat in patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s+', ' ', text).strip(' ,.;')
    return text


def _read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return clean_description(f.read())


def _normalise_record(record):
    # New cleaned schema: annotations_cleaned/*_language.jsonl
    if 'target_tokens' in record or 'context_tokens' in record or 'language' in record:
        target = record.get('target_tokens') or {}
        context = record.get('context_tokens') or {}
        category = clean_description(target.get('category') or record.get('category_hint') or 'object')
        appearance = clean_description(target.get('appearance') or record.get('language') or '')
        description = clean_description(record.get('language') or appearance)
        depth_relation = canonical_relation(context.get('depth_relation'))
        depth_quality = canonical_quality(record.get('depth_quality'))
        occlusion = canonical_occlusion(context.get('occlusion'))
        distractors = context.get('distractors') or []
        if isinstance(distractors, (list, tuple)):
            distractor_text = '; '.join(clean_description(x) for x in distractors if clean_description(x))
        else:
            distractor_text = clean_description(distractors)
        meta = OrderedDict({
            'object_class_name': category,
            'language_category': category,
            'language_appearance': appearance,
            'language_attributes': target.get('attributes') or [],
            'language_context_background': context.get('background') or [],
            'language_depth_relation': depth_relation,
            'language_depth_quality': depth_quality,
            'language_occlusion_state': occlusion,
            'language_distractor_relation': distractor_text or 'none',
            'language_description': description,
            'language_target_tokens': target,
            'language_context_tokens': context,
            'language_source': 'cleaned_jsonl',
        })
        return meta

    # Legacy Qwen3 schema kept for backward compatibility.
    ann = record.get('annotation') or {}
    depth_stats = record.get('depth_stats') or {}
    category = clean_description(ann.get('category') or record.get('category_hint') or 'unknown')
    appearance = clean_description(ann.get('appearance') or record.get('final_description') or '')
    depth_relation = canonical_relation(ann.get('depth_relation') or depth_stats.get('foreground_relation'))
    depth_quality = canonical_quality(ann.get('depth_quality') or depth_stats.get('depth_quality'))
    final_description = clean_description(record.get('final_description') or ann.get('final_description') or appearance)

    return OrderedDict({
        'object_class_name': category,
        'language_category': category,
        'language_appearance': appearance,
        'language_depth_relation': depth_relation,
        'language_depth_quality': depth_quality,
        'language_occlusion_state': canonical_occlusion(ann.get('occlusion_state') or 'unknown'),
        'language_distractor_relation': clean_description(ann.get('distractor_relation') or 'unknown'),
        'language_motion_or_state': clean_description(ann.get('motion_or_state') or 'unknown'),
        'language_description': final_description,
        'language_source': 'legacy_jsonl',
    })


def read_jsonl_annotations(jsonl_path):
    annotations = OrderedDict()
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return annotations
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seq = record.get('sequence_name') or record.get('sequence')
            if not seq:
                continue
            annotations[seq] = _normalise_record(record)
    return annotations


def read_ragstyle_annotations(root):
    annotations = OrderedDict()
    if not root or not os.path.isdir(root):
        return annotations
    for seq in sorted(os.listdir(root)):
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue
        class_path = os.path.join(seq_dir, 'class.txt')
        visible_path = os.path.join(seq_dir, 'visible_description.txt')
        depth_path = os.path.join(seq_dir, 'depth_description.txt')
        sequence_path = os.path.join(seq_dir, 'sequence_description.txt')
        meta = OrderedDict()
        if os.path.isfile(class_path):
            meta['object_class_name'] = _read_text(class_path)
            meta['language_category'] = meta['object_class_name']
        if os.path.isfile(visible_path):
            meta['language_appearance'] = _read_text(visible_path)
        if os.path.isfile(depth_path):
            depth_text = _read_text(depth_path)
            parts = [p.strip() for p in depth_text.split(',')]
            meta['language_depth_relation'] = canonical_relation(parts[0] if parts else depth_text)
            meta['language_depth_quality'] = canonical_quality(parts[1] if len(parts) > 1 else '')
        if os.path.isfile(sequence_path):
            meta['language_description'] = _read_text(sequence_path)
        if meta:
            meta.setdefault('object_class_name', seq.split('_')[0])
            meta.setdefault('language_category', meta['object_class_name'])
            meta.setdefault('language_appearance', meta.get('language_description', ''))
            meta.setdefault('language_depth_relation', 'unknown')
            meta.setdefault('language_depth_quality', 'unknown')
            meta.setdefault('language_occlusion_state', 'unknown')
            meta.setdefault('language_distractor_relation', 'unknown')
            meta.setdefault('language_motion_or_state', 'unknown')
            meta['language_source'] = 'ragstyle'
            annotations[seq] = meta
    return annotations


def load_language_annotations(paths=None, rag_roots=None):
    """Load one or more language annotation sources.

    Later sources override earlier ones for the same sequence. This allows users
    to keep a generic RAG-style directory and then patch it with a corrected JSONL.
    """
    merged = OrderedDict()
    for path in paths or []:
        merged.update(read_jsonl_annotations(path))
    for root in rag_roots or []:
        merged.update(read_ragstyle_annotations(root))
    return merged


def sequence_list_from_jsonl(path):
    return list(read_jsonl_annotations(path).keys())
