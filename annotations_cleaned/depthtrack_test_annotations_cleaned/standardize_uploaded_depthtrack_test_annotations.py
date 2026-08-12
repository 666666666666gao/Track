#!/usr/bin/env python3
import json, re
from pathlib import Path
from collections import Counter

INPUT = Path('/mnt/data/new_upload_annotations/annotations/depthtrack_test_first_qwen3_corrected.jsonl')
OUTDIR = Path('/mnt/data/depthtrack_test_annotations_cleaned/annotations_cleaned')
REPORT = Path('/mnt/data/depthtrack_test_annotations_cleaned/reports/depthtrack_test_annotation_report.txt')
OUT = OUTDIR / 'depthtrack_test_language.jsonl'

DEPTH_REL_MAP = {
    'closer_than_background':'closer_than_background',
    'closer than background':'closer_than_background',
    'close':'closer_than_background',
    'closer':'closer_than_background',
    'farther_than_background':'farther_than_background',
    'farther than background':'farther_than_background',
    'far':'farther_than_background',
    'farther':'farther_than_background',
    'similar_depth':'similar_to_background',
    'same_depth':'similar_to_background',
    'similar_to_background':'similar_to_background',
    'similar depth':'similar_to_background',
    'uncertain':'depth_uncertain',
    'depth_uncertain':'depth_uncertain',
    'unknown':'unknown',
    '': 'unknown',
    None: 'unknown',
}
OCC_MAP = {
    'not_occluded':'none',
    'no occlusion':'none',
    'none':'none',
    'not occluded':'none',
    'partial':'partial',
    'partial occlusion':'partial',
    'partly occluded':'partial',
    'heavy':'heavy',
    'severe occlusion':'heavy',
    'heavily occluded':'heavy',
    'unknown':'unknown',
    '': 'unknown',
    None: 'unknown',
}
QUALITY_MAP = {
    'reliable':'high',
    'good':'high',
    'high':'high',
    'medium':'medium',
    'normal':'medium',
    'moderate':'medium',
    'poor':'low',
    'bad':'low',
    'low':'low',
    'noisy':'low',
    'unreliable':'low',
    'unknown':'unknown',
    '': 'unknown',
    None:'unknown',
}

LEAK_PATTERNS = [
    r'\bwithin the bounding box\b', r'\bin the bounding box\b', r'\binside the bounding box\b',
    r'\bbounding box\b', r'\bbbox\b', r'\bred bounding box\b', r'\bred box\b',
    r'\bannotation box\b', r'\bannotation rectangle\b', r'\bselected object\b', r'\bmarked object\b',
    r'\btarget inside the box\b', r'\bobject inside the box\b', r'\bcoordinates\b', r'\bpixel location\b'
]
LEAK_RE = re.compile('|'.join(LEAK_PATTERNS), re.IGNORECASE)
ABS_RE = re.compile(r'^[A-Za-z]:\\')

ALLOWED_REL = {'closer_than_background','farther_than_background','similar_to_background','depth_uncertain','unknown'}
ALLOWED_OCC = {'none','partial','heavy','unknown'}
ALLOWED_Q = {'high','medium','low','unknown'}

STOP_ATTR = {'a','an','the','with','and','or','possibly','object','target','shape','surface'}

def norm_relation(x):
    key = str(x or '').strip().lower().replace('-', '_')
    key = key.replace('_', ' ')
    if key in DEPTH_REL_MAP: return DEPTH_REL_MAP[key]
    key2 = str(x or '').strip().lower()
    return DEPTH_REL_MAP.get(key2, 'unknown')

def norm_occ(x):
    key = str(x or '').strip().lower().replace('-', '_')
    if key in OCC_MAP: return OCC_MAP[key]
    if key.startswith('partially_') or 'partial' in key: return 'partial'
    if 'heavy' in key or 'severe' in key: return 'heavy'
    if 'not' in key or 'no' in key: return 'none'
    return 'unknown'

def norm_quality(x):
    key = str(x or '').strip().lower().replace('-', '_')
    return QUALITY_MAP.get(key, QUALITY_MAP.get(key.replace('_',' '), 'unknown'))

def remove_paren_category(text, category):
    if not text: return ''
    if category:
        text = re.sub(r'\s*\(' + re.escape(category) + r'\)\s*', ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip(' ,.')

def clean_text(text):
    text = str(text or '')
    original = text
    # Specific rewrites before broad removal.
    text = re.sub(r'\bwithin the bounding box\b', '', text, flags=re.I)
    text = re.sub(r'\bin the bounding box\b', '', text, flags=re.I)
    text = re.sub(r'\binside the bounding box\b', '', text, flags=re.I)
    text = re.sub(r'\bwith a red bounding box\b', '', text, flags=re.I)
    text = re.sub(r'\bred bounding box\b', '', text, flags=re.I)
    text = re.sub(r'\bbounding box\b', '', text, flags=re.I)
    text = re.sub(r'\bbbox\b', '', text, flags=re.I)
    text = re.sub(r'\bred box\b', '', text, flags=re.I)
    text = re.sub(r'\bannotation (box|rectangle)\b', '', text, flags=re.I)
    text = re.sub(r'\bselected object\b', 'target', text, flags=re.I)
    text = re.sub(r'\bmarked object\b', 'target', text, flags=re.I)
    text = re.sub(r'\bby a yellow square\b', 'by a nearby object', text, flags=re.I)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip(' ,')
    # Keep sentence punctuation when present; add a period later if needed.
    return text, (original != text)

def extract_attrs(appearance):
    # Conservative lightweight extraction: split on commas and keep descriptive chunks.
    attrs=[]
    for part in re.split(r'[,;]', appearance or ''):
        p=part.strip(' .')
        if not p: continue
        # Avoid repeating full long noun phrase too much, keep up to 8 words.
        words=p.split()
        if len(words)>8: p=' '.join(words[:8])
        if p.lower() not in STOP_ATTR and p not in attrs:
            attrs.append(p)
    return attrs[:6]

def make_language(category, appearance, rel, occ, quality):
    app = appearance or f'visually identifiable {category if category != "object" else "target object"}'
    # Make relation readable.
    rel_phrase = {
        'closer_than_background':'closer than the surrounding background',
        'farther_than_background':'farther than the surrounding background',
        'similar_to_background':'at a similar depth to the surrounding background',
        'depth_uncertain':'with uncertain depth relation to the background',
        'unknown':'with unknown depth relation to the background',
    }[rel]
    occ_phrase = {
        'none':'no obvious occlusion',
        'partial':'partial occlusion',
        'heavy':'heavy occlusion',
        'unknown':'unknown occlusion status',
    }[occ]
    q_phrase = {'high':'high-quality depth', 'medium':'medium-quality depth', 'low':'low-quality depth', 'unknown':'unknown depth quality'}[quality]
    sent = f'A {app}, {rel_phrase}, with {q_phrase} and {occ_phrase}.'
    sent = re.sub(r'\bA a\b', 'A', sent)
    sent = re.sub(r'\bA an\b', 'An', sent)
    return sent

OUTDIR.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
summary=Counter()
leak_examples=[]
records=[]

with INPUT.open('r',encoding='utf-8-sig') as f:
    for line_no,line in enumerate(f,1):
        if not line.strip(): continue
        d=json.loads(line)
        ann=d.get('annotation') or {}
        seq=d.get('sequence') or d.get('sequence_name') or ''
        category=(ann.get('category') or 'object').strip() or 'object'
        appearance=(ann.get('appearance') or '').strip()
        appearance = remove_paren_category(appearance, category)
        appearance, changed_app = clean_text(appearance)
        rel=norm_relation(ann.get('depth_relation') or d.get('depth_stats',{}).get('foreground_relation'))
        occ=norm_occ(ann.get('occlusion_state'))
        quality=norm_quality(ann.get('depth_quality') or d.get('depth_stats',{}).get('depth_quality'))
        # Start from final_description but regenerate if it contains leak or is empty.
        final=ann.get('final_description') or d.get('final_description') or ''
        cleaned_final, changed_final = clean_text(final)
        if (not cleaned_final) or LEAK_RE.search(cleaned_final):
            cleaned_final = make_language(category, appearance, rel, occ, quality)
            cleaned_final, _ = clean_text(cleaned_final)
        # Clean distractors.
        distractor_raw=str(ann.get('distractor_relation') or '').strip()
        distractor_clean, changed_dis = clean_text(distractor_raw)
        distractors=[]
        if distractor_clean and not re.search(r'^(no\s+(significant\s+)?distractors?|none)$', distractor_clean, re.I):
            distractors=[distractor_clean]
        warnings=[]
        had_leak = any(LEAK_RE.search(str(x or '')) for x in [ann.get('final_description'), ann.get('appearance'), ann.get('distractor_relation'), d.get('final_description')])
        if had_leak:
            warnings.append('Removed or rewrote annotation leakage from source fields.')
            summary['source_bbox_leak'] += 1
            if len(leak_examples)<10: leak_examples.append(seq)
        if ABS_RE.match(str(d.get('rgb_path',''))) or ABS_RE.match(str(d.get('depth_path',''))):
            warnings.append('Ignored Windows absolute paths from source annotation.')
            summary['source_absolute_path'] += 1
        if changed_app or changed_final or changed_dis:
            summary['text_rewritten'] += 1
        # Map final values count.
        if rel not in ALLOWED_REL: warnings.append(f'invalid depth_relation normalized to unknown: {rel}'); rel='unknown'
        if occ not in ALLOWED_OCC: warnings.append(f'invalid occlusion normalized to unknown: {occ}'); occ='unknown'
        if quality not in ALLOWED_Q: warnings.append(f'invalid depth_quality normalized to unknown: {quality}'); quality='unknown'
        rec={
            'dataset':'depthtrack_test',
            'sequence_name':seq,
            'language':cleaned_final if cleaned_final.endswith(('.', '!', '?')) else cleaned_final + '.',
            'target_tokens':{
                'category':category,
                'appearance':appearance or f'visually identifiable {category}',
                'attributes':extract_attrs(appearance),
            },
            'context_tokens':{
                'background':['surrounding background'],
                'distractors':distractors,
                'depth_relation':rel,
                'occlusion':occ,
            },
            'depth_quality':quality,
            'annotation_quality':{
                'has_bbox_leak': bool(LEAK_RE.search(cleaned_final) or LEAK_RE.search(appearance) or any(LEAK_RE.search(x) for x in distractors)),
                'has_absolute_path': False,
                'is_valid': True,
                'warnings': warnings,
            }
        }
        # Final guard: if any leak remains, remove and warn.
        blob=json.dumps(rec,ensure_ascii=False)
        if LEAK_RE.search(blob):
            rec['annotation_quality']['has_bbox_leak']=True
            rec['annotation_quality']['is_valid']=False
        records.append(rec)
        summary['total'] += 1

with OUT.open('w',encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec,ensure_ascii=False)+"\n")

# Validate cleaned file.
valid=Counter(); vals=Counter()
for rec in records:
    blob=json.dumps(rec,ensure_ascii=False)
    if rec['language'].strip(): valid['nonempty_language'] += 1
    if rec['target_tokens']['category'].strip(): valid['nonempty_category'] += 1
    if LEAK_RE.search(blob): valid['cleaned_bbox_leak'] += 1
    if re.search(r'[A-Za-z]:\\', blob): valid['cleaned_absolute_path'] += 1
    if rec['context_tokens']['depth_relation'] not in ALLOWED_REL: valid['bad_depth_relation'] += 1
    if rec['context_tokens']['occlusion'] not in ALLOWED_OCC: valid['bad_occlusion'] += 1
    if rec['depth_quality'] not in ALLOWED_Q: valid['bad_depth_quality'] += 1

report=[]
report.append('DepthTrack test annotation standardization report')
report.append('='*52)
report.append(f'input: {INPUT}')
report.append(f'output: {OUT}')
report.append(f'total records: {summary["total"]}')
report.append(f'source records with bbox/annotation leakage: {summary["source_bbox_leak"]}')
report.append(f'source records with Windows absolute paths: {summary["source_absolute_path"]}')
report.append(f'records with rewritten text fields: {summary["text_rewritten"]}')
report.append('')
report.append('Cleaned validation:')
report.append(f'  empty language: {summary["total"]-valid["nonempty_language"]}')
report.append(f'  empty target category: {summary["total"]-valid["nonempty_category"]}')
report.append(f'  remaining bbox leak: {valid["cleaned_bbox_leak"]}')
report.append(f'  remaining absolute path: {valid["cleaned_absolute_path"]}')
report.append(f'  invalid depth_relation: {valid["bad_depth_relation"]}')
report.append(f'  invalid occlusion: {valid["bad_occlusion"]}')
report.append(f'  invalid depth_quality: {valid["bad_depth_quality"]}')
report.append('')
report.append('Leak example sequences from source: ' + ', '.join(leak_examples))
REPORT.write_text('\n'.join(report)+'\n', encoding='utf-8')
print('\n'.join(report))
