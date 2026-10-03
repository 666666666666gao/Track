"""Join existing Train review images to the exact M107 phrase slots.

The reviewer input excludes GT, candidate indices, sequence names and old labels.
Private sequence/index bindings stay in a separate executor-only file.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


FIELDS = ['audit_id', 'candidate', 'slot', 'query', 'label', 'evidence_note',
          'reviewer', 'human_confirmed']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--weak-labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'reviewer_input.json').exists(), 'Do not overwrite a prepared review.'
    source = base / 'candidate_review_24_v2'
    csv_path = source / 'blind_review_24.csv'
    mapping_path = base / 'sources/candidate_review_private_selection_v2.json'
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        templates = {row['audit_id']: row for row in csv.DictReader(handle)}
    weak = json.loads(args.weak_labels.read_text(encoding='utf-8'))
    initial = {row['sequence']: row for row in weak['initial']}
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))['rows']
    assert len(mapping) == len(templates) == 24
    records, bindings, rows = [], [], []
    for item in mapping:
        audit_id = item['audit_id']
        reference = initial[item['sequence']]
        assert reference['split'] == 'fit'
        phrases = reference['phrases']
        assert 1 <= len(phrases) <= 5
        template = templates[audit_id]
        images = {field: (source / template[field]).resolve() for field in
                  ['initial_full', 'initial_crop', 'current_with_A_B', 'A_crop', 'B_crop']}
        images['multiframe_sheet'] = base / f'multiframe_review/web_candidate/CAND_{audit_id}.jpg'
        images['unmarked_temporal_context'] = base / f'multiframe_review/candidate/{audit_id}_montage.jpg'
        assert all(path.is_file() for path in images.values())
        records.append(dict(audit_id=audit_id, phrases=phrases,
                            images={key: str(value) for key, value in images.items()}))
        bindings.append(dict(audit_id=audit_id, sequence=item['sequence'],
                             frame=item['frame'], a_index=item['a_index'], b_index=item['b_index'],
                             split='fit', phrases=phrases,
                             image_sha256={key: digest(value) for key, value in images.items()}))
        for candidate in ['A', 'B']:
            for slot, query in enumerate(phrases):
                rows.append(dict(audit_id=audit_id, candidate=candidate, slot=slot,
                                 query=query, label='', evidence_note='', reviewer='',
                                 human_confirmed='false'))
    instructions = (
        'Judge each exact query separately against the CURRENT A or B crop. '
        'supported means visible current RGB evidence supports the phrase; conflicting '
        'means clearly observable evidence is incompatible; unknown means blurred, '
        'occluded, outside the crop, or otherwise insufficient. Category agreement is '
        'not proof of the same physical instance. Both candidates may be supported. '
        'Initial images identify the phrase referent. Unmarked temporal context includes '
        'earlier and later Train frames and may explain the scene, but cannot turn an '
        'unobservable current attribute into supported/conflicting. Only RGB is shown; '
        'do not infer depth reliability. Slots are zero-based: category is slot0. '
        'Do not edit the query strings. Model labels '
        'remain weak supervision, with human confirmation pending.'
    )
    packet = dict(status='prepared_model_review_pending', scope='24 DepthTrack Train fitting events',
                  slot_index_base=0, instructions=instructions, cases=records)
    (out / 'reviewer_input.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'executor_private_binding.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out / 'current_phrase_review_blank.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for record in records:
        audit_id = record['audit_id']
        table = ''.join('<tr><td>' + html.escape(row['candidate']) + '</td><td>' +
                        str(row['slot']) + '</td><td>' + html.escape(row['query']) +
                        '</td><td>待核验</td><td></td></tr>' for row in rows if row['audit_id'] == audit_id)
        links = ' · '.join(f'<a href="{Path(path).as_uri()}">{html.escape(key)}</a>'
                           for key, path in record['images'].items() if key != 'multiframe_sheet')
        cards.append(f'<article id="case-{audit_id}"><h2>候选 {audit_id}</h2>'
                     f'<img loading="lazy" src="{Path(record["images"]["multiframe_sheet"]).as_uri()}" '
                     'alt="首帧、当前A/B与无标记多帧上下文"><p>' + links + '</p>'
                     '<table><thead><tr><th>候选</th><th>槽</th><th>实际短语</th><th>当前证据</th>'
                     '<th>理由</th></tr></thead><tbody>' + table + '</tbody></table></article>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<title>当前实际短语：24条候选审核</title><style>'
                'body{font:16px/1.55 sans-serif;max-width:1440px;margin:24px auto;padding:0 16px}'
                'img{max-width:100%;height:auto}article{margin:32px 0;border-top:1px solid #aaa}'
                'table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #ccc}'
                '</style><h1>当前模型实际短语：24条候选审核</h1>'
                '<p>按当前A/B可见RGB逐项填写支持、冲突或未知。首帧用于明确描述对象；'
                '前后多帧可帮助理解场景，但不能代替当前可见性。两候选都支持同一类别是允许的，'
                '类别支持不等于同一实例。此表待模型初审，人工确认尚未完成。</p>'
                '<p>填写文件：current_phrase_review_blank.csv。原24条审核表和既有标签保留。</p>' +
                ''.join(cards) + '</html>')
    (out / 'current_phrase_review.html').write_text(document, encoding='utf-8')
    receipt = dict(status='prepared_model_review_pending', cases=len(records), phrase_candidate_rows=len(rows),
                   fit_events=len(records), development_events=0, human_confirmed=False, slot_index_base=0,
                   input_sha256={'weak_labels': digest(args.weak_labels), 'blank_v2_csv': digest(csv_path),
                                 'private_candidate_binding': digest(mapping_path)},
                   visible_review='RGB only; current crops plus unmarked Train temporal context',
                   excluded_reviewer_fields=['GT', 'IoU', 'model_choice', 'candidate_index',
                                             'sequence_name', 'old_review_labels'],
                   instructions=instructions)
    (out / 'PREPARATION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
