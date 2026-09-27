"""Compare fixed-state cosine choices for three initial-reference origins."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_train_states import overlaps, sha
from audit_visual_similarity import summarize


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--origins', type=Path, required=True)
    args = parser.parse_args()
    preparation = read(args.cache / 'preparation.json')
    assert preparation['training_labels_sha256'] == sha(args.cache / 'training_labels.json')
    labels = read(args.cache / 'training_labels.json')
    references = {}
    for shard in (0, 1):
        receipt = read(args.origins / f'shard{shard}.json')
        path = args.origins / f'shard{shard}.pt'
        assert receipt['status'] == 'complete' and not receipt['smoke']
        assert receipt['source_preparation_sha256'] == sha(args.cache / 'preparation.json')
        assert receipt['feature_sha256'] == sha(path)
        data = torch.load(path, map_location='cpu')
        assert not data['GT_loaded'] and not data['auxiliary_query_committed']
        assert data['records'] == receipt['records']
        for index, item in enumerate(data['records']):
            assert item['sequence'] not in references
            references[item['sequence']] = {
                'split': item['split'],
                't0_template': data['template_rois'][index].float().mean(dim=-2),
                't0_search': data['search_rois'][index].float().mean(dim=-2)}
    assert len(references) == 152
    groups = defaultdict(list)
    rows = []
    for shard in (0, 1):
        receipt = read(args.cache / f'collect_shard{shard}.json')
        assert receipt['status'] == 'complete' and not receipt['smoke']
        assert receipt['preparation_sha256'] == sha(args.cache / 'preparation.json')
        for item in receipt['sequences']:
            path = args.cache / 'features' / f"{item['sequence']}.pt"
            assert sha(path) == item['feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert not data['labels_loaded']
            reference = references[item['sequence']]
            assert reference['split'] == data['split']
            candidate = data['candidate_rois'].float().mean(dim=-2)
            variants = {
                'first_use_template': data['initial_rois'].float().mean(dim=-2),
                't0_template': reference['t0_template'],
                't0_search': reference['t0_search']}
            for variant, initial in variants.items():
                similarity = F.cosine_similarity(candidate, initial[None, None], dim=-1).mean(dim=-1)
                selected = similarity.argmax(dim=1)
                for index, frame in enumerate(data['event_frames']):
                    key = f"{item['sequence']}@{frame}"
                    label = labels[key]
                    assert label['split'] == data['split']
                    target = label['current']
                    row = dict(key=key, variant=variant, split=data['split'], gt_valid=target is not None)
                    if target is not None:
                        iou = overlaps(data['boxes'][index].float(),
                                       torch.tensor(target, dtype=torch.float32))
                        row.update(native_iou=float(iou[0]), visual_iou=float(iou[selected[index]]),
                                   top10_iou=float(iou.max()), visual_choice=int(selected[index]))
                    rows.append(row)
                    groups[f"{variant}:{data['split']}"].append(row)
                    for tag in label['strata']:
                        groups[f"{variant}:{data['split']}:{tag}"].append(row)
    assert len(rows) == 3 * len(labels) == 10506
    previous = read(args.cache / 'visual_similarity_diagnostic.json')
    for split in ('fit', 'development'):
        assert summarize(groups[f'first_use_template:{split}']) == previous['group'][split]
    report = dict(status='complete_train_only',
                  scope='Initial-reference origin cosine comparison on unchanged M90 Train candidates; no fitting or recursive actions.',
                  preparation_sha256=sha(args.cache / 'preparation.json'),
                  origins_receipt_sha256={str(shard): sha(args.origins / f'shard{shard}.json')
                                          for shard in (0, 1)},
                  group={name: summarize(values) for name, values in sorted(groups.items())},
                  rows=rows, no_public_evaluation=True, no_model_training=True)
    path = args.origins / 'cosine_comparison.json'
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value for name, value in report['group'].items()
                      if name.count(':') == 1}, indent=2))


if __name__ == '__main__':
    main()
