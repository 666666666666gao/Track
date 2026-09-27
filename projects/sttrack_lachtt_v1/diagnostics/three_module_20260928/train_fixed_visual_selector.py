"""Train-only fixed-state Top-10 visual selector control on native STTrack states."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analyze_train_states import overlaps, sha


def read(path):
    return json.loads(Path(path).read_text())


def load_panel(root, initial_search=None):
    preparation = read(root / 'preparation.json')
    assert sha(root / 'training_labels.json') == preparation['training_labels_sha256']
    labels = read(root / 'training_labels.json')
    result = {split: defaultdict(list) for split in ('fit', 'development')}
    for shard in (0, 1):
        receipt = read(root / f'collect_shard{shard}.json')
        assert receipt['status'] == 'complete' and not receipt['smoke']
        assert receipt['preparation_sha256'] == sha(root / 'preparation.json')
        for item in receipt['sequences']:
            path = root / 'features' / f"{item['sequence']}.pt"
            assert sha(path) == item['feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert not data['labels_loaded'] and data['split'] in result
            split = result[data['split']]
            candidate = data['candidate_rois'].float().mean(dim=-2)
            initial = data['initial_rois'].float().mean(dim=-2)
            if initial_search is not None:
                initial = initial_search[item['sequence']].float().mean(dim=-2)
            score = data['scores'].float()
            rank = torch.arange(10, dtype=torch.float32)[None, :, None].expand(len(score), -1, -1) / 9
            cell = torch.stack((data['candidate_cells'] % 16,
                                data['candidate_cells'] // 16), dim=-1).float() / 15
            box = data['boxes'].float()
            prior = data['prior_bbox'].float()[:, None, :]
            box_center = box[..., :2] + box[..., 2:] / 2
            prior_center = prior[..., :2] + prior[..., 2:] / 2
            displacement = (box_center - prior_center) / prior[..., 2:].clamp_min(1)
            log_size = torch.log(box[..., 2:].clamp_min(1) / prior[..., 2:].clamp_min(1))
            geometry = torch.cat((score[..., None],
                                  (score - score[:, :1])[..., None], rank,
                                  data['geometry'].float(), cell, displacement,
                                  log_size), dim=-1)
            assert geometry.shape == (len(score), 10, 13)
            for index, frame in enumerate(data['event_frames']):
                key = f"{item['sequence']}@{frame}"
                label = labels[key]
                assert label['split'] == data['split']
                target = label['current']
                if target is None:
                    continue
                iou = overlaps(box[index], torch.tensor(target, dtype=torch.float32))
                split['key'].append(key)
                split['strata'].append(label['strata'])
                split['candidate'].append(candidate[index])
                split['initial'].append(initial)
                split['geometry'].append(geometry[index])
                split['score'].append(score[index])
                split['iou'].append(iou)
    panel = {}
    for name, fields in result.items():
        panel[name] = {key: torch.stack(value) if key not in ('key', 'strata') else value
                       for key, value in fields.items()}
    assert len(panel['fit']['key']) == 2544
    assert len(panel['development']['key']) == 495
    return panel, preparation


def load_initial_search(origins, root):
    references = {}
    for shard in (0, 1):
        receipt = read(origins / f'shard{shard}.json')
        path = origins / f'shard{shard}.pt'
        assert receipt['status'] == 'complete' and not receipt['smoke']
        assert receipt['source_preparation_sha256'] == sha(root / 'preparation.json')
        assert receipt['feature_sha256'] == sha(path)
        data = torch.load(path, map_location='cpu')
        assert not data['GT_loaded'] and not data['auxiliary_query_committed']
        for index, record in enumerate(data['records']):
            assert record['sequence'] not in references
            references[record['sequence']] = data['search_rois'][index]
    assert len(references) == 152
    return references


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(6144, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(77, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, candidate, initial, geometry, visual_enabled):
        initial = initial[:, None].expand_as(candidate)
        if visual_enabled:
            candidate = F.layer_norm(candidate, (768,))
            initial = F.layer_norm(initial, (768,))
            visual = torch.cat((candidate, initial, (candidate - initial).abs(),
                                candidate * initial), dim=-1).flatten(-2)
            visual = self.visual(visual)
        else:
            visual = geometry.new_zeros((*geometry.shape[:2], 64))
        residual = self.head(torch.cat((geometry, visual), dim=-1)).squeeze(-1)
        return geometry[..., 0] + residual


def evaluate(model, panel, visual_enabled, device):
    model.eval()
    grouped = defaultdict(list)
    rows = []
    with torch.no_grad():
        for start in range(0, len(panel['key']), 64):
            end = min(start + 64, len(panel['key']))
            score = model(panel['candidate'][start:end].to(device),
                          panel['initial'][start:end].to(device),
                          panel['geometry'][start:end].to(device), visual_enabled)
            selected = score.argmax(dim=1).cpu()
            for offset, index in enumerate(range(start, end)):
                iou = panel['iou'][index]
                row = dict(key=panel['key'][index], native_iou=float(iou[0]),
                           selected_iou=float(iou[selected[offset]]),
                           oracle_iou=float(iou.max()), selected=int(selected[offset]))
                rows.append(row)
                grouped['all'].append(row)
                for tag in panel['strata'][index]:
                    grouped[tag].append(row)
    metrics = {}
    for tag, values in grouped.items():
        metrics[tag] = dict(valid_gt=len(values),
                            native_iou50=sum(row['native_iou'] >= .5 for row in values),
                            selected_iou50=sum(row['selected_iou'] >= .5 for row in values),
                            oracle_iou50=sum(row['oracle_iou'] >= .5 for row in values),
                            rescues=sum(row['native_iou'] < .5 and row['selected_iou'] >= .5 for row in values),
                            breaks=sum(row['native_iou'] >= .5 and row['selected_iou'] < .5 for row in values),
                            native_mean_iou=sum(row['native_iou'] for row in values) / len(values),
                            selected_mean_iou=sum(row['selected_iou'] for row in values) / len(values))
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('visual', 'geometry'), required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--device', required=True)
    parser.add_argument('--initial-origin', choices=('first_use_template', 't0_search'),
                        default='first_use_template')
    parser.add_argument('--origins', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    initial_search = None
    if args.initial_origin == 't0_search':
        assert args.variant == 'visual' and args.origins is not None
        initial_search = load_initial_search(args.origins, args.root)
    panel, preparation = load_panel(args.root, initial_search)
    device = torch.device(args.device)
    model = Selector().to(device)
    visual_enabled = args.variant == 'visual'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    initial_dev, _ = evaluate(model, panel['development'], visual_enabled, device)
    assert initial_dev['all']['selected_iou50'] == initial_dev['all']['native_iou50']
    history = []
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        permutation = torch.randperm(len(panel['fit']['key']), generator=torch.Generator().manual_seed(2027 + epoch))
        losses = []
        for indices in permutation.split(args.batch_size):
            candidate = panel['fit']['candidate'][indices].to(device)
            initial = panel['fit']['initial'][indices].to(device)
            geometry = panel['fit']['geometry'][indices].to(device)
            target = panel['fit']['iou'][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(candidate, initial, geometry, visual_enabled)
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        state = dict(epoch=epoch + 1, train_bce=sum(losses) / len(losses),
                     seconds=time.time() - started)
        history.append(state)
        print(json.dumps(state), flush=True)
    fit_metrics, _ = evaluate(model, panel['fit'], visual_enabled, device)
    development_metrics, rows = evaluate(model, panel['development'], visual_enabled, device)
    weights = args.output / 'final.pt'
    torch.save(model.state_dict(), weights)
    report = dict(status='complete_train_only', variant=args.variant,
                  initial_origin=args.initial_origin,
                  seed=2027, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  preparation_sha256=sha(args.root / 'preparation.json'),
                  labels_sha256=preparation['training_labels_sha256'],
                  final_weights_sha256=sha(weights),
                  training_scope='Former-fit130 only; fixed native Train states; no public evaluation or recursive action',
                  fit=fit_metrics, development=development_metrics,
                  history=history, development_rows=rows)
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'variant': args.variant, 'fit': fit_metrics['all'],
                      'development': development_metrics['all']}), flush=True)


if __name__ == '__main__':
    main()
