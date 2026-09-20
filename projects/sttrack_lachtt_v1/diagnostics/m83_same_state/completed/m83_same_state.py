"""Read-only M82 content diagnostics. No GT loading and no counterfactual commits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import torch

ROOT = Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
PARENT = Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0, str(PARENT / 'code'))
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack_semantic import STTrackSemantic
from lib.train.dataset.depth_utils import get_rgbd_frame
from lib.utils.box_ops import clip_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan = read(ROOT / 'spec.json')
    assert sha(Path(__file__)) == plan['source_sha256']
    spec = read(PARENT / 'training_spec.json')
    assert sha(PARENT / 'training_spec.json') == plan['training_spec_sha256']
    assert sha(PARENT / 'recursive_result.json') == plan['parent_result_sha256']
    integration = read(PARENT / 'integration.json')
    for name, digest in integration['source_sha256'].items():
        assert sha(PARENT / 'code' / name) == digest
    checkpoint = PARENT / 'training/category/final.pth'
    assert sha(checkpoint) == plan['head_sha256']
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(PARENT / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['native_checkpoint'],
        base_checkpoint_sha256=spec['native_checkpoint_sha256'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrackSemantic(params, str(checkpoint))
    assert tracker.update_intervals == 50 and tracker.update_threshold == .75
    banks = {}
    for arm in ['category', 'empty', 'swapped']:
        source = spec['banks']['development'][arm]
        assert sha(Path(source['path'])) == source['sha256']
        banks[arm] = torch.load(source['path'], map_location='cpu')
    capture = {}
    tracker.network.semantic_adapter.register_forward_pre_hook(lambda module, inputs: capture.update(inputs=inputs))
    tracker.network.box_head.register_forward_hook(lambda module, inputs, output: capture.update(head=output))
    cases = plan['cases'][:1] if args.preflight else plan['cases']
    outdir = ROOT / ('preflight' if args.preflight else 'predictions')
    outdir.mkdir()
    receipts = []
    started = time.time()
    with torch.no_grad():
        for case in cases:
            seq = case['sequence']
            folder = Path(spec['dataset_root']) / seq
            def frame(i):
                return get_rgbd_frame(str(folder/'color'/('%08d.jpg' % (i+1))),
                    str(folder/'depth'/('%08d.png' % (i+1))), dtype='rgbcolormap', depth_clip=True)
            texts = {}
            masks = {}
            for arm, bank in banks.items():
                idx = bank['sequences'].index(seq)
                texts[arm] = bank['tokens'][idx].float().cuda().unsqueeze(0)
                masks[arm] = bank['mask'][idx].bool().cuda().unsqueeze(0)
            assert torch.equal(masks['category'], masks['empty']) and torch.equal(masks['category'], masks['swapped'])
            bank = banks['category']; idx = bank['sequences'].index(seq)
            tracker.initialize(frame(0), dict(init_bbox=case['init_bbox'], text_tokens=bank['tokens'][idx],
                text_mask=bank['mask'][idx], empty_text=bank['empty']))
            sealed = PARENT / 'recursive/category' / (seq + '.json')
            assert sha(sealed) == case['sealed_category_sha256']
            reference = read(sealed)['rows']
            limit = min(case['frames'], 102) if args.preflight else case['frames']
            rows = []
            for i in range(1, limit):
                image = frame(i)
                previous = list(tracker.state)
                actual = tracker.track(image)
                assert actual['target_bbox'] == reference[i]['bbox'], (seq, i, 'bbox parity')
                assert float(actual['best_score']) == reference[i]['score'], (seq, i, 'score parity')
                rgb, depth, fused, initial, text, mask = capture['inputs']
                score, _, size, offset = capture['head']
                outputs = {'category': dict(score_map=score, size_map=size, offset_map=offset)}
                state = list(tracker.state)
                queries = [v.clone() for v in tracker.track_query_before]
                templates = list(tracker.z_dict)
                for arm in ['empty', 'swapped']:
                    enhanced, _ = tracker.network.semantic_adapter.forward(rgb, depth, fused, initial, texts[arm], mask)
                    outputs[arm] = tracker.network.forward_head(enhanced)
                outputs['native'] = tracker.network.forward_head(fused)
                assert tracker.state == state and len(tracker.z_dict) == len(templates)
                assert all(a is b for a, b in zip(tracker.z_dict, templates))
                assert all(torch.equal(a, b) for a, b in zip(tracker.track_query_before, queries))
                side = math.ceil(math.sqrt(previous[2] * previous[3]) * 4.)
                resize = 256. / side
                h, w = image.shape[:2]
                def decode(out, response):
                    box = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4)
                    cx, cy, bw, bh = (box.mean(dim=0) * 256. / resize).tolist()
                    cx += previous[0] + .5 * previous[2] - .5 * 256. / resize
                    cy += previous[1] + .5 * previous[3] - .5 * 256. / resize
                    return clip_box([cx-.5*bw, cy-.5*bh, bw, bh], h, w, margin=10)
                native = outputs['native']['score_map'].flatten(1)
                native = native / native.sum(dim=1, keepdim=True)
                variants = {}
                for arm, out in outputs.items():
                    raw = out['score_map']; hann = raw * tracker.output_window
                    p = raw.flatten(1); mass = p.sum(dim=1, keepdim=True); p = p / mass
                    variants[arm] = dict(hann_bbox=decode(out, hann), raw_bbox=decode(out, raw),
                        raw_peak=int(raw.argmax()), hann_peak=int(hann.argmax()),
                        raw_max=float(raw.max()), hann_max=float(hann.max()), raw_mass=float(mass.item()),
                        native_spatial_kl=float((native*(native.log()-p.log())).sum().item()))
                assert variants['category']['hann_bbox'] == actual['target_bbox'], (seq, i, 'decoder parity')
                rows.append(dict(frame=i, previous_bbox=previous, search_side=side, variants=variants))
            path = outdir / (seq + '.json')
            path.write_text(json.dumps(dict(sequence=seq, rows=rows), separators=(',', ':'), allow_nan=False)+'\n')
            receipt = dict(sequence=seq, positions=len(rows), sha256=sha(path), elapsed_seconds=time.time()-started)
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    report = dict(status='complete', spec_sha256=sha(ROOT/'spec.json'), source_sha256=sha(Path(__file__)),
        head_sha256=sha(checkpoint), mode='preflight' if args.preflight else 'full22',
        positions=sum(v['positions'] for v in receipts), sequences=receipts,
        category_sealed_bbox_score_exact=True, counterfactual_state_not_committed=True,
        no_groundtruth_loaded=True)
    (outdir/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
