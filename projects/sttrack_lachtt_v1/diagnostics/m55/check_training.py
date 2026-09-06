"""Three optimizer steps on real fitting RGB-D frames; no performance claim."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=['control', 'clone'], required=True)
    args = parser.parse_args()
    root, variant = args.root, args.variant
    plan = json.loads((root / 'preparation.json').read_text())
    repo = root / 'code' / variant
    for name, digest in plan['code_sha256'][variant].items():
        assert sha(repo / name) == digest, name
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    assert sha(plan['fitting_manifest']) == plan['fitting_manifest_sha256']
    sys.path.insert(0, str(repo))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.models.sttrack import build_sttrack
    from lib.test.tracker.data_utils import PreprocessorMM
    from lib.train.data.processing_utils import sample_target, transform_image_to_crop
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.utils.box_ops import giou_loss, box_cxcywh_to_xyxy, box_xywh_to_xyxy
    from lib.utils.focal_loss import FocalLoss
    from lib.utils.heapmap_utils import generate_heatmap

    torch.set_num_threads(1)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    # Explicit strict initialization avoids the original training entry's missing SOT path.
    network = build_sttrack(cfg, training=False)
    checkpoint = torch.load(plan['checkpoint'], map_location='cpu')['net']
    network.load_state_dict(checkpoint, strict=True)
    assert all(torch.equal(v, checkpoint[n]) for n, v in network.state_dict().items())
    initial = {n: tensor_sha(v) for n, v in network.state_dict().items()}
    del checkpoint
    network.cuda().train()
    assert network.fix_query_window and network.template_number == 2
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    all_cases = json.loads(Path(plan['fitting_manifest']).read_text())
    cases = [next(c for c in all_cases if c['sequence'] == name and c['split'] == 'fit')
             for name in ['chair01_indoor', 'cube04_indoor']]
    templates, searches, targets = [], [[], [], [], []], [[], [], [], []]
    sources = {}
    for case in cases:
        folder = Path(plan['dataset_root']) / case['sequence']
        gt_path = folder / 'groundtruth.txt'
        sources[str(gt_path)] = sha(gt_path)
        gt = np.loadtxt(gt_path, delimiter=',')[:5]
        assert np.isfinite(gt).all() and (gt[:, 2:] > 0).all()
        assert np.array_equal(gt[0], np.asarray(case['init_bbox']))
        prior = torch.tensor(case['init_bbox'], dtype=torch.float32)
        for frame in range(5):
            rgb, depth = folder / 'color' / ('%08d.jpg' % (frame + 1)), folder / 'depth' / ('%08d.png' % (frame + 1))
            sources[str(rgb)], sources[str(depth)] = sha(rgb), sha(depth)
            image = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
            size, factor = (128, 2.) if frame == 0 else (256, 4.)
            crop, resize, _ = sample_target(image, case['init_bbox'], factor, output_sz=size)
            tensor = preprocessor.process(crop)
            if frame == 0:
                templates.append(tensor)
            else:
                searches[frame - 1].append(tensor)
                targets[frame - 1].append(transform_image_to_crop(torch.tensor(gt[frame], dtype=torch.float32),
                    prior, resize, torch.tensor([256., 256.]), normalize=True))
    template = torch.cat(templates, dim=0)
    inputs = [torch.cat(x, dim=0) for x in searches]
    labels = torch.stack([torch.stack(x) for x in targets]).cuda()
    input_binding = [tensor_sha(x) for x in [template] + inputs + [labels]]
    heatmaps = generate_heatmap(labels, 256, 16)
    focal = FocalLoss()
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in network.named_parameters() if n.startswith('backbone.')], 'lr': 1e-5},
        {'params': [p for n, p in network.named_parameters() if not n.startswith('backbone.')], 'lr': 1e-4},
    ], weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(3):
        torch.cuda.synchronize()
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        outputs = network(template=[template, template], search=inputs, track_query_before=None, keep_rate=1.)
        assert len(outputs) == 4
        losses = []
        for frame, output in enumerate(outputs):
            pred = box_cxcywh_to_xyxy(output['pred_boxes']).reshape(-1, 4)
            target = box_xywh_to_xyxy(labels[frame]).clamp(0., 1.)
            giou, _ = giou_loss(pred, target)
            loss = 2. * giou + 5. * torch.nn.functional.l1_loss(pred, target) + focal(output['score_map'], heatmaps[frame].unsqueeze(1))
            assert torch.isfinite(loss)
            losses.append(loss)
        total = torch.stack(losses).sum()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        row = dict(optimizer_step=step + 1, summed_four_frame_loss=float(total.detach()),
                   grad_norm_before_clip=float(norm), seconds=time.time() - started,
                   query_lengths=[q.shape[1] for q in outputs[-1]['track_query_before']])
        assert row['query_lengths'] == [4, 4]
        records.append(row)
        print(json.dumps(row), flush=True)
        del outputs, losses, total
    groups = defaultdict(lambda: {'parameter_elements': 0, 'gradient_elements': 0, 'nonzero_gradient_tensors': 0})
    missing_gradients = []
    for name, parameter in network.named_parameters():
        group = groups[name.split('.')[0]]
        group['parameter_elements'] += parameter.numel()
        if parameter.grad is None:
            missing_gradients.append(name)
        else:
            group['gradient_elements'] += parameter.numel()
            group['nonzero_gradient_tensors'] += int(bool((parameter.grad != 0).any()))
    changed = [n for n, v in network.state_dict().items() if tensor_sha(v) != initial[n]]
    assert changed
    for name, digest in plan['code_sha256'][variant].items():
        assert sha(repo / name) == digest, name
    assert all(sha(path) == digest for path, digest in sources.items())
    result = dict(status='complete_training_contract', variant=variant,
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), probe_sha256=sha(__file__),
        preparation_sha256=sha(root / 'preparation.json'), model_source_sha256=sha(repo / 'lib/models/sttrack/sttrack.py'),
        base_checkpoint_sha256=plan['checkpoint_sha256'], initial_state_sha256=initial,
        sequences=[c['sequence'] for c in cases], frames_zero_based=[0, 1, 2, 3, 4],
        batch_size=2, search_frames=4, training_steps=3, trainable_parameters=sum(p.numel() for p in network.parameters()),
        input_tensor_sha256=input_binding, raw_input_sha256=sources, records=records,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        gradient_groups=dict(groups), missing_gradient_parameters=missing_gradients, changed_tensors=changed,
        trained_checkpoint_saved=False, public_evaluation=False, development_gt_used=False,
        scope='Three real-data full-network optimizer steps on two fitting clips. Initialization-centered crops and consecutive frames are a contract batch, not the final training sampler or recursive performance evaluation. No surviving/adopted trained weights.')
    output = root / 'contract' / (variant + '_result.json')
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['initial_state_sha256', 'raw_input_sha256', 'changed_tensors']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
