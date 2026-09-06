"""Paired DepthTrack fine-tuning; the only arm difference is the TSG clone patch."""
import argparse
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class PairedClips(torch.utils.data.Dataset):
    """Original causal sampling and ViPT augmentation, with explicit error propagation."""
    def __init__(self, spec, epoch, sample_count):
        from lib.train.dataset.depthtrack import DepthTrack
        from lib.train.data.sampler import TrackingSampler
        from lib.train.data.processing import ViPTProcessing
        from lib.train.data import transforms as tfm
        self.spec = spec
        self.epoch = epoch
        self.sample_count = sample_count
        self.dataset = DepthTrack(root=spec['dataset_root'])
        self.dataset.sequence_list = spec['fit_sequences']
        self.annotations = [self.dataset.get_sequence_info(i) for i in range(len(self.dataset.sequence_list))]
        for name, info in zip(self.dataset.sequence_list, self.annotations):
            assert torch.isfinite(info['bbox'][info['valid']]).all(), name
            assert int(info['visible'].sum()) > 12 and len(info['visible']) >= 20, name
        # The native sampler reads these same annotations on every draw.
        self.dataset.get_sequence_info = self.cached_sequence_info
        self.sampler = TrackingSampler([self.dataset], [1], sample_count, 200, 4, 2)
        self.processor = ViPTProcessing(
            search_area_factor={'template': 2., 'search': 4.},
            output_sz={'template': 128, 'search': 256},
            center_jitter_factor={'template': 0., 'search': 3.},
            scale_jitter_factor={'template': 0., 'search': .25}, mode='sequence',
            transform=tfm.Transform(tfm.ToTensorAndJitter(.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=.5),
                                    tfm.Normalize(mean=spec['mean'], std=spec['std'])),
            joint_transform=tfm.Transform(tfm.ToGrayscale(probability=0.),
                                          tfm.RandomHorizontalFlip(probability=.5)))

    def cached_sequence_info(self, seq_id):
        return self.annotations[seq_id]

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        from lib.utils import TensorDict
        sample_seed = self.spec['data_seed'] + (self.epoch - 1) * self.spec['samples_per_epoch'] + index
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        attempts = 0
        while True:
            attempts += 1
            # Preserve the single-dataset random draw performed by the original sampler.
            dataset = random.choices(self.sampler.datasets, self.sampler.p_datasets)[0]
            seq_id, visible, info = self.sampler.sample_seq_from_dataset(dataset, True)
            search_ids = None
            gap_increase = 0
            while search_ids is None:
                base = self.sampler._sample_visible_ids(visible, 1, min_id=1, max_id=len(visible) - 4)
                previous = self.sampler._sample_visible_ids(visible, 1,
                    min_id=base[0] - 200 - gap_increase, max_id=base[0])
                if previous is None:
                    gap_increase += 5
                    continue
                template_ids = base + previous
                search_ids = self.sampler._sample_visible_ids(visible, 4,
                    min_id=base[0] + 1, max_id=base[0] + 200 + gap_increase)
                gap_increase += 5
            # No sorting or deduplication: those would be a second intervention.
            templates, template_anno, meta = dataset.get_frames(seq_id, template_ids, info)
            searches, search_anno, _ = dataset.get_frames(seq_id, search_ids, info)
            data = self.processor(TensorDict(template_images=templates,
                template_anno=template_anno['bbox'], search_images=searches,
                search_anno=search_anno['bbox'], dataset=dataset.get_name(),
                test_class=meta.get('object_class_name')))
            # Retain only the native processor's explicit invalid-crop resampling rule.
            if data['valid']:
                record = dict(epoch=self.epoch, index=index, seed=sample_seed,
                    sequence=dataset.sequence_list[seq_id], template_ids=template_ids,
                    search_ids=search_ids, attempts=attempts)
                return {key: data[key] for key in ['template_images', 'search_images', 'search_anno']}, record


def collate_clips(items):
    return ({key: torch.stack([item[0][key] for item in items], dim=1)
             for key in ['template_images', 'search_images', 'search_anno']},
            [item[1] for item in items])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=['control', 'clone'], required=True)
    parser.add_argument('--mode', choices=['contract', 'train'], required=True)
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root / 'training_spec.json').read_text())
    assert sha(__file__) == spec['trainer_sha256']
    preparation = json.loads((root / 'preparation.json').read_text())
    assert sha(root / 'preparation.json') == spec['preparation_sha256']
    repo = root / 'code' / args.variant
    for name, digest in preparation['code_sha256'][args.variant].items():
        assert sha(repo / name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(spec['fitting_manifest']) == spec['fitting_manifest_sha256']
    for name, digest in spec['fit_gt_sha256'].items():
        assert sha(Path(spec['dataset_root']) / name / 'groundtruth.txt') == digest, name
    sys.path.insert(0, str(repo))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.models.sttrack import build_sttrack
    from lib.utils.box_ops import giou_loss, box_cxcywh_to_xyxy, box_xywh_to_xyxy
    from lib.utils.focal_loss import FocalLoss
    from lib.utils.heapmap_utils import generate_heatmap
    from lib.utils.ce_utils import adjust_keep_rate

    output = root / ('sampler_contract' if args.mode == 'contract' else 'training') / args.variant
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(spec['model_seed'])
    np.random.seed(spec['model_seed'])
    torch.manual_seed(spec['model_seed'])
    torch.cuda.manual_seed_all(spec['model_seed'])
    update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    network = build_sttrack(cfg, training=False)
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu')['net']
    network.load_state_dict(checkpoint, strict=True)
    assert all(torch.equal(v, checkpoint[n]) for n, v in network.state_dict().items())
    initial_state_sha = {n: tensor_sha(v) for n, v in network.state_dict().items()}
    del checkpoint
    write_json(output / 'initial_state_sha256.json', initial_state_sha)
    network.cuda().train()
    assert network.fix_query_window and network.template_number == 2
    focal = FocalLoss()
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in network.named_parameters() if n.startswith('backbone.')], 'lr': 1e-5},
        {'params': [p for n, p in network.named_parameters() if not n.startswith('backbone.')], 'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    epochs = [1, 10] if args.mode == 'contract' else list(range(1, spec['epochs'] + 1))
    sample_count = 8 if args.mode == 'contract' else spec['samples_per_epoch']
    steps = 0
    microbatches = 0
    started = time.time()
    epoch_records = []
    data_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    write_json(output / 'execution_binding.json', dict(mode=args.mode, variant=args.variant,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_sha256=sha(root / 'training_spec.json'), trainer_sha256=sha(__file__),
        model_source_sha256=sha(repo / 'lib/models/sttrack/sttrack.py'),
        initial_state_file_sha256=sha(output / 'initial_state_sha256.json'),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(), parameter_elements=sum(p.numel() for p in network.parameters()),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        exact_repeat_of_custom_cuda_kernels_guaranteed=False))
    with (output / 'batches.jsonl').open('w') as batch_log, (output / 'steps.jsonl').open('w') as step_log:
        for epoch in epochs:
            epoch_started = time.time()
            dataset = PairedClips(spec, epoch, sample_count)
            generator = torch.Generator().manual_seed(spec['data_seed'] + epoch)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4,
                shuffle=False, drop_last=True, collate_fn=collate_clips, pin_memory=True,
                generator=generator)
            keep_rate = adjust_keep_rate(epoch, warmup_epochs=10, total_epochs=15)
            keep_rate = [float(x) for x in keep_rate]
            epoch_loss = 0.
            update_loss = 0.
            epoch_digest = hashlib.sha256()
            optimizer.zero_grad(set_to_none=True)
            for batch_id, (data, sample_records) in enumerate(loader):
                fingerprints = {key: tensor_sha(value) for key, value in data.items()}
                record = dict(epoch=epoch, batch_id=batch_id, samples=sample_records, tensor_sha256=fingerprints)
                encoded = json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n'
                data_digest.update(encoded.encode())
                epoch_digest.update(encoded.encode())
                batch_log.write(encoded)
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
                heatmaps = generate_heatmap(data['search_anno'], 256, 16)
                predictions = network(template=list(data['template_images'].unbind(0)),
                    search=list(data['search_images'].unbind(0)), track_query_before=None,
                    keep_rate=keep_rate, ce_template_mask=None, return_last_attn=False)
                losses = []
                for frame, prediction in enumerate(predictions):
                    pred = box_cxcywh_to_xyxy(prediction['pred_boxes']).reshape(-1, 4)
                    target = box_xywh_to_xyxy(data['search_anno'][frame]).clamp(0., 1.)
                    giou, _ = giou_loss(pred, target)
                    losses.append(2. * giou + 5. * torch.nn.functional.l1_loss(pred, target)
                        + focal(prediction['score_map'], heatmaps[frame].unsqueeze(1)))
                loss = torch.stack(losses).sum()
                assert torch.isfinite(loss), (epoch, batch_id)
                assert [q.shape[1] for q in predictions[-1]['track_query_before']] == [4, 4]
                (loss / spec['gradient_accumulation']).backward()
                value = float(loss.detach())
                epoch_loss += value
                update_loss += value / spec['gradient_accumulation']
                microbatches += 1
                if (batch_id + 1) % spec['gradient_accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), .1)
                    assert torch.isfinite(norm), (epoch, batch_id)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    row = dict(epoch=epoch, optimizer_step=steps, microbatches=microbatches,
                        mean_summed_four_frame_loss=update_loss, grad_norm_before_clip=float(norm),
                        learning_rates=[g['lr'] for g in optimizer.param_groups], keep_rate=keep_rate,
                        elapsed_seconds=time.time() - started)
                    step_log.write(json.dumps(row, allow_nan=False) + '\n')
                    if steps % 25 == 0 or args.mode == 'contract' or steps == 1:
                        print(json.dumps(row, allow_nan=False), flush=True)
                        step_log.flush()
                        batch_log.flush()
                    update_loss = 0.
                del predictions, losses, loss, data, heatmaps
            assert len(loader) % spec['gradient_accumulation'] == 0
            scheduler.step()
            epoch_row = dict(epoch=epoch, samples=sample_count, optimizer_steps=steps,
                mean_summed_four_frame_loss=epoch_loss / len(loader),
                seconds=time.time() - epoch_started, batch_stream_sha256=epoch_digest.hexdigest())
            epoch_records.append(epoch_row)
            write_json(output / 'epochs.json', epoch_records)
            print(json.dumps(epoch_row, allow_nan=False), flush=True)
            if args.mode == 'train':
                # /root/autodl-tmp has only 2.2 GB free; optimizer state uses the 89 GB shared-memory volume.
                temporary = Path(spec['temporary_checkpoint_root']) / args.variant
                temporary.mkdir(parents=True, exist_ok=True)
                state = dict(net=network.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, optimizer_steps=steps,
                    model_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    spec_sha256=sha(root / 'training_spec.json'), variant=args.variant)
                torch.save(state, temporary / 'latest.tmp')
                (temporary / 'latest.tmp').replace(temporary / 'latest.pth')
                del state
    expected_steps = len(epochs) * sample_count // (2 * spec['gradient_accumulation'])
    assert steps == expected_steps
    for name, digest in preparation['code_sha256'][args.variant].items():
        assert sha(repo / name) == digest, name
    assert sha(__file__) == spec['trainer_sha256']
    for name, digest in spec['fit_gt_sha256'].items():
        assert sha(Path(spec['dataset_root']) / name / 'groundtruth.txt') == digest, name
    final_state = {n: v.detach().cpu() for n, v in network.state_dict().items()}
    changed = [n for n, v in final_state.items() if tensor_sha(v) != initial_state_sha[n]]
    assert changed
    weight = None
    if args.mode == 'train':
        weight = output / 'model_final.pth'
        torch.save(dict(net=final_state, variant=args.variant, spec_sha256=sha(root / 'training_spec.json'),
            optimizer_steps=steps, epochs=spec['epochs']), weight)
    result = dict(status='complete_' + args.mode, mode=args.mode, variant=args.variant,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.time() - started, epochs=epoch_records, optimizer_steps=steps,
        microbatches=microbatches, clips=microbatches * 2, search_frames=microbatches * 8,
        data_stream_sha256=data_digest.hexdigest(), changed_state_tensors=len(changed),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        weight_path=str(weight) if weight else None, weight_sha256=sha(weight) if weight else None,
        spec_sha256=sha(root / 'training_spec.json'), execution_binding_sha256=sha(output / 'execution_binding.json'),
        batches_sha256=sha(output / 'batches.jsonl'), steps_sha256=sha(output / 'steps.jsonl'),
        development_evaluation=False, public_evaluation=False, adopted_weight=False)
    write_json(output / 'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
