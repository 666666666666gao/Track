"""M80 single-seed causal training with fixed blockwise text dropout."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def tensor_state_sha(network, adapter_only=False):
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        if name.startswith('semantic_adapter.') == adapter_only:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['category'], required=True)
    args = parser.parse_args()
    root = Path('/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908')
    assert (root / 'frozen.json').is_file()
    spec_path = root / 'training_spec.json'
    spec = json.loads(spec_path.read_text())
    assert sha(spec_path) == json.loads((root / 'frozen.json').read_text())['training_spec_sha256']
    assert sha(__file__) == spec['training_script_sha256']
    assert sha(root / 'causal_training.py') == spec['causal_script_sha256']
    assert sha(root / 'support_loss.py') == spec['support_loss_sha256']
    assert sha(root / 'window_competition.py') == spec['window_loss_sha256']
    assert sha(spec['banks']['fit'][args.arm]['path']) == spec['banks']['fit'][args.arm]['sha256']
    assert sha(root / 'native_parity' / (args.arm + '_zero.pth')) == spec['initial_checkpoint_sha256'][args.arm]
    assert sha(spec['native_checkpoint']) == spec['native_checkpoint_sha256']
    integration = json.loads((root / 'integration.json').read_text())
    assert sha(root / 'integration.json') == spec['integration_sha256']
    for name, digest in integration['source_sha256'].items():
        assert sha(root / 'code' / name) == digest, name
    sys.path.insert(0, str(root / 'code'))
    sys.path.insert(0, str(root))
    from text_dropout import make_contexts
    assert sha(root / 'text_dropout.py') == spec['text_dropout_source_sha256']
    assert sha(root / 'text_schedule.json') == spec['text_schedule_sha256']
    schedule = json.loads((root / 'text_schedule.json').read_text())['sequences']
    from causal_training import CausalTrainingTracker
    from window_competition import supervision
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    torch.manual_seed(spec['seed'])
    torch.cuda.manual_seed_all(spec['seed'])
    update_config_from_file(str(root / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['native_checkpoint'],
        base_checkpoint_sha256=spec['native_checkpoint_sha256'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = CausalTrainingTracker(params, str(root / 'native_parity' / (args.arm + '_zero.pth')))
    tracker.support_loss_weight = spec['support_loss_weights'][args.arm]
    assert tracker.use_text and tracker.network.semantic_adapter.null_support
    assert sum(p.numel() for p in tracker.network.parameters() if p.requires_grad) == 289154
    base_before = tensor_state_sha(tracker.network)
    initial_sha = tensor_state_sha(tracker.network, adapter_only=True)
    bank = torch.load(spec['banks']['fit'][args.arm]['path'], map_location='cpu')
    assert set(bank['sequences']) == {r['sequence'] for r in spec['sequence_order']}
    output = root / 'training' / args.arm
    output.mkdir(parents=True)
    optimizer = torch.optim.AdamW(tracker.network.semantic_adapter.parameters(),
                                 lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    started = time.time()
    frame_count = steps = 0
    aggregate = Counter()
    visited_digest = hashlib.sha256()
    accumulated = spec['gradient_accumulation_frames']
    receipts = []

    def checkpoint(complete):
        return dict(architecture='semantic_spatial_support_v1', null_support=True, support_loss_weight=spec['support_loss_weights'][args.arm], arm=args.arm, model=tracker.network.semantic_adapter.state_dict(),
            base_checkpoint_sha256=spec['native_checkpoint_sha256'], use_text=tracker.use_text,
            optimizer=optimizer.state_dict(), completed_sequences=len(receipts), optimizer_steps=steps,
            actual_dataset_optimizer_steps=steps, frame_count=frame_count, training_spec_sha256=sha(spec_path),
            status='complete' if complete else 'in_progress', seed=spec['seed'], text_training_protocol='fixed_20_percent_32frame_block_dropout', text_schedule_sha256=spec['text_schedule_sha256'])

    rank_frames = rank_negative_sum = 0
    rank_loss_sum = 0.
    with (output / 'sequence_log.jsonl').open('x') as log, (output / 'sampled_state_trace.jsonl').open('x') as trace:
        for seq_index, row in enumerate(spec['sequence_order']):
            folder = Path(spec['dataset_root']) / row['sequence']
            assert sha(folder / 'groundtruth.txt') == row['groundtruth_sha256']
            gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)
            n = row['rgb_frames']
            if row['sequence'] == 'toy07_indoor_320':
                assert len(gt) == 1406 and n == 1367
                assert row['groundtruth_sha256'] == '683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
                gt = gt[:n]
            assert len(gt) == n
            index = bank['sequences'].index(row['sequence'])
            info = dict(init_bbox=row['first_box'], text_tokens=bank['tokens'][index],
                        text_mask=bank['mask'][index], empty_text=bank['empty'])

            def frame(i):
                return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (i + 1))),
                    str(folder / 'depth' / ('%08d.png' % (i + 1))), dtype='rgbcolormap', depth_clip=True)

            tracker.initialize(frame(0), info)
            category_context, empty_context = make_contexts(tracker, bank['empty'])
            text_counts = Counter()
            sequence_start = time.time()
            labels = Counter()
            loss_sum = 0.
            supervised = writes = pending = 0
            gradient_frames = 0
            max_gradient_norm = 0.
            optimizer.zero_grad(set_to_none=True)
            for frame_index in range(1, n):
                dropped = schedule[row['sequence']][(frame_index - 1) // 32]
                tracker.semantic_context['text'] = empty_context if dropped else category_context
                text_counts['empty' if dropped else 'category'] += 1
                out, state = tracker.step(frame(frame_index))
                state['text_dropped'] = dropped
                # Only now expose the current training target to the loss function.
                loss, diagnostic = supervision(tracker.network, out, gt[frame_index],
                    state['previous_bbox'], state['resize_factor'], 256, output_window=tracker.output_window)
                labels[diagnostic['label']] += 1
                if diagnostic['competition_loss'] is not None:
                    rank_frames += 1
                    rank_loss_sum += diagnostic['competition_loss']
                    rank_negative_sum += diagnostic['competition_negatives']
                pending += 1
                frame_count += 1
                writes += int(state['template_write'])
                if loss is not None:
                    (loss / accumulated).backward()
                    loss_sum += float(loss.detach())
                    supervised += 1
                    gradient_frames += 1
                visited_digest.update(json.dumps([row['sequence'], frame_index, state['previous_bbox'],
                    state['bbox'], state['best_score']], separators=(',', ':')).encode())
                if frame_index % 50 == 0 or frame_index == 1 or frame_index == n - 1:
                    trace.write(json.dumps(dict(sequence=row['sequence'], frame_index=frame_index,
                        optimizer_steps_before_update=steps, **state, **diagnostic)) + '\n')
                if pending == accumulated or frame_index == n - 1:
                    if gradient_frames:
                        for parameter in tracker.network.semantic_adapter.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(accumulated / gradient_frames)
                        norm = torch.nn.utils.clip_grad_norm_(tracker.network.semantic_adapter.parameters(), spec['gradient_clip'])
                        assert bool(torch.isfinite(norm)), (row['sequence'], frame_index)
                        max_gradient_norm = max(max_gradient_norm, float(norm))
                        optimizer.step()
                        steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    pending = gradient_frames = 0
                del out, loss
            aggregate.update(labels)
            record = dict(sequence=row['sequence'], sequence_index=seq_index, total_sequences=len(spec['sequence_order']),
                frames=n, track_calls=n-1, text_condition_calls=dict(text_counts), supervised_frames=supervised, label_counts=dict(labels),
                mean_training_loss=loss_sum/supervised if supervised else None, template_writes=writes,
                cumulative_competition_frames=rank_frames, cumulative_competition_loss_sum=rank_loss_sum, cumulative_competition_negatives=rank_negative_sum,
                maximum_preclip_gradient_norm=max_gradient_norm, total_optimizer_steps=steps,
                total_track_calls=frame_count, seconds=time.time()-sequence_start, elapsed_seconds=time.time()-started)
            receipts.append(record)
            log.write(json.dumps(record) + '\n'); log.flush(); trace.flush()
            torch.save(checkpoint(False), output / 'latest.pth')
            print(json.dumps(record), flush=True)
    assert frame_count == spec['total_training_track_calls']
    assert tensor_state_sha(tracker.network) == base_before
    assert all(p.grad is None for n, p in tracker.network.named_parameters() if not n.startswith('semantic_adapter.'))
    for name, digest in integration['source_sha256'].items():
        assert sha(root / 'code' / name) == digest
    torch.save(checkpoint(True), output / 'final.pth')
    result = dict(status='one_full_causal_fit_pass_complete', arm=args.arm, null_support=True, support_loss_weight=spec['support_loss_weights'][args.arm], sequences=len(receipts),
        total_track_calls=frame_count, optimizer_steps=steps, training_label_counts=dict(aggregate),
        elapsed_seconds=time.time()-started, initial_adapter_state_sha256=initial_sha,
        base_state_before_sha256=base_before, base_parameters_and_buffers_unchanged=True,
        training_spec_sha256=sha(spec_path), final_checkpoint_sha256=sha(output/'final.pth'),
        sequence_log_sha256=sha(output/'sequence_log.jsonl'), sampled_trace_sha256=sha(output/'sampled_state_trace.jsonl'),
        full_visited_state_stream_sha256=visited_digest.hexdigest(),
        gt_after_prediction_for_loss_only=True, gt_reinitialization_after_first_frame=False,
        learned_parameters=289154, backward_through_crops_or_time=False,
        competition_frames=rank_frames, competition_loss_sum=rank_loss_sum, competition_negatives=rank_negative_sum, window_loss_sha256=spec['window_loss_sha256'],
        text_training_protocol='fixed_20_percent_32frame_block_dropout', text_schedule_sha256=spec['text_schedule_sha256'],
        actual_text_condition_calls={key:sum(x['text_condition_calls'].get(key,0) for x in receipts) for key in ['category','empty']},
        evaluation_metrics_computed=False, observed_utc=datetime.now(timezone.utc).isoformat())
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
