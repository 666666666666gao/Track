"""M102 single-candidate RGB identity readout; no teacher labels or state."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from audit_candidate_vlm import MODEL, REGISTER, PACKET


WORDS = ('Yes', 'No', 'Unknown')
PROMPT = ('The first full image marks one object in red, and the second image '
          'is its crop. The third full image marks another object in red, and '
          'the fourth image is its crop. Are these the same physical instance '
          'seen at different times, rather than merely objects of the same '
          'category? Reply with exactly one word: Yes, No, or Unknown. Use '
          'Unknown if the visible evidence cannot determine instance identity.')


def inputs_for(processor, images):
    content = [{'type': 'image', 'image': image,
                'max_pixels': (512 if index % 2 == 0 else 256) * 28 * 28,
                'min_pixels': 128 * 28 * 28} for index, image in enumerate(images)]
    content.append({'type': 'text', 'text': PROMPT})
    messages = [{'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors='pt')


def image_blocks(inputs):
    cursor, blocks = 0, []
    for grid in inputs.image_grid_thw:
        count = int(grid.prod())
        blocks.append((grid, inputs.pixel_values[cursor:cursor+count]))
        cursor += count
    assert cursor == len(inputs.pixel_values) and len(blocks) == 4
    return blocks


def score(model, inputs, token_ids):
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1, token_ids].float()
    assert bool(torch.isfinite(logits).all())
    return dict(restricted_choice=WORDS[int(logits.argmax())],
                logits=dict(zip(WORDS, logits.cpu().tolist())),
                three_token_probabilities=dict(zip(WORDS, logits.softmax(0).cpu().tolist())),
                yes_minus_no=float(logits[0]-logits[1]), input_tokens=inputs.input_ids.numel())


def winner(values):
    gap = values['A']['yes_minus_no'] - values['B']['yes_minus_no']
    return 'A' if gap > 0 else 'B' if gap < 0 else 'tie'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--mode', choices=('sanity', 'full'), required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4); torch.manual_seed(2027)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    encoded = [processor.tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
    assert all(len(ids) == 1 for ids in encoded)
    token_ids = [ids[0] for ids in encoded]
    assert len(set(token_ids)) == 3
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    manifest = json.loads((PACKET/'private_selection.json').read_text())
    assert manifest['status'] == 'prepared_train_only' and len(manifest['rows']) == 24
    items = manifest['rows'][args.shard::2]
    if args.mode == 'sanity':
        items = items[:1]
    register = {row['sequence']:row for row in json.loads(REGISTER.read_text())}
    rows, metrics = [], []
    started = time.time()
    for item in items:
        ref = register[item['sequence']]
        assert ref['split'] == 'fit'
        assert item['positive_iou'] >= .6 and item['native_iou'] <= .1 and item['cross_iou'] <= .1
        assert {item['a_index'], item['b_index']} == {item['native_index'], item['positive_index']}
        target_side = 'A' if item['a_index'] == item['positive_index'] else 'B'
        initial = Image.open(ref['initialization_image']).convert('RGB')
        initial_crop = initial.crop(tuple(ref['generator_target_xyxy']))
        initial_full = initial.copy()
        ImageDraw.Draw(initial_full).rectangle(tuple(ref['generator_target_xyxy']), outline='red', width=2)
        current = Image.open(item['source_image']).convert('RGB')
        conditions = {'initial_first': {}, 'current_first': {}}
        binding = {}
        for side in ('A', 'B'):
            index = item['a_index'] if side == 'A' else item['b_index']
            box = item['native_box'] if index == item['native_index'] else item['positive_box']
            x, y, width, height = box
            full = current.copy()
            ImageDraw.Draw(full).rectangle((x, y, x+width, y+height), outline='red', width=2)
            crop = Image.open(PACKET/'images'/f"{item['audit_id']}_{side}.png").convert('RGB')
            forward = inputs_for(processor, (initial_full, initial_crop, full, crop))
            reverse = inputs_for(processor, (full, crop, initial_full, initial_crop))
            if args.mode == 'sanity':
                a, b = image_blocks(forward), image_blocks(reverse)
                assert all(torch.equal(a[i][0], b[(i+2)%4][0]) and
                           torch.equal(a[i][1], b[(i+2)%4][1]) for i in range(4))
                binding[side] = True
            conditions['initial_first'][side] = score(model, forward, token_ids)
            conditions['current_first'][side] = score(model, reverse, token_ids)
        ranks = {name:winner(values) for name,values in conditions.items()}
        event_metrics = dict(rank_consistent=ranks['initial_first'] == ranks['current_first'],
                             rank_changed=ranks['initial_first'] != ranks['current_first'])
        if args.mode == 'full':
            event_metrics.update(initial_first_correct=ranks['initial_first'] == target_side,
                                 current_first_correct=ranks['current_first'] == target_side,
                                 both_correct=all(rank == target_side for rank in ranks.values()))
        metrics.append(event_metrics)
        row = dict(audit_id=item['audit_id'], conditions=conditions, ranks=ranks)
        if args.mode == 'sanity':
            row['exact_image_pair_swap'] = binding
            row['identical_initial_self_pair'] = score(model, inputs_for(processor,
                                      (initial_full, initial_crop, initial_full, initial_crop)), token_ids)
        rows.append(row)
        print(json.dumps(dict(completed=len(rows), total=len(items), elapsed_seconds=time.time()-started)), flush=True)
    summary = {name:sum(row[name] for row in metrics) for name in metrics[0]}
    summary['events'] = len(rows)
    for order in ('initial_first', 'current_first'):
        summary[order+'_ranks'] = dict(Counter(row['ranks'][order] for row in rows))
        summary[order+'_token_choices'] = dict(Counter(row['conditions'][order][side]['restricted_choice']
                                                       for row in rows for side in ('A', 'B')))
    result = dict(status='complete_single_candidate_sanity_only' if args.mode == 'sanity' else
                        'complete_single_candidate_identity_readout_shard',
                  mode=args.mode, shard=args.shard, seed=2027, dtype='bfloat16',
                  token_ids=dict(zip(WORDS, token_ids)), summary=summary, rows=rows,
                  elapsed_seconds=time.time()-started, no_teacher_label_or_tracker_state=True,
                  no_GT_scores_category_or_candidate_letters_in_prompt=True,
                  target_side_is_localization_proxy_not_human_identity_truth=True,
                  independent_v2_assignment_not_read=True, no_public_benchmark_or_checkpoint=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
