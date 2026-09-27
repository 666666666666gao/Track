"""Read-only Qwen A/B identity and swap-bias probe on M92 Train events."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL = Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct')
REGISTER = Path('/root/autodl-tmp/sttrack_m86_initialization_audit_20260921/evidence/audit_register.json')
PACKET = Path('/root/autodl-tmp/sttrack_m92_candidate_review_20260928')
LETTERS = ('A', 'B', 'N', 'U')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def score(model, processor, initial_full, initial_crop, current_full, a_crop, b_crop,
          category, token_ids):
    prompt = ('The first image marks one target with a red box. The second is a '
              'crop of that same target. The third is a later unmarked full frame. '
              'The fourth is candidate A and the fifth is candidate B. Which '
              'candidate shows the same physical object as the marked target, '
              'not merely an object of the same category? Answer A or B. '
              'Answer N if neither is the target, or U if the visible evidence '
              'cannot decide. Give exactly one letter.')
    if category is not None:
        prompt += f' An automatically generated initial category is "{category}"; it may be wrong.'
    images = (initial_full, initial_crop, current_full, a_crop, b_crop)
    content = [{'type': 'image', 'image': im,
                'max_pixels': (512 if index in (0, 2) else 256) * 28 * 28,
                'min_pixels': 128 * 28 * 28} for index, im in enumerate(images)]
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_images, videos = process_vision_info(messages)
    inputs = processor(text=[chat], images=vision_images, videos=videos,
                       padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]
    selected = logits[token_ids].float()
    assert bool(torch.isfinite(selected).all()), 'Non-finite A/B/N/U logits'
    values = [float(x) for x in selected.cpu()]
    probabilities = [float(x) for x in selected.softmax(dim=0).cpu()]
    return {'choice': LETTERS[int(selected.argmax())],
            'logits': dict(zip(LETTERS, values)),
            'four_token_probabilities': dict(zip(LETTERS, probabilities)),
            'input_tokens': int(inputs.input_ids.numel())}


def flipped(choice):
    return {'A': 'B', 'B': 'A', 'N': 'N', 'U': 'U'}[choice]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    assert not args.output.exists() and args.limit in (1, 24)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    token_ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(x) == 1 for x in token_ids)
    token_ids = [x[0] for x in token_ids]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    manifest = json.loads((PACKET / 'private_selection.json').read_text())
    assert manifest['status'] == 'prepared_train_only' and len(manifest['rows']) == 24
    register = {row['sequence']: row for row in json.loads(REGISTER.read_text())}
    rows = []
    private_metrics = []
    started = time.time()
    for item in manifest['rows'][:args.limit]:
        ref = register[item['sequence']]
        assert ref['split'] == 'fit' and sha(ref['initialization_image']) == ref['image_sha256']
        assert item['positive_iou'] >= 0.6 and item['native_iou'] <= 0.1 and item['cross_iou'] <= 0.1
        assert {item['a_index'], item['b_index']} == {item['native_index'], item['positive_index']}
        target_side = 'A' if item['a_index'] == item['positive_index'] else 'B'
        initial = Image.open(ref['initialization_image']).convert('RGB')
        initial_crop = initial.crop(tuple(ref['generator_target_xyxy']))
        initial_full = initial.copy()
        ImageDraw.Draw(initial_full).rectangle(tuple(ref['generator_target_xyxy']), outline='red', width=2)
        current = Image.open(item['source_image']).convert('RGB')
        audit_id = item['audit_id']
        a = Image.open(PACKET / 'images' / f'{audit_id}_A.png').convert('RGB')
        b = Image.open(PACKET / 'images' / f'{audit_id}_B.png').convert('RGB')
        conditions = {}
        event_metrics = {}
        for name, category in (('visual_only', None), ('auto_category', ref['parsed']['category'])):
            forward = score(model, processor, initial_full, initial_crop, current, a, b,
                            category, token_ids)
            reversed_order = score(model, processor, initial_full, initial_crop, current, b, a,
                                   category, token_ids)
            conditions[name] = {'AB': forward, 'BA': reversed_order}
            event_metrics[name] = {'AB_correct': forward['choice'] == target_side,
                                   'BA_correct': reversed_order['choice'] == flipped(target_side),
                                   'both_correct': (forward['choice'] == target_side and
                                                    reversed_order['choice'] == flipped(target_side)),
                                   'order_consistent': reversed_order['choice'] == flipped(forward['choice'])}
        private_metrics.append(event_metrics)
        rows.append({'audit_id': audit_id, 'initial_image_sha256': ref['image_sha256'],
                     'current_image_sha256': sha(item['source_image']),
                     'auto_category': ref['parsed']['category'], 'conditions': conditions})
        print(json.dumps({'completed': len(rows), 'total': args.limit,
                          'elapsed_seconds': round(time.time() - started, 1)}), flush=True)
    summary = {}
    for name in ('visual_only', 'auto_category'):
        subset = [row[name] for row in private_metrics]
        summary[name] = {'events': len(rows),
                         'AB_correct': sum(row['AB_correct'] for row in subset),
                         'BA_correct': sum(row['BA_correct'] for row in subset),
                         'both_correct': sum(row['both_correct'] for row in subset),
                         'order_consistent': sum(row['order_consistent'] for row in subset),
                         'AB_choices': {letter: sum(row['conditions'][name]['AB']['choice'] == letter for row in rows) for letter in LETTERS},
                         'BA_choices': {letter: sum(row['conditions'][name]['BA']['choice'] == letter for row in rows) for letter in LETTERS}}
    result = {'status': 'complete_read_only_candidate_identity_probe',
              'scope': 'DepthTrack Train fit130, 24 deliberately selected native-error events',
              'seed': 2027, 'dtype': 'bfloat16', 'model': str(MODEL),
              'token_ids': dict(zip(LETTERS, token_ids)),
              'hidden_answer_manifest_published': False,
              'summary': summary, 'rows': rows, 'elapsed_seconds': time.time() - started,
              'limitations': ['Correct side is defined by Train GT candidate IoU, not human phrase review.',
                              'The automatic category is unverified and does not establish semantic contribution.',
                              'Only RGB is shown to Qwen; this is not an RGB-D tracker or recursive recovery.',
                              'The 24 hard events are selected, not a representative event-frequency sample.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
