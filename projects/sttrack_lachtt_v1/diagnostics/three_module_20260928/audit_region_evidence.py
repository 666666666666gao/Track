"""Read-only target-region evidence probe on 12 DepthTrack Train initializations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


REGISTER = Path('/root/autodl-tmp/sttrack_m86_initialization_audit_20260921/evidence/audit_register.json')
MODEL = Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct')
PILOT_IDS = ('002', '005', '006', '007', '010', '011', '012', '013', '023', '024', '026', '029')
FILL = (128, 128, 128)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unrelated_box(size, target):
    width, height = size
    left, top, right, bottom = target
    bw, bh = right - left, bottom - top
    corners = [(0, 0), (width - bw, 0), (0, height - bh), (width - bw, height - bh)]
    options = []
    for x, y in corners:
        box = (x, y, x + bw, y + bh)
        overlap = max(0, min(right, box[2]) - max(left, x)) * max(0, min(bottom, box[3]) - max(top, y))
        if overlap == 0:
            distance = (x + bw / 2 - (left + right) / 2) ** 2 + (y + bh / 2 - (top + bottom) / 2) ** 2
            options.append((distance, box))
    assert options
    return max(options)[1]


def images_for_condition(image, target, condition):
    crop = image.crop(target)
    full = image.copy()
    control = unrelated_box(image.size, target)
    if condition in ('target_full_masked', 'target_both_masked'):
        ImageDraw.Draw(full).rectangle(target, fill=FILL)
    if condition == 'target_both_masked':
        crop = Image.new('RGB', crop.size, FILL)
    if condition == 'unrelated_full_masked':
        ImageDraw.Draw(full).rectangle(control, fill=FILL)
    assert condition in ('original', 'target_full_masked', 'target_both_masked', 'unrelated_full_masked')
    ImageDraw.Draw(full).rectangle(target, outline='red', width=2)
    return full, crop, control


def score(model, processor, full, crop, category, yes_id, no_id):
    question = (f'Does the object marked by the red rectangle in the first image and shown in '
                f'the second image appear to be a {category}? Answer with exactly one word: Yes or No.')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': full, 'max_pixels': 512 * 28 * 28},
        {'type': 'image', 'image': crop, 'min_pixels': 128 * 28 * 28, 'max_pixels': 256 * 28 * 28},
        {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]
    pair = logits[[yes_id, no_id]].float()
    assert bool(torch.isfinite(pair).all()), 'Non-finite Yes/No logits'
    margin = float((pair[0] - pair[1]).item())
    probability = float(pair.softmax(dim=0)[0].item())
    return {'yes_minus_no_logit': margin, 'yes_two_token_probability': probability,
            'input_tokens': int(inputs.input_ids.numel())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='+', choices=PILOT_IDS, default=PILOT_IDS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    yes = processor.tokenizer.encode('Yes', add_special_tokens=False)
    no = processor.tokenizer.encode('No', add_special_tokens=False)
    assert len(yes) == len(no) == 1
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    register = {row['audit_id']: row for row in json.loads(REGISTER.read_text())}
    records = []
    start = time.time()
    for audit_id in args.ids:
        row = register[audit_id]
        assert row['split'] == 'fit' and sha(row['initialization_image']) == row['image_sha256']
        image = Image.open(row['initialization_image']).convert('RGB')
        target = tuple(row['generator_target_xyxy'])
        assert target[2] <= image.width and target[3] <= image.height
        category = row['parsed']['category']
        conditions = {}
        for condition in ('original', 'target_full_masked', 'unrelated_full_masked', 'target_both_masked'):
            full, crop, control = images_for_condition(image, target, condition)
            conditions[condition] = score(model, processor, full, crop, category, yes[0], no[0])
        records.append({'audit_id': audit_id, 'sequence': row['sequence'],
                        'image_sha256': row['image_sha256'], 'auto_category': category,
                        'target_box': target, 'unrelated_mask_box': control,
                        'conditions': conditions,
                        'target_full_minus_original': conditions['target_full_masked']['yes_minus_no_logit'] - conditions['original']['yes_minus_no_logit'],
                        'unrelated_full_minus_original': conditions['unrelated_full_masked']['yes_minus_no_logit'] - conditions['original']['yes_minus_no_logit'],
                        'target_both_minus_original': conditions['target_both_masked']['yes_minus_no_logit'] - conditions['original']['yes_minus_no_logit']})
        print(json.dumps({'completed': len(records), 'total': len(args.ids),
                          'audit_id': audit_id, 'elapsed_seconds': round(time.time() - start, 1)}), flush=True)
    result = {'status': 'complete_read_only_region_evidence_probe', 'scope': 'DepthTrack Train fit130 first frames',
              'seed': 2027, 'model': str(MODEL), 'dtype': 'bfloat16',
              'yes_token': yes[0], 'no_token': no[0], 'model_input': 'red-box full image plus target crop',
              'conditions': ['original', 'target_full_masked', 'unrelated_full_masked', 'target_both_masked'],
              'mask_fill_rgb': FILL, 'records': records, 'elapsed_seconds': time.time() - start,
              'semantic_ground_truth': False, 'limitations': [
                  'Automatic category may be wrong and is not an independently verified label.',
                  'Gray masking changes image distribution; the target-full and unrelated-full masks are the paired full-image control with the same crop.',
                  'The target-both mask also alters the second image, so it is not paired with the unrelated-full mask.',
                  'Yes/No two-token probability is an uncalibrated VLM diagnostic, not identity confidence.',
                  'No tracking model, weights, crops, or final benchmark metrics are changed.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
