"""Verify M94 image-order tensor binding and two short actual replies."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from audit_candidate_vlm import MODEL, REGISTER, PACKET


def inputs_for(processor, images):
    prompt = ('The first image marks one target with a red box. The second is a '
              'crop of that same target. The third is a later unmarked full frame. '
              'The fourth is candidate A and the fifth is candidate B. Which '
              'candidate shows the same physical object as the marked target, '
              'not merely an object of the same category? Answer A or B. '
              'Answer N if neither is the target, or U if the visible evidence '
              'cannot decide. Give exactly one letter.')
    content = [{'type': 'image', 'image': im,
                'max_pixels': (512 if index in (0, 2) else 256) * 28 * 28,
                'min_pixels': 128 * 28 * 28} for index, im in enumerate(images)]
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_images, videos = process_vision_info(messages)
    return processor(text=[chat], images=vision_images, videos=videos,
                     padding=True, return_tensors='pt')


def image_blocks(inputs):
    cursor = 0
    blocks = []
    for grid in inputs.image_grid_thw:
        count = int(grid.prod())
        values = inputs.pixel_values[cursor:cursor + count].contiguous()
        blocks.append({'grid': grid.tolist(), 'patches': count,
                       'pixel_sha256': hashlib.sha256(values.numpy().tobytes()).hexdigest()})
        cursor += count
    assert cursor == inputs.pixel_values.shape[0] and len(blocks) == 5
    return blocks


def main():
    output = Path('/root/autodl-tmp/sttrack_m94_candidate_vlm_20260928/inputs_verified.json')
    assert not output.exists()
    item = json.loads((PACKET / 'private_selection.json').read_text())['rows'][0]
    register = {row['sequence']: row for row in json.loads(REGISTER.read_text())}
    ref = register[item['sequence']]
    initial = Image.open(ref['initialization_image']).convert('RGB')
    crop = initial.crop(tuple(ref['generator_target_xyxy']))
    full = initial.copy()
    ImageDraw.Draw(full).rectangle(tuple(ref['generator_target_xyxy']), outline='red', width=2)
    current = Image.open(item['source_image']).convert('RGB')
    a = Image.open(PACKET / 'images' / f"{item['audit_id']}_A.png").convert('RGB')
    b = Image.open(PACKET / 'images' / f"{item['audit_id']}_B.png").convert('RGB')
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    ab = inputs_for(processor, (full, crop, current, a, b))
    ba = inputs_for(processor, (full, crop, current, b, a))
    ab_blocks, ba_blocks = image_blocks(ab), image_blocks(ba)
    assert ab_blocks[:3] == ba_blocks[:3]
    assert ab_blocks[3] == ba_blocks[4] and ab_blocks[4] == ba_blocks[3]
    assert ab_blocks[3]['pixel_sha256'] != ab_blocks[4]['pixel_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    replies = {}
    for name, inputs in (('AB', ab), ('BA', ba)):
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        replies[name] = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0]
    result = {'status': 'image_swap_binding_verified', 'audit_id': item['audit_id'],
              'AB_image_blocks': ab_blocks, 'BA_image_blocks': ba_blocks,
              'first_three_images_identical': True, 'candidate_images_swapped_exactly': True,
              'candidate_images_different': True, 'actual_short_replies': replies,
              'GT_or_identity_side_in_prompt': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'actual_short_replies': replies}), flush=True)


if __name__ == '__main__':
    main()
