"""Causal local VLM pilot: template references and current ordinal descriptions.

This emits suggestions only. It never edits STTrack, runs an optimizer, or
reports benchmark metrics. Current-frame GT is absent from the model input.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


IDENTITY = (
    'Identify the same physical target by stable visible appearance, markings, shape, '
    'and visual correspondence. Do not use an ordinal position as the identity. '
    'Set similar_count, left_to_right_rank, and relative_description to null. ')
RELATIVE = (
    'Also reason about the target among visually similar objects in the CURRENT frame: '
    'count visible similar objects and describe its current rank from LEFT to RIGHT '
    'using object center x coordinates, and its relation to adjacent objects. '
    'Do not assume its rank in the initialization frame remains fixed: objects and camera '
    'can move or cross. If count or ordering is unclear, set only those fields to null. '
    'Set identity_uncertain to true only if the target identity itself is unclear. ')
COMMON = (
    'You are observing a single-object tracking video causally. Image 1 is the '
    'initialization frame with the target marked by a red box. Image 2 is the '
    'initial target crop. {dynamic} The final two images are the immediately previous '
    'observed frame and the CURRENT frame, in tracking order. The previous frame has '
    'a YELLOW box showing the previous STTrack prediction. This is a causal identity '
    'reference, not ground truth; use image correspondence to follow that instance '
    'into the CURRENT frame and compare its appearance with the initialization. '
    'Backward playback is '
    'possible; do not assume a fixed motion direction. Locate the SAME physical '
    'target in the final CURRENT image; do not just locate an object of the same category. '
    'There is no target mark in the current image. If it is occluded or indistinguishable '
    'from another instance, report uncertainty and a null box. {condition}'
    'Return only one JSON object with these keys: target_visible (boolean), '
    'identity_uncertain (boolean), bbox_xyxy_pixels ([x1,y1,x2,y2] in native PIXELS '
    'of the final CURRENT 640-by-360 image, or null; never normalized), stable_identity (short string), '
    'similar_count (integer or null), left_to_right_rank (one-based integer or null), '
    'relative_description (short string or null). Never use a frame number as identity. '
    'Do not invent a hidden target or copy coordinates from a reference image.')


def load_rgb(folder, index):
    return Image.open(folder / 'color' / f'{index + 1:08d}.jpg').convert('RGB')


def crop(image, box):
    x, y, w, h = box
    return image.crop((max(0, math.floor(x)), max(0, math.floor(y)),
                       min(image.width, math.ceil(x + w)), min(image.height, math.ceil(y + h))))


def prepare_cases(m41, m39):
    inputs = json.loads((m41 / 'inputs.json').read_text())
    result = []
    for sequence in ['cup02_indoor_1', 'toy09_indoor_1', 'shoes02_indoor_1', 'cube05_indoor_5']:
        values = sorted([c for c in inputs if c['sequence'] == sequence], key=lambda c: c['anchor'])
        for event in [values[0], values[len(values) // 2], values[-1]]:
            confidence_path = (m39 / 'default/master/results/sttrack_m39_default_low22/baseline' /
                               sequence / f'{sequence}_{event["anchor"]:08d}_confidence.value')
            confidences = confidence_path.read_text().splitlines()
            for age in ['before10', 'onset']:
                step = max(1, event['progress'] - 10) if age == 'before10' else event['progress']
                writes = [i for i in range(50, step, 50) if float(confidences[i]) > .75]
                last_write = writes[-1] if writes else 0
                dynamic_box = event['expected_boxes'][last_write - 1] if last_write else event['init_bbox']
                result.append(dict(key=event['key'] + ':' + age, sequence=sequence, anchor=event['anchor'],
                    direction=event['direction'], step=step, current_frame=event['anchor'] + event['direction'] * step,
                    init_bbox=event['init_bbox'], dynamic_frame=event['anchor'] + event['direction'] * last_write,
                    previous_bbox=event['init_bbox'] if step == 1 else event['expected_boxes'][step - 2],
                    dynamic_bbox=dynamic_box, dynamic_step=last_write,
                    protected_bbox=event['expected_boxes'][step - 1]))
    assert len(result) == 24
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--m41', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        m39 = Path('/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902')
        cases = prepare_cases(args.m41, m39)
        (root / 'inputs.json').write_text(json.dumps(cases, indent=2) + '\n')
        spec = dict(schema='sttrack_position_text_causal_pilot_v2', count=24,
            selection='First/median/last anchor by numeric anchor order in each of four fixed sequences, onset and onset-minus-ten. Selected before VLM outputs; no favorable-case selection.',
            variants=['initial_identity', 'default_templates_identity', 'default_templates_relative'],
            prompts=dict(common=COMMON, identity=IDENTITY, relative=RELATIVE),
            model='/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct',
            model_config_sha256=hashlib.sha256(Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct/config.json').read_bytes()).hexdigest(),
            sequence_root='/root/autodl-tmp/VOT-RGBD2022/sequences',
            max_new_tokens=320, do_sample=False, no_tracker_commit=True, no_training=True,
            observation_change='v1 returned initialization-scale coordinates and lacked a target mark in the immediately previous image. v2 uses native pixels, explicit image roles, and previous predicted bbox as a causal reference. Same 24 windows, variants, model and decision gate; no GT previous/current target mark.',
            inputs_sha256=hashlib.sha256((root / 'inputs.json').read_bytes()).hexdigest(),
            success_rule='Relative vs default-template identity increases correct localization (IoU>=.5) across at least two sequences and does not increase wrong accepted localization (IoU<=.1). Template reference contribution is separately compared to initial-only identity.',
            limitations='GT-derived time sampling is diagnosis only. Rank/count lack independent instance annotations; report target localization and qualitative rank consistency, not rank accuracy. This is not online tracker or VOT validation.')
        (root / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
        print(json.dumps(dict(prepared_cases=len(cases), calls=len(cases) * 3)))
        return
    spec = json.loads((root / 'spec.json').read_text())
    assert hashlib.sha256((root / 'inputs.json').read_bytes()).hexdigest() == spec['inputs_sha256']
    cases = json.loads((root / 'inputs.json').read_text())
    torch.set_num_threads(2)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(spec['model'], local_files_only=True,
                 torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}).eval()
    processor = AutoProcessor.from_pretrained(spec['model'], local_files_only=True, use_fast=False)
    start = time.monotonic()
    with (root / 'responses.jsonl').open('x') as stream:
        for case in cases:
            folder = Path(spec['sequence_root']) / case['sequence']
            initial = load_rgb(folder, case['anchor'])
            marked = initial.copy()
            x, y, w, h = case['init_bbox']
            ImageDraw.Draw(marked).rectangle((x, y, x + w, y + h), outline='red', width=2)
            initial_crop = crop(initial, case['init_bbox'])
            dynamic = crop(load_rgb(folder, case['dynamic_frame']), case['dynamic_bbox'])
            previous = load_rgb(folder, case['current_frame'] - case['direction'])
            px, py, pw, ph = case['previous_bbox']
            ImageDraw.Draw(previous).rectangle((px, py, px + pw, py + ph), outline='yellow', width=2)
            current = load_rgb(folder, case['current_frame'])
            for variant in spec['variants']:
                images = [marked, initial_crop]
                dynamic_prompt = ''
                if variant != 'initial_identity':
                    images.append(dynamic)
                    dynamic_prompt = ('Image 3 is the latest dynamic template from the default tracker, '
                        'selected strictly before the current frame. It is a model prediction and '
                        'could be wrong; check it against the initialization identity. ')
                images += [previous, current]
                condition = spec['prompts']['relative'] if variant.endswith('relative') else spec['prompts']['identity']
                prompt = spec['prompts']['common'].format(dynamic=dynamic_prompt, condition=condition)
                roles = ['INITIALIZATION: red target box.', 'INITIAL TARGET CROP.']
                if variant != 'initial_identity':
                    roles.append('OLDER DYNAMIC TARGET CROP from default tracker.')
                roles += ['IMMEDIATELY PREVIOUS FRAME: yellow box is the previous model prediction.',
                          'CURRENT FRAME: localize the same target here in native 640x360 pixel coordinates.']
                content = []
                for role, im in zip(roles, images):
                    content.extend([{'type': 'text', 'text': role}, {'type': 'image', 'image': im, 'max_pixels': 640 * 480}])
                content.append({'type': 'text', 'text': prompt})
                message = [{'role': 'user', 'content': content}]
                chat = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(message)
                inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors='pt').to(model.device)
                t0 = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=spec['max_new_tokens'], use_cache=True)
                raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                record = dict(key=case['key'], variant=variant, raw=raw, latency_seconds=time.monotonic() - t0,
                              dynamic_step=case['dynamic_step'], image_size=list(current.size))
                stream.write(json.dumps(record) + '\n'); stream.flush()
                print(json.dumps(dict(key=case['key'], variant=variant, latency_seconds=record['latency_seconds'],
                                      elapsed=time.monotonic() - start)), flush=True)
    (root / 'receipt.json').write_text(json.dumps(dict(status='complete', cases=len(cases), calls=len(cases) * 3,
          elapsed_seconds=time.monotonic() - start, tracker_commits=0, training_steps=0), indent=2) + '\n')


if __name__ == '__main__':
    main()
