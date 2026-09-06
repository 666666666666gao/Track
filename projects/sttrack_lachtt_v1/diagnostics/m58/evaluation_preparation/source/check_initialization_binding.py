"""CPU contract checks on existing Train initialization records, not tracking."""
from datetime import datetime, timezone
import json
from pathlib import Path
import py_compile
import shutil
import tempfile
import time
import unittest

import numpy as np
import torch
import trax

from initialization_text import InitializationTextBank, initialization_key, sha, vot_wire_bbox
from run_semantic_ope import write_predictions


ROOT = Path(__file__).resolve().parent
TRAIN = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
CAPTIONS = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906/initial_captions_v2')
EXPECTED_TRAINING_SPEC = 'c592109d10579efac4dce5d3dd6f4881900c2c264acc3db3b63a6021855ea3b7'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.time()
    torch.set_num_threads(1)
    assert sha(TRAIN / 'training_spec.json') == EXPECTED_TRAINING_SPEC
    spec = json.loads((TRAIN / 'training_spec.json').read_text())
    assert sha(TRAIN / 'data_inventory.json') == spec['inventory_sha256']
    assert sha(TRAIN / 'integration.json') == spec['integration_sha256']
    integration = json.loads((TRAIN / 'integration.json').read_text())
    for name, digest in integration['source_sha256'].items():
        assert sha(TRAIN / 'code' / name) == digest
    inventory = json.loads((TRAIN / 'data_inventory.json').read_text())
    preparation = json.loads((TRAIN / 'text_preparation.json').read_text())
    assert sha(CAPTIONS / 'plan.json') == preparation['caption_plan_sha256']
    assert sha(CAPTIONS / 'records.jsonl') == preparation['caption_records_sha256']
    caption_plan = json.loads((CAPTIONS / 'plan.json').read_text())
    caption_result = json.loads((CAPTIONS / 'result.json').read_text())
    assert caption_result['plan_sha256'] == preparation['caption_plan_sha256']
    assert caption_result['records_sha256'] == preparation['caption_records_sha256']
    for name, digest in preparation['file_sha256'].items():
        assert sha(TRAIN / name) == digest
    for path, digest in preparation['clip_source_sha256'].items():
        assert sha(path) == digest
    protocol = {key: caption_plan[key] for key in ['prompt', 'model_sha256', 'generation',
        'image_pixels', 'torch_dtype', 'attention', 'seed', 'sequence_name_in_model_prompt',
        'category_hint_in_model_prompt', 'subsequent_images_used', 'manual_adjudication_used']}
    protocol.update(format='m58_initialization_text_protocol_v1',
        initialization_only=True, input_modality='RGB', input_region='Actual initialization xywh',
        crop_xyxy='[max(0,int(x)), max(0,int(y)), min(W,int(x+w+.999999)), min(H,int(y+h+.999999))]',
        images=['RGB full image with the crop rectangle drawn red at width 2',
                'Tight crop from the same unmarked RGB image'],
        processor=dict(use_fast=False, padding=True, add_generation_prompt=True),
        decoding=dict(skip_special_tokens=True, clean_up_tokenization_spaces=False),
        json_parser='Strip whitespace, then strip only an exact ```json\\n...\\n``` wrapper; parse once.',
        caption_package_versions=caption_result['package_versions'],
        text_encoder=dict(weight_sha256=preparation['encoder_sha256'],
            python_source_sha256={Path(p).name: h for p, h in preparation['clip_source_sha256'].items()},
            device='cpu', dtype='float32', normalize=False, truncate=False,
            slots=5, channels=768, order='category, then attributes in generated order',
            unused_slots='zero vector and false mask', empty_control='CLIP encoding of the empty string'))
    save(ROOT / 'text_protocol.json', protocol)
    protocol_sha = sha(ROOT / 'text_protocol.json')
    original = {split: torch.load(TRAIN / ('text_' + split + '.pt'), map_location='cpu')
                for split in ['fit', 'development']}
    assert torch.equal(original['fit']['empty'], original['development']['empty'])
    captions = {r['sequence']: r for r in caption_plan['rows']}
    keys, tokens, masks, source_rows = [], [], [], []
    for row in inventory['sequences_detail']:
        bank = original[row['split']]
        index = bank['sequences'].index(row['sequence'])
        image_sha = sha(row['initial_rgb'])
        assert image_sha == captions[row['sequence']]['image_sha256']
        keys.append(initialization_key(image_sha, row['first_box']))
        tokens.append(bank['tokens'][index]); masks.append(bank['mask'][index])
        source_rows.append(dict(sequence=row['sequence'], split=row['split'],
                                image_sha256=image_sha, original_index=index))
    assert len(keys) == len(set(keys)) == 152
    assert len({Path(r['initial_rgb']).name for r in inventory['sequences_detail']}) == 1
    packed = dict(format='initialization_observation_v1', protocol_sha256=protocol_sha,
        keys=keys, tokens=torch.stack(tokens), mask=torch.stack(masks), empty=original['fit']['empty'],
        scope='Existing DepthTrack Train initialization records only; CPU interface fixture',
        original_text_preparation_sha256=sha(TRAIN / 'text_preparation.json'), sources=source_rows)
    bank_path = ROOT / 'train_initialization_fixture.pt'
    torch.save(packed, bank_path)
    bank = InitializationTextBank(bank_path, sha(bank_path), protocol_sha)
    for row in inventory['sequences_detail']:
        info = bank.info(row['initial_rgb'], row['first_box'])
        prior = original[row['split']]
        index = prior['sequences'].index(row['sequence'])
        assert info['init_bbox'] == row['first_box']
        assert torch.equal(info['text_tokens'], prior['tokens'][index])
        assert torch.equal(info['text_mask'], prior['mask'][index])
        assert torch.equal(info['empty_text'], prior['empty'])
    # The real TraX constructor proves the conversion, including decimal inputs.
    wire_cases = [[1.1, 2.2, 30.3, 40.4]] + [r['first_box'] for r in inventory['sequences_detail']]
    for bbox in wire_cases:
        assert list(trax.Rectangle.create(*bbox).bounds()) == vot_wire_bbox(bbox)
    sample = inventory['sequences_detail'][0]
    with tempfile.TemporaryDirectory(prefix='m58_eval_contract_', dir=str(ROOT)) as tmp:
        folder = Path(tmp)
        copied = folder / 'different_path.jpg'
        shutil.copyfile(sample['initial_rgb'], copied)
        assert torch.equal(bank.info(copied, sample['first_box'])['text_tokens'], tokens[0])
        # Same image, two distinct initialized objects: the box must select the text.
        bbox_a = [1.1, 2.2, 30.3, 40.4]
        bbox_b = [11.1, 2.2, 30.3, 40.4]
        wire_a, wire_b = vot_wire_bbox(bbox_a), vot_wire_bbox(bbox_b)
        synthetic = dict(packed)
        synthetic.update(keys=[initialization_key(sha(copied), b) for b in [wire_a, wire_b]],
                         tokens=packed['tokens'][:2], mask=packed['mask'][:2])
        assert not torch.equal(synthetic['tokens'][0], synthetic['tokens'][1])
        synthetic_path = folder / 'two_regions.pt'
        torch.save(synthetic, synthetic_path)
        dual = InitializationTextBank(synthetic_path, sha(synthetic_path), protocol_sha)
        for i, bbox in enumerate([wire_a, wire_b]):
            info = dual.info(copied, list(trax.Rectangle.create(*bbox).bounds()))
            assert info['init_bbox'] == bbox
            assert torch.equal(info['text_tokens'], synthetic['tokens'][i])
        test = unittest.TestCase()
        test.assertRaises(KeyError, dual.info, copied, bbox_a)
        test.assertRaises(KeyError, dual.info, copied, [21.1, 2.2, 30.3, 40.4])
        boxes = np.array([[1.1, 2.2, 30.3, 40.4], [1.23456789, 2.3, 29.7654321, 41.2],
                          [1.34567891, 2.4, 31.23456789, 40.2]])
        scores = np.array([1., .533012345, .060412345])
        output_receipt = write_predictions(folder, 'cpu_fixture', boxes, scores)
        restored = np.loadtxt(folder / 'cpu_fixture_all_scores.txt')
        assert np.array_equal(np.argsort(scores), np.argsort(restored))
        roundtrip_error = float(np.abs(restored - scores).max())
    names = ['initialization_text.py', 'semantic_runtime.py', 'run_semantic_ope.py',
             'run_semantic_vot.py', 'check_initialization_binding.py', 'm39_vot_bridge.py']
    for name in names:
        py_compile.compile(str(ROOT / name), doraise=True)
    assert sha(ROOT / 'm39_vot_bridge.py') == '230acf10f378a6babfacf9979ea07a1ce89c34952cc0c6b9568376249e265316'
    assert not torch.cuda.is_initialized()
    assert sha(TRAIN / 'training_spec.json') == EXPECTED_TRAINING_SPEC
    for name, digest in integration['source_sha256'].items():
        assert sha(TRAIN / 'code' / name) == digest
    result = dict(status='cpu_initialization_and_serialization_contract_passed',
        observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started,
        training_spec_sha256=EXPECTED_TRAINING_SPEC, original_inference_files_unchanged=len(integration['source_sha256']),
        original_text_banks_unchanged=True, train_initializations_verified=152,
        same_filename_across_all_inputs=True, exact_token_mask_and_bbox_matches=152,
        actual_trax_wire_comparisons=len(wire_cases), noninteger_wire_example=dict(input=wire_cases[0],
            actual=list(trax.Rectangle.create(*wire_cases[0]).bounds())),
        same_image_distinct_region_routes=2, same_image_different_path_route=True,
        unknown_or_unconverted_region_rejected=True, ope_fixture_frames=output_receipt['frames'],
        maximum_confidence_roundtrip_error=roundtrip_error, confidence_rank_preserved=True,
        text_protocol_sha256=protocol_sha, train_fixture_sha256=sha(bank_path), train_fixture_bytes=bank_path.stat().st_size,
        source_sha256={name:sha(ROOT / name) for name in names}, cuda_initialized=False,
        optimizer_steps=0, actual_semantic_tracking_calls=0, public_test_images_opened=0,
        new_captions_generated=0, final_evaluation_bundle_created=False,
        gpu_wrapper_parity_verified=False, trax_server_exchange_verified=False,
        independent_review_pass=False,
        scope='CPU data-routing/output checks only; no tracking gain, final-weight promotion, or public evaluation.')
    save(ROOT / 'cpu_contract_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
