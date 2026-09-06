"""Freeze same-slot lexical interventions without observing tracking outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import py_compile
import sys
import time

import torch

from run_controls import ROOT, PARENT, INTERFACE, TRAINING_SHA, RECURSIVE_SHA, parent_plans, sha, write


def main():
    started = time.time()
    torch.set_num_threads(1)
    assert not (ROOT / 'spec.json').exists()
    assert not (PARENT / 'recursive_result.json').exists()
    training, recursive, integration = parent_plans()
    preparation = json.loads((PARENT / 'text_preparation.json').read_text())
    records_path = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906/initial_captions_v2/records.jsonl')
    assert sha(records_path) == preparation['caption_records_sha256']
    records = {r['sequence']:r for r in [json.loads(line) for line in records_path.read_text().splitlines()]}
    assert len(records) == 152
    original_banks = {}
    for split in ['fit', 'development']:
        path = PARENT / ('text_' + split + '.pt')
        assert sha(path) == preparation['file_sha256'][path.name]
        original_banks[split] = torch.load(path, map_location='cpu')
    assert torch.equal(original_banks['fit']['empty'], original_banks['development']['empty'])
    source = {}
    for name, record in records.items():
        bank = original_banks[record['split']]
        index = bank['sequences'].index(name)
        phrases = [record['category']] + record['attributes']
        assert bank['initialization_phrases'][index] == phrases
        assert int(bank['mask'][index].sum()) == len(phrases)
        source[name] = dict(tokens=bank['tokens'][index], mask=bank['mask'][index], phrases=phrases)
    assert sha(PARENT / 'data_inventory.json') == training['inventory_sha256']
    inventory = {r['sequence']:r for r in json.loads((PARENT / 'data_inventory.json').read_text())['sequences_detail']}
    cases = recursive['cases']
    assert len(cases) == 22 and sum(c['frames'] for c in cases) == 33130
    for case in cases:
        row = inventory[case['sequence']]
        assert row['split'] == 'development' and row['first_box'] == case['init_bbox']
        assert sha(row['initial_rgb']) == records[case['sequence']]['image_sha256']
    sys.path.insert(0, str(INTERFACE))
    from initialization_text import InitializationTextBank, initialization_key
    interface_result = json.loads((INTERFACE / 'cpu_contract_result.json').read_text())
    assert interface_result['status'] == 'cpu_initialization_and_serialization_contract_passed'
    for name, digest in interface_result['source_sha256'].items():
        assert sha(INTERFACE / name) == digest
    protocol_sha = sha(INTERFACE / 'text_protocol.json')
    assert protocol_sha == interface_result['text_protocol_sha256']
    mappings = {name:[] for name in ['original', 'empty', 'category', 'mismatch', 'attributes']}
    excluded_attributes = []
    bank_shas = {}
    route_checks = 0
    for control in mappings:
        tokens, masks, keys = [], [], []
        for case in cases:
            name = case['sequence']
            target = records[name]
            count = len(source[name]['phrases'])
            changed = source[name]['tokens'].clone()
            donor = None
            if control == 'empty':
                changed[:count] = original_banks['fit']['empty']
            elif control == 'category':
                changed[1:count] = original_banks['fit']['empty']
            elif control in ['mismatch', 'attributes']:
                candidates = [n for n, r in records.items() if n != name
                    and len(source[n]['phrases']) == count
                    and ((r['category'] != target['category']) if control == 'mismatch'
                         else (r['category'] == target['category'] and r['attributes'] != target['attributes']))]
                if control == 'attributes' and not candidates:
                    excluded_attributes.append(dict(sequence=name, category=target['category'],
                        slots=count, reason='No different-attribute donor with the same automatic caption category and slot count'))
                    continue
                assert candidates
                donor = min(candidates, key=lambda n:hashlib.sha256(
                    ('2026|' + control + '|' + name + '|' + n).encode()).hexdigest())
                if control == 'mismatch':
                    changed[:count] = source[donor]['tokens'][:count]
                else:
                    changed[1:count] = source[donor]['tokens'][1:count]
                    assert torch.equal(changed[0], source[name]['tokens'][0])
            assert torch.equal(changed[count:], source[name]['tokens'][count:])
            assert torch.isfinite(changed).all()
            assert control == 'original' or not torch.equal(changed, source[name]['tokens'])
            if control in ['category', 'attributes']:
                assert torch.equal(changed[0], source[name]['tokens'][0])
            tokens.append(changed); masks.append(source[name]['mask'])
            keys.append(initialization_key(target['image_sha256'], case['init_bbox']))
            mapping = dict(sequence=name, slots=count, original_phrases=source[name]['phrases'],
                donor_sequence=donor, donor_split=records[donor]['split'] if donor else None,
                donor_phrases=source[donor]['phrases'] if donor else None)
            mappings[control].append(mapping)
        bank = dict(format='initialization_observation_v1', protocol_sha256=protocol_sha,
            keys=keys, tokens=torch.stack(tokens), mask=torch.stack(masks), empty=original_banks['fit']['empty'],
            control=control, caption_records_sha256=sha(records_path),
            sequences=[r['sequence'] for r in mappings[control]])
        path = ROOT / (control + '.pt')
        torch.save(bank, path)
        bank_shas[control] = sha(path)
        loaded = InitializationTextBank(path, bank_shas[control], protocol_sha)
        for i, row in enumerate(mappings[control]):
            case = next(c for c in cases if c['sequence'] == row['sequence'])
            info = loaded.info(inventory[row['sequence']]['initial_rgb'], case['init_bbox'])
            assert info['init_bbox'] == case['init_bbox']
            assert torch.equal(info['text_tokens'], tokens[i])
            assert torch.equal(info['text_mask'], source[row['sequence']]['mask'])
            route_checks += 1
    assert [len(mappings[n]) for n in mappings] == [22, 22, 22, 22, 14]
    assert len(excluded_attributes) == 8
    write(ROOT / 'mappings.json', dict(seed=2026, controls=mappings, attribute_exclusions=excluded_attributes,
        donor_rule='Lowest SHA256 of 2026|control|target|donor among eligible initialization captions; donor reuse allowed.',
        true_semantic_category_or_wrong_attributes_verified=False, new_captions_generated=0))
    files = ['prepare_controls.py', 'run_controls.py', 'queue_controls.py']
    for name in files:
        py_compile.compile(str(ROOT / name), doraise=True)
    spec = dict(status='frozen_before_parent_training_completion_and_development_results',
        observed_utc=datetime.now(timezone.utc).isoformat(), parent_training_spec_sha256=TRAINING_SHA,
        parent_recursive_spec_sha256=RECURSIVE_SHA, caption_records_sha256=sha(records_path),
        source_sha256={name:sha(ROOT / name) for name in files},
        interface_sha256=interface_result['source_sha256'], interface_cpu_result_sha256=sha(INTERFACE / 'cpu_contract_result.json'),
        text_protocol_sha256=protocol_sha, bank_sha256=bank_shas, mappings_sha256=sha(ROOT / 'mappings.json'),
        cases=cases, controls={name:dict(sequences=[r['sequence'] for r in mappings[name]],
            frames=sum(c['frames'] for c in cases if c['sequence'] in {r['sequence'] for r in mappings[name]}))
            for name in ['empty', 'category', 'mismatch', 'attributes']},
        workers={'0':['empty', 'mismatch'], '1':['category', 'attributes']},
        prefix_parity_sequences=['bag05_indoor', 'cup08_indoor', 'mobilephone02_indoor'], prefix_parity_frames=102,
        primary_content_gates={'empty':.001, 'mismatch':.001},
        gate_definition='Original automatic text pooled mean IoU minus each fixed-head control must be at least 0.001, after all original parent gates pass.',
        secondary_comparisons='Category retention with empty attribute vectors; same automatic-caption-category attribute donor subset. Report all, without a prespecified independent attribute claim.',
        checkpoint_policy='Reuse the single completed text final in the parent directory; no new training, head copy, or intermediate checkpoint.',
        same_slot_mask=True, initial_rgbd_reference_unchanged=True, template_motion_policy_unchanged=True,
        source_pool='All 152 existing DepthTrack Train initialization captions; 130 fit and 22 reused development.',
        limitations=['Automatic captions contain documented errors; original does not mean semantic ground truth.',
            'Same caption category is not verified same physical class; donor attributes are not verified false for the target.',
            'Cross-category mismatches are not a pure attribute test or a permutation; donor reuse is explicit.',
            'Empty/mismatched text can be outside the trained text distribution; compare with the separately trained visual control too.',
            'One seed; reused development22; no public test or generalization claim.'],
        monitoring_seconds=240, independent_review_pass=False,
        after_result='Low22 technical entry and current-anchor text preparation only if parent and primary content gates pass; no automatic full public evaluation.')
    assert not (PARENT / 'recursive_result.json').exists()
    write(ROOT / 'spec.json', spec)
    assert not torch.cuda.is_initialized()
    parent_plans()
    result = dict(status='content_banks_and_frozen_plan_ready', spec_sha256=sha(ROOT / 'spec.json'),
        mappings_sha256=sha(ROOT / 'mappings.json'), source_sha256=spec['source_sha256'],
        bank_sha256=bank_shas, bank_bytes=sum((ROOT / (n + '.pt')).stat().st_size for n in bank_shas),
        initialization_routes_checked=route_checks, parent_prediction_result_existed=False,
        controls=spec['controls'], attribute_exclusions=len(excluded_attributes),
        optimizer_steps=0, new_captions_generated=0, actual_tracking_calls=0, cuda_initialized=False,
        unchanged_parent_inference_files=len(integration['source_sha256']),
        elapsed_seconds=time.time() - started, independent_review_pass=False)
    write(ROOT / 'preparation_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
