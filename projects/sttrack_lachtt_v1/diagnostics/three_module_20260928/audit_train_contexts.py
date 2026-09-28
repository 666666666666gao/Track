"""Audit complete M98 input coverage/linkage without GT or text."""

import argparse
import json
from pathlib import Path

import torch

from collect_train_states import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prep = json.loads((args.cache / 'preparation.json').read_text())
    assert prep['inference_inputs_sha256'] == sha(args.cache / 'inference_inputs.json')
    plans = {r['sequence']: r for r in json.loads((args.cache/'inference_inputs.json').read_text())}
    rows, seen = [], set()
    for shard in (0, 1):
        receipt = json.loads((args.contexts / f'shard{shard}.json').read_text())
        original = json.loads((args.cache / f'collect_shard{shard}.json').read_text())
        assert receipt['status'] == 'complete_context_collection_only' and not receipt['smoke']
        assert receipt['source_preparation_sha256'] == sha(args.cache/'preparation.json')
        assert receipt['checkpoint_sha256'] == original['checkpoint_sha256']
        assert not receipt['GT_loaded'] and not receipt['text_loaded'] and receipt['optimizer_steps'] == 0
        assert not receipt['auxiliary_query_committed'] and not receipt['prototype_state_committed']
        originals = {r['sequence']: r for r in original['sequences']}
        assert {r['sequence'] for r in receipt['sequences']} == set(originals)
        assert receipt['frames'] == original['frames'] and receipt['events'] == original['events']
        assert receipt['bytes'] == sum(r['bytes'] for r in receipt['sequences'])
        for row in receipt['sequences']:
            sequence = row['sequence']
            assert sequence not in seen and plans[sequence]['shard'] == shard
            seen.add(sequence)
            path = args.contexts / 'features' / f'{sequence}.pt'
            assert sha(path) == row['feature_sha256'] and path.stat().st_size == row['bytes']
            assert row['original_feature_sha256'] == originals[sequence]['feature_sha256']
            assert sha(args.cache/'features'/f'{sequence}.pt') == row['original_feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert data['sequence'] == sequence and data['split'] == row['split'] == plans[sequence]['split']
            assert data['event_frames'] == plans[sequence]['event_frames']
            assert data['original_feature_sha256'] == row['original_feature_sha256']
            assert not data['GT_loaded'] and not data['text_loaded']
            assert not data['auxiliary_query_committed'] and not data['prototype_state_committed']
            assert row['candidate_regions_exact'] and row['candidate_boxes_exact'] and row['initial_region_exact']
            assert row['maximum_box_error_px'] <= 1e-4 and row['maximum_score_error'] <= 1e-6
            events = len(data['event_frames'])
            assert events == row['events'] and row['frames'] == max(data['event_frames'])
            shapes = dict(initial_context=(2, 12, 768), initial_mask=(16,), initial_context_mask=(12,),
                          initial_depth_valid=(), contexts=(events, 10, 2, 12, 768),
                          candidate_mask=(events, 10, 16), context_mask=(events, 10, 12), depth_valid=(events, 10))
            for key, shape in shapes.items():
                assert tuple(data[key].shape) == shape and bool(torch.isfinite(data[key]).all())
                expected_dtype = torch.bool if 'mask' in key else torch.float16 if key in ('contexts', 'initial_context') else torch.float32
                assert data[key].dtype == expected_dtype
            for key in ('initial_depth_valid', 'depth_valid'):
                assert bool(((data[key] >= 0) & (data[key] <= 1)).all())
            rows.append(dict(sequence=sequence, split=row['split'], events=events, frames=row['frames'],
                             candidate_samples_without_support=int((~data['candidate_mask'].any(-1)).sum()),
                             context_samples_without_support=int((~data['context_mask'].any(-1)).sum()),
                             maximum_box_error_px=row['maximum_box_error_px'], maximum_score_error=row['maximum_score_error']))
    assert seen == set(plans) and len(rows) == 152
    assert sum(r['events'] for r in rows) == 3502 and sum(r['frames'] for r in rows) == 219194
    result = dict(status='complete_context_input_audit_only', sequences=len(rows),
                  events=sum(r['events'] for r in rows), frames=sum(r['frames'] for r in rows),
                  maximum_box_error_px=max(r['maximum_box_error_px'] for r in rows),
                  maximum_score_error=max(r['maximum_score_error'] for r in rows),
                  records=rows, GT_loaded=False, text_loaded=False, optimizer_steps=0,
                  no_public_evaluation=True, no_semantic_or_recursive_gain_claim=True)
    path = args.contexts / 'input_audit.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
