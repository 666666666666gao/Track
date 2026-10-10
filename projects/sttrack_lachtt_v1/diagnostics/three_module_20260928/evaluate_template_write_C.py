"""Complete own-history inference on the prespecified C development32.

Only first-box initialization enters the tracker; GT is never opened here.
Current/future C use identical frozen P1, text and feature interfaces.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from analyze_train_states import sha
from full_dense_tracker import state_digest
from run_template_write_pilot import context
from template_write_C import TemplateWriteC
from template_write_features import CONTRACT, INPUT_DIM
from trusted_template_tracker import TrustedTemplateTracker

SPLIT_SHA = 'ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6'


def main():
    parser = argparse.ArgumentParser()
    for name in ['spec', 'trained-result', 'repository', 'checkpoint', 'clip-weight', 'bank', 'labels',
                 'final', 'C-final', 'C-result', 'split', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=['current', 'future'], required=True)
    parser.add_argument('--physical-gpu', choices=['0', '1'], required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.physical_gpu
    assert not args.output.exists() and sha(args.split) == SPLIT_SHA
    trained_C = json.loads(args.C_result.read_text())
    assert trained_C['status'] == 'complete_M122_C_fixed_teacher_fit120'
    assert trained_C['arm'] == args.arm and trained_C['epochs'] == 10 and trained_C['seed'] == 2027
    assert trained_C['input_contract'] == CONTRACT and trained_C['input_dim'] == INPUT_DIM
    assert trained_C['final_sha256'] == sha(args.C_final) and trained_C['split_sha256'] == SPLIT_SHA
    assert trained_C['model_source_sha256'] == sha(Path(__file__).with_name('template_write_C.py'))
    split = json.loads(args.split.read_text())
    spec, bank, actor, image, inputs = context(args)
    assert [r['sequence'] for r in spec['sequence_order']] == [r['sequence'] for r in split['sequence_order']]
    writer = TemplateWriteC().cuda().eval().requires_grad_(False)
    writer.load_state_dict(torch.load(args.C_final, map_location='cpu'), strict=True)
    tracker = TrustedTemplateTracker(actor, writer, args.arm)
    frozen = actor.frozen_digest(); decoder_before = state_digest(actor.decoder); C_before = state_digest(writer)
    args.output.mkdir()
    calls = 0; records = []; started = time.time()
    with (args.output / 'predictions.jsonl').open('w') as log:
        for index, case in enumerate(spec['sequence_order']):
            if split['sequence_order'][index]['C_split'] != 'development':
                continue
            tracker.initialize(image(case, 0), case['first_box'], bank['sequences'].index(case['sequence']))
            writes = qualified = local_calls = 0
            for frame in range(1, case['rgb_frames']):
                _, _, record = tracker.step(image(case, frame))
                assert record['frame'] == frame and record['C_arm'] == args.arm
                assert not record['template_write'] or record['native_rule_write_qualified']
                calls += 1; local_calls += 1
                writes += record['template_write']; qualified += record['native_rule_write_qualified']
                log.write(json.dumps(dict(sequence=case['sequence'], **record)) + '\n')
            log.flush()
            row = dict(sequence=case['sequence'], sequence_index=index, calls=local_calls,
                       native_rule_qualified_events=qualified, accepted_writes=writes)
            records.append(row); print(json.dumps(row), flush=True)
    assert len(records) == 32
    assert calls == sum(c['rgb_frames'] - 1 for i, c in enumerate(spec['sequence_order'])
                        if split['sequence_order'][i]['C_split'] == 'development')
    after = actor.frozen_digest(); decoder_after = state_digest(actor.decoder); C_after = state_digest(writer)
    assert frozen == after and decoder_before == decoder_after and C_before == C_after
    receipt = dict(status='complete_M122_C_development32_own_history_predictions', arm=args.arm, seed=2027,
        inputs=inputs, C_final_sha256=sha(args.C_final), C_result_sha256=sha(args.C_result),
        C_initial_state_sha256=trained_C['initial_state_sha256'], C_dataset_sha256=trained_C['common_dataset_sha256'],
        C_order_sha256=trained_C['order_sha256'], split_sha256=SPLIT_SHA,
        source_sha256=sha(Path(__file__)), tracker_source_sha256=sha(Path(__file__).with_name('trusted_template_tracker.py')),
        model_source_sha256=trained_C['model_source_sha256'], feature_source_sha256=sha(Path(__file__).with_name('template_write_features.py')),
        sequence_records=records, sequences=32, calls=calls, predictions_sha256=sha(args.output / 'predictions.jsonl'),
        frozen_states_before=frozen, frozen_states_after=after, decoder_before=decoder_before, decoder_after=decoder_after,
        C_before=C_before, C_after=C_after, GT_opened_after_first_box=False, GT_reinitializations_after_first_frame=0,
        optimizations=0, same_selected_candidate_score_box_feature_query=True,
        template_rule='native-qualified at interval50/.75, then C-current>.5 or C-future>0; no score multiplication',
        input_contract=CONTRACT, physical_gpu_environment=args.physical_gpu, cuda_device=torch.cuda.current_device(),
        public_Test_CDTB_VOT_used=False, current_IoU_or_future_reward_in_deployment_input=False,
        elapsed_seconds=time.time() - started)
    (args.output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
