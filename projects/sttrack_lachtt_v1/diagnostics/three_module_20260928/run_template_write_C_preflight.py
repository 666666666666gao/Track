"""R3 target preflight only: constant-action fixtures, not trained C results.

Run after the terminal R2 raw audit and separate source/deployment admission.
Two own histories cover the fixed first 256 noninitialization Train frames.
No GT beyond the legal initialization box is read. No optimizer is created.
"""
import argparse
from copy import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from analyze_train_states import sha
from full_dense_tracker import state_digest
from run_template_write_pilot import context
from template_write_features import CONTRACT, INPUT_DIM
from template_write_forks import (assert_same_submission, capture_rng, copy_tracker_state,
                                  restore_rng)
from trusted_template_tracker import TrustedTemplateTracker

FRAMES = 256


class ConstantActionFixture:
    """An explicit parameter-free test control; no C weights are constructed."""
    training = False

    def __init__(self, value):
        self.value = value
        self.inputs = []

    def parameters(self):
        return ()

    def __call__(self, features):
        assert features.shape == (1, INPUT_DIM) and features.dtype == torch.float32
        assert bool(torch.isfinite(features).all())
        self.inputs.append(features.detach().cpu().clone())
        return features.new_full((1,), self.value)


def assert_same_rng(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2]) and torch.equal(left[3], right[3])


def assert_same_output(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def assert_same_record(left, right, write):
    assert left.keys() <= right.keys()
    expected = dict(left, template_write=write)
    assert expected == {key: right[key] for key in left}


def saved_history(actor, record):
    return dict(record=dict(record),
        query=[tensor.detach().cpu().clone() for tensor in actor.native.track_query_before],
        selected_feature=actor.selected_feature.detach().cpu().clone())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for name in ['spec', 'trained-result', 'repository', 'checkpoint', 'clip-weight', 'bank',
                 'labels', 'final', 'teacher-root', 'teacher-audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--physical-gpu', choices=['0', '1'], required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.physical_gpu and not args.output.exists()
    audit = json.loads(args.teacher_audit.read_text())
    assert audit['review_call_status'] == 'completed' and audit['blocking_count'] == 0
    assert audit['R2_terminal_raw_audit_complete']
    assert sha(args.teacher_root / 'result.json') == audit['audited_teacher_result_sha256']
    assert (args.teacher_root / 'controller.exit').read_text().strip() == '0'
    teacher = json.loads((args.teacher_root / 'result.json').read_text())
    assert teacher['status'] == 'complete_M122_full_train_template_teacher'
    spec, bank, actor, image, inputs = context(args)
    assert teacher['inputs'] == inputs
    case = spec['sequence_order'][0]
    assert case['sequence'] == 'cube04_indoor' and case['rgb_frames'] > FRAMES
    frozen_before = actor.frozen_digest()
    decoder_before = state_digest(actor.decoder)
    initial_image = image(case, 0)
    bank_index = bank['sequences'].index(case['sequence'])
    rng = capture_rng()
    actor.initialize(initial_image, case['first_box'], bank_index)
    original_initialized_rng = capture_rng()
    other = copy(actor)
    other.native = copy(actor.native)
    assert other is not actor and other.native is not actor.native
    always = ConstantActionFixture(1.)
    wrapped = TrustedTemplateTracker(other, always, 'future')
    restore_rng(rng)
    wrapped.initialize(initial_image, case['first_box'], bank_index)
    assert_same_rng(original_initialized_rng, capture_rng())
    assert actor.native.state == other.native.state == list(case['first_box'])
    assert actor.native.frame_id == other.native.frame_id == 0
    assert actor.native.track_query_before is other.native.track_query_before is None
    assert actor.native.state is not other.native.state
    assert actor.native.z_dict is not other.native.z_dict
    assert actor.native.z_patch_arr is not other.native.z_patch_arr
    assert actor.initial is not other.initial
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actor.native.z_dict, other.native.z_dict))
    assert all(torch.equal(a, b) for a, b in zip(actor.native.z_dict, other.native.z_dict))
    assert np.array_equal(actor.native.z_patch_arr, other.native.z_patch_arr)
    assert actor.initial.keys() == other.initial.keys()
    assert all(torch.equal(actor.initial[key], other.initial[key]) for key in actor.initial)
    for key in ['words', 'word_mask', 'empty']:
        assert torch.equal(getattr(actor, key), getattr(other, key))
    restore_rng(original_initialized_rng)
    rows = []
    rejects = []
    writes = 0
    calls = 0
    started = time.time()
    args.output.mkdir()
    with (args.output / 'records.jsonl').open('w') as log:
        for frame in range(1, FRAMES + 1):
            current_image = image(case, frame)
            before_rng = capture_rng()
            if frame % 50 == 0:
                before = copy_tracker_state(other, other.native.z_dict, other.native.z_patch_arr)
            original_out, _, original_record = actor.step(current_image)
            calls += 1
            original_out = {key: tensor.detach().clone() for key, tensor in original_out.items()}
            original_post = copy_tracker_state(actor, actor.native.z_dict, actor.native.z_patch_arr)
            original_saved = saved_history(actor, original_record)
            original_rng = capture_rng()
            restore_rng(before_rng)
            wrapped_out, _, wrapped_record = wrapped.step(current_image)
            calls += 1
            wrapped_rng = capture_rng()
            assert_same_rng(original_rng, wrapped_rng)
            assert_same_output(original_out, wrapped_out)
            assert_same_record(original_record, wrapped_record, original_record['template_write'])
            assert_same_submission(actor, original_post)
            assert_same_submission(original_post, other)
            saved = dict(frame=frame, original=original_saved,
                         always_write=saved_history(other, wrapped_record))
            if original_record['template_write']:
                writes += 1
                assert frame % 50 == 0 and wrapped_record['C_prediction'] == 1.
                reject_fixture = ConstantActionFixture(0.)
                reject_tracker = TrustedTemplateTracker(before, reject_fixture, 'future')
                reject_tracker.reference = wrapped.reference
                templates = [tensor.detach().clone() for tensor in before.native.z_dict]
                patch = before.native.z_patch_arr.copy()
                restore_rng(before_rng)
                rejected_out, _, rejected_record = reject_tracker.step(current_image)
                calls += 1
                assert_same_rng(original_rng, capture_rng())
                assert_same_output(original_out, rejected_out)
                assert_same_record(original_record, rejected_record, False)
                assert rejected_record['native_rule_write_qualified'] and rejected_record['C_prediction'] == 0.
                assert before.native.state == other.native.state
                assert before.native.frame_id == other.native.frame_id
                assert len(before.native.track_query_before) == len(other.native.track_query_before)
                assert all(torch.equal(a, b) for a, b in zip(before.native.track_query_before, other.native.track_query_before))
                assert torch.equal(before.selected_feature, other.selected_feature)
                assert all(torch.equal(a, b) for a, b in zip(templates, before.native.z_dict))
                assert np.array_equal(patch, before.native.z_patch_arr)
                assert torch.equal(always.inputs[-1], reject_fixture.inputs[0])
                rejects.append(dict(frame=frame, C_input=reject_fixture.inputs[0],
                    retained_templates=[tensor.detach().cpu().clone() for tensor in before.native.z_dict],
                    retained_patch=before.native.z_patch_arr.copy(), retained_history=saved_history(before, rejected_record)))
            restore_rng(original_rng)
            rows.append(saved)
            log.write(json.dumps(dict(frame=frame, original=original_record, always_write=wrapped_record)) + '\n')
            if frame % 50 == 0:
                log.flush()
                print(json.dumps(dict(frame=frame, native_qualified_writes=writes, calls=calls)), flush=True)
    assert writes >= 1 and len(always.inputs) == len(rejects) == writes
    assert calls == 2 * FRAMES + writes
    frozen_after = actor.frozen_digest()
    decoder_after = state_digest(actor.decoder)
    assert frozen_before == frozen_after and decoder_before == decoder_after
    raw_path = args.output / 'raw_histories.pt'
    torch.save(dict(full_256_frame_histories=rows, actual_qualified_rejection_checks=rejects,
                    always_write_C_inputs=always.inputs, fixture_values=[1., 0.]), raw_path)
    receipt = dict(status='complete_M122_R3_constant_action_preflight256', inputs=inputs, seed=2027,
        sequence=case['sequence'], frames_per_history=FRAMES, track_calls=calls, native_qualified_writes=writes,
        same_GPU_for_all_calls=True, physical_gpu_environment=args.physical_gpu, cuda_device=torch.cuda.current_device(),
        own_original_and_always_write_histories=True, always_write_output_and_submission_exact=True,
        rejection_keeps_current_output_query_and_previous_templates_exact=True,
        frozen_states_before=frozen_before, frozen_states_after=frozen_after,
        decoder_before=decoder_before, decoder_after=decoder_after, optimizations=0,
        C_MLPs_constructed=0, constant_action_fixture_not_trained_C=True,
        GT_opened_after_initialization=False, GT_reinitializations_after_first_frame=0,
        teacher_result_sha256=sha(args.teacher_root / 'result.json'), teacher_audit_sha256=sha(args.teacher_audit),
        source_sha256=sha(Path(__file__)), tracker_source_sha256=sha(Path(__file__).with_name('trusted_template_tracker.py')),
        feature_source_sha256=sha(Path(__file__).with_name('template_write_features.py')),
        raw_histories_bytes=raw_path.stat().st_size, raw_histories_sha256=sha(raw_path),
        records_sha256=sha(args.output / 'records.jsonl'), input_contract=CONTRACT,
        full_development32_or_official_metrics=False, C_performance_claim=False,
        elapsed_seconds=time.time() - started)
    (args.output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
