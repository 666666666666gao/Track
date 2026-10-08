"""Offline R1 helpers for the fixed M122 P1 template-write experiment.

No runner or GPU job is started here. Current inference remains unchanged.
The caller runs the original step and supplies its pre-step template state;
the original step's two template writes are then reconstructed as an event.
"""
from copy import copy
from dataclasses import dataclass
import random

import numpy as np
import torch


def copy_templates(templates):
    """Own tensor storage while preserving the native initial-slot alias."""
    owned = {}
    for tensor in templates:
        if id(tensor) not in owned:
            owned[id(tensor)] = tensor.detach().clone()
    return [owned[id(tensor)] for tensor in templates]


def copy_tracker_state(source, templates, template_patch):
    """Share frozen models/immutable initial evidence; own mutable history."""
    assert not source.decoder.training and not source.native.network.training
    assert not source.clip.training
    assert source.native.track_query_before is not None
    assert source.native.num_template == len(templates) == 2
    branch = copy(source)
    branch.native = copy(source.native)
    branch.native.state = list(source.native.state)
    branch.native.z_dict = copy_templates(templates)
    branch.native.z_patch_arr = template_patch.copy()
    branch.native.track_query_before = [tensor.detach().clone() for tensor in source.native.track_query_before]
    branch.selected_feature = source.selected_feature.detach().clone()
    branch.initial = dict(source.initial)
    return branch


def assert_same_submission(left, right):
    assert left.native.state == right.native.state
    assert left.native.frame_id == right.native.frame_id
    assert left.native.update_intervals == right.native.update_intervals == 50
    assert left.native.update_threshold == right.native.update_threshold == .75
    assert left.native.keep_rate == right.native.keep_rate
    assert len(left.native.z_dict) == len(right.native.z_dict) == 2
    assert all(torch.equal(a, b) for a, b in zip(left.native.z_dict, right.native.z_dict))
    assert np.array_equal(left.native.z_patch_arr, right.native.z_patch_arr)
    assert len(left.native.track_query_before) == len(right.native.track_query_before)
    assert all(torch.equal(a, b) for a, b in zip(left.native.track_query_before, right.native.track_query_before))
    assert torch.equal(left.selected_feature, right.selected_feature)
    assert left.initial.keys() == right.initial.keys()
    assert all(torch.equal(left.initial[key], right.initial[key]) for key in left.initial)
    for key in ['words', 'word_mask', 'empty']:
        assert torch.equal(getattr(left, key), getattr(right, key))


@dataclass
class WriteEvent:
    pre_write_tracker: object
    written_template: torch.Tensor
    written_patch: np.ndarray
    record: dict


def event_from_native_write(actor, templates_before_step, patch_before_step, record):
    """Undo only original step's z_dict/patch replacement in an owned copy.

    The actor keeps its actual native-write trajectory. No frame, bbox,
    candidate, query or selected feature is reverted to a previous frame.
    """
    assert record['template_write'] and record['frame'] == actor.native.frame_id
    assert record['bbox'] == actor.native.state
    assert actor.native.frame_id % actor.native.update_intervals == 0
    assert record['native_same_position_response'] > actor.native.update_threshold
    assert actor.native.update_intervals == 50 and actor.native.update_threshold == .75
    assert torch.equal(actor.native.z_dict[0], templates_before_step[0])
    before = copy_tracker_state(actor, templates_before_step, patch_before_step)
    return WriteEvent(before, actor.native.z_dict[1].detach().clone(), actor.native.z_patch_arr.copy(), dict(record))


def fork_event(event, write):
    before = event.pre_write_tracker
    branch = copy_tracker_state(before, before.native.z_dict, before.native.z_patch_arr)
    assert_same_submission(before, branch)
    if write:
        branch.native.z_dict[1] = event.written_template.detach().clone()
        branch.native.z_patch_arr = event.written_patch.copy()
    return branch


def capture_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


@torch.no_grad()
def rollout_pair(left, right, images):
    """Two sequential same-GPU histories; future images contain no GT."""
    assert len(images) <= 32
    assert left.selected_feature.device == right.selected_feature.device
    assert left.selected_feature.device.index == torch.cuda.current_device()
    assert left.native.frame_id == right.native.frame_id
    assert left.native.frame_id % 50 == 0
    rng = capture_rng()
    trajectories = []
    for branch in [left, right]:
        restore_rng(rng)
        rows = []
        for image in images:
            _, _, record = branch.step(image)
            assert not record['template_write']
            rows.append(dict(record=record,
                query=[tensor.detach().clone() for tensor in branch.native.track_query_before],
                selected_feature=branch.selected_feature.detach().clone()))
        trajectories.append(rows)
    restore_rng(rng)
    return trajectories


def probe_keep_keep(event, images):
    left, right = fork_event(event, False), fork_event(event, False)
    assert_same_submission(left, right)
    first, second = rollout_pair(left, right, images)
    for a, b in zip(first, second):
        assert a['record'] == b['record'], (a['record']['frame'], 'K/K prediction or control record differs')
        assert len(a['query']) == len(b['query'])
        assert all(torch.equal(x, y) for x, y in zip(a['query'], b['query'])), (a['record']['frame'], 'K/K query differs')
        assert torch.equal(a['selected_feature'], b['selected_feature']), (a['record']['frame'], 'K/K selected feature differs')
    assert_same_submission(left, right)
    return first


def write_keep_rollouts(event, images):
    """Return predictions only; any future GT label must be computed outside."""
    return rollout_pair(fork_event(event, True), fork_event(event, False), images)
