"""Tensor-only R1 event storage; frozen model objects are never serialized."""
import torch
from template_write_forks import WriteEvent, capture_rng


def cpu_tensor(tensor):
    return tensor.detach().cpu().clone()


def move_templates(templates, device):
    owned = {}
    for tensor in templates:
        if id(tensor) not in owned:
            owned[id(tensor)] = tensor.detach().to(device).clone()
    return [owned[id(tensor)] for tensor in templates]


def pack_event(event):
    actor = event.pre_write_tracker
    assert actor.native.box_mask_z is None
    return dict(state=dict(bbox=list(actor.native.state), frame=actor.native.frame_id,
        templates=move_templates(actor.native.z_dict, 'cpu'), patch=actor.native.z_patch_arr.copy(),
        query=[cpu_tensor(x) for x in actor.native.track_query_before],
        selected_feature=cpu_tensor(actor.selected_feature),
        initial={k:cpu_tensor(v) for k,v in actor.initial.items()},
        words=cpu_tensor(actor.words), word_mask=cpu_tensor(actor.word_mask), empty=cpu_tensor(actor.empty)),
        written_template=cpu_tensor(event.written_template), written_patch=event.written_patch.copy(),
        record=dict(event.record), rng=capture_rng())


def unpack_event(actor, payload):
    """Actor has only been initialized with the legal first frame and box."""
    state = payload['state']
    assert actor.native.box_mask_z is None and actor.native.num_template == 2
    assert actor.native.update_intervals == 50 and actor.native.update_threshold == .75
    device = torch.device('cuda', torch.cuda.current_device())
    actor.native.state = list(state['bbox'])
    actor.native.frame_id = state['frame']
    actor.native.z_dict = move_templates(state['templates'], device)
    actor.native.z_patch_arr = state['patch'].copy()
    actor.native.track_query_before = [x.to(device).clone() for x in state['query']]
    actor.selected_feature = state['selected_feature'].to(device).clone()
    actor.initial = {k:v.to(device).clone() for k,v in state['initial'].items()}
    actor.words = state['words'].to(device).clone()
    actor.word_mask = state['word_mask'].to(device).clone()
    actor.empty = state['empty'].to(device).clone()
    assert actor.native.state == payload['record']['bbox']
    assert actor.native.frame_id == payload['record']['frame']
    return WriteEvent(actor, payload['written_template'].to(device).clone(), payload['written_patch'].copy(), dict(payload['record']))


def cpu_trajectory(rows):
    return [dict(record=dict(row['record']), query=[cpu_tensor(x) for x in row['query']],
        selected_feature=cpu_tensor(row['selected_feature'])) for row in rows]
