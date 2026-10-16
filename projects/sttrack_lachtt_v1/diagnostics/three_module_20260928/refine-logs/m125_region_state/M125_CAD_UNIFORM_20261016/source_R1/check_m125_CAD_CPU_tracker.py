"""Synthetic tensor actors test write routing; no learned model or dataset is run."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from trusted_template_tracker import TrustedTemplateTracker
from profiled_template_tracker import ProfiledTemplateTracker


class MockDecoder:
    training = False
    axis = torch.linspace(-.75, .75, 4)
    roi_xy = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(16, 2)
    native = {name:(lambda value:value[..., :64]) for name in ['rgb', 'depth', 'fused']}
    semantic_projection = staticmethod(lambda value:value[..., :64])
    position = staticmethod(lambda xy:torch.zeros(len(xy), 64))

    def __call__(self, data):
        frame = data['fixture_frame']
        response = torch.full((1, 256), .1); response[0, 7] = .9
        features = torch.full((1, 256, 128), float(frame))
        quality = torch.zeros(1, 256)
        return dict(raw_boxes=data['boxes'].clone(), selected_index=torch.tensor([7]),
            response=response, selected_score=response[:, 7], spatial_features=features,
            selected_feature=features[:, 7], quality_logits=quality,
            selected_quality=quality.sigmoid()[:, 7], observation_logits=torch.tensor([2.]))


class MockWriter:
    training = False
    def __init__(self):self.inputs = []
    def parameters(self):return iter(())
    def __call__(self, features):
        self.inputs.append(features.clone())
        frame = int(features[0, 0])
        return torch.tensor([.5 if frame % 20 == 0 else .55])


class MockActor:
    def __init__(self):
        self.decoder = MockDecoder()
        self.clip = SimpleNamespace(training=False)
        self.native = SimpleNamespace(network=SimpleNamespace(training=False), update_intervals=50,
            update_threshold=.75, preprocessor=SimpleNamespace(process=lambda value:torch.from_numpy(value.copy())))
        self.samples = []

    def initialize(self, image, box, index):
        self.native.state = list(box); self.native.frame_id = 0
        self.native.track_query_before = None
        self.native.z_dict = [torch.tensor([-1]), torch.tensor([-2])]
        self.initial = {'initial_'+name:torch.ones(1, 256, 768) for name in ['rgb','depth','fused','clip']}
        self.initial.update(initial_native_valid=torch.ones(1,256),initial_clip_valid=torch.ones(1,256),
            initial_box=torch.tensor([box]),initial_origin=torch.tensor([[0.,0.,128.,128.]]))
        self.words = torch.ones(1,5,768);self.word_mask = torch.ones(1,5,dtype=torch.bool)
        self.empty = torch.zeros(1,5,768)

    def observe(self, image, prior, query):
        frame = self.native.frame_id
        boxes = torch.tensor([[[20.+frame*.02,20.,20.,20.]]]).repeat(1,256,1)
        selected_response = .6 if frame % 25 == 0 else (.7 if frame % 10 == 5 else .8)
        if frame in [50,100]:selected_response = .9
        native_response = torch.full((1,256),.1)
        native_response[0,3] = .99
        native_response[0,7] = selected_response
        data = dict(boxes=boxes,image_shape=torch.tensor([[128.,128.]]),native_response=native_response,
                    origin=torch.tensor([[0.,0.,128.,128.]]),fixture_frame=frame)
        native = dict(track_query_before=[torch.tensor([[float(frame)]])])
        return data,boxes.double(),native,1.

    def sample_target(self, image, box, factor, output_sz):
        assert factor == 2. and output_sz == 128 and box == self.native.state
        self.samples.append(self.native.frame_id)
        return np.full((128,128,6),self.native.frame_id,dtype=np.uint8),1.,None


def main(output):
    output.mkdir()
    image = np.zeros((128,128,6),dtype=np.uint8)
    old_actor,new_actor = MockActor(),MockActor()
    old_writer,new_writer = MockWriter(),MockWriter()
    old = TrustedTemplateTracker(old_actor,old_writer,'current')
    new = ProfiledTemplateTracker(new_actor,new_writer)
    for tracker in [old,new]:tracker.initialize(image,[20.,20.,20.,20.],0)
    checks = 0
    for frame in range(1,101):
        old_out,_,old_record = old.step(image)
        new_out,_,new_record = new.step(image)
        assert set(old_out) == set(new_out)
        assert all(torch.equal(old_out[key],new_out[key]) for key in old_out)
        assert old_record == {key:new_record[key] for key in old_record}
        assert old_actor.native.state == new_actor.native.state
        assert all(torch.equal(a,b) for a,b in zip(old_actor.native.track_query_before,new_actor.native.track_query_before))
        assert all(torch.equal(a,b) for a,b in zip(old_actor.native.z_dict,new_actor.native.z_dict))
        assert torch.equal(new_actor.native.z_dict[0],torch.tensor([-1]))
        checks += 7
    assert old_actor.samples == new_actor.samples == [50]
    assert len(old_writer.inputs) == len(new_writer.inputs) == 2
    assert all(torch.equal(a,b) for a,b in zip(old_writer.inputs,new_writer.inputs))
    rows=[dict(profile='U0',counts=new.counts,write_frames=new_actor.samples)]
    for profile,expected in [('U1',[10,30,50,70,90]),
        ('U2',[5,10,15,30,35,45,50,55,65,70,85,90,95]),
        ('U3',[5,10,15,25,30,35,45,50,55,65,70,75,85,90,95]),
        ('U4',list(range(5,101,5)))]:
        actor,writer=MockActor(),MockWriter()
        tracker=ProfiledTemplateTracker(actor,writer,profile)
        tracker.initialize(image,[20.,20.,20.,20.],0)
        for frame in range(1,101):
            out,_,record=tracker.step(image)
            assert record['best_score'] == float(out['selected_score'][0])
            assert record['selected'] == 7
            assert record['bbox'] == actor.native.state
            assert torch.equal(actor.native.z_dict[0],torch.tensor([-1])) and len(actor.native.z_dict)==2
            assert record['inference_profile_id']==profile
            if profile in ['U1','U2'] and frame in [25,75]:
                assert not record['profile_write_qualified'] and not record['template_write']
            checks += 5
        assert actor.samples == expected,(profile,actor.samples,expected)
        assert tracker.counts['template_writes']==len(expected)
        assert tracker.counts['tracking_calls']==100
        if profile=='U4':assert not writer.inputs and tracker.counts['C_evaluations']==0
        rows.append(dict(profile=profile,counts=tracker.counts,write_frames=actor.samples))
    result=dict(status='complete_synthetic_tensor_template_routing_checks',checks=checks,
        scope='mock actors and scalar C; no learned network or trajectory quality evidence',
        U0_old_tracker_outputs_state_query_C_inputs_and_template_slots_exact=True,
        profiles=rows,synthetic_step_calls=600,STTrack_CLIP_instances=0,decoder_network_forwards=0,
        real_tracking_calls=0,dataset_images_opened=0,optimizer_updates=0,GPU_queries=0,
        original_R0_progress_queries=0,evaluation_type='synthetic_tracker_interface_only')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
