"""Train A+B+C on complete own-history sequences; frozen observations on GPU1."""
import torch

from dense_target_decoder import clip_boxes
from template_write_C import accepts_write
from template_write_features import initial_reference, write_features


class JointFullABCTracker:
    def __init__(self, actor, writer):
        self.actor = actor
        self.writer = writer
        self.observation_device = next(actor.native.network.parameters()).device
        self.learning_device = next(actor.decoder.parameters()).device
        assert self.observation_device.index == 1 and self.learning_device.index == 0
        assert next(writer.parameters()).device == self.learning_device
        assert not actor.native.network.training and not actor.clip.training
        assert all(p.requires_grad for p in actor.decoder.parameters())
        assert all(p.requires_grad for p in writer.parameters())

    def initialize(self, image, box, bank_index):
        actor = self.actor
        with torch.cuda.device(self.observation_device):
            actor.initialize(image, box, bank_index)
        actor.initial = {name: value.to(self.learning_device) for name, value in actor.initial.items()}
        actor.words = actor.words.to(self.learning_device)
        actor.word_mask = actor.word_mask.to(self.learning_device)
        actor.empty = actor.empty.to(self.learning_device)
        actor.decoder.eval()
        self.reference = initial_reference(actor)
        actor.decoder.train()
        self.writer.train()

    def step(self, image):
        actor = self.actor
        previous = actor.selected_feature.detach().clone() if actor.native.frame_id > 0 else None
        prior = list(actor.native.state)
        actor.native.frame_id += 1
        with torch.cuda.device(self.observation_device):
            data, base, native, resize = actor.observe(image, prior, actor.native.track_query_before)
        data = {name: value.to(self.learning_device) if torch.is_tensor(value) else value
                for name, value in data.items()}
        base = base.to(self.learning_device)
        data.update(actor.initial)
        data.update(query=actor.words, query_mask=actor.word_mask, empty_query=actor.empty)
        out = actor.decoder(data)
        boxes = clip_boxes(base + (out['raw_boxes'] - data['boxes']).double(), data['image_shape'].double())
        selected = out['selected_index']
        row = torch.arange(1, device=selected.device)
        out.update(boxes=boxes, selected_box=boxes[row, selected])
        assert all(bool(torch.isfinite(value).all()) for value in out.values())
        assert torch.equal(out['selected_score'], out['response'][row, selected])
        assert torch.equal(out['selected_feature'], out['spatial_features'][row, selected])
        assert torch.equal(out['selected_quality'], out['quality_logits'].sigmoid()[row, selected])
        actor.native.state = out['selected_box'][0].detach().cpu().tolist()
        actor.native.track_query_before = [value.detach() for value in native['track_query_before']]
        actor.selected_feature = out['selected_feature'].detach()
        confidence = float(data['native_response'][0, selected[0]])
        qualified = actor.native.frame_id % actor.native.update_intervals == 0 and confidence > actor.native.update_threshold
        record = dict(frame=actor.native.frame_id, previous_bbox=prior, bbox=list(actor.native.state),
                      selected=int(selected[0]), best_score=float(out['selected_score'][0].detach()),
                      selected_quality=float(out['selected_quality'][0].detach()),
                      native_same_position_response=confidence,
                      observation_intersection_probability=float(out['observation_logits'][0].detach().sigmoid()),
                      template_write=False, native_rule_write_qualified=bool(qualified),
                      resize_factor=float(resize), crop_origin=data['origin'][0].tolist(), C_prediction=None)
        prediction = None
        if previous is not None:
            features = write_features(actor.selected_feature, previous, self.reference, record)
            prediction = self.writer(features)
            assert bool(torch.isfinite(prediction).all())
            record['C_prediction'] = float(prediction[0].detach())
        if qualified:
            assert actor.native.update_intervals == 50 and actor.native.update_threshold == .75
            assert prediction is not None
            record['template_write'] = accepts_write(prediction.detach(), 'current')
            if record['template_write']:
                with torch.cuda.device(self.observation_device):
                    patch, _, _ = actor.sample_target(image, actor.native.state, 2., output_sz=128)
                    actor.native.z_patch_arr = patch
                    template = actor.native.preprocessor.process(patch)
                    actor.native.z_dict.append(template)
                    actor.native.z_dict.pop(1)
        return out, data, record, prediction
