"""Frozen A+B observation/selection, C decision before the template crop.

The submitted frame box, query, selected feature and reported score follow the
bound FullDenseTracker.step path. Only its dynamic-template write is gated.
"""
import torch
from dense_target_decoder import clip_boxes
from template_write_features import initial_reference, write_features
from template_write_C import accepts_write


class TrustedTemplateTracker:
    def __init__(self, actor, writer, arm):
        assert arm in ['current', 'future']
        assert not actor.decoder.training and not actor.native.network.training and not actor.clip.training
        assert not writer.training and not any(p.requires_grad for p in writer.parameters())
        self.actor = actor
        self.writer = writer
        self.arm = arm

    @torch.no_grad()
    def initialize(self, image, box, bank_index):
        self.actor.initialize(image, box, bank_index)
        self.reference = initial_reference(self.actor)

    @torch.no_grad()
    def step(self, image):
        actor = self.actor
        previous_feature = actor.selected_feature.detach().clone() if actor.native.frame_id > 0 else None
        prior = list(actor.native.state)
        actor.native.frame_id += 1
        data, base, native, resize = actor.observe(image, prior, actor.native.track_query_before)
        data.update(actor.initial)
        data.update(query=actor.words, query_mask=actor.word_mask, empty_query=actor.empty)
        out = actor.decoder(data)
        boxes = clip_boxes(base + (out['raw_boxes'] - data['boxes']).double(), data['image_shape'].double())
        selected = out['selected_index']
        row = torch.arange(1, device=selected.device)
        out.update(boxes=boxes, selected_box=boxes[row, selected])
        assert all(bool(torch.isfinite(x).all()) for x in out.values())
        assert torch.equal(out['selected_score'], out['response'][row, selected])
        assert torch.equal(out['selected_feature'], out['spatial_features'][row, selected])
        assert torch.equal(out['selected_quality'], out['quality_logits'].sigmoid()[row, selected])
        actor.native.state = out['selected_box'][0].detach().cpu().tolist()
        actor.native.track_query_before = [x.detach() for x in native['track_query_before']]
        actor.selected_feature = out['selected_feature'].detach()
        confidence = float(data['native_response'][0, selected[0]])
        qualified = actor.native.frame_id % actor.native.update_intervals == 0 and confidence > actor.native.update_threshold
        record = dict(frame=actor.native.frame_id, previous_bbox=prior, bbox=list(actor.native.state),
            selected=int(selected[0]), best_score=float(out['selected_score'][0].detach()),
            selected_quality=float(out['selected_quality'][0].detach()), native_same_position_response=confidence,
            observation_intersection_probability=float(out['observation_logits'][0].detach().sigmoid()),
            template_write=False, resize_factor=float(resize), crop_origin=data['origin'][0].tolist(),
            native_rule_write_qualified=bool(qualified), C_arm=self.arm, C_prediction=None)
        if qualified:
            assert actor.native.update_intervals == 50 and actor.native.update_threshold == .75
            assert previous_feature is not None
            features = write_features(actor.selected_feature, previous_feature, self.reference, record)
            prediction = self.writer(features)
            record['C_prediction'] = float(prediction[0])
            record['template_write'] = accepts_write(prediction, self.arm)
            if record['template_write']:
                patch, _, _ = actor.sample_target(image, actor.native.state, 2., output_sz=128)
                actor.native.z_patch_arr = patch
                template = actor.native.preprocessor.process(patch)
                actor.native.z_dict.append(template)
                actor.native.z_dict.pop(1)
        return out, data, record

    @torch.no_grad()
    def track(self, image, info=None):
        _, _, record = self.step(image)
        return dict(target_bbox=record['bbox'], best_score=record['best_score'])
