"""CADTrack observations feeding the existing M125 region decoder and C writer."""
import sys
from types import SimpleNamespace

import torch
from PIL import Image
from torch.nn import functional as F

from dense_region_encoding import dense_patch_features
from dense_target_decoder import clip_boxes
from full_dense_tracker import FullDenseTracker
from region_patch_evidence import crop_geometry
from template_write_C import accepts_write
from template_write_features import initial_reference, write_features


class CADTrackActor(FullDenseTracker):
    """Observe native forward outputs without editing CADTrack source or weights."""

    def __init__(self, repository, checkpoint, config, clip_weight, bank, decoder):
        import clip
        sys.path.insert(0, str(repository))
        from lib.config.cadtrack.config import cfg, update_config_from_file
        from lib.test.tracker.cadtrack import CADTrack
        from lib.train.data.processing_utils import sample_target

        update_config_from_file(str(config))
        params = SimpleNamespace(cfg=cfg, checkpoint=str(checkpoint),
            template_factor=2., template_size=128, search_factor=4., search_size=256,
            save_all_boxes=False, debug=0)
        self.native = CADTrack(params)  # Native strict=True checkpoint loading.
        self.native.network.eval().requires_grad_(False)
        self.clip, self.clip_preprocess = clip.load(str(clip_weight), device='cuda', jit=False)
        self.clip = self.clip.float().eval().requires_grad_(False)
        self.decoder, self.bank, self.sample_target = decoder, bank, sample_target
        self.cell_response = torch.eye(256, device='cuda').reshape(256, 1, 16, 16)
        assert self.native.feat_sz == 16 and self.native.num_template == 2
        assert self.native.update_intervals == 5 and self.native.update_threshold == .65
        assert self.clip.visual.input_resolution == 224
        self.features = {}
        self.hooks = [
            self.native.network.backbone.register_forward_hook(self._backbone_features),
            self.native.network.decode_fuse_search.register_forward_hook(self._fused_features)]

    def _backbone_features(self, module, args, output):
        tokens, _, lengths = output
        rgb, depth = tokens.chunk(2, dim=1)
        assert lengths[1] == 256
        self.features['rgb'] = rgb[:, -256:].detach()
        self.features['depth'] = depth[:, -256:].detach()

    def _fused_features(self, module, args, output):
        self.features['fused'] = output.flatten(2).transpose(1, 2).detach()

    @torch.no_grad()
    def observe(self, image, prior, query):
        assert query is None  # CADTrack source explicitly returns no temporal query.
        patch, resize, mask = self.sample_target(image, prior, 4., output_sz=256)
        native = self.native.network(template=self.native.z_dict,
            search=[self.native.preprocessor.process(patch)], track_query_before=None)[0]
        assert native['track_query_before'] is None
        dense, _ = dense_patch_features(self.clip,
            self.clip_preprocess(Image.fromarray(patch[:, :, :3]))[None].cuda())
        valid = torch.from_numpy(~mask).float()[None, None].cuda()
        boxes = self.pixels(native, prior, resize, image.shape)
        data = {name: value.half().float() for name, value in self.features.items()}
        assert all(value.shape == (1, 256, 768) for value in data.values())
        data.update(clip=dense.half().float(),
            native_valid=F.avg_pool2d(valid, 16, 16).reshape(1, 256),
            clip_valid=F.avg_pool2d(F.interpolate(valid, size=(224, 224), mode='area'),
                14, 14).reshape(1, 256), boxes=boxes.float(),
            native_score=native['score_map'].reshape(1, 256),
            native_response=(self.native.output_window * native['score_map']).reshape(1, 256),
            origin=torch.tensor([crop_geometry(prior)], device='cuda', dtype=torch.float32),
            image_shape=torch.tensor([image.shape[:2]], device='cuda', dtype=torch.float32),
            condition='human_text')
        return data, boxes, native, resize


class CADTrackABC:
    """Own-history A/B selection and detached current-IoU C supervision."""

    def __init__(self, actor, writer):
        self.actor, self.writer = actor, writer
        self.observation_device = next(actor.native.network.parameters()).device
        self.learning_device = next(actor.decoder.parameters()).device
        assert next(writer.parameters()).device == self.learning_device

    def initialize(self, image, box, bank_index):
        actor = self.actor
        with torch.cuda.device(self.observation_device):
            actor.initialize(image, box, bank_index)
        actor.initial = {name: value.to(self.learning_device) for name, value in actor.initial.items()}
        actor.words = actor.words.to(self.learning_device)
        actor.word_mask = actor.word_mask.to(self.learning_device)
        actor.empty = actor.empty.to(self.learning_device)
        was_training = actor.decoder.training
        actor.decoder.eval()
        self.reference = initial_reference(actor)
        actor.decoder.train(was_training)
        self.previous = None

    def step(self, image, *, gate_writes=True):
        actor = self.actor
        prior = list(actor.native.state)
        actor.native.frame_id += 1
        with torch.cuda.device(self.observation_device):
            data, base, _, resize = actor.observe(image, prior, None)
        data = {name: value.to(self.learning_device) if torch.is_tensor(value) else value
                for name, value in data.items()}
        data.update(actor.initial, query=actor.words, query_mask=actor.word_mask, empty_query=actor.empty)
        out = actor.decoder(data)
        boxes = clip_boxes(base.to(self.learning_device) +
            (out['raw_boxes'] - data['boxes']).double(), data['image_shape'].double())
        selected = out['selected_index']
        out.update(boxes=boxes, selected_box=boxes[torch.arange(1, device=self.learning_device), selected])
        assert all(bool(torch.isfinite(value).all()) for value in out.values())
        actor.native.state = out['selected_box'][0].detach().cpu().tolist()
        current = out['selected_feature'].detach()
        confidence = float(data['native_response'][0, selected[0]])
        qualified = actor.native.frame_id % 5 == 0 and confidence > .65
        record = dict(frame=actor.native.frame_id, previous_bbox=prior, bbox=list(actor.native.state),
            selected=int(selected[0]), best_score=float(out['selected_score'][0].detach()),
            selected_quality=float(out['selected_quality'][0].detach()),
            native_same_position_response=confidence,
            observation_intersection_probability=float(out['observation_logits'][0].detach().sigmoid()),
            native_rule_write_qualified=qualified, template_write=False, C_prediction=None,
            resize_factor=float(resize), crop_origin=data['origin'][0].tolist())
        prediction = None
        if self.previous is not None:
            prediction = self.writer(write_features(current, self.previous, self.reference, record))
            record['C_prediction'] = float(prediction[0].detach())
        self.previous = current
        if qualified:
            assert prediction is not None
            record['template_write'] = not gate_writes or accepts_write(prediction.detach(), 'current')
            if record['template_write']:
                with torch.cuda.device(self.observation_device):
                    patch, _, _ = actor.sample_target(image, actor.native.state, 2., output_sz=128)
                    actor.native.z_patch_arr = patch
                    actor.native.z_dict.append(actor.native.preprocessor.process(patch))
                    actor.native.z_dict.pop(1)
        return out, data, record, prediction

    @torch.no_grad()
    def track(self, image, info=None):
        _, _, record, _ = self.step(image)
        return dict(target_bbox=record['bbox'], best_score=record['best_score'])


def load_final_tracker(repository, checkpoint, config, clip_weight, bank, final):
    """One final/profile for all datasets; caller sets the visible CUDA device."""
    from region_conditioned_target_decoder import RegionConditionedTargetDecoder
    from template_write_C import TemplateWriteC
    from train import BASELINE_SHA, sha
    weights = torch.load(final, map_location='cpu')
    assert weights['protocol']['format'] == 'CADTrack_M125_port_v1'
    assert weights['protocol']['mode'] == 'train', 'Smoke weights are not formal finals.'
    assert sha(checkpoint) == weights['protocol']['baseline_sha256'] == BASELINE_SHA
    assert sha(config) == weights['protocol']['inputs']['config']
    assert sha(clip_weight) == weights['protocol']['inputs']['clip_weight']
    decoder = RegionConditionedTargetDecoder('roi').cuda().eval()
    writer = TemplateWriteC().cuda().eval()
    decoder.load_state_dict(weights['A_B'], strict=True)
    writer.load_state_dict(weights['C'], strict=True)
    decoder.requires_grad_(False)
    writer.requires_grad_(False)
    actor = CADTrackActor(repository, checkpoint, config, clip_weight, bank, decoder)
    return CADTrackABC(actor, writer)
