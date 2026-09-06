"""STTrack with a separately trained dense phrase/RGB-D residual module."""
import torch

from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_initial_instance_observation import extract_initial_instance_roi


class STTrackSemantic(STTrack):
    def __init__(self, params, checkpoint_path):
        super().__init__(params)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint['architecture'] == 'semantic_spatial_v1'
        assert checkpoint['base_checkpoint_sha256'] == params.base_checkpoint_sha256
        self.use_text = checkpoint['use_text']
        adapter = SemanticSpatialAdapter()
        adapter.load_state_dict(checkpoint['model'], strict=True)
        self.network.semantic_adapter = adapter.cuda().eval()

    def initialize(self, image, info):
        result = super().initialize(image, info)
        initial = extract_initial_instance_roi(self, image, info['init_bbox'])
        text = info['text_tokens']
        if not self.use_text:
            text = info['empty_text'].reshape(1, -1).expand_as(text)
        self.semantic_context = {
            'initial': initial.half().float().unsqueeze(0),
            'text': text.float().cuda().unsqueeze(0),
            'mask': info['text_mask'].bool().cuda().unsqueeze(0),
        }
        return result
