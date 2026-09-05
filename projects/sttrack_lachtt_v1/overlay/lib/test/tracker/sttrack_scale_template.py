"""Native STTrack with one additional confident scale-change template write."""
import math

from lib.test.tracker.sttrack import STTrack
from lib.train.data.processing_utils import sample_target


class STTrackScaleTemplate(STTrack):
    def __init__(self, params, scale_change=1.25):
        super().__init__(params)
        assert self.num_template == 2
        self.scale_change = scale_change

    def initialize(self, image, info):
        result = super().initialize(image, info)
        self.template_scale = math.sqrt(self.state[2] * self.state[3])
        return result

    def track(self, image, info=None):
        previous_template = self.z_dict[1]
        reference_scale = self.template_scale
        result = super().track(image, info)
        current_scale = math.sqrt(self.state[2] * self.state[3])
        ratio = max(current_scale / reference_scale, reference_scale / current_scale)
        update = None
        if self.z_dict[1] is not previous_template:
            update = 'native'
        elif float(result['best_score']) > self.update_threshold and ratio >= self.scale_change:
            patch, _, _ = sample_target(image, self.state, self.params.template_factor,
                                        output_sz=self.params.template_size)
            self.z_patch_arr = patch
            self.z_dict[1] = self.preprocessor.process(patch)
            update = 'scale'
        if update is not None:
            self.template_scale = current_scale
        return dict(result, template_update=update, template_scale_ratio=ratio,
                    template_reference_scale=reference_scale)
