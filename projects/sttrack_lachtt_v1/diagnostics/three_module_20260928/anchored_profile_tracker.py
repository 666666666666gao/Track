"""Same R0 final and CAD template profile, with native-anchored selection for all datasets."""
from m125_CAD_official_runtime import OfficialProfileTracker
from native_anchored_selection import NativeAnchoredDecoder


class OfficialAnchoredTracker(OfficialProfileTracker):
    def __init__(self, plan, bundle):
        super().__init__(plan, bundle)
        decoder = NativeAnchoredDecoder(self.actor.decoder, bundle['anchor_margin'], bundle['anchor_radius']).eval()
        self.actor.decoder = decoder
        self.trusted.counts['anchored_overrides'] = 0
        step = self.trusted.step

        def counted_step(image):
            out, data, record = step(image)
            record['anchored_override'] = bool(out['anchored_override'][0])
            self.trusted.counts['anchored_overrides'] += int(record['anchored_override'])
            return out, data, record

        self.trusted.step = counted_step

    def initialize(self, image, bbox, rgb_path):
        super().initialize(image, bbox, rgb_path)
        self.trusted.counts['anchored_overrides'] = 0
