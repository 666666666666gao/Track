"""Native-anchored selection: a learned non-local jump must beat the native peak by a margin.

The frozen STTrack Hann peak keeps the final choice unless the learned A+B response
either stays within `radius` cells of it or exceeds its response by `(1+margin)`.
Boxes, scores, quality and features still come from the same selected index.
"""
import torch
from torch import nn


class NativeAnchoredDecoder(nn.Module):
    def __init__(self, decoder, margin, radius=1):
        super().__init__()
        self.decoder = decoder
        self.margin = margin
        self.radius = radius

    def forward(self, data):
        out = self.decoder(data)
        response = out['response']
        learned = out['selected_index']
        native = data['native_response'].argmax(-1)
        row = torch.arange(len(learned), device=learned.device)
        distance = torch.maximum((learned // 16 - native // 16).abs(), (learned % 16 - native % 16).abs())
        jump = distance > self.radius
        weak = response[row, learned] < (1 + self.margin) * response[row, native]
        selected = torch.where(jump & weak, native, learned)
        out.update(selected_index=selected, selected_box=out['boxes'][row, selected],
                   selected_score=response[row, selected],
                   selected_quality=out['quality_logits'].sigmoid()[row, selected],
                   selected_feature=out['spatial_features'][row, selected],
                   native_index=native, anchored_override=jump & weak)
        return out
