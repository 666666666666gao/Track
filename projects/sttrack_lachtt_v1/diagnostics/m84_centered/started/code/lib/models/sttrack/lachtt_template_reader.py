"""Compare two actual template-conditioned predictions using local RGB-D evidence."""
import torch
from torch import nn


class TemplateReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32), nn.GELU())
        self.reference_role = nn.Embedding(2, 32)
        self.match = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.local = nn.Sequential(nn.Linear(128, 64), nn.GELU())
        self.response = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 32), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(165, 64), nn.GELU(), nn.Linear(64, 1))
        self.source_bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.zeros_(self.quality[-1].bias)

    def forward(self, rois, references, maps, geometry, scores):
        # RoIs: B,2 views,2 modalities,16 cells,768. References: initial/previous.
        batch = rois.shape[0]
        query = self.cell(rois)
        ref = self.cell(references) + self.reference_role.weight[None, :, None, None, :]
        ref = ref.permute(0, 2, 1, 3, 4).reshape(batch, 2, 32, 32)
        ref = ref[:, None].expand(-1, 2, -1, -1, -1).reshape(batch * 4, 32, 32)
        query = query.reshape(batch * 4, 16, 32)
        matched, _ = self.match(query, ref, ref, need_weights=False)
        local = self.local(torch.cat([query, matched, query - matched, query * matched], dim=-1))
        local = local.mean(1).reshape(batch, 2, 128)
        response = self.response(maps.reshape(batch * 2, 2, 16, 16)).reshape(batch, 2, 32)
        log_scores = scores.clamp_min(1e-6).log()
        evidence = torch.cat([local, response, geometry, log_scores[..., None]], dim=-1)
        return self.quality(evidence).squeeze(-1) + log_scores + self.source_bias
