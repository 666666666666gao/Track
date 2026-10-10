"""One shared C architecture; current quality and future utility differ by labels."""
import torch
from torch import nn
from template_write_features import INPUT_DIM, CONTRACT


class TemplateWriteC(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.ReLU(),
                                     nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        assert features.ndim == 2 and features.shape[1] == INPUT_DIM
        assert features.dtype == torch.float32 and bool(torch.isfinite(features).all())
        prediction = self.network(features)[:, 0]
        assert bool(torch.isfinite(prediction).all())
        return prediction


def accepts_write(prediction, arm):
    assert arm in ['current', 'future']
    assert prediction.numel() == 1 and bool(torch.isfinite(prediction).all())
    return bool(prediction[0] > (.5 if arm == 'current' else 0.))
