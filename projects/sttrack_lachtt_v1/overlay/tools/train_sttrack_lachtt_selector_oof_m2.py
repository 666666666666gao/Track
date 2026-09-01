#!/usr/bin/env python3
"""M2 selector: add candidate-to-immutable-native-template identity evidence.

All M1 labels, sequence folds, losses, thresholds, and safety gates are reused.
The only model-input change is twelve causal statistics comparing each
candidate's own native RGB/depth feature with the initialized target's frozen
native STTrack token banks.
"""

import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import train_sttrack_lachtt_selector_oof as base


IDENTITY_DIM = 12
BASE_TRAINER_PATH = Path(base.__file__).resolve()


def native_statistics(candidate, bank):
    candidate = F.normalize(candidate.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    similarity = torch.einsum("acd,td->act", candidate, bank)
    mean = similarity.mean(dim=-1)
    maximum = similarity.max(dim=-1).values
    top4 = similarity.topk(4, dim=-1).values.mean(dim=-1)
    deviation = similarity.std(dim=-1, unbiased=False)
    return torch.stack((mean, maximum, top4, deviation), dim=-1)


def native_identity_block(features, anchor):
    rgb = native_statistics(
        features["native_rgb"], anchor["native_template_rgb_tokens"])
    depth = native_statistics(
        features["native_depth"], anchor["native_template_depth_tokens"])
    combined = torch.stack((
        torch.minimum(rgb[..., 2], depth[..., 2]),
        rgb[..., 2] * depth[..., 2],
        torch.abs(rgb[..., 2] - depth[..., 2]),
        0.5 * (rgb[..., 1] + depth[..., 1]),
    ), dim=-1)
    result = torch.cat((rgb, depth, combined), dim=-1)
    if list(result.shape) != [5, 6, IDENTITY_DIM]:
        raise RuntimeError("M2 native identity feature shape drifted")
    if not torch.isfinite(result).all().item():
        raise RuntimeError("M2 native identity feature is non-finite")
    return result.half()


def load_native_index(spec):
    root = Path(spec["native_anchor_root"]).resolve()
    manifest_path = root / "manifest.json"
    index_path = root / "index.jsonl"
    bindings = spec.get("bindings", {})
    if (base.sha256_file(manifest_path) !=
            bindings["native_anchor_manifest"]["sha256"] or
            base.sha256_file(index_path) !=
            bindings["native_anchor_index"]["sha256"]):
        raise ValueError("M2 native anchor artifact binding mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("complete") is not True or
            manifest.get("future_ground_truth_opened") is not False or
            manifest.get("future_frame_opened") is not False or
            manifest.get("metric_computed") is not False or
            int(manifest.get("sequence_count", -1)) != 152):
        raise ValueError("M2 native anchor manifest is unsafe or incomplete")
    rows = {}
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sequence = row["sequence"]
            path = root / row["path"]
            if (sequence in rows or base.sha256_file(path) != row["sha256"] or
                    path.stat().st_size != row["bytes"]):
                raise ValueError("M2 native anchor index/file mismatch")
            rows[sequence] = path
    if len(rows) != 152:
        raise ValueError("M2 native anchor sequence coverage drifted")
    return rows


def validate_formal_source_bindings(spec):
    paths = {
        "m1_trainer_base": BASE_TRAINER_PATH,
        "source_selector_spec": Path(spec["source_selector_spec"]),
        "labeled_actions": (Path(spec["gate_a_root"]) /
                            "labeled_actions.jsonl.gz"),
        "collection_manifest_shard0": (Path(spec["collection_root"]) /
                                       "shard0/manifest.json"),
        "collection_manifest_shard1": (Path(spec["collection_root"]) /
                                       "shard1/manifest.json"),
    }
    for name, path in paths.items():
        expected = spec.get("bindings", {}).get(name)
        if (not isinstance(expected, dict) or not path.is_file() or
                path.stat().st_size != expected.get("bytes") or
                base.sha256_file(path) != expected.get("sha256")):
            raise ValueError("M2 formal source binding mismatch: %s" % name)


_load_data_m1 = base.load_data


def load_data(spec, smoke=False):
    if not smoke:
        validate_formal_source_bindings(spec)
    data = _load_data_m1(spec, smoke=smoke)
    paths = load_native_index(spec)
    cache = {}
    blocks = []
    for index, sequence in enumerate(data["sequences"]):
        if sequence not in cache:
            value = torch.load(paths[sequence], map_location="cpu")
            required = ("native_template_rgb_tokens",
                        "native_template_depth_tokens")
            if any(list(value[name].shape) != [64, 768]
                   for name in required):
                raise ValueError("M2 native anchor token shape drifted")
            cache[sequence] = value
        event_features = {
            "native_rgb": data["features"]["native_rgb"][index],
            "native_depth": data["features"]["native_depth"][index],
        }
        blocks.append(native_identity_block(event_features, cache[sequence]))
    data["features"]["native_identity"] = torch.stack(blocks, dim=0)
    if list(data["features"]["native_identity"].shape[1:]) != [5, 6, 12]:
        raise RuntimeError("M2 stacked identity feature shape drifted")
    return data


class NativeIdentityAssociationSelector(nn.Module):
    """M1 architecture plus explicit candidate-to-native-init evidence."""

    def __init__(self):
        super().__init__()
        projection = 24
        hidden = 96
        names = ("native_rgb", "native_depth", "native_fused",
                 "clip_image", "query_rgb", "query_depth")
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(768, projection), nn.GELU(),
                                nn.LayerNorm(projection)) for name in names
        })
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.scalar_encoder = nn.Sequential(
            nn.Linear(15, 32), nn.GELU(), nn.Linear(32, 16),
            nn.LayerNorm(16))
        self.identity_encoder = nn.Sequential(
            nn.Linear(IDENTITY_DIM, 24), nn.GELU(), nn.LayerNorm(24))
        input_size = projection * len(names) + 16 + 16 + 2 + 24
        self.fusion = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=192,
            dropout=0.10, activation="gelu", batch_first=True,
            norm_first=True)
        self.distractor = nn.TransformerEncoder(layer, num_layers=1)
        self.beneficial_head = nn.Linear(hidden, 1)
        self.catastrophic_head = nn.Linear(hidden, 1)
        self.gain_head = nn.Linear(hidden, 1)

    def forward(self, features, anchor_image, anchor_text):
        parts = [self.projections[name](features[name].float())
                 for name in self.projections]
        batch, ages, candidates = features["clip_image"].shape[:3]
        depth = features["raw_depth"].float().reshape(
            batch * ages * candidates, 2, 16, 16)
        depth = self.depth_encoder(depth).reshape(
            batch, ages, candidates, 16)
        scalars = self.scalar_encoder(torch.asinh(features["scalars"].float()))
        identity = self.identity_encoder(features["native_identity"].float())
        clip_image = F.normalize(features["clip_image"].float(), dim=-1)
        init = F.normalize(anchor_image.float(), dim=-1)[:, None, None, :]
        text = F.normalize(anchor_text.float(), dim=-1)[:, None, None, :]
        similarities = torch.stack((
            (clip_image * init).sum(dim=-1),
            (clip_image * text).sum(dim=-1)), dim=-1)
        fused = self.fusion(torch.cat(
            parts + [depth, scalars, similarities, identity], dim=-1))
        temporal_input = fused.permute(0, 2, 1, 3).reshape(
            batch * candidates, ages, -1)
        _, state = self.temporal(temporal_input)
        candidate_state = state[-1].reshape(batch, candidates, -1)
        associated = self.distractor(candidate_state)
        return {
            "beneficial_logit": self.beneficial_head(associated).squeeze(-1),
            "catastrophic_logit": self.catastrophic_head(associated).squeeze(-1),
            "gain": torch.tanh(self.gain_head(associated).squeeze(-1)),
        }


base.AssociationSelector = NativeIdentityAssociationSelector
base.load_data = load_data
base.__file__ = __file__


if __name__ == "__main__":
    base.main()
