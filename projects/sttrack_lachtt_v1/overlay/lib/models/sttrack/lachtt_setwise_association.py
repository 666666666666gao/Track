"""Permutation-equivariant setwise target--distractor association."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class SetwiseCandidateAssociation(nn.Module):
    """Compare candidate trajectories jointly and retain an abstain action.

    Candidate order, source name, peak rank, sequence ID and frame ID are not
    encoded.  With dropout disabled, permuting candidates must permute every
    candidate output while leaving the abstain output unchanged.
    """

    def __init__(self, feature_dim=128, scalar_dim=9, hidden_dim=128,
                 heads=4, layers=2):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scalar_dim = int(scalar_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.scalar_encoder = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, 32),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            feature_dim + 32, hidden_dim, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(
            block, num_layers=layers, norm=nn.LayerNorm(hidden_dim))
        self.abstain_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.abstain_token, std=0.02)
        self.beneficial_head = nn.Linear(hidden_dim, 1)
        self.catastrophic_head = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.selection_head = nn.Linear(hidden_dim, 1)
        self.abstain_head = nn.Linear(hidden_dim, 1)

    def forward(self, trajectory_features, trajectory_scalars,
                candidate_valid):
        if (trajectory_features.ndim != 4 or
                trajectory_scalars.ndim != 4 or candidate_valid.ndim != 2):
            raise ValueError("setwise input rank drifted")
        batch, horizon, candidates, width = trajectory_features.shape
        if (width != self.feature_dim or
                list(trajectory_scalars.shape[:3]) !=
                [batch, horizon, candidates] or
                trajectory_scalars.shape[-1] != self.scalar_dim or
                list(candidate_valid.shape) != [batch, candidates]):
            raise ValueError("setwise input shape drifted")
        if candidate_valid.dtype != torch.bool:
            raise ValueError("candidate_valid must be boolean")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("every event needs at least one valid candidate")
        for value in (trajectory_features, trajectory_scalars):
            if not torch.isfinite(value.float()).all().item():
                raise ValueError("setwise input is non-finite")

        visual = self.feature_norm(trajectory_features.float())
        scalars = self.scalar_encoder(trajectory_scalars.float())
        temporal_input = torch.cat((visual, scalars), dim=-1)
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(
            batch * candidates, horizon, -1)
        _, state = self.temporal(temporal_input)
        candidate_tokens = self.temporal_norm(
            state[-1].reshape(batch, candidates, self.hidden_dim))
        abstain = self.abstain_token.expand(batch, -1, -1)
        tokens = torch.cat((candidate_tokens, abstain), dim=1)
        padding = torch.cat((~candidate_valid,
                             torch.zeros(batch, 1, dtype=torch.bool,
                                         device=candidate_valid.device)), dim=1)
        encoded = self.set_encoder(tokens, src_key_padding_mask=padding)
        candidates_encoded = encoded[:, :candidates]
        abstain_encoded = encoded[:, candidates]
        selection = self.selection_head(candidates_encoded).squeeze(-1)
        selection = selection.masked_fill(~candidate_valid, -float("inf"))
        selection = torch.cat((selection,
                               self.abstain_head(abstain_encoded)), dim=1)
        result = {
            "beneficial_logits": self.beneficial_head(
                candidates_encoded).squeeze(-1),
            "catastrophic_logits": self.catastrophic_head(
                candidates_encoded).squeeze(-1),
            "gain": self.gain_head(candidates_encoded).squeeze(-1),
            "selection_logits": selection,
        }
        if any(not torch.isfinite(value[candidate_valid].float()).all().item()
               for key, value in result.items() if key != "selection_logits"):
            raise RuntimeError("setwise candidate output is non-finite")
        if not torch.isfinite(result["selection_logits"]).logical_or(
                torch.isneginf(result["selection_logits"])).all().item():
            raise RuntimeError("setwise selection output is invalid")
        return result


def setwise_association_loss(outputs, gain_target, beneficial_target,
                             catastrophic_target, candidate_valid,
                             rank_margin=0.10,
                             selection_beneficial_event_weight=1.0,
                             beneficial_bce_positive_weight=1.0,
                             catastrophic_bce_positive_weight=1.0,
                             gain_beneficial_candidate_weight=1.0,
                             pairwise_beneficial_candidate_weight=1.0,
                             normalize_selection_event_weight=True,
                             gate_aligned_margins=None,
                             beneficial_gate_weight=0.0,
                             catastrophic_gate_weight=0.0,
                             gain_gate_weight=0.0):
    for value in (gain_target, beneficial_target, catastrophic_target):
        if value.shape != candidate_valid.shape:
            raise ValueError("setwise target shape drifted")
    balance = {
        "selection_beneficial_event_weight":
            selection_beneficial_event_weight,
        "beneficial_bce_positive_weight": beneficial_bce_positive_weight,
        "catastrophic_bce_positive_weight":
            catastrophic_bce_positive_weight,
        "gain_beneficial_candidate_weight":
            gain_beneficial_candidate_weight,
        "pairwise_beneficial_candidate_weight":
            pairwise_beneficial_candidate_weight,
    }
    if any(not math.isfinite(float(value)) or float(value) < 1.0
           for value in balance.values()):
        raise ValueError("setwise balance weights must be finite and >= 1")
    gate_weights = (beneficial_gate_weight, catastrophic_gate_weight,
                    gain_gate_weight)
    if any(not math.isfinite(float(value)) or float(value) < 0.0
           for value in gate_weights):
        raise ValueError("setwise gate weights must be finite and >= 0")
    valid = candidate_valid.float()
    denominator = valid.sum().clamp_min(1.0)
    beneficial = (F.binary_cross_entropy_with_logits(
        outputs["beneficial_logits"], beneficial_target.float(),
        reduction="none", pos_weight=outputs["beneficial_logits"].new_tensor(
            float(beneficial_bce_positive_weight))) * valid).sum() / denominator
    catastrophic = (F.binary_cross_entropy_with_logits(
        outputs["catastrophic_logits"], catastrophic_target.float(),
        reduction="none", pos_weight=outputs["catastrophic_logits"].new_tensor(
            float(catastrophic_bce_positive_weight))) * valid).sum() / denominator
    gain_weights = valid * torch.where(
        beneficial_target.bool(),
        valid.new_tensor(float(gain_beneficial_candidate_weight)),
        valid.new_tensor(1.0))
    gain = (F.smooth_l1_loss(
        outputs["gain"], gain_target.float(), reduction="none") *
        gain_weights).sum() / gain_weights.sum().clamp_min(1.0)

    batch, candidates = candidate_valid.shape
    selection_targets = []
    selection_weights = []
    pairwise_terms = []
    pairwise_weights = []
    candidate_selection = outputs["selection_logits"][:, :candidates]
    for index in range(batch):
        eligible = torch.nonzero(
            candidate_valid[index] & beneficial_target[index].bool(),
            as_tuple=False).flatten()
        if eligible.numel():
            best = eligible[gain_target[index, eligible].argmax()]
            selection_targets.append(best)
            selection_weights.append(float(selection_beneficial_event_weight))
        else:
            selection_targets.append(torch.tensor(
                candidates, device=candidate_valid.device))
            selection_weights.append(1.0)
        valid_indexes = torch.nonzero(
            candidate_valid[index], as_tuple=False).flatten()
        for left_position in range(valid_indexes.numel()):
            for right_position in range(left_position + 1,
                                        valid_indexes.numel()):
                left = valid_indexes[left_position]
                right = valid_indexes[right_position]
                difference = gain_target[index, left] - gain_target[index, right]
                if torch.abs(difference).item() < 0.05:
                    continue
                direction = torch.sign(difference)
                score_difference = (candidate_selection[index, left] -
                                    candidate_selection[index, right])
                pairwise_terms.append(F.relu(
                    float(rank_margin) - direction * score_difference))
                pair_has_beneficial = bool(
                    beneficial_target[index, left].item() or
                    beneficial_target[index, right].item())
                pairwise_weights.append(
                    float(pairwise_beneficial_candidate_weight)
                    if pair_has_beneficial else 1.0)
    selection_target = torch.stack(selection_targets).long()
    selection_weight = outputs["selection_logits"].new_tensor(
        selection_weights)
    weighted_selection = (F.cross_entropy(
        outputs["selection_logits"], selection_target, reduction="none") *
        selection_weight)
    if normalize_selection_event_weight:
        selection = (weighted_selection.sum() /
                     selection_weight.sum().clamp_min(1.0))
    else:
        selection = weighted_selection.mean()
    zero = candidate_selection[candidate_valid].sum() * 0.0
    if pairwise_terms:
        pairwise_weight = candidate_selection.new_tensor(pairwise_weights)
        pairwise = (torch.stack(pairwise_terms) * pairwise_weight).sum()
        pairwise = pairwise / pairwise_weight.sum().clamp_min(1.0)
    else:
        pairwise = zero

    beneficial_gate = zero
    catastrophic_gate = zero
    gain_gate = zero
    if gate_aligned_margins:
        required = {
            "beneficial_positive_logit_floor",
            "beneficial_negative_logit_ceiling",
            "catastrophic_positive_logit_floor",
            "catastrophic_negative_logit_ceiling",
            "beneficial_gain_floor",
        }
        if set(gate_aligned_margins) != required:
            raise ValueError("gate-aligned margin contract drifted")
        if any(not math.isfinite(float(gate_aligned_margins[key]))
               for key in required):
            raise ValueError("gate-aligned margins must be finite")
        beneficial_gate_terms = torch.where(
            beneficial_target.bool(),
            F.relu(float(gate_aligned_margins[
                "beneficial_positive_logit_floor"]) -
                   outputs["beneficial_logits"]),
            F.relu(outputs["beneficial_logits"] - float(
                gate_aligned_margins[
                    "beneficial_negative_logit_ceiling"])))
        beneficial_gate = (beneficial_gate_terms * valid).sum() / denominator
        catastrophic_gate_terms = torch.where(
            catastrophic_target.bool(),
            F.relu(float(gate_aligned_margins[
                "catastrophic_positive_logit_floor"]) -
                   outputs["catastrophic_logits"]),
            F.relu(outputs["catastrophic_logits"] - float(
                gate_aligned_margins[
                    "catastrophic_negative_logit_ceiling"])))
        catastrophic_gate = ((catastrophic_gate_terms * valid).sum() /
                             denominator)
        beneficial_valid = valid * beneficial_target.float()
        gain_gate = (F.relu(float(gate_aligned_margins[
            "beneficial_gain_floor"]) - outputs["gain"]) *
                     beneficial_valid).sum()
        gain_gate = gain_gate / beneficial_valid.sum().clamp_min(1.0)
    total = (selection + catastrophic + 0.5 * beneficial +
             0.5 * gain + 0.5 * pairwise +
             float(beneficial_gate_weight) * beneficial_gate +
             float(catastrophic_gate_weight) * catastrophic_gate +
             float(gain_gate_weight) * gain_gate)
    return {
        "total": total,
        "selection": selection,
        "beneficial": beneficial,
        "catastrophic": catastrophic,
        "gain": gain,
        "pairwise": pairwise,
        "beneficial_gate": beneficial_gate,
        "catastrophic_gate": catastrophic_gate,
        "gain_gate": gain_gate,
    }


__all__ = ["SetwiseCandidateAssociation", "setwise_association_loss"]
