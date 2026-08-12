"""
MPLT_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lib.models.layers.head import build_box_head, conv
from lib.models.mplt_track.vit_mplt_care import vit_base_patch16_224_mplt
from lib.utils.box_ops import box_xyxy_to_cxcywh


class DepthReliableLanguageMoE(nn.Module):
    """RGB-D-L expert routing for target/context language alignment."""

    def __init__(self, hidden_dim, dropout=0.1, router_temperature=1.0,
                 matching_strength=0.10, expert_residual_weight=0.35,
                 token_keep_ratio=0.75, token_mask_strength=0.05,
                 route_language_max=0.35, match_logit_scale_max=20.0,
                 match_logit_scale_trainable=True,
                 reliability_residual_gate=True):
        super().__init__()
        self.matching_strength = float(matching_strength)
        self.expert_residual_weight = float(expert_residual_weight)
        self.token_keep_ratio = float(token_keep_ratio)
        self.token_mask_strength = float(token_mask_strength)
        self.route_language_max = float(route_language_max)
        self.router_temperature = max(float(router_temperature), 1e-4)
        self.match_logit_scale_max = max(float(match_logit_scale_max), 1.0)
        self.reliability_residual_gate = bool(reliability_residual_gate)
        self.match_logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        if not bool(match_logit_scale_trainable):
            self.match_logit_scale.requires_grad_(False)

        def expert():
            return nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            )

        self.rgb_expert = expert()
        self.depth_expert = expert()
        self.shared_expert = expert()
        self.language_to_feature = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.match_norm = nn.LayerNorm(hidden_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5 + 6),
            nn.Linear(hidden_dim * 5 + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @staticmethod
    def _global_pool(feat):
        return F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)

    def _candidate_gate(self, targetness_score, height, width):
        bs, num_tokens = targetness_score.shape
        return self._candidate_gate_with_strength(targetness_score, height, width, self.token_mask_strength)

    def _candidate_gate_with_strength(self, targetness_score, height, width, token_mask_strength):
        bs, num_tokens = targetness_score.shape
        token_mask_strength = float(max(token_mask_strength, 0.0))
        if self.token_keep_ratio >= 1.0 or token_mask_strength <= 0.0:
            gate = targetness_score.new_ones(bs, num_tokens)
            keep_mask = gate
        else:
            keep_count = max(1, min(num_tokens, int(math.ceil(num_tokens * self.token_keep_ratio))))
            _, keep_idx = torch.topk(targetness_score, k=keep_count, dim=1, largest=True)
            keep_mask = targetness_score.new_zeros(bs, num_tokens)
            keep_mask.scatter_(1, keep_idx, 1.0)
            gate = 1.0 - token_mask_strength * (1.0 - keep_mask)
            gate = gate / gate.mean(dim=1, keepdim=True).clamp_min(1e-6)
            gate = gate.clamp(0.5, 1.5)
        return gate.view(bs, 1, height, width), keep_mask

    def _cap_language_route(self, route_weights):
        if self.route_language_max >= 1.0:
            return route_weights
        lang = route_weights[:, 2:3].clamp(max=max(self.route_language_max, 0.0))
        other = torch.cat([route_weights[:, :2], route_weights[:, 3:]], dim=1)
        other = other / other.sum(dim=1, keepdim=True).clamp_min(1e-6) * (1.0 - lang)
        return torch.cat([other[:, :2], lang, other[:, 2:]], dim=1)

    def forward(self, rgb_feat, depth_feat, shared_feat, language_features):
        bs, channels, height, width = shared_feat.shape
        target_feature = language_features['target']
        context_feature = language_features['context']
        reliability = language_features['reliability'].to(dtype=shared_feat.dtype)
        runtime_gate = language_features.get('runtime_gate', None)
        if runtime_gate is None:
            runtime_gate = reliability.new_ones(reliability.shape[0], 1)
        runtime_gate = runtime_gate.to(dtype=shared_feat.dtype).view(bs, 1).clamp(0.0, 1.0)
        train_gate = language_features.get('train_gate', None)
        if train_gate is None:
            train_gate = reliability.new_ones(reliability.shape[0], 1)
        train_gate = train_gate.to(dtype=shared_feat.dtype).view(bs, 1).clamp(0.0, 1.0)

        token_feat = shared_feat.flatten(2).transpose(1, 2)
        token_feat = F.normalize(self.match_norm(token_feat), dim=-1)
        target_norm = F.normalize(target_feature, dim=-1)
        context_norm = F.normalize(context_feature, dim=-1)

        match_scale = self.match_logit_scale.exp().clamp(1.0, self.match_logit_scale_max)
        target_score = match_scale * (token_feat * target_norm[:, None, :]).sum(dim=-1)
        context_score = match_scale * (token_feat * context_norm[:, None, :]).sum(dim=-1)
        targetness_score = target_score - 0.7 * context_score
        target_map = torch.sigmoid(target_score).view(bs, 1, height, width)
        context_map = torch.sigmoid(context_score).view(bs, 1, height, width)
        targetness_map = torch.sigmoid(targetness_score).view(bs, 1, height, width)

        language_basis = self.language_to_feature(
            torch.cat([target_feature, context_feature, target_feature - context_feature, reliability], dim=1))
        language_feat = language_basis.view(bs, channels, 1, 1) * (1.0 + targetness_map)

        rgb_expert = self.rgb_expert(rgb_feat)
        depth_expert = self.depth_expert(depth_feat)
        shared_expert = self.shared_expert(shared_feat)

        rgb_global = self._global_pool(rgb_expert)
        depth_global = self._global_pool(depth_expert)
        shared_global = self._global_pool(shared_expert)
        depth_variance = depth_feat.flatten(2).var(dim=2, unbiased=False).mean(dim=1, keepdim=True)
        rgb_depth_cos = F.cosine_similarity(rgb_global, depth_global, dim=1, eps=1e-6).unsqueeze(1)
        sim_target = target_score.mean(dim=1, keepdim=True)
        sim_context = context_score.mean(dim=1, keepdim=True)
        contrast = (target_score - context_score).mean(dim=1, keepdim=True)

        router_input = torch.cat([
            rgb_global,
            depth_global,
            shared_global,
            target_feature,
            context_feature,
            reliability,
            sim_target,
            sim_context,
            contrast,
            depth_variance,
            rgb_depth_cos,
        ], dim=1)
        route_logits = self.router(router_input) / self.router_temperature
        route_weights_raw = torch.softmax(route_logits, dim=1)
        route_weights = self._cap_language_route(route_weights_raw)
        route = route_weights.view(bs, 4, 1, 1, 1)

        expert_stack = torch.stack([rgb_expert, depth_expert, language_feat, shared_expert], dim=1)
        expert_delta = (route * expert_stack).sum(dim=1)
        residual_gate = (runtime_gate * train_gate).view(bs, 1, 1, 1)
        if not self.reliability_residual_gate:
            residual_gate = train_gate.view(bs, 1, 1, 1)
        fused = shared_feat + self.expert_residual_weight * residual_gate * expert_delta
        targetness_centered = targetness_map - targetness_map.flatten(2).mean(dim=2).view(bs, 1, 1, 1)
        if self.matching_strength > 0.0:
            fused = fused * (1.0 + self.matching_strength * residual_gate * targetness_centered)
        token_mask_strength = self.token_mask_strength * float(train_gate.detach().mean().item())
        candidate_gate, keep_mask = self._candidate_gate_with_strength(
            targetness_score, height, width, token_mask_strength)
        fused = fused * candidate_gate

        shared_norm = shared_feat.flatten(1).norm(dim=1).clamp_min(1e-6)
        delta_norm = (self.expert_residual_weight * residual_gate * expert_delta).flatten(1).norm(dim=1) / shared_norm

        return fused, {
            'language_target_map': target_map,
            'language_context_map': context_map,
            'language_targetness_map': targetness_map,
            'language_candidate_gate': candidate_gate,
            'language_candidate_keep_mask': keep_mask.view(bs, 1, height, width),
            'language_target_logits': target_score.view(bs, 1, height, width),
            'language_context_logits': context_score.view(bs, 1, height, width),
            'language_targetness_logits': targetness_score.view(bs, 1, height, width),
            'language_reliability': reliability,
            'language_runtime_gate': runtime_gate.detach(),
            'language_train_gate': train_gate.detach(),
            'moe_route_weights_raw': route_weights_raw,
            'moe_route_weights': route_weights,
            'language_match_scale': match_scale.detach(),
            'language_match_scale_loss': match_scale,
            'moe_expert_residual_weight': shared_feat.new_tensor(self.expert_residual_weight),
            'moe_matching_strength': shared_feat.new_tensor(self.matching_strength),
            'moe_token_keep_ratio': shared_feat.new_tensor(self.token_keep_ratio),
            'moe_token_mask_strength': shared_feat.new_tensor(self.token_mask_strength),
            'moe_effective_token_mask_strength': shared_feat.new_tensor(token_mask_strength),
            'moe_delta_norm_ratio': delta_norm.detach(),
            'moe_residual_gate': residual_gate.flatten().detach(),
        }


class MPLTTrack(nn.Module):
    """ This is the base class for MPLTTrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", language_cfg=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.hidden_dim = hidden_dim
        self.backbone = transformer
        self.mplt_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head
        self.use_language = bool(language_cfg is not None and getattr(language_cfg, "USE", False))
        self.language_tokenizer = None
        self.language_encoder = None
        self.language_proj = None
        self.language_moe = None
        self.language_cfg = language_cfg
        self.freeze_language_encoder = True

        if self.use_language:
            model_name = getattr(language_cfg, "MODEL_NAME_OR_PATH", "roberta-base")
            local_files_only = bool(getattr(language_cfg, "LOCAL_FILES_ONLY", True))
            random_init = bool(getattr(language_cfg, "RANDOM_INIT_IF_MISSING", False))
            try:
                self.language_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=local_files_only, use_fast=True)
                self.language_encoder = AutoModel.from_pretrained(
                    model_name, local_files_only=local_files_only)
            except Exception as exc:
                if not random_init:
                    raise RuntimeError(
                        "Failed to load RoBERTa language encoder from '{}'. Put the model files locally "
                        "or set MODEL.LANGUAGE.LOCAL_FILES_ONLY to False if network access is available."
                        .format(model_name)) from exc
                roberta_config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
                self.language_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=local_files_only, use_fast=True)
                self.language_encoder = AutoModel.from_config(roberta_config)

            self.freeze_language_encoder = bool(getattr(language_cfg, "FREEZE", True))
            if self.freeze_language_encoder:
                for p in self.language_encoder.parameters():
                    p.requires_grad = False

            language_dim = self.language_encoder.config.hidden_size
            self.language_input_dim = language_dim
            language_dropout = float(getattr(language_cfg, "DROPOUT", 0.1))
            self.language_max_length = int(getattr(language_cfg, "MAX_LENGTH", 48))
            self.language_proj = nn.Sequential(
                nn.LayerNorm(language_dim),
                nn.Dropout(language_dropout),
                nn.Linear(language_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.language_moe = DepthReliableLanguageMoE(
                hidden_dim,
                dropout=language_dropout,
                router_temperature=float(getattr(language_cfg, "ROUTER_TEMPERATURE", 1.0)),
                matching_strength=float(getattr(language_cfg, "MATCHING_STRENGTH", 0.10)),
                expert_residual_weight=float(getattr(language_cfg, "EXPERT_RESIDUAL_WEIGHT", 0.35)),
                token_keep_ratio=float(getattr(language_cfg, "TOKEN_KEEP_RATIO", 0.75)),
                token_mask_strength=float(getattr(language_cfg, "TOKEN_MASK_STRENGTH", 0.05)),
                route_language_max=float(getattr(language_cfg, "ROUTE_LANGUAGE_MAX", 0.35)),
                match_logit_scale_max=float(getattr(language_cfg, "MATCH_LOGIT_SCALE_MAX", 20.0)),
                match_logit_scale_trainable=bool(getattr(language_cfg, "MATCH_LOGIT_SCALE_TRAINABLE", True)),
                reliability_residual_gate=bool(getattr(language_cfg, "RELIABILITY_RESIDUAL_GATE", True)),
            )

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                language_target=None,
                language_context=None,
                language_reliability=None,
                language_runtime_gate=None,
                language_train_gate=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        self._sync_language_runtime_cfg()
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        language_features = None
        if self.use_language:
            language_features = self.encode_language_pair(
                language_target, language_context, language_reliability, language_runtime_gate,
                language_train_gate, feat_last.device)
        out = self.forward_head(feat_last, None, language_features=language_features)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def _sync_language_runtime_cfg(self):
        if not self.use_language or self.language_moe is None or self.language_cfg is None:
            return
        cfg = self.language_cfg
        self.language_moe.matching_strength = float(getattr(cfg, "MATCHING_STRENGTH", self.language_moe.matching_strength))
        self.language_moe.expert_residual_weight = float(
            getattr(cfg, "EXPERT_RESIDUAL_WEIGHT", self.language_moe.expert_residual_weight))
        self.language_moe.token_keep_ratio = float(getattr(cfg, "TOKEN_KEEP_RATIO", self.language_moe.token_keep_ratio))
        self.language_moe.token_mask_strength = float(
            getattr(cfg, "TOKEN_MASK_STRENGTH", self.language_moe.token_mask_strength))
        self.language_moe.route_language_max = float(
            getattr(cfg, "ROUTE_LANGUAGE_MAX", self.language_moe.route_language_max))
        self.language_moe.router_temperature = max(
            float(getattr(cfg, "ROUTER_TEMPERATURE", self.language_moe.router_temperature)), 1e-4)
        self.language_moe.match_logit_scale_max = max(
            float(getattr(cfg, "MATCH_LOGIT_SCALE_MAX", self.language_moe.match_logit_scale_max)), 1.0)
        self.language_moe.reliability_residual_gate = bool(
            getattr(cfg, "RELIABILITY_RESIDUAL_GATE", self.language_moe.reliability_residual_gate))

    def encode_language_pair(self, language_target, language_context, language_reliability,
                             language_runtime_gate, language_train_gate, device):
        target_feature = self.encode_language(language_target, device)
        context_feature = self.encode_language(language_context, device)
        if target_feature is None and context_feature is None:
            return None
        if target_feature is None:
            target_feature = context_feature
        if context_feature is None:
            context_feature = target_feature
        batch_size = target_feature.shape[0]
        if language_reliability is None:
            reliability = target_feature.new_ones(batch_size, 1)
        else:
            if torch.is_tensor(language_reliability):
                reliability = language_reliability.to(device=device, dtype=target_feature.dtype)
            else:
                reliability = torch.tensor(language_reliability, device=device, dtype=target_feature.dtype)
            reliability = reliability.view(batch_size, 1).clamp(0.0, 1.0)
        if language_runtime_gate is None:
            runtime_gate = target_feature.new_ones(batch_size, 1)
        else:
            if torch.is_tensor(language_runtime_gate):
                runtime_gate = language_runtime_gate.to(device=device, dtype=target_feature.dtype)
            else:
                runtime_gate = torch.tensor(language_runtime_gate, device=device, dtype=target_feature.dtype)
            runtime_gate = runtime_gate.view(batch_size, 1).clamp(0.0, 1.0)
        if language_train_gate is None:
            train_gate = target_feature.new_ones(batch_size, 1)
        else:
            if torch.is_tensor(language_train_gate):
                train_gate = language_train_gate.to(device=device, dtype=target_feature.dtype)
            else:
                train_gate = torch.tensor(language_train_gate, device=device, dtype=target_feature.dtype)
            train_gate = train_gate.view(batch_size, 1).clamp(0.0, 1.0)
        return {
            'target': target_feature,
            'context': context_feature,
            'reliability': reliability,
            'runtime_gate': runtime_gate,
            'train_gate': train_gate,
        }

    def encode_language(self, language, device):
        if language is None:
            return None
        if torch.is_tensor(language):
            feature = language.to(device=device, dtype=next(self.language_proj.parameters()).dtype)
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            if feature.dim() == 3:
                feature = feature[:, 0, :]
            if feature.shape[-1] == self.language_input_dim:
                return self.language_proj(feature)
            if feature.shape[-1] == self.hidden_dim:
                return feature
            raise ValueError("Unsupported language feature dimension: {}".format(feature.shape[-1]))
        if isinstance(language, tuple):
            language = list(language)
        if isinstance(language, str):
            language = [language]
        if not isinstance(language, list):
            language = list(language)
        language = [text if isinstance(text, str) and text else "object" for text in language]
        tokens = self.language_tokenizer(
            language,
            padding=True,
            truncation=True,
            max_length=self.language_max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        if self.freeze_language_encoder:
            self.language_encoder.eval()
        with torch.set_grad_enabled(not self.freeze_language_encoder):
            encoded = self.language_encoder(**tokens).last_hidden_state[:, 0]
        return self.language_proj(encoded)

    def forward_head(self, cat_feature, gt_score_map=None, language_features=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        bs = enc_opt1.shape[0]
        Nq = 1
        rgb_feat = enc_opt1.transpose(1, 2).contiguous().view(bs, -1, self.feat_sz_s, self.feat_sz_s)
        depth_feat = enc_opt2.transpose(1, 2).contiguous().view(bs, -1, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.mplt_fuse_search(torch.cat([rgb_feat, depth_feat], dim=1))
        language_maps = {}
        if language_features is not None and self.language_moe is not None:
            opt_feat, language_maps = self.language_moe(rgb_feat, depth_feat, opt_feat, language_features)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            out.update(language_maps)
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            out.update(language_maps)
            return out
        else:
            raise NotImplementedError


def build_mplt_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    full_model_pretrain = ''
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrain_file = os.path.expanduser(cfg.MODEL.PRETRAIN_FILE)
        if not os.path.isabs(pretrain_file):
            pretrain_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        if os.path.isfile(pretrain_file):
            try:
                checkpoint_probe = torch.load(pretrain_file, map_location='cpu')
                if isinstance(checkpoint_probe, dict) and 'net' in checkpoint_probe:
                    full_model_pretrain = pretrain_file
                del checkpoint_probe
            except Exception:
                full_model_pretrain = ''
    if cfg.MODEL.PRETRAIN_FILE and not full_model_pretrain and ('MPLTTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_mplt':
        backbone = vit_base_patch16_224_mplt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            mplt_loc=cfg.MODEL.BACKBONE.MPLT_LOC,
                                            mplt_drop_path=cfg.TRAIN.MPLT_DROP_PATH
                                            )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = MPLTTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        language_cfg=getattr(cfg.MODEL, "LANGUAGE", None),
    )

    if full_model_pretrain and training:
        checkpoint = torch.load(full_model_pretrain, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load full MPLTTrack model from: ' + full_model_pretrain)
        print('Missing keys: {}'.format(missing_keys))
        print('Unexpected keys: {}'.format(unexpected_keys))
    elif 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
