from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class MPLTTrackActor(BaseActor):
    """ Actor for training MPLT_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    @staticmethod
    def _to_text_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            out = []
            for item in value:
                out.extend(MPLTTrackActor._to_text_list(item))
            return out
        return [str(value)]

    @staticmethod
    def _first_nonempty(*values):
        for value in values:
            if isinstance(value, str) and value:
                return value
        return ''

    @staticmethod
    def _depth_reliability(depth_quality):
        quality = depth_quality.lower()
        if any(k in quality for k in ('reliable', 'good', 'clear', 'valid')):
            return 1.0
        if any(k in quality for k in ('poor', 'unreliable', 'missing', 'invalid', 'noisy', 'low')):
            return 0.25
        return 0.6

    @staticmethod
    def _first_present(container, *keys):
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
        return None

    @staticmethod
    def _feature_batch_size(feature):
        if torch.is_tensor(feature):
            return feature.shape[0] if feature.dim() > 1 else 1
        return 1

    @staticmethod
    def _estimate_depth_image_reliability(depth_img, bbox):
        if depth_img is None or bbox is None:
            return None, {}
        with torch.no_grad():
            depth_img = depth_img.detach()
            bbox = bbox.detach()
            batch_size, _, height, width = depth_img.shape
            reliability = depth_img.new_zeros(batch_size, 1)
            contrast_values = depth_img.new_zeros(batch_size, 1)
            target_std_values = depth_img.new_zeros(batch_size, 1)
            global_std_values = depth_img.flatten(2).std(dim=2, unbiased=False).mean(dim=1, keepdim=True)
            for i in range(batch_size):
                x, y, bw, bh = bbox[i].tolist()
                x0 = max(int(round(x * width)), 0)
                y0 = max(int(round(y * height)), 0)
                x1 = min(int(round((x + max(bw, 1e-6)) * width)), width)
                y1 = min(int(round((y + max(bh, 1e-6)) * height)), height)
                if x1 <= x0 or y1 <= y0:
                    reliability[i, 0] = 0.25
                    continue
                target = depth_img[i:i + 1, :, y0:y1, x0:x1]
                pad_x = max(int(round((x1 - x0) * 0.5)), 4)
                pad_y = max(int(round((y1 - y0) * 0.5)), 4)
                cx0 = max(x0 - pad_x, 0)
                cy0 = max(y0 - pad_y, 0)
                cx1 = min(x1 + pad_x, width)
                cy1 = min(y1 + pad_y, height)
                context = depth_img[i:i + 1, :, cy0:cy1, cx0:cx1].clone()
                context[:, :, y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 0
                target_mean = target.mean(dim=(0, 2, 3))
                context_pixels = context.flatten(2)
                nonzero = context_pixels.abs().sum(dim=1, keepdim=True).gt(1e-6).float()
                denom = nonzero.sum().clamp_min(1.0)
                context_mean = (context_pixels * nonzero).sum(dim=(0, 2)) / denom
                contrast = (target_mean - context_mean).abs().mean()
                target_std = target.flatten(2).std(dim=2, unbiased=False).mean()
                contrast_values[i, 0] = contrast
                target_std_values[i, 0] = target_std
                contrast_score = (contrast / 0.45).clamp(0.0, 1.0)
                variance_score = (global_std_values[i, 0] / 0.75).clamp(0.0, 1.0)
                reliability[i, 0] = (0.7 * contrast_score + 0.3 * variance_score).clamp(0.0, 1.0)
            diagnostics = {
                'input_depth_bbox_contrast': contrast_values.mean(),
                'input_depth_bbox_target_std': target_std_values.mean(),
                'input_depth_global_std_score': (global_std_values / 0.75).clamp(0.0, 1.0).mean(),
                'input_depth_image_reliability': reliability.mean(),
            }
            return reliability, diagnostics

    def _build_language_inputs(self, data):
        visible = data['visible']
        target_feature = self._first_present(
            visible,
            'language_target_feature',
            'language_target_roberta',
            'language_target_roberta_feature',
            'roberta_target_feature',
        )
        context_feature = self._first_present(
            visible,
            'language_context_feature',
            'language_context_roberta',
            'language_context_roberta_feature',
            'roberta_context_feature',
        )
        if target_feature is not None or context_feature is not None:
            depth_qualities = self._to_text_list(visible.get('language_depth_quality', ''))
            batch_size = max(
                self._feature_batch_size(target_feature) if target_feature is not None else 1,
                self._feature_batch_size(context_feature) if context_feature is not None else 1,
            )
            reliability = []
            for i in range(batch_size):
                depth_quality = depth_qualities[i] if i < len(depth_qualities) else ''
                reliability.append(self._depth_reliability(depth_quality))
            return target_feature, context_feature, reliability

        descriptions = self._to_text_list(visible.get('language_description', ''))
        appearances = self._to_text_list(visible.get('language_appearance', ''))
        categories = self._to_text_list(visible.get('language_category', visible.get('test_class', 'object')))
        depth_relations = self._to_text_list(visible.get('language_depth_relation', ''))
        depth_qualities = self._to_text_list(visible.get('language_depth_quality', ''))
        occlusions = self._to_text_list(visible.get('language_occlusion_state', ''))
        distractors = self._to_text_list(visible.get('language_distractor_relation', ''))
        batch_size = max(len(descriptions), len(appearances), len(categories), len(depth_relations), 1)

        def get(items, idx):
            return items[idx] if idx < len(items) else ''

        target_texts, context_texts, reliability = [], [], []
        for i in range(batch_size):
            category = get(categories, i)
            appearance = get(appearances, i)
            description = get(descriptions, i)
            depth_relation = get(depth_relations, i)
            depth_quality = get(depth_qualities, i)
            occlusion = get(occlusions, i)
            distractor = get(distractors, i)
            target_parts = []
            if category:
                target_parts.append('target object: {}'.format(category))
            if appearance:
                target_parts.append('appearance: {}'.format(appearance))
            if depth_relation:
                target_parts.append('target depth geometry: {}'.format(depth_relation))
            target_texts.append(self._first_nonempty('; '.join(target_parts), description, 'target object'))
            context_parts = ['non-target surrounding background and distractor regions']
            if distractor:
                context_parts.append('distractors/context to suppress: {}'.format(distractor))
            if occlusion:
                context_parts.append('occlusion and surrounding state: {}'.format(occlusion))
            if depth_quality:
                context_parts.append('depth reliability cue: {}'.format(depth_quality))
            context_parts.append('not the tracked target')
            context_texts.append('; '.join(context_parts))
            reliability.append(self._depth_reliability(depth_quality))
        return target_texts, context_texts, reliability

    def _language_train_gate(self, data):
        language_cfg = getattr(getattr(self.cfg, "MODEL", None), "LANGUAGE", None)
        if language_cfg is None:
            return None
        warmup_iters = int(getattr(language_cfg, "TRAIN_WARMUP_ITERS", 0))
        ramp_iters = int(getattr(language_cfg, "TRAIN_RAMP_ITERS", 0))
        if warmup_iters <= 0 and ramp_iters <= 0:
            return None
        epoch = int(data.get('epoch', 1))
        iter_id = int(data.get('iter', 1))
        total_iter = max(int(data.get('total_iter', 1)), 1)
        global_iter = max((epoch - 1) * total_iter + iter_id, 1)
        if global_iter <= warmup_iters:
            return 0.0
        if ramp_iters <= 0:
            return 1.0
        return min(max((global_iter - warmup_iters) / float(ramp_iters), 0.0), 1.0)

    def _build_route_target(self, pred_dict, route):
        reliability = pred_dict.get('language_reliability', None)
        if reliability is None:
            reliability = route.new_ones(route.shape[0], 1)
        else:
            reliability = reliability.detach().to(device=route.device, dtype=route.dtype).view(route.shape[0], 1)
        reliability = reliability.clamp(0.0, 1.0)

        def cfg_float(name, default):
            return float(getattr(self.cfg.TRAIN, name, default))

        rgb_low = cfg_float("LANGUAGE_ROUTE_RGB_LOW_RELIABILITY", 0.50)
        rgb_high = cfg_float("LANGUAGE_ROUTE_RGB_HIGH_RELIABILITY", 0.28)
        depth_low = cfg_float("LANGUAGE_ROUTE_DEPTH_LOW_RELIABILITY", 0.15)
        depth_high = cfg_float("LANGUAGE_ROUTE_DEPTH_HIGH_RELIABILITY", 0.42)
        lang_low = cfg_float("LANGUAGE_ROUTE_LANG_LOW_RELIABILITY", 0.005)
        lang_high = cfg_float("LANGUAGE_ROUTE_LANG_HIGH_RELIABILITY", 0.03)

        target_rgb = rgb_low * (1.0 - reliability) + rgb_high * reliability
        target_depth = depth_low * (1.0 - reliability) + depth_high * reliability
        target_lang = lang_low * (1.0 - reliability) + lang_high * reliability
        language_cfg = getattr(getattr(self.cfg, "MODEL", None), "LANGUAGE", None)
        if language_cfg is not None:
            route_language_max = float(getattr(language_cfg, "ROUTE_LANGUAGE_MAX", 1.0))
            target_lang = target_lang.clamp(max=max(route_language_max, 0.0))
        target_shared = (1.0 - target_rgb - target_depth - target_lang).clamp_min(1e-4)
        target = torch.cat([target_rgb, target_depth, target_lang, target_shared], dim=1)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return target.detach()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['visible']['template_images']) == 1
        assert len(data['visible']['search_images']) == 1 or len(data['visible']['search_images']) == 2

        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)        
        
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_v_last = data['visible']['search_images'][1].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i_last = data['infrared']['search_images'][1].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        language_kwargs = {}
        if getattr(getattr(self.cfg.MODEL, "LANGUAGE", None), "USE", False):
            language_target, language_context, language_reliability = self._build_language_inputs(data)
            depth_reliability_diag = {}
            language_cfg = getattr(self.cfg.MODEL, "LANGUAGE", None)
            if bool(getattr(language_cfg, "USE_DEPTH_IMAGE_RELIABILITY", False)):
                image_reliability, depth_reliability_diag = self._estimate_depth_image_reliability(
                    search_img_i, data['visible']['search_anno'][0])
                if image_reliability is not None:
                    image_reliability = image_reliability.view(-1).tolist()
                    if language_reliability is None:
                        language_reliability = image_reliability
                    else:
                        text_reliability = torch.as_tensor(language_reliability, dtype=search_img_i.dtype)
                        image_reliability_tensor = torch.as_tensor(image_reliability, dtype=search_img_i.dtype)
                        weight = float(getattr(language_cfg, "DEPTH_IMAGE_RELIABILITY_WEIGHT", 0.5))
                        language_reliability = (
                            (1.0 - weight) * text_reliability + weight * image_reliability_tensor).tolist()
            language_kwargs = {
                'language_target': language_target,
                'language_context': language_context,
                'language_reliability': language_reliability,
                'language_train_gate': self._language_train_gate(data),
            }

        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i, search_img_v_last, search_img_i_last],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            **language_kwargs)
        out_dict['input_rgb_mean'] = search_img_v.detach().mean()
        out_dict['input_rgb_std'] = search_img_v.detach().std(unbiased=False)
        out_dict['input_depth_mean'] = search_img_i.detach().mean()
        out_dict['input_depth_std'] = search_img_i.detach().std(unbiased=False)
        if getattr(getattr(self.cfg.MODEL, "LANGUAGE", None), "USE", False):
            depth_reliability, depth_reliability_diag = self._estimate_depth_image_reliability(
                search_img_i, data['visible']['search_anno'][0])
            out_dict.update(depth_reliability_diag)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][0]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap([gt_dict['search_anno'][0]], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        target_heatmap = F.interpolate(gt_gaussian_maps, size=pred_dict['score_map'].shape[-2:],
                                       mode='bilinear', align_corners=False).clamp(0.0, 1.0)
        language_target_loss = torch.tensor(0.0, device=l1_loss.device)
        language_context_loss = torch.tensor(0.0, device=l1_loss.device)
        language_route_loss = torch.tensor(0.0, device=l1_loss.device)
        language_route_entropy_loss = torch.tensor(0.0, device=l1_loss.device)
        language_route_supervision_loss = torch.tensor(0.0, device=l1_loss.device)
        language_route_target = None
        language_match_scale_loss = torch.tensor(0.0, device=l1_loss.device)
        if 'language_target_logits' in pred_dict:
            target_logits = pred_dict['language_target_logits']
            target_heatmap = F.interpolate(gt_gaussian_maps, size=target_logits.shape[-2:], mode='bilinear',
                                           align_corners=False).clamp(0.0, 1.0)
            pos = target_heatmap.gt(0.2).float().sum().clamp(min=1.0)
            neg = target_heatmap.le(0.2).float().sum().clamp(min=1.0)
            pos_weight = (neg / pos).clamp(min=1.0, max=16.0).detach()
            language_target_loss = F.binary_cross_entropy_with_logits(
                target_logits, target_heatmap, pos_weight=pos_weight)
            if 'language_targetness_logits' in pred_dict:
                language_target_loss = language_target_loss + 0.5 * F.binary_cross_entropy_with_logits(
                    pred_dict['language_targetness_logits'], target_heatmap, pos_weight=pos_weight)
        elif 'language_targetness_map' in pred_dict:
            targetness = pred_dict['language_targetness_map']
            target_heatmap = F.interpolate(gt_gaussian_maps, size=targetness.shape[-2:], mode='bilinear',
                                           align_corners=False).clamp(0.0, 1.0)
            language_target_loss = F.mse_loss(targetness, target_heatmap)
        if 'language_context_logits' in pred_dict:
            context_logits = pred_dict['language_context_logits']
            context_heatmap = F.interpolate(gt_gaussian_maps, size=context_logits.shape[-2:], mode='bilinear',
                                            align_corners=False).clamp(0.0, 1.0)
            context_target = 1.0 - context_heatmap
            context_weight = 1.0 + 8.0 * context_heatmap
            language_context_loss = F.binary_cross_entropy_with_logits(
                context_logits, context_target, weight=context_weight)
            context_fg_mask = context_heatmap.gt(0.3)
            context_bg_mask = context_heatmap.lt(0.05)
            if context_fg_mask.any() and context_bg_mask.any():
                context_fg_logit = context_logits[context_fg_mask].mean()
                context_bg_logit = context_logits[context_bg_mask].mean()
                language_context_loss = language_context_loss + F.relu(
                    0.05 + context_fg_logit - context_bg_logit)
        elif 'language_context_map' in pred_dict:
            context_map = pred_dict['language_context_map']
            context_target = 1.0 - F.interpolate(gt_gaussian_maps, size=context_map.shape[-2:], mode='bilinear',
                                                 align_corners=False).clamp(0.0, 1.0)
            language_context_loss = F.mse_loss(context_map, context_target)
        if 'moe_route_weights' in pred_dict:
            route = pred_dict['moe_route_weights'].clamp(1e-6, 1.0)
            entropy = -(route * route.log()).sum(dim=1).mean()
            language_route_entropy_loss = -entropy
            route_supervision_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_ROUTE_SUPERVISION_WEIGHT", 0.0))
            route_entropy_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_ROUTE_ENTROPY_WEIGHT", 1.0))
            if route_supervision_weight > 0.0:
                language_route_target = self._build_route_target(pred_dict, route)
                language_route_supervision_loss = F.kl_div(
                    route.log(), language_route_target, reduction='batchmean')
                language_route_supervision_loss = language_route_supervision_loss + 0.25 * F.l1_loss(
                    route, language_route_target)
            language_route_loss = (
                route_entropy_weight * language_route_entropy_loss
                + route_supervision_weight * language_route_supervision_loss)
        if 'language_match_scale_loss' in pred_dict:
            scale_target = float(getattr(self.cfg.TRAIN, "LANGUAGE_MATCH_SCALE_TARGET", 10.0))
            language_match_scale_loss = F.relu(pred_dict['language_match_scale_loss'] - scale_target).pow(2)

        language_target_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_TARGET_WEIGHT", 0.0))
        language_context_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_CONTEXT_WEIGHT", 0.0))
        language_route_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_ROUTE_WEIGHT", 0.0))
        language_match_scale_weight = float(getattr(self.cfg.TRAIN, "LANGUAGE_MATCH_SCALE_WEIGHT", 0.0))
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = loss + language_target_weight * language_target_loss
        loss = loss + language_context_weight * language_context_loss
        loss = loss + language_route_weight * language_route_loss
        loss = loss + language_match_scale_weight * language_match_scale_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/lang_target": language_target_loss.item(),
                      "Loss/lang_context": language_context_loss.item(),
                      "Loss/lang_route": language_route_loss.item(),
                      "Loss/lang_route_entropy": language_route_entropy_loss.item(),
                      "Loss/lang_route_sup": language_route_supervision_loss.item(),
                      "Loss/lang_match_scale": language_match_scale_loss.item(),
                      "IoU": mean_iou.item()}
            if 'language_targetness_map' in pred_dict:
                targetness = pred_dict['language_targetness_map'].detach()
                lang_heatmap = F.interpolate(gt_gaussian_maps, size=targetness.shape[-2:], mode='bilinear',
                                             align_corners=False).clamp(0.0, 1.0)
                fg_mask = lang_heatmap.gt(0.3)
                bg_mask = lang_heatmap.lt(0.05)
                status.update({
                    "Lang/targetness_mean": targetness.mean().item(),
                    "Lang/targetness_max": targetness.max().item(),
                    "Lang/targetness_min": targetness.min().item(),
                })
                if fg_mask.any() and bg_mask.any():
                    target_fg = targetness[fg_mask].mean()
                    target_bg = targetness[bg_mask].mean()
                    status.update({
                        "Lang/targetness_fg": target_fg.item(),
                        "Lang/targetness_bg": target_bg.item(),
                        "Lang/targetness_gap": (target_fg - target_bg).item(),
                    })
            if 'language_target_map' in pred_dict:
                target_map = pred_dict['language_target_map'].detach()
                status["Lang/target_match_mean"] = target_map.mean().item()
                if 'language_targetness_map' in pred_dict and fg_mask.any() and bg_mask.any():
                    target_fg = target_map[fg_mask].mean()
                    target_bg = target_map[bg_mask].mean()
                    status["Lang/target_match_gap"] = (target_fg - target_bg).item()
            if 'language_context_map' in pred_dict:
                context_map = pred_dict['language_context_map'].detach()
                status["Lang/context_match_mean"] = context_map.mean().item()
                if 'language_targetness_map' in pred_dict and fg_mask.any() and bg_mask.any():
                    context_fg = context_map[fg_mask].mean()
                    context_bg = context_map[bg_mask].mean()
                    status["Lang/context_bg_minus_fg"] = (context_bg - context_fg).item()
            if 'language_targetness_logits' in pred_dict:
                targetness_logits = pred_dict['language_targetness_logits'].detach()
                status["Lang/targetness_logit_std"] = targetness_logits.std(unbiased=False).item()
            if 'language_candidate_gate' in pred_dict:
                candidate_gate = pred_dict['language_candidate_gate'].detach()
                status.update({
                    "Lang/candidate_gate_mean": candidate_gate.mean().item(),
                    "Lang/candidate_gate_min": candidate_gate.min().item(),
                    "Lang/candidate_gate_max": candidate_gate.max().item(),
                })
            if 'language_candidate_keep_mask' in pred_dict:
                keep_mask = pred_dict['language_candidate_keep_mask'].detach()
                status["Lang/candidate_keep_ratio"] = keep_mask.mean().item()
            if 'language_match_scale' in pred_dict:
                status["Lang/match_scale"] = float(pred_dict['language_match_scale'].item())
            if 'language_reliability' in pred_dict:
                status["Lang/depth_reliability"] = pred_dict['language_reliability'].detach().mean().item()
            if 'language_runtime_gate' in pred_dict:
                status["Lang/runtime_gate"] = pred_dict['language_runtime_gate'].detach().mean().item()
            if 'language_train_gate' in pred_dict:
                status["Lang/train_gate"] = pred_dict['language_train_gate'].detach().mean().item()
            if 'moe_expert_residual_weight' in pred_dict:
                status["MoE/residual_weight"] = float(pred_dict['moe_expert_residual_weight'].item())
            if 'moe_matching_strength' in pred_dict:
                status["MoE/matching_strength"] = float(pred_dict['moe_matching_strength'].item())
            if 'moe_token_keep_ratio' in pred_dict:
                status["MoE/token_keep_ratio_cfg"] = float(pred_dict['moe_token_keep_ratio'].item())
            if 'moe_token_mask_strength' in pred_dict:
                status["MoE/token_mask_strength"] = float(pred_dict['moe_token_mask_strength'].item())
            if 'moe_effective_token_mask_strength' in pred_dict:
                status["MoE/effective_token_mask_strength"] = float(
                    pred_dict['moe_effective_token_mask_strength'].item())
            if 'moe_delta_norm_ratio' in pred_dict:
                status["MoE/delta_norm_ratio"] = pred_dict['moe_delta_norm_ratio'].mean().item()
            if 'moe_residual_gate' in pred_dict:
                status["MoE/residual_gate"] = pred_dict['moe_residual_gate'].mean().item()
            if 'moe_route_weights' in pred_dict:
                route = pred_dict['moe_route_weights'].detach()
                route_max = route.max(dim=1).values
                status.update({
                    "MoE/w_rgb": route[:, 0].mean().item(),
                    "MoE/w_depth": route[:, 1].mean().item(),
                    "MoE/w_lang": route[:, 2].mean().item(),
                    "MoE/w_shared": route[:, 3].mean().item(),
                    "MoE/entropy": (-(route.clamp(1e-6, 1.0) * route.clamp(1e-6, 1.0).log()).sum(dim=1)).mean().item(),
                    "MoE/route_max": route_max.mean().item(),
                })
                if language_route_target is not None:
                    route_target = language_route_target.detach()
                    route_l1 = (route - route_target).abs().mean(dim=1)
                    status.update({
                        "MoE/target_w_rgb": route_target[:, 0].mean().item(),
                        "MoE/target_w_depth": route_target[:, 1].mean().item(),
                        "MoE/target_w_lang": route_target[:, 2].mean().item(),
                        "MoE/target_w_shared": route_target[:, 3].mean().item(),
                        "MoE/route_target_l1": route_l1.mean().item(),
                    })
                    if 'language_reliability' in pred_dict:
                        reliability = pred_dict['language_reliability'].detach().view(-1).to(route.device)
                        reliable_mask = reliability >= 0.75
                        unreliable_mask = reliability <= 0.35
                        if reliable_mask.any():
                            status["MoE/reliable_w_depth"] = route[reliable_mask, 1].mean().item()
                            status["MoE/reliable_w_rgb"] = route[reliable_mask, 0].mean().item()
                        if unreliable_mask.any():
                            status["MoE/unreliable_w_depth"] = route[unreliable_mask, 1].mean().item()
                            status["MoE/unreliable_w_rgb"] = route[unreliable_mask, 0].mean().item()
            if 'moe_route_weights_raw' in pred_dict:
                route_raw = pred_dict['moe_route_weights_raw'].detach()
                status["MoE/raw_w_lang"] = route_raw[:, 2].mean().item()
            for diag_key, log_key in [
                ('input_rgb_mean', 'Input/rgb_mean'),
                ('input_rgb_std', 'Input/rgb_std'),
                ('input_depth_mean', 'Input/depth_mean'),
                ('input_depth_std', 'Input/depth_std'),
                ('input_depth_bbox_contrast', 'Input/depth_bbox_contrast'),
                ('input_depth_bbox_target_std', 'Input/depth_bbox_target_std'),
                ('input_depth_global_std_score', 'Input/depth_global_std_score'),
                ('input_depth_image_reliability', 'Input/depth_image_reliability'),
            ]:
                if diag_key in pred_dict:
                    status[log_key] = pred_dict[diag_key].item()
            return loss, status
        else:
            return loss
