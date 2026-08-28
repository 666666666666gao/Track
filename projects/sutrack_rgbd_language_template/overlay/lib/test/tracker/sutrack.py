from lib.test.tracker.basetracker import BaseTracker
import math

import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.sutrack import build_sutrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.test.tracker.safe_template_update import (
    SafeTemplateUpdatePolicy,
    nms_response_margin,
)
import clip
import numpy as np


class SUTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SUTRACK, self).__init__(params)
        network = build_sutrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        safe_template_config = self.cfg.TEST.SAFE_TEMPLATE_UPDATE
        self.safe_template_update = bool(safe_template_config.USE)
        self.safe_template_policy = None
        self.safe_template_nms_kernel = int(safe_template_config.NMS_KERNEL)
        self.safe_template_apply_tensor_blend = bool(
            safe_template_config.APPLY_TENSOR_BLEND)
        if self.safe_template_update:
            if self.num_template < 2:
                raise ValueError(
                    'Safe template update requires at least two template slots')
            self.safe_template_policy = SafeTemplateUpdatePolicy.from_config(
                safe_template_config)
            if (self.safe_template_nms_kernel <= 0 or
                    self.safe_template_nms_kernel % 2 == 0):
                raise ValueError('Safe template NMS kernel must be a positive odd integer')

        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)

        if (self.cfg.TEST.RGBD_LANGUAGE.USE and
                (not self.multi_modal_language or not self.use_nlp)):
            raise ValueError(
                'Bound RGB-D language requires active language and NLP paths')
        if self.safe_template_update and not self.multi_modal_vision:
            raise ValueError(
                'RGB-D safe template update requires multi-modal vision')

        self.task_index_batch = None
        online_language_config = self.cfg.TEST.ONLINE_LANGUAGE_UPDATE
        self.online_language_evidence_enabled = bool(
            online_language_config.USE)
        self._last_online_response_margin = None


    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.static_template = template
        self.template_list = [self.static_template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.static_template_anno = prev_box_crop.to(template.device).unsqueeze(0)
        self.template_anno_list = [self.static_template_anno]
        self.frame_id = 0
        self.sequence_name = str(info.get('sequence_name', 'unknown'))
        self._last_online_response_margin = None

        if self.safe_template_update:
            self.safe_template_policy.initialize(
                image, info['init_bbox'], info.get('depth_path'))

        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                init_nlp = info.get("init_nlp")
            else:
                init_nlp = None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        prior_state = list(self.state)
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            enc_opt = self.network.forward_encoder(self.template_list,
                                                   search_list,
                                                   self.template_anno_list,
                                                   self.text_src,
                                                   self.task_index_batch)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=enc_opt)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if self.online_language_evidence_enabled:
            selected_index = int(
                response.detach().reshape(-1).argmax().item())
            self._last_online_response_margin = nms_response_margin(
                response.detach(), selected_index,
                kernel=int(self.cfg.TEST.ONLINE_LANGUAGE_UPDATE.NMS_KERNEL))
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # Update only the bounded dynamic slot.  In the ported configuration,
        # simultaneous temporal, response, immutable-RGB and raw-depth
        # evidence replaces the baseline interval/confidence rule.
        safe_template_decision = None
        applied_template_blend_weight = None
        candidate_state = list(self.state)
        online_state_evidence = None
        if self.safe_template_update:
            confidence = float(conf_score.detach().reshape(-1).max().item())
            selected_index = int(response.detach().reshape(-1).argmax().item())
            response_margin = nms_response_margin(
                response.detach(), selected_index,
                kernel=self.safe_template_nms_kernel)
            safe_template_decision = self.safe_template_policy.observe(
                self.frame_id, image, self.state, confidence,
                response_margin, None if info is None else info.get('depth_path'))
            if safe_template_decision.drop_dynamic:
                self._drop_dynamic_template()
            if safe_template_decision.rollback_state:
                self.state = prior_state
            if safe_template_decision.replace_dynamic:
                applied_template_blend_weight = (
                    float(self.safe_template_policy.blend_weight)
                    if self.safe_template_apply_tensor_blend else 1.0)
                self._replace_dynamic_template(
                    image,
                    blend_weight=(
                        applied_template_blend_weight
                        if self.safe_template_apply_tensor_blend else None))
                self.safe_template_policy.commit(self.frame_id)
            online_state_evidence = {
                'frame_id': int(self.frame_id),
                'prior_bbox': [float(value) for value in prior_state],
                'candidate_bbox': [float(value) for value in candidate_state],
                'confidence': confidence,
                'response_margin': float(response_margin),
                'identity_similarity': safe_template_decision.identity_similarity,
                'normalized_center_jump': (
                    safe_template_decision.normalized_center_jump),
                'log_depth_change': safe_template_decision.log_depth_change,
                'checked': bool(safe_template_decision.checked),
                'eligible': bool(safe_template_decision.eligible),
                'dynamic_active': bool(safe_template_decision.dynamic_active),
                'stable_frames': int(safe_template_decision.stable_frames),
                'consecutive_state_rollbacks': int(
                    safe_template_decision.consecutive_state_rollbacks),
                'reasons': list(safe_template_decision.reasons),
            }
        elif self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if image.shape[-1] == 6:
            image_show = image[:,:,:3]
        else:
            image_show = image
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        output = {"target_bbox": self.state,
                  "best_score": conf_score}
        if safe_template_decision is not None:
            output['safe_template_decision'] = safe_template_decision
            output['applied_template_blend_weight'] = (
                applied_template_blend_weight)
            # Trace-only consumers may inspect these online quantities, but
            # they do not feed back into tracking and never contain GT.
            output['online_state_evidence'] = online_state_evidence
        return output

    def _drop_dynamic_template(self):
        """Atomically restore every online slot to the immutable template."""
        self.template_list = [self.static_template] * self.num_template
        self.template_anno_list = [self.static_template_anno]

    def _replace_dynamic_template(self, image, blend_weight=None):
        """Replace one slot, optionally interpolating with the static tensor."""
        z_patch_arr, resize_factor = sample_target(
            image, self.state, self.params.template_factor,
            output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and template.size(1) == 3:
            template = torch.cat((template, template), axis=1)

        prev_box_crop = transform_image_to_crop(
            torch.tensor(self.state), torch.tensor(self.state), resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True)
        template_anno = prev_box_crop.to(template.device).unsqueeze(0)

        if blend_weight is not None:
            try:
                blend_weight = float(blend_weight)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('Template blend weight must be finite') from error
            if (not math.isfinite(blend_weight) or
                    not 0.0 < blend_weight <= 1.0):
                raise ValueError('Template blend weight must lie in (0, 1]')
            if (template.shape != self.static_template.shape or
                    template_anno.shape != self.static_template_anno.shape):
                raise ValueError('Static/candidate template shapes differ')
            template = torch.lerp(
                self.static_template, template, blend_weight)
            template_anno = torch.lerp(
                self.static_template_anno, template_anno, blend_weight)

        self.template_list.append(template)
        if len(self.template_list) > self.num_template:
            self.template_list.pop(1)
        self.template_anno_list.append(template_anno)
        if len(self.template_anno_list) > self.num_template:
            self.template_anno_list.pop(1)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
        else:
            nlp_ids = clip.tokenize(nlp).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks

def get_tracker_class():
    return SUTRACK
