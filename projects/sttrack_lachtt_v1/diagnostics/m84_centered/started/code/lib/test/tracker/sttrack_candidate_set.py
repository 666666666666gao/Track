"""Causal candidate-set association on the ordinary STTrack recursive path."""
import torch
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_candidate_set_observation import observe_candidate_set
from lib.test.tracker.sttrack_local_spatial_observation import NativeReferenceBank
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation,select_candidate
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class STTrackCandidateSet(STTrack):
    def __init__(self,params,association_checkpoint):
        super().__init__(params)
        assert not params.save_all_boxes and params.debug==0
        checkpoint=torch.load(association_checkpoint,map_location='cpu')
        self.association=CandidateSetAssociation(checkpoint['variant']=='geometry').cuda().eval()
        self.association.load_state_dict(checkpoint['model'],strict=True)

    def initialize(self,image,info):
        super().initialize(image,info)
        self.reference_bank=NativeReferenceBank(info['init_bbox'])
        self.previous_set=None;self.previous_choice=0

    def track(self,image,info=None):
        height,width=image.shape[:2];self.frame_id+=1;prior=list(self.state);dynamic=self.z_dict[1]
        patch,resize,_=sample_target(image,self.state,self.params.search_factor,output_sz=self.params.search_size)
        search=self.preprocessor.process(patch)
        with torch.no_grad():
            output=self.network.forward(template=self.z_dict,search=[search],ce_template_mask=self.box_mask_z,
                track_query_before=self.track_query_before,keep_rate=self.keep_rate,return_candidate_features=True)[0]
            self.track_query_before=output['track_query_before'];features=output['candidate_features']
            response=self.output_window*output['score_map']
            current=observe_candidate_set(output,self.output_window,prior,resize,image.shape)
            refs=self.reference_bank.before_decision(features,dynamic)
            chosen=0;none=False
            if self.previous_set is not None:
                geometry=torch.cat([current['geometry'],self.previous_set['geometry']]).cuda()[None]
                scores=torch.cat([current['scores'],self.previous_set['scores']]).cuda()[None]
                logits,_=self.association(current['rois'][None].half().float(),self.previous_set['rois'][None].half().float(),
                    refs[None,:2].half().float(),geometry,scores,torch.tensor([self.previous_choice],device='cuda'))
                chosen=int(select_candidate(logits)[0]);none=int(logits.argmax(1)[0])==10
            if chosen==0:
                boxes=self.network.box_head.cal_bbox(response,output['size_map'],output['offset_map'])
                confidence=response.flatten(1).max(1,keepdim=True).values
            else:
                item=current['candidates'][chosen];row,col=item['grid_row'],item['grid_column'];offset=output['offset_map'];size=output['size_map']
                boxes=torch.stack([(col+offset[0,0,row,col])/self.feat_sz,(row+offset[0,1,row,col])/self.feat_sz,
                    size[0,0,row,col],size[0,1,row,col]]).reshape(1,4)
                confidence=response[0,0,row,col].reshape(1,1)
            box=(boxes.view(-1,4).mean(0)*self.params.search_size/resize).tolist()
            self.state=clip_box(self.map_box_back(box,resize),height,width,margin=10)
            if self.frame_id%self.update_intervals==0 and confidence>self.update_threshold:
                patch,_,_=sample_target(image,self.state,self.params.template_factor,output_sz=self.params.template_size)
                self.z_patch_arr=patch;self.z_dict.append(self.preprocessor.process(patch))
                if len(self.z_dict)>self.num_template:self.z_dict.pop(1)
            self.reference_bank.after_decision(features,prior,resize,self.state,self.z_dict[1])
            self.previous_set=current;self.previous_choice=chosen
        return dict(target_bbox=self.state,best_score=confidence.cpu().numpy()[0][0],association_candidate=chosen,association_none=none)
