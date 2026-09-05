#!/usr/bin/env python3
"""Check object-relative geometry and original cache hashes without a GPU."""
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT=Path('/root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905')
M44=Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
REPO=Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
sys.path.insert(0,str(REPO))
from lib.models.sttrack.lachtt_relative_geometry import relative_geometry,RelativeCandidateSetAssociation
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026);control=CandidateSetAssociation(True)
    torch.manual_seed(2026);relative=RelativeCandidateSetAssociation(True)
    assert all(torch.equal(v,relative.state_dict()[k]) for k,v in control.state_dict().items())
    assert sum(p.numel() for p in relative.parameters())==448739
    receipts=[json.loads((M44/f'shard{i}_receipt.json').read_text()) for i in [0,1]]
    pairs=0;records=[];max_error=0.
    for receipt in receipts:
        for item in receipt['sequences']:
            p=M44/'features'/(item['sequence']+'.pt');assert sha(p)==item['feature_sha256']
            data=torch.load(p,map_location='cpu');geometry=data['geometry'].double()
            assert (geometry[...,2:]>0).all() and torch.isfinite(geometry).all()
            choices=torch.arange(len(geometry))%10
            result=relative_geometry(geometry,choices)
            assert torch.isfinite(result).all()
            transformed=geometry.clone();transformed[...,:2]=transformed[...,:2]*1.7+.13;transformed[...,2:]*=1.7
            difference=float((result-relative_geometry(transformed,choices)).abs().max())
            assert difference<1e-10;max_error=max(max_error,difference)
            indexes=torch.arange(len(geometry))
            assert torch.equal(result[indexes,10+choices],torch.zeros(len(geometry),4,dtype=torch.float64))
            records.append(dict(sequence=data['sequence'],pairs=len(geometry),feature_sha256=item['feature_sha256'],
                                minimum_normalized_dimension=float(geometry[...,2:].min()),maximum_absolute_relative_value=float(result.abs().max())))
            pairs+=len(geometry)
    assert len(records)==85 and pairs==2101
    out=dict(status='PASS',gpu_used=False,gt_labels_read=False,parameters=448739,matched_initial_parameters=True,
             feature_files_verified=85,pairs=pairs,nonzero_previous_choices_checked=True,
             common_translation_scale_invariance_max_error=max_error,source_sha256=sha(Path(__file__)),sequences=records)
    (ROOT/'geometry_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='sequences'},indent=2))


if __name__=='__main__':main()
