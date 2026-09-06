"""GPU contract for t0 extraction; execute only after the visual base is fixed."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_sha(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(bytes.fromhex(tensor_sha(value)))
    return digest.hexdigest()


def rng_state():
    return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        torch=tensor_sha(torch.get_rng_state()), cuda=tensor_sha(torch.cuda.get_rng_state()))


def initial_tracker_state(tracker):
    assert tracker.frame_id == 0 and tracker.track_query_before is None
    assert tracker.box_mask_z is None and not tracker.network.training
    return dict(frame_id=tracker.frame_id, bbox=list(tracker.state),
        templates=[tensor_sha(value) for value in tracker.z_dict],
        template_alias=tracker.z_dict[0] is tracker.z_dict[1],
        template_patch=hashlib.sha256(tracker.z_patch_arr.tobytes()).hexdigest(),
        output_window=tensor_sha(tracker.output_window),
        preprocessing=[tensor_sha(tracker.preprocessor.mean),tensor_sha(tracker.preprocessor.std)],
        network=model_sha(tracker.network),
        module_training={name:module.training for name,module in tracker.network.named_modules()},
        random_state=rng_state())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert sha(__file__) == spec['checker_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == spec['cuda_visible_devices']
    code, output = Path(spec['code_root']), Path(spec['output'])
    assert not output.exists()
    for name,digest in spec['source_sha256'].items():
        assert sha(code/name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(spec['initialization_manifest']) == spec['initialization_manifest_sha256']
    assert sha(spec['extractor']) == spec['extractor_sha256']
    cases = json.loads(Path(spec['initialization_manifest']).read_text())
    assert len(cases) == 1
    case = cases[0]
    assert set(case) == {'sequence','split','frames','init_bbox'}
    assert case['sequence'] == 'chair01_indoor'
    assert case['split'] == 'fit' and case['frames'] >= 102
    assert spec['sequence'] == 'chair01_indoor' and spec['frames'] == 102
    sys.path.insert(0,str(code))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_local_spatial_observation import NativeReferenceBank
    from lib.train.dataset.depth_utils import get_rgbd_frame
    module_spec = importlib.util.spec_from_file_location('m57_t0_extractor',spec['extractor'])
    extractor = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(extractor)
    update_config_from_file(str(code/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    assert cfg.DATA.TEMPLATE.NUMBER == 2
    assert cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
    params = SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,
        template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    torch.set_num_threads(1)
    folder = Path(spec['dataset_root'])/case['sequence']
    image_hashes = {}

    def image_at(frame):
        rgb=folder/'color'/('%08d.jpg' % (frame+1))
        depth=folder/'depth'/('%08d.png' % (frame+1))
        image=get_rgbd_frame(str(rgb),str(depth),dtype='rgbcolormap',depth_clip=True)
        current={str(rgb):sha(rgb),str(depth):sha(depth)}
        for path,digest in current.items():
            if path in image_hashes:
                assert image_hashes[path] == digest
            image_hashes[path]=digest
        return image

    output.mkdir()
    phases = {}
    for phase in ['native','observation_t1','initial_t0']:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        torch.cuda.manual_seed(2026)
        tracker=STTrack(params)
        first=image_at(0)
        tracker.initialize(first,dict(init_bbox=list(case['init_bbox'])))
        bank=NativeReferenceBank(case['init_bbox'])
        initial_reference_sha=None
        t0_neutral=None
        if phase == 'initial_t0':
            templates=list(tracker.z_dict)
            before=initial_tracker_state(tracker)
            reference=extractor.extract_initial_instance_roi(tracker,first,case['init_bbox'])
            after=initial_tracker_state(tracker)
            assert before == after
            assert all(left is right for left,right in zip(templates,tracker.z_dict))
            assert tuple(reference.shape) == (2,16,768)
            assert not reference.requires_grad and torch.isfinite(reference).all()
            bank.initial=reference
            assert bank.dynamic is None and bank.encoded_dynamic is None and bank.previous is None
            initial_reference_sha=tensor_sha(reference)
            t0_neutral=dict(exact_state_and_random_state_equal=True,
                template_object_identity_preserved=True,network_state_sha256=before['network'],
                reference_shape=list(reference.shape),reference_dtype=str(reference.dtype))
        observed={}
        new_dynamic_reads=[]
        if phase != 'native':
            original_forward=tracker.network.forward

            def observing_forward(*values,**kwargs):
                kwargs['return_candidate_features']=True
                result=original_forward(*values,**kwargs)
                observed['features']=result[0]['candidate_features']
                if bank.encoded_dynamic is not None and bank.encoded_dynamic is not tracker.z_dict[1]:
                    new_dynamic_reads.append(tracker.frame_id)
                bank.before_decision(observed['features'],tracker.z_dict[1])
                return result

            # Install only after the t0 forward; t0 must not initialize dynamic references.
            tracker.network.forward=observing_forward
        rows=[dict(frame=0,bbox=list(tracker.state),score=1.,frame_id=0)]
        writes=[]
        for frame in range(1,102):
            image=image_at(frame)
            prior=list(tracker.state)
            prior_dynamic=tracker.z_dict[1]
            result=tracker.track(image)
            if tracker.z_dict[1] is not prior_dynamic:
                writes.append(frame)
            if phase != 'native':
                resize=256/math.ceil(math.sqrt(prior[2]*prior[3])*4.)
                bank.after_decision(observed['features'],prior,resize,tracker.state,tracker.z_dict[1])
                if phase == 'initial_t0':
                    assert tensor_sha(bank.initial) == initial_reference_sha
            rows.append(dict(frame=frame,bbox=[float(x) for x in result['target_bbox']],
                score=float(result['best_score']),frame_id=tracker.frame_id,
                query_sha256=[tensor_sha(value) for value in tracker.track_query_before],
                template_sha256=[tensor_sha(value) for value in tracker.z_dict],
                template_alias=tracker.z_dict[0] is tracker.z_dict[1]))
        if phase != 'native':
            del tracker.network.forward
            assert new_dynamic_reads == [frame+1 for frame in writes]
        path=output/(phase+'.json')
        path.write_text(json.dumps(dict(phase=phase,rows=rows),indent=2,allow_nan=False)+'\n')
        phases[phase]=dict(rows=rows,predictions_sha256=sha(path),default_template_write_frames=writes,
            new_dynamic_template_read_frames=new_dynamic_reads,
            t0_neutral=t0_neutral,initial_reference_sha256=initial_reference_sha)
        print(json.dumps(dict(phase=phase,frames=len(rows),template_writes=writes)),flush=True)
        del tracker
    assert phases['native']['rows'] == phases['observation_t1']['rows']
    assert phases['native']['rows'] == phases['initial_t0']['rows']
    for name,digest in spec['source_sha256'].items():
        assert sha(code/name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(spec['extractor']) == spec['extractor_sha256']
    result=dict(status='complete_measurement',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_sha256=sha(args.spec),checker_sha256=sha(__file__),checkpoint_sha256=spec['checkpoint_sha256'],
        extractor_sha256=spec['extractor_sha256'],sequence=case['sequence'],frames_per_phase=102,
        exact_native_prefix_parity=True,subsequent_gt_opened=False,formal_features_saved=False,
        template_write_preservation_observed=bool(phases['initial_t0']['default_template_write_frames']),
        phases={name:{k:v for k,v in row.items() if k!='rows'} for name,row in phases.items()},
        observed_image_sha256=image_hashes,
        scope='One fitting prefix; t0/observation state contract only. No language semantics or tracking performance claim. Zero writes means template-write coverage remains incomplete.')
    (output/'contract.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['phases','observed_image_sha256']}),flush=True)


if __name__ == '__main__':
    main()
