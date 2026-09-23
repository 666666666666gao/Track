"""Short real GPU witness on the migrated host, using saved Train prefixes."""
from pathlib import Path
import sys
import torch
import prepare_full as common

sys.path.insert(0,str(common.INTERFACE))
from semantic_runtime import checked_plan,make_tracker,text_bank

def check(name):
    common.checked(name)
    root=common.ROOT/name/'preflight'
    plan,bundle=checked_plan(root/'category_plan.json')
    tracker=make_tracker(bundle);bank=text_bank(plan,bundle)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    case=common.read(root/'cases.json')[0]
    expected=common.read(root/'category_direct.json')[0]['rows']
    folder=Path(plan['dataset_root'])/case['sequence'];errors=[]
    for frame in range(16):
        stem='%08d'%(frame+1);rgb=folder/'color'/(stem+'.jpg');depth=folder/'depth'/(stem+'.png')
        image=get_rgbd_frame(str(rgb),str(depth),dtype='rgbcolormap',depth_clip=True)
        if frame==0:tracker.initialize(image,bank.info(rgb,case['init_bbox']))
        else:
            out=tracker.track(image);ref=expected[frame]
            error=max(abs(float(a)-float(b)) for a,b in zip(out['target_bbox'],ref['bbox']))
            score_error=abs(float(out['best_score'])-ref['score'])
            assert error<=1e-5 and score_error<=1e-6,(name,frame,error,score_error)
            errors.append(dict(frame=frame,box_error=error,score_error=score_error))
    common.write(common.ROOT/(name+'_migration_preflight.json'),dict(status='pass',model=name,frames=16,
        comparisons=errors,bundle_sha256=common.sha(common.ROOT/name/'bundle.json'),
        checkpoint_sha256=common.sha(bundle['adapter_checkpoint']),base_checkpoint_sha256=common.sha(bundle['base_checkpoint']),
        source_sha256=common.sha(__file__),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        new_training_steps=0,formal_metric=False,subsequent_gt_opened=False))

if __name__=='__main__':
    import subprocess
    if len(sys.argv)==2:check(sys.argv[1])
    else:
        # Separate processes avoid sharing imported modules across model trees.
        for name in ['M67','M82']:subprocess.run([sys.executable,__file__,name],check=True)
        print('Both migrated final models reproduce saved Train prefixes.',flush=True)
