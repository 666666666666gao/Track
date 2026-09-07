from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path('/root/autodl-tmp/sttrack_m66_same_state_diagnostic_20260907')
PARENT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
assert json.loads((ROOT/'result.json').read_text())['status']=='complete_post_sealing_same_state_diagnostic'
case=next(c for c in json.loads((ROOT/'spec.json').read_text())['cases'] if c['sequence']=='glass03_indoor')
folder=Path(json.loads((PARENT/'training_spec.json').read_text())['dataset_root'])/case['sequence']
gtpath=folder/'groundtruth.txt';assert hashlib.sha256(gtpath.read_bytes()).hexdigest()==case['gt_sha256']
gt=np.loadtxt(gtpath,delimiter=',').reshape(-1,4)
rows={a:json.loads((PARENT/'recursive'/a/(case['sequence']+'.json')).read_text())['rows'] for a in ['control','null']}
trace=json.loads((ROOT/'null'/(case['sequence']+'.json')).read_text())
geo={x['frame']:x for x in trace['geometry']};probes={x['frame']:x for x in trace['probes']}
fig,axes=plt.subplots(3,3,figsize=(14,10))
frames=[0,600,640,650,656,657,658,660,674]
for ax,f in zip(axes.flat,frames):
    image=Image.open(folder/'color'/('%08d.jpg'%(f+1)))
    ax.imshow(image)
    valid=bool(np.isfinite(gt[f]).all() and (gt[f,2:]>0).all())
    boxes=[('Control',rows['control'][f]['bbox'],'cyan','-'),('Null',rows['null'][f]['bbox'],'magenta','-')]
    if valid:boxes.append(('GT',gt[f],'lime','-'))
    if f in geo:boxes.append(('Null search',geo[f]['search_rectangle'],'white','--'))
    if f in probes:
        h=probes[f]['heads']['unadapted_head_same_current_features']
        boxes.append(('Unadapted on Null state',h['dense_boxes'][h['selected_index']],'orange',':'))
    for label,b,color,style in boxes:
        ax.add_patch(Rectangle((b[0],b[1]),b[2],b[3],fill=False,edgecolor=color,linestyle=style,linewidth=1.5))
    ax.set_xlim(0,image.width);ax.set_ylim(image.height,0)
    ax.set_title('Frame %d%s'%(f,'' if valid else ' | GT invalid'),fontsize=11);ax.axis('off')
fig.suptitle('glass03_indoor | green: GT; cyan: Control; magenta: Null; orange: unadapted head on Null state\nWhite dashed: Null search crop. Zero-based frames; head probes do not alter tracking.',fontsize=12)
fig.tight_layout(rect=[0,0,1,.95]);out=ROOT/'glass03_frames.png';fig.savefig(out,dpi=140);plt.close(fig)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size)))
