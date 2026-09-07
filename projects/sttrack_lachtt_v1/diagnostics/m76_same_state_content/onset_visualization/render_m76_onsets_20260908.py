"""Render the four already-sealed M76 onsets; no tracker or model is executed."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json,math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

B=Path('/root/autodl-tmp');R=B/'sttrack_m76_same_state_content_20260907'
P=B/'sttrack_m73_paired_lexical_replication_20260907/seed2027'
O=R/'onset_visualization_20260908';O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert sha(R/'result.json')=='a8150272c28093b9be9e444c27c0d5e8a50c2e13b81ca41ba0509562ed55d9fd'
assert sha(R/'completed_evidence_audit.json')=='3297d636109949ca08325d6d48c9ea98b7686c4211fb18480d0cdae16eb39239'
spec,receipt,result=read(R/'spec.json'),read(R/'receipt.json'),read(R/'result.json')
source=B/'m76_same_state_content_20260907.py';assert sha(source)==spec['source_sha256']
loader=importlib.util.spec_from_file_location('sealed_m76_overlap',str(source));helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
training=read(P/'training_spec.json');case_map={v['sequence']:v for v in spec['cases']}
sealed={v['sequence']:v for v in receipt['sequences']};panels=[];rows=[]
for onset in result['onsets']:
    seq=onset['sequence'];frame=onset['H10_start'];path=R/'replay'/(seq+'.json')
    assert sha(path)==sealed[seq]['sha256'];data=read(path)
    probe=next(v for v in data['probes'] if v['frame']==frame)
    geometry=next(v for v in data['geometry'] if v['frame']==frame)
    folder=Path(training['dataset_root'])/seq;gt_path=folder/'groundtruth.txt'
    assert sha(gt_path)==case_map[seq]['gt_sha256']
    gt=np.loadtxt(gt_path,delimiter=',').reshape(-1,4)[frame]
    assert gt.tolist()==onset['probe']['GT'] and np.isfinite(gt).all() and (gt[2:]>0).all()
    image=folder/'color'/('%08d.jpg'%(frame+1));assert image.is_file()
    for name,head in probe['heads'].items():
        iou=helper.many_iou(head['dense_boxes'],gt);h=head['selected_index'];r=head['raw_selected_index']
        expected=onset['probe']['heads'][name]
        assert abs(iou[h]-expected['selected_iou'])<1e-10
        assert abs(iou[r]-expected['raw_selected_iou'])<1e-10
        assert abs(max(iou)-expected['dense_best_iou'])<1e-10
        assert h==int(np.argmax(head['hann_scores'])) and r==int(np.argmax(head['raw_scores']))
        rows.append(dict(sequence=seq,frame_zero_based=frame,head=name,hann_selected_iou=float(iou[h]),raw_selected_iou=float(iou[r]),
            dense_best_iou=float(max(iou)),hann_index=h,raw_index=r,raw_score_at_hann_peak=head['raw_scores'][h],
            raw_score_at_raw_peak=head['raw_scores'][r],hann_score_at_hann_peak=head['hann_scores'][h],hann_score_at_raw_peak=head['hann_scores'][r]))
    head=probe['heads']['category'];h=head['selected_index'];r=head['raw_selected_index']
    assert np.max(np.abs(np.asarray(head['dense_boxes'][h])-geometry['bbox']))<1e-4
    panels.append(dict(sequence=seq,frame_zero_based=frame,image=str(image),image_sha256=sha(image),GT=gt.tolist(),
        search_rectangle=geometry['search_rectangle'],previous_bbox=geometry['previous_bbox'],hann_bbox=head['dense_boxes'][h],raw_bbox=head['dense_boxes'][r],
        hann_index=h,raw_index=r,raw_scores=head['raw_scores'],hann_scores=head['hann_scores'],
        statistics=onset['probe']['heads']['category'],replay_sha256=sha(path)))

assert len(panels)==4 and len(rows)==16
colors={'GT':'#00df8d','Hann':'#ff4edb','Raw':'#ffbb38','Search':'white'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(4,4,figsize=(20,15),gridspec_kw={'width_ratios':[1.55,1,1,1]})
fig.subplots_adjust(left=.025,right=.975,bottom=.095,top=.90,wspace=.17,hspace=.35)
fig.suptitle('M76: the same Category state, before and after the Hann window',fontsize=21,y=.985,fontweight='bold')
fig.text(.5,.953,'Four preselected Train development onsets; original images and sealed responses. No M77 results, new forward passes or recursive recovery claims.',ha='center',fontsize=11)
legend=[Line2D([0],[0],color=colors['GT'],lw=2.4,label='GT'),Line2D([0],[0],color=colors['Hann'],lw=2.4,label='Hann top1 (actual output)'),
        Line2D([0],[0],color=colors['Raw'],lw=2.4,ls='--',label='Raw top1 (one-frame counterfactual)'),Line2D([0],[0],color='#777777',lw=2,ls=':',label='Search region')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.944),ncol=4,frameon=False)
def box(ax,b,color,style='-',scale=1.,offset=(0,0)):
    ax.add_patch(Rectangle(((b[0]-offset[0])*scale,(b[1]-offset[1])*scale),b[2]*scale,b[3]*scale,
        fill=False,edgecolor=color,linewidth=2,linestyle=style))
for n,panel in enumerate(panels):
    image=Image.open(panel['image']).convert('RGB');image.load();a,b,c,d=axes[n]
    a.imshow(image);a.set_xlim(0,image.width);a.set_ylim(image.height,0)
    for key,col,style in [('GT','GT','-'),('hann_bbox','Hann','-'),('raw_bbox','Raw','--'),('search_rectangle','Search',':')]:box(a,panel[key],colors[col],style)
    stats=panel['statistics'];a.set_title('%s | frame %d (zero-based)'%(panel['sequence'],panel['frame_zero_based']),fontsize=12,fontweight='bold')
    x,y,w,h=panel['search_rectangle'];assert w==h
    crop=image.crop((x,y,x+w,y+h)).resize((256,256),Image.BILINEAR)
    b.imshow(crop);b.set_xlim(0,256);b.set_ylim(256,0)
    for key,col,style in [('GT','GT','-'),('hann_bbox','Hann','-'),('raw_bbox','Raw','--')]:box(b,panel[key],colors[col],style,256/w,(x,y))
    b.set_title('Search RGB (display resize)\nDense best IoU %.3f'%stats['dense_best_iou'])
    for axis,key,title in [(c,'raw_scores','Raw response'),(d,'hann_scores','Hann x raw response')]:
        values=np.asarray(panel[key]).reshape(16,16);assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
        im=axis.imshow(values,vmin=0,vmax=1,cmap='magma',interpolation='nearest',origin='upper')
        for name,index,marker in [('Hann',panel['hann_index'],'o'),('Raw',panel['raw_index'],'x')]:
            row,col=divmod(index,16);axis.scatter([col],[row],marker=marker,s=110,facecolors='none' if marker=='o' else colors[name],edgecolors=colors[name],linewidths=2)
        axis.set_xticks([0,5,10,15]);axis.set_yticks([0,5,10,15]);axis.tick_params(labelsize=8)
        iou=stats['raw_selected_iou'] if key=='raw_scores' else stats['selected_iou']
        axis.set_title(title+'\nTop1 decoded IoU %.3f'%iou)
    for axis in [a,b]:axis.set_xticks([]);axis.set_yticks([])
    for axis in [a,b,c,d]:
        for spine in axis.spines.values():spine.set_color('#bbbbbb')
bar=fig.colorbar(im,cax=fig.add_axes([.70,.045,.24,.012]),orientation='horizontal');bar.set_label('Actual score scale shared by every response panel (0 to 1)',fontsize=9)
fig.text(.025,.053,'book06 / cup08: no correct decoded dense box (IoU >=0.5) at this onset.\ncup10 / egg: raw top1 is correct here, but windowed top1 is wrong. This does not justify removing Hann globally.',fontsize=11,va='center')
png=O/'M76_same_state_onsets.png';pdf=O/'M76_same_state_onsets.pdf'
fig.savefig(png,dpi=180,facecolor='white');fig.savefig(pdf,facecolor='white');plt.close(fig)
with (O/'onset_head_scores.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(O/'figure_data.json').write_text(json.dumps(dict(panels=panels,score_display='vmin=0 vmax=1; no per-panel or metric normalization'),indent=2)+'\n')
Image.open(png).verify()
report=dict(status='rendered_and_numeric_sources_verified',observed_utc=datetime.now(timezone.utc).isoformat(),renderer_sha256=sha(__file__),
    M76_result_sha256=sha(R/'result.json'),M76_audit_sha256=sha(R/'completed_evidence_audit.json'),
    source_images={p['image']:p['image_sha256'] for p in panels},events=4,head_score_rows=16,frame_numbering='zero-based; image filename is frame+1',
    new_tracking_calls=0,new_optimizer_steps=0,new_seeds=[],public_evaluation_started=False,
    scope='Visualization of four already-selected historical M76 Train development onsets, not an unbiased failure survey, M77 result or recursive rescue.',
    output_sha256={p.name:sha(p) for p in [png,pdf,O/'onset_head_scores.csv',O/'figure_data.json']})
(O/'render_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
