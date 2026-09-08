from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil, tarfile

B=Path('/root/autodl-tmp'); R=B/'sttrack_m80_block_text_dropout_20260908'; M=B/'sttrack_m81_multistart_20260908'
O=R/'development_completed_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='e26a1ab6c3e8e72b4c6feb0dcbaacdb0facde3ae8d5469b4d370f6df59a2a823'
assert b'\n## 5.165 ' not in old and not O.exists()
result=read(R/'result.json'); audit=read(R/'saved_development_audit.json')
assert result['status']=='completed_M80_development_and_content'
assert audit['result_sha256']==sha(R/'result.json')=='d923839d8a8ecab302ee1a1313fe0c8cea92282eb279fe3c0b045f2bd352b6bc'
assert audit['trajectories']==66 and audit['positions']==99390
assert audit['primary_pass_count']==4 and audit['content_pass_count']==1 and audit['dropout_pass_count']==0
for name in ['training_category','eval_category','eval_empty','eval_swapped','analysis','controller']:
    assert (R/(name+'.exit')).read_text().strip()=='0'
launch=read(M/'launch.json')
assert launch['training_started'] and launch['dependency_result_sha256']==sha(R/'result.json')
workers=[]
for pid,arm in [(130723,'category'),(130724,'empty')]:
    p=Path('/proc')/str(pid)
    assert p.exists() and (p/'cwd').resolve()==M
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'train_causal.py --arm '+arm in cmd
    workers.append(dict(pid=pid,arm=arm,command=cmd))
assert not (M/'controller.exit').exists() and not (M/'result.json').exists()
status=dict(observed_utc=datetime.now(timezone.utc).isoformat(),M80_status=result['status'],M81_status='both_training_processes_verified',
            M81_workers=workers,M81_training_spec_sha256=sha(M/'training_spec.json'),free_disk_bytes=shutil.disk_usage(R).free,
            independent_model_review_pass=False)
table=[]
for key,label in [('native','原生STTrack'),('M78_category','M78 Category，无文本dropout'),('M78_empty_trained','M78独立Empty训练'),('category','M80 Category'),('empty','M80同权重Empty'),('swapped','M80同权重Swapped')]:
    a=result['aggregates'][key]
    table.append('| %s | %.6f | %.6f | %d | %d |'%(label,a['mean_iou'],a['macro_sequence_mean_iou'],a['low_iou_frames'],a['failure_episodes']))
note='''

## 5.165 M80完整开发与内容对照失败，M81固定多起点配对已实际接续

M80于2026-09-08 12:09:39 UTC完成三组开发与内容评测及分析，training、Category、Empty、Swapped、analysis和controller退出码均为0。三组使用同一个固定最终head，每组22条完整轨迹、33130个位置、28897个有效非初始化GT位置。当前只涉及反复使用的DepthTrack Train开发22，不是DepthTrack Test、CDTB或VOT指标。

| 条件 | 按有效帧平均IoU | 序列等权平均IoU | IoU≤0.1帧 | H10段 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

冻结条件：主条件4/10、同权重内容条件1/8、相对M78的dropout增量条件0/4，三个门均未通过。M80类别条件相对M78按帧IoU下降7.236887个百分点，低重叠帧增加2475，H10增加8；相对原生按帧IoU下降0.558655个百分点，低重叠帧增加301，H10增加6。不得为了本结果事后修改门槛。

类别内容未优于同权重空词或替换类别的均值、宏平均与低重叠帧，只有相对空词H10更少。原生及M78独立Empty训练原本零H10的container01_indoor，在M80 Category新增1段H10；这也是两个成功轨迹保护条件均失败的原因。glass03_indoor在M80 Category/Empty/Swapped的平均IoU分别为0.271884/0.580104/0.916563，H10分别为4/2/0；替换词表现好不能倒推为语义正确，也不能据此解释全部轨迹因果。完整逐序列结果一并发布。

这轮只否定§5.160冻结的20%按32帧片段训练文本dropout实现，不能扩大为所有文本dropout或语言方法均无效。训练器与原M78保留相同Raw竞争、seed2027、130次首帧初始化、186694次跟踪调用和5798次优化；dropout并未增加初始化覆盖。Empty仍保留已训练adapter，不能称为原生回退。M80不进入VOT低22或正式三数据集，也不根据替换类别结果选择部署文字。

独立CPU标量核验在全部66个轨迹文件和3份receipt封存后，先检查全部SHA、帧索引、初始框、预测框有限性、置信度和输入绑定，再读取开发GT，重新计算99390个位置对应的逐序列IoU、无效标注中断的H10、聚合及冻结条件；与原汇总一致。它没有重新推理或训练。核验报告与脚本、原始汇总、receipt及逐序列CSV发布于diagnostics/m80_block_text_dropout/development_completed/。确定性核验不等于独立模型审阅通过；gpt-6-astra/max审阅仍无完成态PASS。

M80结果SHA256为`d923839d8a8ecab302ee1a1313fe0c8cea92282eb279fe3c0b045f2bd352b6bc`；最终head保持`d853f51ff5820e563d895af220cbe5659a60e3c2d20d5a36c027610fdd36c908`。三个receipt及训练/评测协议绑定见result.json。正式三数据集仍保留原生STTrack已发布结果，未产生新模型完整指标。

M81调度器于2026-09-08 12:12:40 UTC（北京时间20:12:40）在M80正常结束、两个GPU各1MiB且磁盘足够后，按§5.162冻结队列启动。已通过/proc命令行、cwd及GPU进程核实Category PID130723、Empty PID130724，两组实际训练而非仅排队；GPU各约2454MiB。M81是否启动与M80指标通过无关，方案在M80完成前已冻结。

M81仍为固定seed2027、426个合法训练episode、186694次跟踪调用、5798次优化；额外296次GT初始化及其前向成本已明确披露。它保留M78 Raw损失，不混入M80 dropout、反向、跨初始化一致性、新记忆或模板规则。首帧与多起点开发协议、九组输出和32项判定保持原方案，全部封存后再评价。当前只有启动证据，没有M81完成态指标；预计训练约4小时，后续开发与内容评测另需约2小时，以实际日志为准。CPU观察器将按冻结时间安排核验训练保存产物。两套Qwen及基线/参照权重保留，未为此次发布删除权重。
'''
O.mkdir(); F=O/'files'; F.mkdir()
sources=[(R/'result.json','result.json'),(R/'saved_development_audit.json','saved_development_audit.json'),
         (B/'audit_m80_development_complete_20260908.py','audit_m80_development_complete_20260908.py'),
         (M/'launch.json','M81_launch.json'),(Path(__file__),Path(__file__).name)]
sources += [(R/('receipt_'+arm+'.json'),'receipt_'+arm+'.json') for arm in ['category','empty','swapped']]
for src,name in sources:shutil.copyfile(src,F/name)
(F/'verified_status.json').write_text(json.dumps(status,indent=2)+'\n')
(F/'handoff_append.md').write_text(note)
with (F/'per_sequence.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['condition','sequence','valid_frames','mean_iou','low_iou_frames','H10_episodes','invalid_gt_frames'])
    for arm,seqs in result['per_sequence'].items():
        for seq,a in seqs.items():writer.writerow([arm,seq,a['valid_frames'],a['mean_iou'],a['low_iou_frames'],a['failure_episodes'],a['invalid_gt_frames']])
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())]
(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(F.iterdir()):tar.add(p,arcname=p.name)
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
