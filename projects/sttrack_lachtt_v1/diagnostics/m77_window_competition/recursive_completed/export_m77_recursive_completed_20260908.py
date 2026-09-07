from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, io, json, shutil, tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m77_window_competition_20260907';C=R/'content_followup';O=R/'recursive_completed_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists()
assert sha(R/'recursive_result.json')=='d3d33dac98c510f76711949f12b01d497157d25f2085fac9c83e0937018ff9c1'
assert sha(C/'completed_training_audit.json')=='7cd9538ba22d4454f7e4742b74e93ebbad6ebe2bff1a2b50251659ac2cc23015'
d=read(R/'recursive_result.json');a=read(C/'completed_training_audit.json');prefix=read(C/'prefix/receipt.json')
assert d['status']=='complete_recursive_development' and not d['primary_pass']
assert len(d['gates'])==15 and sum(d['gates'].values())==2 and a['gates']==d['gates']
assert a['result_sha256']==sha(R/'recursive_result.json') and not a['development_gates_pass']
assert prefix['exact_category_prefix_parity'] and prefix['total_frames']==306
assert prefix['head_sha256']==sha(R/'training/category/final.pth')
for arm in ['empty','category']:
 t=read(R/'training'/arm/'result.json')
 assert t['sequences']==130 and t['total_track_calls']==186694 and t['optimizer_steps']==5798
 assert sha(R/'training'/arm/'final.pth')==t['final_checkpoint_sha256']==a['training'][arm]['head_sha256']
for name in ['training_empty.exit','training_category.exit','empty_recursive.exit','category_recursive.exit','recursive_analysis.exit','controller.exit']:
 assert (R/name).read_text().strip()=='0'
assert (C/'training_audit.exit').read_text().strip()=='0' and (C/'prefix.exit').read_text().strip()=='0'
processes=[]
for name,pid,ticks in [('empty',503842,'4583354567'),('swapped',503843,'4583354567'),('content_controller',496607,'4581958397'),('evaluation_controller',497894,'4582162682')]:
 p=Path('/proc')/str(pid);st=(p/'stat').read_text().split()
 assert st[21]==ticks and st[2]!='Z'
 processes.append(dict(name=name,pid=pid,start_ticks=st[21],state=st[2],cwd=str((p/'cwd').resolve())))
now=datetime.now(timezone.utc).isoformat();O.mkdir()
snapshot=dict(observed_utc=now,processes=processes,content_stage=read(C/'stage.json'),content_result_complete=(C/'result.json').exists(),free_bytes=shutil.disk_usage(R).free)
(O/'process_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
files={name:R/name for name in ['recursive_result.json','category_recursive_receipt.json','empty_recursive_receipt.json','training_empty.exit','training_category.exit','empty_recursive.exit','category_recursive.exit','recursive_analysis.exit','controller.exit']}
files.update({name:C/name for name in ['completed_training_audit.json','activation.json','training_audit.exit','prefix.exit']})
files['prefix_receipt.json']=C/'prefix/receipt.json'
files['process_snapshot.json']=O/'process_snapshot.json'
files['export_m77_recursive_completed_20260908.py']=Path(__file__)
for arm in ['empty','category']:
 files[arm+'__training_result.json']=R/'training'/arm/'result.json'
 files[arm+'__sequence_log.jsonl']=R/'training'/arm/'sequence_log.jsonl'
out=io.StringIO(newline='');writer=csv.writer(out);writer.writerow(['sequence','arm','mean_iou','low_iou_frames','H10','valid_frames'])
for seq in d['per_sequence']['category']:
 for arm in ['native','empty','category']:
  v=d['per_sequence'][arm][seq];writer.writerow([seq,arm,format(v['mean_iou'],'.12f'),v['low_iou_frames'],v['failure_episodes'],v['valid_frames']])
(O/'per_sequence.csv').write_bytes(out.getvalue().encode('utf-8'));files['per_sequence.csv']=O/'per_sequence.csv'
tracker='''# M77 completed training and development audit — 2026-09-08

This completion snapshot supersedes RUNNING/QUEUED entries in the preserved launch snapshots. Seed2027 only; no additional seed or checkpoint selection.

| Stage | Status | Evidence |
|---|---|---|
| Category / Empty fit | COMPLETE, both exit0 | 130 sequences,186694 calls,5798 optimizer steps each; final checkpoint hashes in training results |
| Train development22 | COMPLETE, both exit0 | 33130 frames per arm; sealed receipts and independently recomputed metrics |
| Checkpoint / scalar audit | COMPLETE, exit0 | completed_training_audit.json; this is not independent model review |
| Frozen development conditions | FAIL,2/15 | Category head is not promoted |
| Same-head Category prefix | COMPLETE, exact parity | Three sequences x102 frames; no optimization |
| Same-head Empty / Swapped | RUNNING at process_snapshot timestamp | Both actual process identities verified |
| VOT low22 / full datasets | NOT OPEN | Failed development conditions remain failed regardless of content outcome |
'''
(O/'EXPERIMENT_TRACKER.md').write_bytes(tracker.encode('utf-8'));files['EXPERIMENT_TRACKER.md']=O/'EXPERIMENT_TRACKER.md'
agg=d['aggregates'];m73=d['parent_M73_aggregates']['category']
rows=[]
for label,v in [('原生STTrack',agg['native']),('M73类别文本（历史参照）',m73),('M77空文本训练对照',agg['empty']),('M77类别文本',agg['category'])]:
 rows.append('| %s | %.6f | %.6f | %d | %d |'%(label,v['mean_iou'],v['macro_sequence_mean_iou'],v['low_iou_frames'],v['failure_episodes']))
table='\n'.join(rows)
c,e=agg['category'],agg['empty'];delta=(c['mean_iou']-e['mean_iou'])*100;parent_delta=(c['mean_iou']-m73['mean_iou'])*100
harms=[v for v in a['strict_category_H10_reference_correct_every_frame'] if v['reference']=='native']
assert len(harms)==4 and sum(v['frames'] for v in harms)==250
note=f'''

## 5.149 M77单seed配对训练与开发递归完成：竞争损失未通过，内容诊断继续

本节为完成态结果，覆盖130条DepthTrack Train拟合序列和既有Train开发22；不是VOT低22、官方DepthTrack Test或CDTB结果。两组仅seed2027，不补其他seed。M77架构、训练计划及预注册条件见§5.144—5.147，本节不重复完整架构图：原生STTrack底座冻结，只训练既定结构的289154参数语义适配器；增加Hann最终空间竞争损失，推理仍用原Hann选峰及原生模板更新。

两组均完成130条序列、186694次真实预测裁剪跟踪调用、5798次优化；使用同一初始化张量和物理序列顺序，递归状态由各自预测形成。类别组于UTC18:45:30、空文本组于18:46:57完成，训练退出均为0。初始化后没有GT重置，状态detach，无跨时间/crop反向传播。最终权重及训练覆盖通过既定检查；基线冻结证据是训练末端张量哈希断言，并非再次完整重放训练。

两组完整开发22各33130帧（33108次跟踪调用），有效非初始化帧28897；输出全部封存后才读取后续GT计算指标。以下均值是0—1尺度，H10是连续IoU≤0.1至少10帧的代理失败段，不等同于VOT ROB。

| 配置 | 按有效帧平均IoU | 序列等权平均IoU | IoU≤0.1帧 | H10段 |
|---|---:|---:|---:|---:|
{table}

M77类别相对同预算空文本对照的按帧均值变化为{delta:.4f}个百分点；相对M73类别为{parent_delta:.4f}个百分点。类别组只通过15项冻结条件中的2项：均值超过原生至少0.001、低重叠帧不多于原生。其余13项未通过，包含宏平均、持续失败和成功序列保护。不能将原生到M77类别的微小均值增加写成语言改进，更不能将M77空文本的收益转给类别组。

类别组损害原生无H10的mobilephone02；损害本次空文本成功的bag05、glass03、mobilephone02；历史M65 Control保护集合中bag05、glass03也未保住。独立标量核对识别出4个严格持续损害区间：bag05[697,713)、cup10[2070,2090)、mobilephone01[940,950)、mobilephone02[497,701)，合计250帧；每个区间内独立原生和本次空文本路径逐帧IoU均≥0.5。两个参照各产生一条审计记录，不能将8条记录相加误报500个不同受损帧。全22条逐序列数据随per_sequence.csv发布，不仅挑选正例。

这证明本次“额外监督最终窗口竞争”的训练干预没有改善类别条件路径的完整递归效果；不证明Hann应删除，也不证明全部文字无效。训练中各组访问状态不同，两个最终权重的差异不能直接归因到某个词。尤其空文本组相对其M73历史对照提高，而类别组下降，须保留这种不对称结果，不将其包装为稳定语义竞争机制。

后续继续已经预先规定的固定类别最终权重内容诊断：类别前缀3×102帧与原类别递归逐值完全一致，exit0；同权重Empty和Swapped全22于UTC19:20:04启动，各33108次调用，无优化、无新caption、无新增seed。完成后比较具体词义的增量与伤害，再决定是否研究监督冲突或语义条件分支。此诊断无论结果如何都不撤销本次失败；条件评测队列不会因内容正例越过开发门，M77本版不进入VOT低22或三数据集正式评测。没有新的正式指标，目标仍未完成。

已执行完成态检查包括：最终/最新checkpoint及优化状态、训练序列和调用覆盖、同初始化/预算、160份集成源码及文字bank SHA、竞争监督计数、两套标量实现重算、全部15项条件逐项重算。检查通过意味着证据自洽，不是方法晋升，也不是gpt-6-astra独立模型审阅PASS。

类别最终权重SHA256：`{sha(R/'training/category/final.pth')}`；空文本最终权重：`{sha(R/'training/empty/final.pth')}`。
递归结果SHA256：`{sha(R/'recursive_result.json')}`；完成态审计：`{sha(C/'completed_training_audit.json')}`。

完成态源码/记录/CSV发布至`projects/sttrack_lachtt_v1/diagnostics/m77_window_competition/recursive_completed/`；原始预测、完整训练trace及权重保留服务器，未复制大文件到Git。当前状态以本节及该目录EXPERIMENT_TRACKER.md为准，旧启动快照按原SHA保留。磁盘本次实测{snapshot['free_bytes']}字节可用，未删除文件，两份Qwen保留。
'''
(O/'handoff_append.md').write_bytes(note.encode('utf-8'));files['handoff_append.md']=O/'handoff_append.md'
published=O/'published';published.mkdir()
for name,p in files.items():(published/name).write_bytes(p.read_bytes())
(published/'manifest.json').write_text(json.dumps([dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(files.items())],indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(str(archive),'w:gz') as tar:
 for p in sorted(published.iterdir()):tar.add(str(p),arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
assert sha(master)=='f75324bc984deda63168541c0f4cb249009181658be76f28523ca64cbce18985'
old=master.read_bytes();assert b'5.149 ' not in old;master.write_bytes(old+note.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(files))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
