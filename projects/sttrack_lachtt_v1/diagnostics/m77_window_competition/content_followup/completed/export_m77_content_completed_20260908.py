from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,shutil,tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m77_window_competition_20260907';C=R/'content_followup';Q=C/'analysis_recovery_20260908';E=R/'candidate_evaluation/deferred_execution';O=C/'content_completed_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists() and not (C/'result.json').exists()
assert sha(Q/'result.json')=='12d4d5792fe171f10bba6201f5285d8a0f65910ce13512dfca7b9e2d9c0a3b08'
d=read(Q/'result.json');prep=read(Q/'preparation.json');audit=read(C/'completed_training_audit.json')
for p,h in prep['sealed_inputs'].items():assert sha(p)==h
assert (Q/'analysis.exit').read_text().strip()=='0'
assert (C/'empty.exit').read_text().strip()==(C/'swapped.exit').read_text().strip()=='0'
assert (C/'content_analysis.exit').read_text().strip()==(C/'controller.exit').read_text().strip()==(E/'controller.exit').read_text().strip()=='1'
assert d['independent_scalar_recomputation'] and d['head_sha256']==sha(R/'training/category/final.pth')
assert not d['development_gate_pass'] and not d['descriptive_criteria_pass']
assert sum(v for group in d['descriptive_criteria'].values() for v in group.values())==0
assert not (R/'candidate_evaluation/bundle').exists() and not (R/'candidate_evaluation/full_evaluation').exists()
for pid in [503842,503843,496607,497894]:assert not (Path('/proc')/str(pid)).exists()
O.mkdir()
decision=dict(status='M77_content_analysis_recovered_not_promoted',observed_utc=datetime.now(timezone.utc).isoformat(),result_sha256=sha(Q/'result.json'),original_analysis_exit=1,recovered_analysis_exit=0,original_content_controller_exit=1,original_evaluation_controller_exit=1,development_conditions_passed=sum(audit['gates'].values()),development_conditions_total=15,content_conditions_passed=0,content_conditions_total=8,public_evaluation_started=False,restart_public_evaluation=False,new_tracking_calls=0,new_optimizer_steps=0,seed=2027,additional_seeds=[],original_logs_and_predictions_unchanged=True,goal_achieved=False,free_bytes=shutil.disk_usage(R).free)
(O/'resolution.json').write_text(json.dumps(decision,indent=2)+'\n')
files={'result.json':Q/'result.json','recovery_preparation.json':Q/'preparation.json','analysis_function.diff':Q/'analysis_function.diff','recovery_analysis.exit':Q/'analysis.exit','recovery_analysis.txt':Q/'analysis.log','original_analysis_failure.txt':C/'content_analysis.log','original_content_failure.txt':C/'controller.log','original_evaluation_failure.txt':E/'controller.log','original_analysis.exit':C/'content_analysis.exit','original_content_controller.exit':C/'controller.exit','original_evaluation_controller.exit':E/'controller.exit','empty.exit':C/'empty.exit','swapped.exit':C/'swapped.exit','empty_receipt.json':C/'empty/receipt.json','swapped_receipt.json':C/'swapped/receipt.json','resolution.json':O/'resolution.json','m77_content_analysis_recovery_20260908.py':B/'m77_content_analysis_recovery_20260908.py','prepare_m77_content_recovery_20260908.py':B/'prepare_m77_content_recovery_20260908.py','export_m77_content_completed_20260908.py':Path(__file__)}
out=io.StringIO(newline='');w=csv.writer(out);w.writerow(['sequence','content','mean_iou','low_iou_frames','H10','valid_frames'])
for seq in d['per_sequence']['category']:
 for arm in ['category','empty','swapped']:
  v=d['per_sequence'][arm][seq];w.writerow([seq,arm,format(v['mean_iou'],'.12f'),v['low_iou_frames'],v['failure_episodes'],v['valid_frames']])
(O/'per_sequence.csv').write_bytes(out.getvalue().encode('utf-8'));files['per_sequence.csv']=O/'per_sequence.csv'
tracker='''# M77 final content status —2026-09-08

| Stage | Final status |
|---|---|
| Seed2027 paired training / development | COMPLETE; development2/15, not promoted |
| Fixed Category head Empty / Swapped tracking | COMPLETE; both exit0; no optimization |
| Original content analysis | ERROR: bbox array supplied where frame records were required; logs and exit1 preserved |
| CPU-only recovered analysis | COMPLETE,exit0; scalar(rows,gt); sealed trajectories rechecked with two metric implementations |
| Same-head content conditions | FAIL,0/8; recovered result in this directory |
| Original conditional evaluation controller | Exit1 on upstream analysis failure; not rewritten as success |
| Public evaluation decision after recovery | NOT STARTED and NOT RESTARTED; both development and content conditions failed |

This completion snapshot supersedes prior RUNNING records. It does not turn empty-content diagnostic gains into a text contribution or a promoted benchmark result. BothQwen checkpoints remain preserved; no new seed or tracking run was introduced by the analysis repair.
'''
(O/'EXPERIMENT_TRACKER.md').write_bytes(tracker.encode());files['EXPERIMENT_TRACKER.md']=O/'EXPERIMENT_TRACKER.md'
table=[]
for label,arm in [('原自动类别','category'),('同权重空文本','empty'),('同权重替换类别','swapped')]:
 v=d['aggregates'][arm];table.append('| %s | %.6f | %.6f | %d | %d | %d |'%(label,v['mean_iou'],v['macro_sequence_mean_iou'],v['low_iou_frames'],v['failure_episodes'],d['reconstructed_template_writes'][arm]))
table='\n'.join(table);c,e,s=(d['aggregates'][n] for n in ['category','empty','swapped'])
harm=[v for v in d['strict_category_H10_reference_correct_every_frame'] if v['reference']=='empty'];assert len(harm)==7 and sum(v['frames'] for v in harm)==360
note=f'''

## 5.150 M77固定权重内容诊断完成：原类别伤害明确，统计接口已最小修复

本节固定§5.149的同一个M77类别最终权重`dee56368…`，保留5个槽、mask、padding、CLIP空文本向量、Hann和模板更新规则，仅改变文字内容。下表“空文本”是类别权重在空内容下的运行，不是上一节另行训练的Empty权重。两条新增对照各完成Train开发22的33130帧/33108次跟踪调用，均exit0；不新增优化、caption或seed。全部预测封存后计算指标，仍为开发诊断而非正式测试。

| 同一类别权重的内容条件 | 按帧平均IoU | 序列等权平均IoU | IoU≤0.1帧 | H10段 | 模板写入 |
|---|---:|---:|---:|---:|---:|
{table}

空文本相对原类别按帧均值提高{100*(e['mean_iou']-c['mean_iou']):.4f}个百分点，宏平均提高{100*(e['macro_sequence_mean_iou']-c['macro_sequence_mean_iou']):.4f}个百分点，低重叠帧减少3407、H10减少21段；替换类别也比原类别高{100*(s['mean_iou']-c['mean_iou']):.4f}个百分点。原类别的8项预定义内容条件全部未通过。这个固定权重对照直接支持“本版具体文字条件造成整体伤害”，不能再仅归结为两个训练权重不同；但不代表所有序列、所有文本机制都无用，原类别仍有6/22条均值高于同权重空文本。

mobilephone02原类别IoU0.563046/H10=1，空文本0.843958/0，替换类别0.845014/0；glass03分别0.485374/3、0.894962/1、0.579842/2。原类别相对同权重空文本存在7个严格持续损害区间、共360帧，参照在每帧IoU≥0.5。cup10则保留局部反例：原类别整序列IoU0.887822高于空文本0.841383，H10为2对3，尽管其中[2070,2090)仍有一段明确伤害。全部逐序列结果均发布，不能把整体负结果改写为每条都负。

原类别与空文本的总模板写入数为199和198，差异很小；这排除了“本轮整体差距只需用总写入数量解释”的简单说法。写入时机、裁剪内容和query仍可随轨迹改变，因此不能据此声称模板完全无关。当前也没有证明自动类别都是语义真值，不能用替换类别变好反推其描述更正确。

统计工程问题及处理必须保留：原分析将bbox数组传给M63独立标量函数，而该函数明确读取`rows[f]['bbox']`，导致IndexError；原分析/内容控制器exit1，后续条件评测控制器亦因上游非零退出而exit1。两个跟踪进程本身均exit0，完整收据及预测未损坏。恢复脚本仅将独立统计调用改为`scalar(rows,gt)`，并把结果输出到独立analysis_recovery_20260908目录；原源码、错误日志和exit1原样封存，不覆盖为成功。修复没有运行GPU跟踪或优化。

恢复过程再次核验三组完整预测的帧数/顺序、初始框、文件SHA、同一权重、文字bank和GT SHA，在66个序列-条件组合上逐项比较两套指标实现；有效非初始化帧均为28897，类别结果与此前独立开发审计一致。恢复exit0。代码差分、恢复源码、原错误和恢复输出随本节发布，这属于执行器统计修复及证据检查，不是独立模型审阅PASS。

M77最终结论：结构/训练对照2/15、内容对照0/8，未晋升。原条件队列因统计异常退出是工程事实；即使恢复统计，开发和内容条件仍双重失败，所以不重启M77公开评测，不改写旧退出码。已验证未创建candidate bundle或full_evaluation目录，两张GPU空闲。强空文本诊断结果仅保留为后续研究对照，不能事后当作原预注册类别主模型通过，也不能把关闭文字后的收益写成文本创新。

用户随后提出“原始响应竞争/最终Hann竞争”的同栈对照，下一步优先按这个单一变化隔离损失作用：M73原定位目标、M77加窗竞争作为已封存参照，新增相同初始化/预算的原始响应竞争Category与Empty训练，推理仍保持Hann。先判断负样本排序本身与加入位置先验的训练影响，再决定是否增加实例语义或记忆模块。未来新实验预先确定主要参照和整体收益/持续损害条件，不继续累加所有历史版本的逐序列优势；这不更改M73/M77原有判定。文字监督需使用可核验的支持或冲突，不能把任意捐赠属性直接当真负例。仍只在DepthTrack Train训练、固定seed2027，不选择早期checkpoint；完整递归和内容对照通过后再冻结低22及同一模型三数据集评测。该计划尚未启动新训练，不声称已有新的语言收益。

恢复结果SHA256：`{sha(Q/'result.json')}`；恢复分析源码：`{d['source_sha256']}`。发布目录`projects/sttrack_lachtt_v1/diagnostics/m77_window_competition/content_followup/completed/`，含完整逐序列CSV和最终状态说明。原始轨迹/权重保留远端，两份Qwen保留；本次磁盘可用{decision['free_bytes']}字节，无删除。正式三数据集指标未更新，项目目标继续保持未完成。
'''
(O/'handoff_append.md').write_bytes(note.encode());files['handoff_append.md']=O/'handoff_append.md'
published=O/'published';published.mkdir()
for n,p in files.items():(published/n).write_bytes(p.read_bytes())
(published/'manifest.json').write_text(json.dumps([dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(files.items())],indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(str(archive),'w:gz') as tar:
 for p in sorted(published.iterdir()):tar.add(str(p),arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');assert sha(master)=='ae789bcfca7a80652c5dee52fb2956f02608b8a4e7aaa7e316e48477298dc4ba'
old=master.read_bytes();assert b'5.150 ' not in old;master.write_bytes(old+note.encode())
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(files))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
