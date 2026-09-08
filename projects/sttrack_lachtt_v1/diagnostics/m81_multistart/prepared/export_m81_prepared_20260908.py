"""Export completed text binding and frozen, resource-queued M81 training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';O=R/'prepared_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not O.exists()
f=read(R/'frozen.json');t=read(R/'training_spec.json');c=read(R/'causal_check.json');bind=read(R/'banks/binding_result.json');q=read(R/'queue_state.json')
assert sha(R/'training_spec.json')==f['training_spec_sha256'] and sha(R/'causal_check.json')==f['causal_check_sha256']
assert c['status']=='completed_M81_native_parity_and_causal_smoke' and c['later_initialization_frame']==1536
assert bind['status']=='complete_paired_multistart_text_binding' and bind['new_vs_existing_empty_max_abs']==0
for bank in bind['banks'].values():assert sha(bank['path'])==bank['sha256']
assert q['status']=='waiting_for_live_M80' and q['dependency_screen_live']
p=Path('/proc')/str(q['scheduler_pid']);assert p.is_dir() and str(B/'queue_m81_after_m80_20260908.py').encode() in (p/'cmdline').read_bytes().split(b'\0')
m=Path('/proc/120925');assert m.is_dir() and (m/'cwd').resolve()==B/'sttrack_m80_block_text_dropout_20260908'
assert not (R/'training').exists() and not (R/'launch.json').exists()
status=dict(status='M81_frozen_and_queued_behind_live_M80',observed_utc=datetime.now(timezone.utc).isoformat(),scheduler_pid=q['scheduler_pid'],M80_training_pid=120925,queue_state=q,M81_training_started=False,free_disk_bytes=shutil.disk_usage(R).free)
note=f'''

## 5.162 M81文字绑定完成，配对多起点训练冻结并排队

§5.161的348条新起点描述全部生成，CPU CLIP编码完成；生成、编码和描述控制器均退出0。无重试改写或事后人工改caption。500个初始化episode完成图像、深度、框、key与文字bank绑定：原152个t0输入逐张量保持，348个新增起点使用各自图像的输出。Category/Empty五槽mask、padding完全配对，Category属性槽内容置空；新编码与原CLIP空文本向量最大绝对差为0。这里只验证绑定与编码，不保证自动类别在语义上正确。

| 新完成产物 | 范围 | 状态 |
| --- | --- | --- |
| 初始化描述与CPU编码 | 348个新增Train起点，470个含空词的独立短语 | 已完成 |
| 拟合Category / Empty bank | 各426个episode | 张量与绑定检查通过 |
| 开发Category / Empty bank | 各74个episode | 张量与绑定检查通过 |
| 中途初始化因果检查 | cube04_indoor第1536帧起，101帧零残差与96帧优化检查 | 已通过，检查权重未保存 |
| 正式M81训练及指标 | Category / Empty，均seed2027 | 已冻结并排队，尚未开始，无指标 |

当前训练入口由封存M78衍生。每条序列仍遍历原完整转移帧；仅在冻结清单的episode边界，用该时刻初始化框和文字重新建立模板/query/语义参考。段内不按GT或跟踪失败重置。记录同时包含全局frame_index、episode_start与episode_frame_id；模板50步计数按episode重新开始。边界对齐原32帧累积区间并显式检查无悬挂梯度，所以两组各186694次调用、预期5798次优化，额外初始化前向与GT框数量另行注明。底座/Center Head冻结，仍训练289154参数密集适配器；M78原始竞争目标、Hann、默认模板条件均不变，没有M80 dropout。

检查使用拟合集预先选定的合法中途起点cube04:1536：两组各101帧与同起点原生bbox、score、query、模板逐项一致，默认模板写入时刻一致；随后每组96帧、3次优化，验证原定位＋原始竞争损失组合、有限梯度、RGB/Depth/Text投影梯度及冻结底座。合成例验证同一实例的候选不被误作负例。检查为代码与数值验证，不是独立模型审阅PASS；没有将检查得到的权重用于正式训练。

完整评测队列已实现，共9个预测族。t0协议包括新Category、新独立Empty、新Category权重下空词与替换词，分别完整运行原开发22；多起点协议包括新Category、新Empty、原生、封存M78 Category与封存M78 Empty，五组共用74个episode。所有9组完整输出封存后才读取指标GT。多起点统计按共同边界截断H10段，不能叫VOT ROB，也不能拿它与未重初始化的原生t0直接相减作为模型收益。

冻结条件：两个协议各自对原生与本轮匹配Empty检查按帧均值（至少＋0.002/＋0.001）、宏均值不低、低重叠帧/H10不高、各自零H10序列保护，共20项；t0同权重原类别对空词及替换词的8项内容检查；同一多起点协议中新Category对M78 Category的4项覆盖检查（按帧至少＋0.001、宏均值不低、低重叠/H10不高）。M78 t0结果作为描述性对照，不积累所有历史成功序列并集。所有内容与多起点预测都运行，无论早期结果是否可能不利；不通过内容选择、早期checkpoint或改门晋升。没有自动正式外部评测。

UTC{status['observed_utc']}核实M81调度进程{q['scheduler_pid']}活跃，正在等待实际存活的M80控制器；M80训练进程120925仍活跃。M81尚无training目录或正式launch记录。调度每240秒检查：M80完整训练与内容评测控制器正常结束、完成结果存在、GPU0/1均空闲且磁盘足够后，启动固定M81 Category/Empty配对；若M80异常则保留错误并停止调度，不重启任何训练。调度不根据M80指标改变M81方案，也不抢占GPU。

预计M81两组并行训练约4小时，9族开发预测额外约2小时，实际耗时以记录为准。后续核对训练初始化数、全部转移覆盖、基线冻结、最终head、所有预测绑定与32项条件；通过后再单独审计低22资格，同一最终模型的DepthTrack Test、CDTB、VOT完整目标仍需验证。

关键指纹：training_spec `{sha(R/'training_spec.json')}`；evaluation_spec `{sha(R/'evaluation_spec.json')}`；因果检查`{sha(R/'causal_check.json')}`；bank绑定`{sha(R/'banks/binding_result.json')}`。启动队列与完成文字记录、bank哈希、衍生源码和检查记录发布于projects/sttrack_lachtt_v1/diagnostics/m81_multistart/prepared/。本节没有重复旧三数据集指标。两份Qwen继续保留，本轮没有删除权重或下载模型。
'''
O.mkdir();files=O/'files';files.mkdir()
for name in ['prepared_training_spec.json','training_spec.json','evaluation_spec.json','frozen.json','training_preparation.json','train_causal.py','training_change.diff','multistart.py','evaluate.py','run.sh','check_causal.py','causal_check.json','smoke_episode.json','queue_spec.json','queue_state.json','wait_queue.sh','integration.json','causal_training.py','window_competition.py','recursive_metric.py','support_loss.py']:
    shutil.copyfile(R/name,files/name)
for name in ['generation_result.json','encoding_result.json','records.jsonl']:shutil.copyfile(R/'captions'/name,files/name)
for name in ['binding_result.json','bindings.json']:shutil.copyfile(R/'banks'/name,files/name)
for name in ['bind_m81_text_banks_20260908.py','prepare_m81_training_20260908.py','queue_m81_after_m80_20260908.py']:
    shutil.copyfile(B/name,files/name)
shutil.copyfile(__file__,files/Path(__file__).name)
(files/'.gitattributes').write_text('training_change.diff whitespace=-blank-at-eol\n')
write(files/'verified_queue_status.json',status);(files/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())];write(files/'manifest.json',manifest)
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files.iterdir()):tar.add(p,arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='ecf6de906664e487282e770931e9980c1bb2d91f2318502e832570c9e47c86f5' and b'\n## 5.162 ' not in old
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest));write(O/'publication_record.json',record)
print(json.dumps(dict(publication=record,status=status)))
