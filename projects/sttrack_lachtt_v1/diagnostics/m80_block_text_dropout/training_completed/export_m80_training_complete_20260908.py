from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m80_block_text_dropout_20260908';O=R/'training_completed_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists()
train=read(R/'training/category/result.json');audit=read(R/'saved_training_audit.json')
assert (R/'training_category.exit').read_text().strip()=='0'
assert train['total_track_calls']==186694 and train['optimizer_steps']==5798 and train['sequences']==130
assert sha(R/'training/category/final.pth')==train['final_checkpoint_sha256']=='d853f51ff5820e563d895af220cbe5659a60e3c2d20d5a36c027610fdd36c908'
assert audit['formal_training_completion_verified'] and audit['arms']['category']['sampled_trace_rows']==3917
assert audit['arms']['category']['final_checkpoint_sha256']==train['final_checkpoint_sha256']
procs=[]
for pid in [128432,128433]:
    p=Path('/proc')/str(pid);assert p.exists() and (p/'cwd').resolve()==R
    procs.append(dict(pid=pid,cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
assert not (R/'controller.exit').exists() and not (R/'result.json').exists()
status=dict(observed_utc=datetime.now(timezone.utc).isoformat(),status='training_verified_content_evaluation_running',processes=procs,head_sha256=train['final_checkpoint_sha256'],free_disk_bytes=shutil.disk_usage(R).free)
note='''

## 5.164 M80训练完成，保存产物核验通过，内容评测接续运行

M80于2026-09-08 11:09:45 UTC完成训练，training_category.exit=0；耗时13267.13秒。沿用§5.160冻结的单seed2027文字片段dropout方案，无中途改动或checkpoint选择。此前模型架构与计划不重复记录。

| 完成态训练及核验项目 | 结果 |
| --- | ---: |
| DepthTrack Train拟合序列 | 130 |
| 实际跟踪调用 | 186694 |
| 实际优化次数 | 5798 |
| 实际类别／空词调用 | 149777 / 36917 |
| 保存的逐序列记录／抽样状态 | 130 / 3917 |
| 训练退出码／独立保存产物核验退出码 | 0 / 0 |

§5.163的CPU观察器在训练正常结束后执行--complete：最终checkpoint、初始化协议、优化累积边界、所有计划抽样位置、模板检查与原生坐标监督均通过。检查没有调用跟踪推理或优化。底座未改变的证据仍限定为训练器完成态断言、记录的张量指纹及未改变的原checkpoint，不声称另存了一份完整训练后底座。确定性核验不等于独立模型审阅PASS。

最终head SHA256为`d853f51ff5820e563d895af220cbe5659a60e3c2d20d5a36c027610fdd36c908`，适配器张量SHA256为`f65dd8c453c4bcf86bbc24aadbe37562955b021f8b6af193e3868d8f7b8e9d7e`。完整日志与抽样轨迹保留于远端，报告中已封存其SHA256；训练结果、核验报告与逐序列日志发布于diagnostics/m80_block_text_dropout/training_completed/。

Category与同权重Empty开发评测已自动启动，/proc实际命令行及cwd核实；评测入口检查并加载上述最终head。随后执行同权重Swapped，三组完整输出全部封存后再计算指标与冻结条件。当前没有开发完成态指标或新的正式三数据集成绩，不能凭训练完成宣布性能晋升。M81仍等待M80完整开发与内容流程正常结束；它的固定多起点Category/Empty配对方案不依赖M80指标临时改变。
'''
O.mkdir();F=O/'files';F.mkdir()
for src,name in [(R/'training/category/result.json','training_result.json'),(R/'saved_training_audit.json','saved_training_audit.json'),(R/'training/category/sequence_log.jsonl','sequence_log.jsonl'),(Path(__file__),Path(__file__).name)]:shutil.copyfile(src,F/name)
(F/'verified_status.json').write_text(json.dumps(status,indent=2)+'\n');(F/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())];(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(F.iterdir()):t.add(p,arcname=p.name)
m=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=m.read_bytes()
assert sha(m)=='fe5291bc4102a2cf4288be4a6b3684bdf0bdacaa48902d4cc2df29c37652acb2' and b'\n## 5.164 ' not in old
m.write_bytes(old+note.encode())
record=dict(archive_sha256=sha(archive),master_sha256=sha(m),master_bytes=m.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
