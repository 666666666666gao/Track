from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,shutil,subprocess,tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m79_vot_content_diagnostic_20260908';O=R/'published_start'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
source=B/'m79_vot_content_diagnostic_20260908.py'
module_spec=importlib.util.spec_from_file_location('m79_publication_check',str(source))
m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m);s=m.checked()
assert not O.exists() and not (R/'controller.exit').exists()
assert (R/'binding.exit').read_text().strip()=='0'
launch=read(R/'launch.json');p=Path('/proc')/str(launch['pid']);st=(p/'stat').read_text().split()
assert st[21]==launch['start_ticks'] and st[2]!='Z'
workers=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
    pid,memory=[int(v.strip()) for v in line.split(',')];proc=Path('/proc')/str(pid)
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    if b'm79_semantic_vot' not in cmd:continue
    env=dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
    cwd=str((proc/'cwd').resolve());gpu=env[b'CUDA_VISIBLE_DEVICES'].decode()
    pythonpath=env[b'PYTHONPATH'].decode()
    arm=next(arm for arm in m.ARMS if pythonpath==str(R/arm))
    assert int(gpu)==s['arms'][arm]['gpu']
    workers.append(dict(pid=pid,gpu=int(gpu),arm=arm,cwd=cwd,pythonpath=pythonpath,memory_MiB=memory))
assert len(workers)==8 and all(sum(w['arm']==arm for w in workers)==4 for arm in m.ARMS)
progress={}
for arm in m.ARMS:
    files=list((R/arm/'run').glob('shard-*/results/**/*_time.value'))
    progress[arm]=dict(completed_anchors=len(files),completed_frame_positions=sum(len(p.read_text().splitlines()) for p in files))
snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),parent_pid=launch['pid'],parent_start_ticks=st[21],parent_state=st[2],
    workers=workers,progress=progress,binding_exit=0,controller_complete=False,disk_free_bytes=shutil.disk_usage(R).free,
    two_Qwen_directories_preserved=all((B/'qwen'/n).is_dir() for n in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']))
assert snapshot['two_Qwen_directories_preserved']
note=f'''

## 5.157 M79启动：固定M78权重的VOT低22内容归因

§5.156已经确认M78低22为54.045964 / 75.697432 / 68.721059、140个失败，救回5个、新增21个；原六项条件只有完整性通过，原低22到全量队列正常结束且没有启动全量。M78开发集及训练集同权重内容对照的收益保持原记录，但没有迁移到本次VOT低22。现在新增M79诊断，不更改M78结论或门槛，不选择替代checkpoint，不直接晋升任意内容条件。

M79只固定M78 Category最终权重与运行代码，复用已封存的原类别303条轨迹，新增完整Empty、Swapped各303个anchor。Empty将有效词槽替换为既有CLIP空文本向量；Swapped只替换槽0，属性空词、padding、mask、五槽结构与初始化key保持完全相同。按初始化key排序，从循环顺序中取第一个不同序列且槽0向量不同的donor；规则和完整303行映射在运行前封存，不依据GT或跟踪结果挑选。不同类别向量不等于已核验的语义冲突，因此只能称为替换类别诊断。

两组均为seed2027对应的同一个已训练权重，新增seed、参数、优化步骤、caption及embedding调用均为0。固定head SHA256为`{s['head_sha256']}`，base为`{s['base_sha256']}`。每组显式使用不同文本协议与bank哈希，bundle中除文本协议路径及哈希外其余字段完全相同；不能把内容被替换的对照宣称成原正式文本协议。底座、adapter、Center Head、Hann、模板interval50/score>0.75、query及crop规则均未改变。

准备检查验证两组共606个初始化路由、全部bank key/mask/padding/属性槽一致、303个替换类别向量实际发生变化、所有模型与入口字节保持不变；CPU准备没有初始化CUDA或进行跟踪。随后通过启动前绑定核验，已有真实GPU进程见process_snapshot.json。运行8个分片工作进程，Empty四分片在GPU0，Swapped四分片在GPU1；这些是工作划分与内容对照，不是多seed。

主脚本PID{launch['pid']}、启动标识{launch['start_ticks']}，screen会话为track_m79_vot_content_20260908，工作目录`{R}`。启动记录采样UTC{launch['observed_utc']}；初次PID查询同时返回主bash及两个arm子shell，已按PPID找出唯一主进程并补录，未重复启动、未修改实验。发布快照UTC{snapshot['observed_utc']}，Empty已封存{progress['empty']['completed_anchors']}/303，Swapped已封存{progress['swapped']['completed_anchors']}/303；尚无完成态指标。

每组220483个计划帧位置，总新增440966；复用原四分片、toolkit配置、anchor分配及合并工具，控制器每240秒检查。两组完整保存输出后分别运行官方analysis，核对每组909份文件及303条失败轨迹，再汇总与原类别、原生STTrack的EAO/ACC/ROB差值、救回、新增失败和逐序列结果。M79没有晋升门或自动全量队列，任何结果都不改变M78已失败的六项条件。该诊断用于区分具体词义作用与适配权重整体迁移问题，不能仅凭某个对照更好就改成新的最终模型。

spec SHA256：`{sha(R/'spec.json')}`；诊断源码：`{sha(source)}`；运行脚本：`{sha(B/'m79_vot_content_diagnostic_20260908.sh')}`。启动检查属于可复核的程序/输入/进程检查，不是独立模型审阅PASS。预计每组约3小时，两个GPU并行；应在预计收尾前按实际封存速度检查，不高频重复查询。

本节及启动证据发布于projects/sttrack_lachtt_v1/diagnostics/m79_vot_content_diagnostic/。当前磁盘可用{snapshot['disk_free_bytes']}字节，两份Qwen目录均保留。VOT低22已反复用于开发诊断，不是完全未见测试；最终目标仍需训练所得同一完整模型在三个数据集的正式验证，本节不宣称已达成。
'''
O.mkdir();files=O/'files';files.mkdir()
inputs={n:R/n for n in ['spec.json','preparation_result.json','donor_mapping.json','launch.json','binding.exit']}
inputs.update({source.name:source,'m79_vot_content_diagnostic_20260908.sh':B/'m79_vot_content_diagnostic_20260908.sh','run_vot_failure_family_shards.py':R/'run_vot_failure_family_shards.py',Path(__file__).name:Path(__file__)})
for arm in m.ARMS:
    for n in ['plan.json','bundle.json','text_protocol.json','run/shard_manifest.json']:
        inputs[arm+'__'+n.replace('/','__')]=R/arm/n
for name,src in inputs.items():shutil.copyfile(src,files/name)
(files/'process_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(files/'handoff_append.md').write_bytes(note.encode('utf-8'))
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())]
(files/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='f32d7170ef7a2d8084aec74e930e4f0348bc3ac46fe70e2be5556bf6c84510c3' and b'\n## 5.157 ' not in old
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
