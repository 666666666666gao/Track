from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m82_native_preservation_20260909';O=R/'launch_publication';F=O/'files'
M=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=M.read_bytes()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(M)=='767334a7c91849bdf2488fa29135399f7a7bf0f5d788b897426f6b1f36cc9fae'
assert b'\n## 5.170 ' not in old and not O.exists()
h=json.loads((R/'first_training_health.json').read_text());l=json.loads((R/'launch.json').read_text())
assert {x['arm'] for x in h['processes']}=={'category','empty'} and h['source_hashes_unchanged']
assert all(x['completed_sequences']>=1 and x['exit'] is None for x in h['arms'].values())
assert not h['performance_results_exist'] and not (R/'recursive_result.json').exists()
note='''

## 5.170 M82配对训练已启动，首批优化与运行资源核验

§5.169冻结方案发布后，M82于2026-09-09北京时间03:00:27启动持久screen任务m82_native_preservation，GPU0训练Empty、GPU1训练Category，固定seed2027。没有调整已冻结损失、输入、源码或晋升条件。

03:04:36首轮检查时，两组均完成cube04_indoor的2575次跟踪、2513个有效监督帧、81次优化。Category保持项启用251帧、KL累计23.297601；Empty启用250帧、KL累计23.155654。两组最大原梯度范数分别28.472816、28.249765，按原规则裁剪；没有非有限值或退出错误。当前数字只证明训练与保持项实际运行，不是开发性能，更不是语言收益。

按首条序列约14.8—14.9次跟踪/秒估计，训练约北京时间06:30结束；完整130条序列的可见性与监督比例不同，时间会变化。下一次训练进度核查安排在预计结束前几分钟，不反复高频读取相同日志。训练完成后自动执行原冻结的四组开发22，全部轨迹封存后计算指标；另有独立CPU标量复算任务，每240秒检查控制器是否完成，仅在成功完成后核验训练产物、132520个开发位置和预先声明条件。该核验当前排队中，不能写成已通过，也不等于独立模型审阅。

检查时远程盘剩余1442684928字节（约1.44GB）。Qwen3_8B保留5个分片、16381516776字节；Qwen2.5-VL-3B-Instruct保留2个分片、7509337976字节。本轮不复制STTrack底座或Qwen，不保留逐轮大权重，仅按冻结规则保存两组latest及fixed-final适配器/优化器。正式三数据集成绩没有变化。

启动receipt、首批优化记录、独立复算源码及等待脚本发布于diagnostics/m82_native_preservation/launched/。当前训练与后续自动评测继续运行，不因聊天交接而中断；待完成结果与独立复算后，再判断是否值得进入外部验证。
'''
F.mkdir(parents=True)
for n in ['launch.json','first_training_health.json']:shutil.copyfile(R/n,F/n)
for n in ['audit_m82_completed_20260909.py','watch_m82_audit_20260909.py','check_m82_training_health_20260909.py']:shutil.copyfile(B/n,F/n)
shutil.copyfile(Path(__file__),F/Path(__file__).name)
(F/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())]
(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(F.iterdir()):t.add(p,arcname=p.name)
M.write_bytes(old+note.encode('utf-8'))
record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha(archive),master_sha256=sha(M),master_bytes=M.stat().st_size,files=len(manifest),formal_training_started=True,performance_results_exist=False)
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
