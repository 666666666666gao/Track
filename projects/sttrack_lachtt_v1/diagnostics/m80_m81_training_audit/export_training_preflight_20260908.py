from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');O=B/'sttrack_m80_m81_preflight_publication_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
reports={x:json.loads((B/('sttrack_m80_block_text_dropout_20260908' if x=='m80' else 'sttrack_m81_multistart_20260908')/'saved_training_audit_preflight.json').read_text()) for x in ['m80','m81']}
source=B/'audit_m80_m81_training_20260908.py'
for r in reports.values():
    assert r['status']=='preflight_frozen_training_inputs_verified' and r['auditor_sha256']==sha(source)
    assert not r['formal_training_completion_verified']
note='''

## 5.163 M80/M81冻结训练输入独立复算完成

新增只读核验脚本audit_m80_m81_training_20260908.py，分别在CPU执行M80、M81 preflight，两者通过。脚本独立读取130条拟合序列GT、帧数和冻结清单，复算每个32帧累积区间是否存在有效监督，确认两项方案均包含186694次跟踪调用和5798个有效优化窗口。核验初始化文件、RGB/Depth文件、源代码、文字bank与初始适配器张量的指纹；不运行跟踪推理、不执行优化、不改变运行中的方案。

M80原130次初始化不变；32帧文字条件安排复算为36917次空词与149777次类别调用。M81共426次初始化，逐段检查边界连续、32帧对齐、GT初始化框与图像绑定，原186694次转移没有遗漏或重复监督。两者跟踪调用与有效优化窗口预算相同，但M81额外296次初始化的GT监督及前向计算仍须单独报告，不能声称总计算严格相同。

当前通过的是冻结输入与预期预算复算，尚未核验完成态训练。脚本另含--complete模式，准备在每个训练臂正常结束后读取最终checkpoint、逐序列日志和抽样状态轨迹，复核实际调用、优化、episode重置、模板检查和监督坐标。该模式尚未执行，不能称为已通过；它不会重新训练或改写checkpoint。底座保持证据将明确区分训练器断言、记录的张量指纹与未改变的原始checkpoint，不宣称存在另存的完整训练后底座。

两项preflight与源码封存在projects/sttrack_lachtt_v1/diagnostics/m80_m81_training_audit/。本项为确定性文件与数值核验，不是独立模型审阅PASS，也没有新增开发集或正式三数据集指标。训练继续固定seed2027，两个Qwen保留。
'''
O.mkdir();files=O/'files';files.mkdir()
shutil.copyfile(source,files/source.name)
shutil.copyfile(__file__,files/Path(__file__).name)
for name,r in reports.items():(files/(name+'_preflight.json')).write_text(json.dumps(r,indent=2)+'\n')
(files/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())]
(files/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files.iterdir()):tar.add(p,arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='465daabbfc4cea11a78b72af3eae174ad89bd441d958fd86101551ff85f6a39e' and b'\n## 5.163 ' not in old
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
