from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';E=R/'candidate_evaluation';O=E/'published_prepared'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists();s=read(E/'spec.json');p=read(E/'prepared_entry_and_full.json');c=read(E/'evaluation_preparation_check.json')
assert s['training_spec_sha256']==sha(R/'training_spec.json') and p['entry_spec_sha256']==sha(E/'spec.json')
assert c['no_evaluation_files_created'] and c['category_transform_exact'] and not c['cuda_initialized']
assert not (E/'bundle.json').exists() and not (E/'full_evaluation').exists() and not (R/'recursive_result.json').exists()
for n,h in p['source_files'].items():assert sha(B/n)==h
O.mkdir();files={n:B/n for n in p['source_files']}
files['prepare_m78_evaluation_20260908.py']=B/'prepare_m78_evaluation_20260908.py'
files['export_m78_evaluation_20260908.py']=Path(__file__)
for n in ['spec.json','cpu_preparation_result.json','full_preparation_readiness.json','evaluation_preparation_check.json','prepared_entry_and_full.json']:
    files[n]=E/n
for n in list(s['interface_sha256'])+['text_protocol.json']:
    files[n]=E/'interface'/n
files['premature_full_pipeline.txt']=E/'premature_full_pipeline.log'
tracker='''# M78 external evaluation readiness

CPU preparation complete while paired training is running. No final checkpoint loaded, no final bundle bound, no external tracking or new captions generated. Category routing was checked against all152 Train initializations and303 existing low22 observations; the five interface source files remain byte-identical to the sealed parent interface. Real GPU OPE/TraX parity remains pending until a qualified final head exists.

Order: all10 M78 development conditions and completed audit, all8 fixed-head content comparisons, bind the fixed Category final, real OPE/TraX Train-prefix parity, complete303-anchor low22, then the same bundle on DepthTrack Test/CDTB/full127 only if low22 qualifies. No new seeds and no automatic evaluation controller has been launched.

Full source checks cover50 DepthTrack Test sequences/76373 frames,80 CDTB sequences/101956 frames, and127 VOT sequences/1765 anchors. Full initialization banks are still missing; generate them causally only after low22 passes. The completed-output verifier rechecks all saved results and requires strict VOT EAO>77.9,ACC>82.1,ROB>93.7. Readiness is not benchmark performance or a reviewer PASS.
'''
(O/'EXPERIMENT_TRACKER.md').write_text(tracker);files['EXPERIMENT_TRACKER.md']=O/'EXPERIMENT_TRACKER.md'
note=f'''

## 5.153 M78三数据集评测入口准备完成：同一模型绑定尚未激活

在M78配对训练运行期间，已将封存M77评测入口对应到M78固定Category最终权重，并将资格检查明确调整为本轮10项开发条件和8项同权重内容条件。仅生成和核对CPU侧计划、接口及缓存路由，没有加载尚不存在的最终权重、没有创建bundle，也没有启动新的外部跟踪、caption或优化。

| 准备项 | 已核实范围 | 仍待完成 |
|---|---|---|
| 初始化文本路由 | 152条Train初始化、开发22路由与训练类别银行一致；303个既有VOT低22初始化保持原tokens、mask、空词和key | 合格最终权重的实际GPU入口复现 |
| DepthTrack Test | 50条、76373帧，指标源码与原生参照绑定 | 因果首帧文本银行与新权重完整预测 |
| CDTB | 80条、101956帧，同一P/R/F实现与保存输出检查 | 因果首帧文本银行与新权重完整预测 |
| VOT-RGBD2022 | 127条、1765个anchor；保留原低22的303项，剩余1462项使用统一首帧协议 | 低22通过后补齐初始化银行与完整正式跟踪 |

部署仍使用原Hann最大值作为报告置信度及原生模板条件，无Null乘积、分数重标定或评测规则改写。五个推理接口源码与封存父接口逐字节一致；实际OPE与TraX是否产生相同轨迹，要等最终权重存在并通过资格审计后，在3条Train前缀各202帧验证，不能把CPU检查写成GPU保真已通过。

完整评测脚本的提前调用已在CPU检查中因缺少真实内容激活记录而退出，确认没有产生评测目录或预测文件。检查SHA`{sha(E/'evaluation_preparation_check.json')}`；该预期拒绝不是训练失败。最终验证脚本绑定同一base、adapter、文本协议和运行配置，从保存输出复算DepthTrack/CDTB及VOT结果，并按用户要求对VOT严格检查EAO>77.9、ACC>82.1、ROB>93.7；当前无任何新数值结果。

执行顺序保持：M78完整开发及审计→同权重内容诊断→固定最终权重绑定→真实Train入口一致性→303-anchor低22→低22通过后完整三数据集。低22门沿用对原生的聚合与7条零失败序列要求，不要求支配全部历史模型。没有启动自动外部评测控制器。下一步仍等待M78完整训练结果，不在训练期间改损失、文本或模板策略，继续仅seed2027、保留两个Qwen。
'''
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='7b768020c4c1cdd2c826e33d3db3a906cc20d8ceef69e1b3cf54a40e665b0b69' and b'## 5.153 ' not in old
(O/'master_append.md').write_bytes(note.encode());files['master_append.md']=O/'master_append.md'
master.write_bytes(old+note.encode())
manifest=[]
for n,p in files.items():
    q=O/n
    if p!=q:shutil.copyfile(p,q)
    manifest.append(dict(path=n,bytes=q.stat().st_size,sha256=sha(q)))
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=B/'m78_evaluation_prepared.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(O.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,manifest_files=len(manifest))))
