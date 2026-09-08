"""Publish completed low22 evidence only after the saved-output audit passes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil,tarfile

B=Path('/root/autodl-tmp')
E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation'
A=E/'low22_completed_evidence';Q=E/'full_followup';O=E/'low22_completed_publication'
M=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    assert not O.exists()
    previous=M.read_bytes()
    assert sha(M)=='5b9df0965c64eba308dd4a6305b0359f3bc47ec9213ad972d221a22ed000fef2'
    assert b'\n## 5.156 ' not in previous
    a=read(A/'audit.json');r=read(E/'low22_result.json')
    assert a['status']=='completed_low22_saved_evidence_verified'
    assert a['result_sha256']==sha(E/'low22_result.json')
    assert a['metrics_percent']==r['metrics_percent']
    assert a['frozen_gate_checks']==r['gate_checks']
    allowed=a['full_three_dataset_evaluation_allowed']
    assert allowed==all(a['frozen_gate_checks'].values())
    snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),stage=read(Q/'stage.json'),
        controller_exit=(Q/'controller.exit').read_text().strip() if (Q/'controller.exit').exists() else None,
        result=read(Q/'result.json') if (Q/'result.json').exists() else None,
        full_directory_exists=(E/'full_evaluation').exists(),disk_free_bytes=shutil.disk_usage(E).free)
    if not allowed:
        assert not snapshot['full_directory_exists']
    rows=[]
    for key in ['EAO','ACC','ROB']:
        rows.append('| '+key+' | '+format(a['native_metrics_percent'][key],'.6f')+' | '+format(a['metrics_percent'][key],'.6f')+' | '+format(a['delta_metrics_percent'][key],'+.6f')+' |')
    metric_table='\n'.join(rows)
    gate_text='；'.join(k+'='+('PASS' if v else 'FAIL') for k,v in a['frozen_gate_checks'].items())
    conclusion=('低22全部冻结条件通过，具有进入同bundle完整三数据集验证的资格。后续执行由既有条件队列负责，不重复启动。' if allowed else
        '低22未通过全部冻结条件，本次不进入完整三数据集评测。保留开发集和内容对照的正结果，同时明确记录外部开发集合上的收益与损害；不改变本轮门槛，不更换权重或文本来改写结论。')
    broken='、'.join(a['broken_native_success_sequences']) or '无'
    seq_rows=[]
    for name,v in sorted(r['per_sequence_failures'].items()):
        seq_rows.append('| `'+name+'` | '+str(v['anchors'])+' | '+str(v['confirmed_failures'])+' |')
    note=f'''

## 5.156 M78完整低22结果与保存输出核验

本节记录完成态VOT低22（22序列、303个multi-start anchor），不是完整127序列成绩。沿用§5.154通过开发10/10及同权重内容8/8的M78 Category最终权重；固定seed2027、既有五槽类别保留协议和原生Hann/模板/query规则。没有增加seed、checkpoint选择、在线caption或新推理模块。

| 指标（%） | M39原生STTrack低22 | M78 Category低22 | 差值（百分点） |
| --- | ---: | ---: | ---: |
{metric_table}

确认失败anchor：{a['native_confirmed_failures']}→{a['confirmed_failures']}；救回{a['rescued_count']}个，新增{a['new_failure_count']}个。原生七条零失败序列中新增失败的序列：{broken}。这些数量按完整303个anchor逐一配对，不能当作全127指标贡献。

冻结条件：{gate_text}。

{conclusion}

| 序列 | anchor数 | M78确认失败数 |
| --- | ---: | ---: |
{chr(10).join(seq_rows)}

逐序列原生对照、救回与新增失败见per_sequence_failures.csv；完整303个anchor、方向、长度及失败配对见anchor_comparison.csv。上述表为失败统计，不冒充逐序列EAO/ACC/ROB；三项聚合指标直接核对已有official toolkit analysis产物。

完成核验重新绑定全部303个初始化key，验证909份保存输出的SHA256，重算全部303条轨迹的失败及总长度220483，核对指标、救回/新增数与全部冻结条件。核验没有新跟踪调用、caption调用或优化步骤，也没有重跑官方analysis。检查完成不等于独立模型审阅PASS。

绑定bundle SHA256：`{a['bundle_sha256']}`；最终head：`{a['head_sha256']}`；完成结果：`{a['result_sha256']}`；保存输出核验：`{sha(A/'audit.json')}`。

发布时条件队列快照见followup_snapshot.json，采样时间UTC{snapshot['observed_utc']}；该快照不代表未来完成态。完整评测目录当时存在：{snapshot['full_directory_exists']}。无论队列进入何阶段，只有同一模型、文本协议与运行策略完成DepthTrack Test、CDTB和全VOT并核对目标后，才能判定项目目标达成。本节不宣称目标完成。

本轮证据发布于projects/sttrack_lachtt_v1/diagnostics/m78_raw_competition/low22_completed/。已有Train开发22及VOT低22均为反复使用的开发集合，不能称为完全未见测试。当前磁盘可用{snapshot['disk_free_bytes']}字节；本次核验与导出不删除权重，两份Qwen继续保留。
'''
    O.mkdir();files=O/'files';files.mkdir()
    for row in read(A/'files/manifest.json'):
        src=A/'files'/row['path']
        assert sha(src)==row['sha256'] and src.stat().st_size==row['bytes']
        shutil.copyfile(src,files/row['path'])
    (files/'handoff_append.md').write_bytes(note.encode('utf-8'))
    (files/'followup_snapshot.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(__file__,files/Path(__file__).name)
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())]
    (files/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=O/'published.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files.iterdir()):
            assert p.is_file() and p.stat().st_size<3000000
            tar.add(p,arcname=p.name)
    M.write_bytes(previous+note.encode('utf-8'))
    record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(M),master_bytes=M.stat().st_size,files=len(manifest))
    (O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
