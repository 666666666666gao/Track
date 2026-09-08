"""Publish audited M79 results and reconstruct unchanged template control."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,io,json,math,shutil,statistics,tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m79_vot_content_diagnostic_20260908'
P=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation';A=R/'completed_evidence';O=R/'completed_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode('utf-8'))

def main():
    assert not O.exists() and (R/'completed_saved_audit.exit').read_text().strip()=='0'
    audit=read(A/'audit.json');result=read(R/'result.json')
    assert audit['status']=='completed_M79_all_saved_content_evidence_verified' and audit['result_sha256']==sha(R/'result.json')
    assert audit['total_saved_files_verified']==2727 and audit['total_anchor_trajectories_verified']==909
    source=B/'m79_vot_content_diagnostic_20260908.py'
    spec=importlib.util.spec_from_file_location('m79_publication_frozen_check',str(source))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.checked();parent,_=m.parent()
    bundle=read(P/'bundle.json');base_source=Path(bundle['repository'])/'lib/test/tracker/sttrack.py'
    assert sha(base_source)==bundle['source_sha256']['lib/test/tracker/sttrack.py']
    code=base_source.read_text()
    for text in ['self.frame_id = 0','self.frame_id += 1','(self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold)']:
        assert text in code
    controls={}
    for arm in ['category','empty','swapped']:
        root=P if arm=='category' else R/arm;run=root/('low22_run' if arm=='category' else 'run')
        merge=read(run/'merge_result.json');scores=[];checks=[];per={};files=0
        for rel,h in merge['result_sha256'].items():
            if not rel.endswith('_confidence.value'):continue
            p=run/'master'/rel;assert sha(p)==h;lines=p.read_text().splitlines()
            assert lines[0]=='' and len(lines)>1
            values=[float(v) for v in lines[1:]];assert all(math.isfinite(v) for v in values)
            assert len(lines)==len(p.with_name(p.name.replace('_confidence.value','_time.value')).read_text().splitlines())
            scheduled=[float(lines[i]) for i in range(50,len(lines),50)]
            writes=sum(v>.75 for v in scheduled);seq=p.parent.name
            x=per.setdefault(seq,dict(anchors=0,scheduled_checks=0,reconstructed_writes=0))
            x['anchors']+=1;x['scheduled_checks']+=len(scheduled);x['reconstructed_writes']+=writes
            scores.extend(values);checks.extend(scheduled);files+=1
        assert files==303 and len(scores)==220180
        gap=min(abs(v-.75) for v in checks);assert gap>1e-6
        controls[arm]=dict(confidence_files=files,tracked_frame_scores=len(scores),scheduled_checks=len(checks),
            reconstructed_writes=sum(v>.75 for v in checks),score_median=statistics.median(scores),minimum_scheduled_score_distance_to_threshold=gap,per_sequence=per)
    native=read(parent.M39/'m39_result.json')['arms']['default']
    aggregate=result['aggregates'];table=[]
    native_metrics={k:native['metrics_percent'][k.lower()] for k in ['EAO','ACC','ROB']}
    table.append('| 原生STTrack | '+' | '.join(format(native_metrics[k],'.6f') for k in ['EAO','ACC','ROB'])+' | '+str(native['confirmed_failures'])+' | — |')
    labels={'category':'M78原类别','empty':'同权重空文本','swapped':'同权重替换类别'}
    for arm in ['category','empty','swapped']:
        x=aggregate[arm];table.append('| '+labels[arm]+' | '+' | '.join(format(x['metrics_percent'][k],'.6f') for k in ['EAO','ACC','ROB'])+' | '+str(x['confirmed_failures'])+' | '+str(controls[arm]['reconstructed_writes'])+' |')
    confidence_report=dict(status='complete_saved_confidence_template_write_reconstruction',observed_utc=datetime.now(timezone.utc).isoformat(),
        base_tracker_source_sha256=sha(base_source),interval=50,threshold=.75,comparison='strictly_greater',
        initialization_score_row_excluded=True,frame_id_resets_per_anchor=True,conditions=controls,
        minimum_boundary_margin_verified_greater_than_1e_minus6=True,template_content_or_future_utility_not_measured=True,
        inference_calls=0,causal_attribution_claimed=False)
    note=f'''

## 5.159 M79完成：训练集词义收益没有迁移到VOT低22

M79于UTC{result['observed_utc']}完成Empty与Swapped各303个anchor；跟踪、analysis及主控制器均退出0。复用M78封存Category结果后，共三种内容条件。只使用seed2027对应的同一个M78 Category最终权重，没有新训练、checkpoint选择、caption或embedding生成。该实验是失败后的内容诊断，没有晋升权限或全量队列。

| 完整VOT低22条件 | EAO（%） | ACC（%） | ROB（%） | 确认失败anchor | 默认模板重建写入数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

空文本相对原类别：EAO＋1.341476、ACC＋0.062880、ROB＋1.589842个百分点，救回12个、又新增7个，净减少5个失败。替换类别相对原类别：EAO＋0.539806、ACC−0.125231、ROB＋0.872686个百分点，救回12个、又新增9个，净减少3个失败。因此本轮原类别未稳定优于空词或替换词，不能将§5.154开发集的同权重类别收益直接外推到VOT。

同时，空文本相对原生仍为EAO−1.748553、ROB−2.711499个百分点，失败135对124；它不是恢复原生STTrack，也不能作为达标模型。此处支持两个观察：具体原类别在本VOT开发集合没有带来预期增量；清空输入也未恢复原生性能。不能进一步把总差值精确分摊为“词义损失”和“视觉适配损失”，因为空词同样是对Category训练权重的内容干预，三条递归状态也已不同。

cup02_indoor_1、toy09_indoor_1、shoes02_indoor_1、cube05_indoor_5在三种内容条件下都分别失败36、26、12、9次，合计83次。单纯改类别内容没有解决这些核心困难族。局部差异仍存在：yogurt为10/8/9，cube02_indoor_2为3/1/2，two_tennis_balls_3为4/2/3（顺序均为原类别/空词/替换）；完整22条结果见per_sequence_failures.csv，完整303条配对见anchor_comparison.csv，不只报告有利样例。

另按封存confidence.value与实际源码重建默认模板行为：每个anchor从frame_id=0开始，track每帧加1，只在50的倍数、分数严格大于0.75时更新。所有909个confidence文件及长度经核对，每种内容有220180个非初始化分数；检查分数与阈值的最小距离均大于1e-6，避免边界舍入歧义。上表写入数来自这一条件重建，不表示更新框质量、视图清晰度或未来效用。三种内容同时改变定位、分数、crop/query和模板内容，不能只凭写入次数作因果归因；原生写入数未在本项重建，以“—”标示。

独立保存输出核验于UTC{audit['observed_utc']}完成：重新绑定606个对照初始化，逐张量验证内容干预，检查2727份保存文件，重算909条轨迹及661449个帧位置、官方指标、全部救回/新增失败配对。未重新跟踪或重跑official analysis。该核验是程序与保存证据检查，不是独立模型审阅PASS。

最终head SHA256仍为`{s['head_sha256']}`；M79结果SHA256为`{sha(R/'result.json')}`；完成核验为`{sha(A/'audit.json')}`。M78原失败结论保持不变，M79的三个内容条件均不进入全量评测。正式三数据集目标仍未完成，不把历史SRTrack的DepthTrack/CDTB成绩与当前STTrack拼接。

下一步应回到DepthTrack Train训练阶段，优先验证训练初始化覆盖与实例证据使用方式，继续保留同参数视觉/空词和固定权重内容对照。§5.158只证明首帧正向训练存在覆盖差异，未证明该差异就是本次退化的原因；新的训练方案需先固定数据、预算和评价条件，不能通过在VOT上选择较好的内容、阈值或checkpoint替代训练验证。主线继续是能泛化的文本交互，不改为长期空文本部署，也不启动全量重caption。

本节及全部完成证据发布于projects/sttrack_lachtt_v1/diagnostics/m79_vot_content_diagnostic/completed/。VOT低22是重复使用的开发集合，不能称为完全未见测试。本次没有删除权重，两份Qwen继续保留。
'''
    O.mkdir();files=O/'files';files.mkdir()
    for row in read(A/'files/manifest.json'):
        p=A/'files'/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];shutil.copyfile(p,files/row['path'])
    write(files/'template_update_reconstruction.json',confidence_report)
    (files/'handoff_append.md').write_bytes(note.encode('utf-8'));shutil.copyfile(__file__,files/Path(__file__).name)
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())]
    write(files/'manifest.json',manifest);archive=O/'published.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files.iterdir()):
            assert p.is_file() and p.stat().st_size<3000000
            tar.add(p,arcname=p.name)
    master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
    assert sha(master)=='0464242522367e01303e6561218b1a047d47478bc69eb9cef530e18cc718a6b9' and b'\n## 5.159 ' not in old
    master.write_bytes(old+note.encode('utf-8'))
    record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
    write(O/'publication_record.json',record);print(json.dumps(dict(confidence_summary={arm:{k:v for k,v in x.items() if k!='per_sequence'} for arm,x in controls.items()},publication=record),indent=2))

if __name__=='__main__':main()
