"""Sealed M78 failure accounting by direction and initialization position."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,shutil,tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';E=R/'candidate_evaluation'
O=R/'direction_initialization_diagnostic';M=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode('utf-8'))

def main():
    assert not O.exists()
    assert sha(E/'low22_result.json')=='64f23763ee6df91d80e81754e7db3e654196eb377b624e258768fddd970d5fc3'
    native_path=B/'sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902/m39_result.json'
    assert sha(native_path)=='cf953c0d3c69609bcd83c11cb24ba57f37e30b38d3b3bcad32860b3a9ba9c1b5'
    current=read(E/'low22_result.json');native=read(native_path)['arms']['default']
    assert not current['full_three_dataset_evaluation_allowed'] and set(current['failure_outcomes'])==set(native['failure_outcomes'])
    training=read(R/'training_spec.json');trainer=R/'train_causal.py'
    assert sha(trainer)==training['training_script_sha256']=='9338d6a72c2038bc9e4caad687afc5e9d225d596851dc344e2e22e9d0d96903c'
    lines=trainer.read_text().splitlines()
    expected=['for seq_index, row in enumerate(spec[\'sequence_order\']):','tracker.initialize(frame(0), info)','for frame_index in range(1, n):']
    line_evidence={x:[i+1 for i,s in enumerate(lines) if x in s] for x in expected}
    assert all(len(v)==1 for v in line_evidence.values()) and len(training['sequence_order'])==130
    train_calls={}
    for arm in ['category','empty']:
        records=[json.loads(line) for line in (R/'training'/arm/'sequence_log.jsonl').read_text().splitlines()]
        assert len(records)==130 and [r['sequence'] for r in records]==[r['sequence'] for r in training['sequence_order']]
        assert all(r['track_calls']==r['frames']-1 for r in records)
        train_calls[arm]=sum(r['track_calls'] for r in records);assert train_calls[arm]==186694
    groups={}
    for name in ['forward','backward','first_frame_forward','later_forward','all_later']:
        groups[name]=dict(anchors=0,native_failures=0,M78_failures=0,rescued=0,new_failures=0,frame_positions=0)
    for key,r in current['failure_outcomes'].items():
        old=native['failure_outcomes'][key]
        assert all(r[k]==old[k] for k in ['sequence','anchor','direction','run_length'])
        assert r['direction'] in ['forward','backward']
        tags=[r['direction']]
        if r['anchor']==0:
            assert r['direction']=='forward';tags.append('first_frame_forward')
        else:
            tags.append('all_later')
            if r['direction']=='forward':tags.append('later_forward')
        for tag in tags:
            g=groups[tag];g['anchors']+=1;g['native_failures']+=int(old['failed']);g['M78_failures']+=int(r['failed'])
            g['rescued']+=int(old['failed'] and not r['failed']);g['new_failures']+=int(r['failed'] and not old['failed']);g['frame_positions']+=r['run_length']
    for g in groups.values():
        g['net_added_failures']=g['M78_failures']-g['native_failures']
        assert g['net_added_failures']==g['new_failures']-g['rescued']
        g['native_success_anchors']=g['anchors']-g['native_failures']
        g['new_failure_fraction_among_native_success']=g['new_failures']/g['native_success_anchors']
    assert groups['forward']['anchors']+groups['backward']['anchors']==303
    assert groups['first_frame_forward']['anchors']+groups['all_later']['anchors']==303
    assert groups['forward']['new_failures']+groups['backward']['new_failures']==21
    O.mkdir()
    report=dict(status='complete_posthoc_M78_direction_initialization_accounting',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),M78_result_sha256=sha(E/'low22_result.json'),native_result_sha256=sha(native_path),
        training_spec_sha256=sha(R/'training_spec.json'),training_source_sha256=sha(trainer),training_source_line_evidence=line_evidence,
        training_initializations_per_arm=130,training_initial_frame_index=0,training_direction='forward',actual_training_calls=train_calls,groups=groups,
        new_failures_at_later_anchors_fraction=groups['all_later']['new_failures']/21,later_anchor_fraction=groups['all_later']['anchors']/303,
        interpretation=['Net deterioration occurs in both directions, so reverse-only mismatch does not explain all new failures.',
            'Most new failures are at later initializations, but later anchors also dominate exposure; counts alone do not establish causal concentration.',
            'Single-start forward-only training is a verified coverage difference, not proof that multi-start or reverse training will improve performance.',
            'Groups overlap: forward/backward and first-frame/all-later are separate partitions. Rates here are anchor counts, not VOT ROB.'],
        source_trajectories_modified=False,new_inference_calls=0,new_training_steps=0,new_caption_calls=0,M79_modified=False,goal_achieved=False)
    write(O/'result.json',report)
    stream=io.StringIO(newline='');columns=list(next(iter(groups.values())))
    writer=csv.writer(stream,lineterminator='\n');writer.writerow(['group']+columns)
    for name,g in groups.items():writer.writerow([name]+[g[k] for k in columns])
    (O/'groups.csv').write_bytes(stream.getvalue().encode('utf-8'))
    note=f'''

## 5.158 M78封存结果补查：方向与初始化位置

M79仍保持§5.157的冻结方案。本项只对M78完整303个anchor及M39配对记录重新分组，检查“退化是否主要由反向跟踪或中途初始化造成”的假设；不读取M79部分指标，不修改任何训练、文本或推理规则。

| 分组 | anchor数 | 原生失败 | M78失败 | 救回 | 新增 | 净增失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    labels={'forward':'全部正向','backward':'全部反向','first_frame_forward':'首帧初始化正向','later_forward':'中途初始化正向','all_later':'全部中途初始化'}
    for name,g in groups.items():note+='| '+labels[name]+' | '+' | '.join(str(g[k]) for k in ['anchors','native_failures','M78_failures','rescued','new_failures','net_added_failures'])+' |\n'
    note+=f'''
正向和反向各净增8个失败，因此“只是没有训练反向序列”不足以解释本轮退化。首帧初始化22个anchor只新增1个失败，中途初始化新增20个，但中途anchor本来就占281/303（92.74%）；不能仅凭20/21的数量占比认定中途初始化具有更高的因果风险。以原生成功anchor为分母，首帧新增失败为1/13，中途为20/166；这仍是不同序列/长度/可见性混合后的描述统计，不是受控训练干预效果，也不是VOT ROB。

同时核对了实际训练源码及两组已完成sequence_log：每组130次序列初始化，源码以frame(0)初始化后执行range(1,n)，每组186694次真实跟踪调用。因此首帧单起点、正向训练的覆盖范围是事实；增加训练起点或反向片段是否有效仍需新的DepthTrack Train配对实验，不能用本统计预先宣称有效。当前先完成M79内容归因，不提前启动新的训练或修改其对照。

训练源码SHA256为`{sha(trainer)}`，对应证据行号见result.json；分组CSV使用完整封存结果，无新增跟踪、caption或优化步骤。两个分组表系相互重叠：正向/反向是一种划分，首帧/中途是另一种，不能把所有表行相加。结果与源码发布于projects/sttrack_lachtt_v1/diagnostics/m78_raw_competition/direction_initialization_diagnostic/。项目目标仍未完成。
'''
    (O/'handoff_append.md').write_bytes(note.encode('utf-8'));files=O/'files';files.mkdir()
    for p in [O/'result.json',O/'groups.csv',O/'handoff_append.md',Path(__file__)]:shutil.copyfile(p,files/p.name)
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())];write(files/'manifest.json',manifest)
    archive=O/'published.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files.iterdir()):tar.add(p,arcname=p.name)
    old=M.read_bytes();assert sha(M)=='97783884dda33eaed8eb51c73387aacc50cfbcef02b5dc1e652754301f6b4edb' and b'\n## 5.158 ' not in old
    M.write_bytes(old+note.encode('utf-8'))
    record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(M),master_bytes=M.stat().st_size,files=len(manifest))
    write(O/'publication_record.json',record);print(json.dumps(dict(report=report,publication=record),indent=2))

if __name__=='__main__':main()
