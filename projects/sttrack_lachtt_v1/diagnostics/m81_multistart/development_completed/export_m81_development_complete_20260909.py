from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil, tarfile

B = Path('/root/autodl-tmp')
R = B / 'sttrack_m81_multistart_20260908'
O = R / 'development_completed_publication'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
master = Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
old = master.read_bytes()
assert sha(master) == '5cca5a9ee498d89f599476d1e9c21815f0ec4fce32b8f90c1e185eb3b76dca85'
assert b'\n## 5.168 ' not in old and not O.exists()
result = read(R / 'result.json')
audit = read(R / 'saved_development_audit.json')
contrasts = read(R / 'outcome_contrasts.json')
assert contrasts['result_sha256'] == sha(R / 'result.json') == '586437dcd37772d2c1ae73714a593d1e15c80e0d67b12aee7ae07ec3d17f365a'
assert result['status'] == 'complete_M81_t0_and_multistart_development'
assert audit['status'] == 'complete_independent_scalar_M81_metric_and_receipt_verification'
assert audit['result_sha256'] == sha(R / 'result.json')
assert audit['source_sha256'] == sha(B / 'audit_m81_development_complete_20260908.py') == 'd6bf14dad0809ff16961cfa9ea848785575beae0e39f31f91d304d2c96698975'
assert (audit['families'], audit['episodes'], audit['positions']) == (9, 458, 298430)
assert audit['noninitialization_frame_coverage_exactly_once'] and audit['valid_positions_per_family'] == 28897
families = ['t0_category', 't0_empty_trained', 't0_empty_content', 't0_swapped', 'multi_category', 'multi_empty_trained', 'multi_native', 'multi_M78_category', 'multi_M78_empty_trained']
for name in ['training_category', 'training_empty', 'analysis', 'controller'] + ['eval_' + f for f in families]:
    assert (R / (name + '.exit')).read_text().strip() == '0'
for f in families:
    assert sha(R / ('receipt_' + f + '.json')) == result['receipt_sha256'][f]
gate_groups = [('primary_gates', 'primary_pass_count', 20), ('content_gates', 'content_pass_count', 8), ('initialization_coverage_gates', 'coverage_pass_count', 4)]
for key, count_key, expected in gate_groups:
    assert len(result[key]) == expected and sum(result[key].values()) == audit[count_key]
status = dict(observed_utc=datetime.now(timezone.utc).isoformat(), result_sha256=sha(R / 'result.json'), all_checks_pass=result['all_checks_pass'], free_disk_bytes=shutil.disk_usage(R).free, independent_model_review_pass=False)

def table(entries):
    lines = ['| 条件 | 按有效帧平均IoU | 序列等权平均IoU | IoU≤0.1帧 | H10段 |', '| --- | ---: | ---: | ---: | ---: |']
    for key, label in entries:
        a = result['aggregates'][key]
        lines.append('| %s | %.6f | %.6f | %d | %d |' % (label, a['mean_iou'], a['macro_sequence_mean_iou'], a['low_iou_frames'], a['failure_episodes']))
    return '\n'.join(lines)

t0 = table([('t0_native', '原生STTrack'), ('t0_category', 'M81 Category'), ('t0_empty_trained', 'M81独立Empty训练'), ('t0_empty_content', 'M81同权重Empty内容'), ('t0_swapped', 'M81同权重Swapped内容')])
multi = table([('multi_native', '原生STTrack'), ('multi_M78_category', 'M78 Category'), ('multi_M78_empty_trained', 'M78独立Empty训练'), ('multi_category', 'M81 Category'), ('multi_empty_trained', 'M81独立Empty训练')])
outcome = '全部通过' if result['all_checks_pass'] else '未全部通过'
note = '''

## 5.168 M81首帧与多起点开发评测完成，独立复算结果

本节记录§5.162冻结的M81完成态结果，模型结构、训练计划及§5.167的训练产物不重复记录。九组评测、analysis与controller均正常结束。全部轨迹封存后，独立CPU核验复算了458条episode、298430个位置、每组28897个有效非初始化位置，以及全部32项冻结条件。核验未重新推理或优化，不等同于独立模型审阅通过。

以下均为反复使用的DepthTrack Train开发22，首帧与多起点是不同协议，不是DepthTrack Test、CDTB或VOT正式指标。

首帧协议，每组22条完整轨迹；原生参照复用已绑定的封存结果：

''' + t0 + '''

多起点协议，每组74条episode，五组共享初始化边界，非初始化物理帧各覆盖一次：

''' + multi + '''

H10在共同episode边界及无效GT处中断，多起点H10不能写成VOT ROB，也不能将两套协议H10直接相减来宣称恢复收益。

''' + '冻结条件：主条件%d/20、同权重内容条件%d/8、初始化覆盖增量条件%d/4；总体%s。逐项条件与成功序列保护受损名单随原始result.json封存，未根据结果修改判定规则。\n\n' % (audit['primary_pass_count'], audit['content_pass_count'], audit['coverage_pass_count'], outcome) + '''
Category与独立Empty训练比较、同Category权重下的内容干预、M81与M78的多起点比较分别回答不同问题。Empty内容不等于移除adapter或恢复原生历史轨迹。训练中增加固定初始化同时改变参照视角和递归历史长度，不能仅凭聚合差值将全部作用归因于某个语义绑定机制。

完成态判断：M81没有通过本轮晋升条件，不进入新的正式外部评测。首帧Category相对原生按帧IoU下降0.709478个百分点、宏平均下降2.911514个百分点，低重叠增加374帧、H10增加9段；相对原M78首帧Category下降7.387710个百分点，H10从73增至84。当前约512帧固定重建状态的多起点训练没有保住原M78的首帧完整递归收益，不能宣称初始化覆盖已解决泛化。

在相同74个episode的多起点评测中，M81 Category相对M78 Category按帧下降0.160510个百分点、宏平均下降1.661238个百分点，低重叠增加88帧、H10增加9段，覆盖增量四项均未通过。原生与本轮独立Empty均优于Category的四项聚合指标。与此同时，M81独立Empty相对M78独立Empty按帧提高1.025576个百分点、宏平均提高2.582511个百分点，低重叠减少323帧、H10减少2段；这一条件具有局部正结果，不能把多起点训练概括为对所有条件都无效，也不能把Empty的改善归给文字。

同Category权重下，原类别相对Empty内容仅按帧提高0.326100个百分点，宏平均反而下降4.405939个百分点，低重叠增加57帧、H10增加2段。移除notebook02后，剩余序列的按帧类别增量为-4.708350个百分点。原类别相对Swapped则四项均更好，按帧提高4.971379个百分点，移除任一单序列后该对比仍为正，最小增量0.373922个百分点。这支持具体内容敏感性，但未形成全面优于空词的稳定增量；不能把内容条件5/8误写成完全不使用文字或全部通过。

成功轨迹保护：首帧协议相对原生和本轮独立Empty的零H10序列保护均通过；多起点协议两者都因notebook02失败。该序列多起点原生IoU为0.870504、H10=0，本轮Category为0.871244、H10=1，说明均值微升与新增持续失败可以并存。glass03首帧Category为0.169020/H10=5，原生为0.757594/H10=3；同权重Empty为0.584220/H10=2。cup08则存在正例，首帧原生0.831395/H10=1变为Category0.900454/H10=0。以上属于当前权重轨迹，不套用旧M65图片的因果解释。

下一步研究选择：不晋升M81多起点训练作为新的Category基础，保留其数据覆盖负结果与Empty正例。按用户新要求，下一项最小配对采用M78的首帧、Raw竞争训练协议及相同初始状态，研究可靠训练样本上的同状态原生空间功能保持；复用匹配的M78 Category/Empty封存对照，不以M78训练后权重替代原定零残差初始化。固定seed2027，不增加反向训练、dropout、记忆模块、Hann改动或模板阈值调整。具体损失归一化、可靠性判据、系数和预定评测条件需在新训练前写入独立实验计划，本节不声称已经启动或验证该方法。

源码可行性已经确认：现有一次冻结STTrack前向已返回未适配score/size/offset，教师可直接复用，无需额外ViT或Head。教师参考框必须按Hann响应选峰、用预测前previous_bbox还原，GT可靠性只在适配预测与状态提交后用于训练；教师不提交bbox、query或模板。仅保护同状态空间输出不等于恢复独立原生历史，归一化空间KL也不约束0.75模板门所依赖的绝对分数。PromptSRC官方trainers/promptsrc.py的非AMP路径使用特征L1及输出KL（sum后除以logits.numel），CoPrompt官方trainers/coprompt.py的cosine配置使用双侧特征余弦一致性；这里只借鉴功能保持原则，不直接搬用分类系数或声称KL本身为新增贡献。固定参考提交分别为bb95c77b634d63488f2cad81ff4a72d53bdd06d5与a2a6c12e3622cfe4b3a127f4ba0c4f3eb841418c。

完整汇总、九份receipt、逐序列及逐episode CSV、冻结条件CSV和独立核验材料发布于diagnostics/m81_multistart/development_completed/。最终权重仍为§5.167的固定产物，没有按开发结果重新挑选checkpoint。当前未自动进入外部全量评测，没有新的正式三数据集成绩；后续方案需根据初始化覆盖、内容增量及持续损害分别决定。

''' + '结果SHA256：`%s`。独立核验完成时间：`%s`。\n' % (sha(R / 'result.json'), audit['observed_utc'])

O.mkdir()
F = O / 'files'
F.mkdir()
sources = [(R / 'result.json', 'result.json'), (R / 'saved_development_audit.json', 'saved_development_audit.json'), (B / 'audit_m81_development_complete_20260908.py', 'audit_m81_development_complete_20260908.py'), (Path(__file__), Path(__file__).name)]
sources += [(R / ('receipt_' + f + '.json'), 'receipt_' + f + '.json') for f in families]
sources += [(R / 'outcome_contrasts.json', 'outcome_contrasts.json'), (B / 'report_m81_outcome_20260909.py', 'report_m81_outcome_20260909.py')]
for src, name in sources:
    shutil.copyfile(src, F / name)
(F / 'verified_status.json').write_text(json.dumps(status, indent=2) + '\n')
(F / 'handoff_append.md').write_text(note)
with (F / 'per_sequence.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['condition', 'sequence', 'valid_frames', 'mean_iou', 'low_iou_frames', 'H10_episodes'])
    for family, seqs in result['per_sequence'].items():
        for seq, a in seqs.items():
            writer.writerow([family, seq, a['valid_frames'], a['mean_iou'], a['low_iou_frames'], a['failure_episodes']])
with (F / 'per_episode.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['condition', 'episode', 'valid_frames', 'mean_iou', 'low_iou_frames', 'H10_episodes', 'invalid_gt_frames'])
    for family, episodes in result['episode_metrics'].items():
        for episode, a in episodes.items():
            writer.writerow([family, episode, a['valid_frames'], a['mean_iou'], a['low_iou_frames'], a['failure_episodes'], a['invalid_gt_frames']])
with (F / 'gates.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['group', 'condition', 'passed'])
    for key, _, _ in gate_groups:
        for name, passed in result[key].items():
            writer.writerow([key, name, passed])
manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(F.iterdir())]
(F / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = O / 'published.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(F.iterdir()):
        tar.add(p, arcname=p.name)
master.write_bytes(old + note.encode('utf-8'))
record = dict(archive_sha256=sha(archive), master_sha256=sha(master), master_bytes=master.stat().st_size, files=len(manifest))
(O / 'publication_record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
