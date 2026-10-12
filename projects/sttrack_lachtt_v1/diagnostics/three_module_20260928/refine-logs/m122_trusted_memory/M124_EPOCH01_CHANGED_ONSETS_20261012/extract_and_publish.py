"""Reorganize already captured CPU event rows; no SSH or tracker inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess

R = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902')
A = R / '.aris/m122_full_causal_20261007'
D = R / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
P = Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING')
paths = [A / 'CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json', A / 'M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json']
hashes = ['bcdc6791cf67d71b77e08cacad8b91cefed1c7a89260f98ade04183dfe64de72', '238b30bab4d573e3a4bde0ee00e63a235aa6c26fbc9cdc1127ed80dbea47b4f7']
for path, expected in zip(paths, hashes):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
old, new = [json.loads(path.read_bytes()) for path in paths]
assert old['sequences'] == new['sequences'] == 80 and old['frames'] == new['frames'] == 101956
assert new['final_sha256'] == '75aee7ac912b799f9160baadf95ccc6ff406b6b338760a239da6249bfce505db'
before = {row['sequence']: row for row in old['per_sequence'] if row['model'] == 'CurrentC'}
after = {row['sequence']: row for row in new['per_sequence']}
assert len(before) == len(after) == 80 and set(before) == set(after)
names = ['cart_room_occ_1', 'boxes_backpack_room_occ_1', 'robot_corridor_occ_1', 'jug']
rows = [{ 'sequence': name, 'CurrentC': before[name], 'M124_epoch01': after[name]} for name in names]
for item in rows:
    assert item['CurrentC']['frames'] == item['M124_epoch01']['frames']
    assert item['CurrentC']['valid_GT_frames'] == item['M124_epoch01']['valid_GT_frames']
assert [(s['start'], s['end']) for s in before[names[0]]['long_low_IoU_segments']] == [(177,546),(556,593),(607,624),(685,999)]
assert [(s['start'], s['end']) for s in after[names[0]]['long_low_IoU_segments']] == [(178,261),(395,418),(479,506),(522,546),(556,593),(607,624),(685,999)]
assert before[names[1]]['valid_GT_reappearance_events'][0] == dict(frame=196,center_in_crop=False,IoU=0.)
assert after[names[1]]['valid_GT_reappearance_events'][0] == dict(frame=196,center_in_crop=True,IoU=0.)
assert before[names[1]]['serious_low_IoU_frames'] == after[names[1]]['serious_low_IoU_frames'] == 849
assert before[names[2]]['serious_low_IoU_frames'] == 1084 and after[names[2]]['serious_low_IoU_frames'] == 57
assert after[names[2]]['long_low_IoU_segments'][0]['start'] == 1362
assert after[names[2]]['long_low_IoU_segments'][0]['length'] == 54
assert after[names[2]]['long_low_IoU_segments'][0]['center_outside_frames'] == 0
assert before[names[3]]['long_low_IoU_segments'] == []
assert [(s['start'], s['end']) for s in after[names[3]]['long_low_IoU_segments']] == [(286,315),(326,717),(754,874)]
assert before[names[3]]['serious_low_IoU_frames'] == 18 and after[names[3]]['serious_low_IoU_frames'] == 548

pub = D / 'refine-logs/m122_trusted_memory/M124_EPOCH01_CHANGED_ONSETS_20261012'
assert not pub.exists()
pub.mkdir()
scope = 'Copied already captured real-GT CPU rows, no re-reading remote predictions or NN. Segment onset is the beginning of a >=10-frame valid-GT IoU<=0.1 run, not necessarily first causal divergence. Centre inside reconstructed crop does not prove whole target visibility, adequate resolution, a good candidate, correct semantics use or a unique failure cause. Different weights/protocols/own histories, not a module or text ablation. No incomplete epoch02 or VOT result.'
value = dict(status='actual_local_M124_epoch01_changed_CDTB_onsets',generated_at=datetime.now(timezone.utc).isoformat(),input_sha256={str(path):expected for path,expected in zip(paths,hashes)},selected_sequences=rows,NN_calls=0,GPU_queries=0,NN_progress_queries=0,server_queries=0,training_source_changed=False,training_restarted=False,limits=scope)
(pub / 'captured_event_rows.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
note = '''# M124第1轮：失败起点与旧模型不能混用

只重新整理§5.499已捕获的官方GT/保存预测CPU统计，不调用服务器、不运行神经网络、不修改训练。帧号零基，区间右端不包含。以下起点是至少10帧有效GT且IoU≤0.1段的起点，不保证就是首次因果分叉。

| 序列/事件 | CurrentC | M124第1轮 | 诊断含义 |
|---|---|---|---|
| cart_room_occ_1首个长低重叠段 | [177,546)，369帧，起点中心在crop内 | [178,261)，83帧，起点中心在crop内 | 首段缩短，不等于首次跳框已经消除或已经可靠恢复 |
| cart_room_occ_1后续重现段 | [556,593)、[607,624)、[685,999)，均起点中心在crop外 | 三个区间及起点crop外状态相同 | 全框R改善不能掩盖剩余重现失败 |
| boxes_backpack_room_occ_1，第196帧 | GT中心在crop外、IoU=0 | GT中心在crop内、IoU=0；同一[196,509)段313帧，之后301帧中心在外 | 旧模型的“首帧不可达”不能直接套给新权重；中心在内也不证明有合格候选 |
| robot_corridor_occ_1，第549帧 | 中心在外、IoU=0；全序列1084个严重低帧 | 中心在内、IoU=0.007502；全序列57个严重低帧 | 长轨迹确实改善，但不是重现首帧已准确定位 |
| robot_corridor_occ_1，第1362帧起 | [1362,1750)，388帧，中心一直在外 | [1362,1416)，54帧，中心一直在内 | 新权重剩余这段不能仅用中心出crop解释，应查局部候选和选择 |
| jug，第286/326帧重现 | 两帧中心均在crop内，全序列18个严重低帧，无≥10帧低段 | 两帧中心均在crop外；新增[286,315)、[326,717)、[754,874)，全序列548个严重低帧 | 新增损害主要表现为重现时局部中心不可达；需追溯更早的状态变化 |

推车旧模型175→176标注中心约24px、预测约404px的跳转仍只属于CurrentC/P1保存轨迹。没有新权重逐帧框/候选回放证据时，不能把这个404px数值写成M124第1轮也一样。第1轮首个长低段起点由177变为178，只能证明段定义变化，不能定位具体首次内部错误。

“中心在crop内”只描述按上一保存框重建的几何范围；不保证整个目标可见、足够分辨率或候选IoU≥0.5。GT无效→有效也不单独证明物理遮挡/出画。后续应对最终选定权重的自身历史固定同状态比较候选存在、响应分项和精修前后，不把旧权重失败原因复制给新模型。当前CurrentC候选回放仍只准备未执行；新的NN诊断等待整个M124套件终态及GPU释放。

两模型权重、训练协议和递归历史不同，这些统计不是文字或某模块的独立增量。完整指标仍见§5.493/5.499，不重复、拼接或推断新的P/R/F。M124第2轮指标与VOT仍未产生。本次0 SSH、0 GPU、0 NN、0进度查询；原timer34413/PID42640及13:23:59.580152检查节点保持。
'''
(pub / 'README.md').write_bytes(note.encode('utf-8'))
shutil.copyfile(Path(__file__),pub/'extract_and_publish.py')
local = P / '实验诊断/M124第1轮CDTB召回轨迹_20261012'
assert local.is_dir()
for source, target in [(pub/'README.md',local/'失败起点变化说明.md'),(pub/'captured_event_rows.json',local/'四条轨迹失败起点.json')]:
    shutil.copyfile(source,target)
    assert source.read_bytes() == target.read_bytes()
master=R/'docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md'
content=master.read_text(encoding='utf-8')
assert '### 5.500 ' in content and '### 5.501 ' not in content
extra='\n\n### 5.501 M124第1轮失败起点变化：不能套用旧权重原因（2026-10-12，已有CPU记录本地整理）\n\n'+note.split('\n',1)[1]
master.write_bytes((content+extra).encode('utf-8'))
shutil.copyfile(master,P/master.name)
tracker=D/'refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md'
with tracker.open('a',encoding='utf-8') as stream:
    stream.write('\nOct12 sec5.501 local captured E1/C onset comparison only: cart firstlong369->[178,261)83 but three later reappearance ranges unchanged/outside; backpack196 Coutside versusE1inside yetIoU0/849severelow both; robot549 E1insideIoU.0075(notfirstcorrect), late1362 E1low54allinside versusC388outside; jug286/326 newlyoutside/E1low548vs18. No oldcart404px jump transplantedtoE1/no candidatecause or module/textclaim. Four copiedeventrows/rawSHA bound, noSSH/GPU/NN/progressquery/sourcechange/restart; original34413/PID42640/functions1682 due13:23:59.580152 preserved; goalACTIVE_UNMET.\n')
owned=[path.relative_to(R).as_posix() for path in [pub/'captured_event_rows.json',pub/'README.md',pub/'extract_and_publish.py',master,tracker]]
with (R/'.gitattributes').open('a',encoding='utf-8') as stream:
    for name in owned[:3]:stream.write('\n'+name+' -text whitespace=cr-at-eol,-blank-at-eof\n')
owned.append('.gitattributes')
subprocess.run(['git','add','--',*owned],cwd=R,check=True)
for name in owned:
    assert subprocess.check_output(['git','cat-file','blob',':'+name],cwd=R)==(R/name).read_bytes(),name
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().rstrip('\0').split('\0')
assert set(staged)==set(owned)
subprocess.run(['git','diff','--cached','--check'],cwd=R,check=True)
subprocess.run(['git','commit','-m','Distinguish M124 epoch-one failure onsets from protected CurrentC'],cwd=R,check=True)
subprocess.run(['git','push','origin','main'],cwd=R,check=True)
receipt=dict(status='actual_M124_epoch01_changed_onsets_published',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),master_sha256=hashlib.sha256(master.read_bytes()).hexdigest(),owned_paths=owned,remote_synced=False,NN_calls=0,GPU_queries=0,NN_progress_queries=0,training_source_changed=False,generated_at=datetime.now(timezone.utc).isoformat())
(A/'M124_epoch01_changed_onsets_publication_20261012.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
state_path=A/'CURRENT_STATUS.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(M124_epoch01_changed_onsets_publication=receipt)
state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
