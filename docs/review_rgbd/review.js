import {serializeCSV, parseCSV} from './review-csv.mjs?v=20261007categories';
import {answerStorageKey, savedAnswer, INITIAL_ROUND} from './review-state.mjs?v=20261007categories';
const qs = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const datasets = ['depthtrack', 'depthtrack_test', 'cdtb', 'vot'];
const cache = new Map();
const statusText = {supported:'有图像支持', corrected:'建议修正', conflicting:'存在冲突', uncertain:'不确定',not_visualized:'尚未实际看图'};
let manifest, cases = [], current = 0, detailImage;

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function reviewer() { return $('reviewer').value.trim(); }
function storageKey(item, dataset = manifest.dataset, name = reviewer(), round = cache.get(dataset).human_review_round) {
  return answerStorageKey(dataset,name,item.id,round);
}
function saved(item, dataset = manifest.dataset) {
  return savedAnswer(item,cache.get(dataset),reviewer(),localStorage);
}
function message(text, error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}
function rememberReviewer(name) {
  const names = new Set(JSON.parse(localStorage.getItem('rgbd-reviewer-list') || '[]'));
  if (name) names.add(name);
  localStorage.setItem('rgbd-reviewer-list', JSON.stringify([...names]));
  $('reviewer-list').innerHTML = [...names].map(value => `<option value="${escapeHTML(value)}"></option>`).join('');
}
function bounds() {
  return [Math.max(1, Number($('from').value) || 1), Math.min(manifest.count, Number($('to').value) || manifest.count)];
}
function updateProgress() {
  const count = reviewer() ? manifest.rows.filter(item => saved(item).status).length : 0;
  $('progress').textContent = reviewer() ? `我的人工审核 ${count} / ${manifest.count}` : '填写审核者编号，开始人工核对';
  $('progress-bar').value = count / manifest.count * 100;
  const uncertain = manifest.rows.filter(item => item.model_review.status === 'uncertain').length;
  const submitted = manifest.rows.filter(item => Object.keys(item.human_reviews).length).length;
  $('model-progress').textContent = `GPT ${manifest.count}条 · 不确定 ${uncertain} · 已提交人审 ${submitted}/${manifest.count}`;
}
function applyFilter(preferredId) {
  if (!manifest) return;
  const [first, last] = bounds();
  const query = $('search').value.trim().toLowerCase(), filter = $('filter').value;
  cases = manifest.rows.filter(item => {
    const model = item.model_review;
    const matches = !query || `${item.number} ${item.sequence} ${item.id} ${item.category} ${model.category} ${model.evidence}`.toLowerCase().includes(query);
    const answer = reviewer() ? saved(item) : {};
    const included = filter === 'all' || (filter === 'pending' && !answer.status) ||
      (filter === 'reviewed' && answer.status) || (filter === 'model_uncertain' && model.status === 'uncertain') ||
      (filter === 'model_changed' && ['corrected','conflicting'].includes(model.status));
    return item.number >= first && item.number <= last && matches && included;
  });
  current = Math.max(0, cases.findIndex(item => item.id === preferredId));
  render();
}
function shareState() {
  qs.set('dataset', $('dataset').value);
  qs.set('from', $('from').value || '1');
  qs.set('to', $('to').value || String(manifest.count));
  qs.set('filter', $('filter').value);
  if ($('search').value) qs.set('search', $('search').value); else qs.delete('search');
  if (cases[current]) qs.set('item', cases[current].id); else qs.delete('item');
  history.replaceState(null, '', `?${qs}`);
}
function render() {
  $('previous').disabled = current <= 0;
  $('next').disabled = !cases.length || current >= cases.length - 1;
  $('position').textContent = cases.length ? `${current + 1} / ${cases.length} 项 · 全集 ${manifest.count}项` : '没有匹配的样本';
  updateProgress();
  if (!cases.length) { $('case').innerHTML = '<div class="empty-state">当前范围内没有匹配样本。可以切换筛选或调整范围。</div>'; return; }
  const item = cases[current], model = item.model_review, answer = reviewer() ? saved(item) : {};
  const humanStatus = answer.status ? `我的判断：${{supported:'支持',conflicting:'冲突',uncertain:'不确定'}[answer.status]}` : '本轮待人工确认';
  const submitted = Object.entries(item.human_reviews);
  const proposalReview = manifest.review_subject === 'gpt_proposal';
  $('case').innerHTML = `
    <div class="case-head"><div><h2>${escapeHTML(item.sequence)} <span class="muted">#${item.number}</span></h2><div class="muted">${escapeHTML(manifest.display_name)} · 初始化帧 ${escapeHTML(item.frame)} · 序列共 ${item.sequence_frames}帧${proposalReview ? ' · 本轮重新核对GPT提议' : ''}</div><div class="key">${escapeHTML(item.id)}</div></div><span id="human-status" class="badge">${humanStatus}</span></div>
    <div class="evidence"><div>
      <img class="board" id="board" src="${escapeHTML(item.board)}" alt="初始化红框、目标放大区与无标记邻近帧，点击查看清晰图" loading="lazy" decoding="async" tabindex="0">
      <div class="board-tools"><button type="button" id="open-board">加载高清图 / 放大查看</button><a href="${escapeHTML(item.board_detail)}" target="_blank" rel="noopener">原尺寸拼图 ↗</a></div>
      ${item.temporal_board ? `<details class="temporal-evidence"><summary>查看GPT用到的多帧图</summary><p>${escapeHTML(item.temporal_board_label)}</p><p>零基帧：${escapeHTML(item.temporal_board_frames)}</p><a href="${escapeHTML(item.temporal_board)}" target="_blank" rel="noopener"><img data-src="${escapeHTML(item.temporal_board)}" alt="多帧审核证据，点击查看原尺寸" loading="lazy"></a></details>` : ''}
      ${item.original_frames ? `<details class="original-description"><summary>打开原始JPEG核对细节</summary><p>${item.original_frames.map(frame => `<a href="${escapeHTML(frame.path)}" target="_blank" rel="noopener">帧 ${frame.frame_index}</a>`).join(' · ')}</p></details>` : ''}
      <div class="video-area"><button type="button" id="load-video" class="secondary">▶ 查看整段范围的抽样视频</button><p>${item.video_sample_fps || 8}fps抽样预览；后帧无目标标注，仅作审核上下文。</p><video id="video" class="video" controls preload="none" hidden></video></div>
      ${manifest.reference_caption_available ? `<details class="original-description"><summary>原始自动描述 · 与GPT意见对照</summary><p>类别：<strong>${escapeHTML(item.category)}</strong></p><p>属性：${escapeHTML(item.attributes.join(' / ') || '无')}</p></details>` : '<p class="muted">本次未提供原自动caption；核对右侧GPT提出的首帧描述。</p>'}
    </div><div>
      <section class="model-card"><div class="section-heading"><h3>GPT初审意见</h3><span class="badge ${model.status}">${statusText[model.status]}</span></div>
      <p class="category-label">${escapeHTML(model.category)}</p>
      <p class="model-line"><strong>初始可见属性</strong><br>${escapeHTML(model.stable_attributes || '未确认')}</p>
      <p class="model-line"><strong>仍需核对</strong><br>${escapeHTML(model.uncertain_attributes || '无额外记录')}</p>
      ${model.conflicting_attributes ? `<p class="model-line"><strong>属性冲突</strong><br>${escapeHTML(model.conflicting_attributes)}</p>` : ''}
      <details class="model-evidence"><summary>展开判断依据与多帧备注</summary><p>${escapeHTML(model.evidence)}</p>${model.evidence_frames ? `<p>参考帧：${escapeHTML(model.evidence_frames)}</p>` : ''}${model.initialization_observability ? `<p>初始化可辨性：${escapeHTML(model.initialization_observability)}</p>` : ''}<p><strong>多帧上下文（不自动作为初始化属性）</strong><br>${escapeHTML(model.context_only_notes || '无额外记录')}</p></details>
      <p class="model-source">${escapeHTML(model.reviewer)} · ${escapeHTML(model.review_date)} · GPT提议需由人工核对</p></section>
      ${submitted.length ? `<section class="submitted-review"><h3>已提交的人工核对</h3>${submitted.map(([name,record]) => `<div><strong>${escapeHTML(name)} · ${{supported:'支持',conflicting:'需修正／有冲突',uncertain:'不确定'}[record.status]}</strong><p>${escapeHTML(record.confirmed_category)}</p><p>首帧属性：${escapeHTML(record.stable_attributes || '未记录')}</p><p>不确定：${escapeHTML(record.uncertain_attributes || '未记录')}</p>${record.note ? `<p>${escapeHTML(record.note)}</p>` : ''}</div>`).join('')}</section>` : ''}
      <form id="review-form" class="review-form"><h3>我的人工核对</h3><p class="muted">${proposalReview ? '本轮请核对GPT重审提议。' : '可参考已提交记录继续核对。'}只有明确选择判断才计入已审核。</p>
      <div class="form-grid">
        <label>确认类别<input name="confirmed_category" value="${escapeHTML(answer.confirmed_category)}" placeholder="无法确认可留空"></label>
        <label>初始化可见属性<input name="stable_attributes" value="${escapeHTML(answer.stable_attributes)}" placeholder="例如 striped; white"></label>
        <label class="wide">不确定或暂时不可见<input name="uncertain_attributes" value="${escapeHTML(answer.uncertain_attributes)}"></label>
        <label class="wide">依据与备注<textarea name="note" placeholder="哪些帧帮助判断？哪些细节无法确认？">${escapeHTML(answer.note)}</textarea></label>
      </div>
      <div class="status-options" role="group" aria-label="人工判断">
        <label><input type="radio" name="status" value="supported" ${answer.status === 'supported' ? 'checked' : ''}>${proposalReview ? 'GPT提议有支持' : '原描述支持目标'}</label>
        <label><input type="radio" name="status" value="conflicting" ${answer.status === 'conflicting' ? 'checked' : ''}>需修正／有冲突</label>
        <label><input type="radio" name="status" value="uncertain" ${answer.status === 'uncertain' ? 'checked' : ''}>不确定</label>
      </div><div class="form-actions"><button type="button" id="use-model" class="secondary">采用初审内容</button><button type="button" id="save-next">保存并下一项 →</button></div><p class="saved" id="saved-state">${answer.status ? (answer._published ? '已载入你提交的人工记录；修改后保存在当前浏览器' : '本轮人工判断已保存在当前浏览器') : '选择判断后自动保存；草稿不计入已审核'}</p></form>
    </div></div>`;
  $('open-board').addEventListener('click', () => openEvidence().catch(error => message(error.message, true)));
  $('board').addEventListener('click', () => $('open-board').click());
  $('board').addEventListener('keydown', event => { if (event.key === 'Enter') $('open-board').click(); });
  $('load-video').addEventListener('click', () => {
    const player = $('video'); player.hidden = false; player.src = item.video;
    player.addEventListener('loadedmetadata', () => { player.currentTime = Math.min(item.video_start_seconds, player.duration); }, {once:true});
    player.load(); $('load-video').textContent = '视频已加载 · 下方播放';
  });
  $('review-form').addEventListener('input', saveCurrent);
  document.querySelectorAll('.temporal-evidence').forEach(details => details.addEventListener('toggle', () => {
    if (details.open) { const image = details.querySelector('img'); image.src = image.dataset.src; }
  }, {once:true}));
  $('review-form').addEventListener('submit', event => event.preventDefault());
  $('use-model').addEventListener('click', () => {
    const form = $('review-form');
    form.elements.confirmed_category.value = model.category;
    form.elements.stable_attributes.value = model.stable_attributes;
    form.elements.uncertain_attributes.value = model.uncertain_attributes;
    saveCurrent();
  });
  $('save-next').addEventListener('click', () => {
    if (!reviewer()) { $('reviewer').focus(); message('请先填写审核者编号。', true); return; }
    if (!new FormData($('review-form')).get('status')) { message('请明确选择支持、冲突或不确定。', true); return; }
    saveCurrent();
    const nextId = cases[current + 1]?.id;
    applyFilter(nextId || item.id);
    if (!nextId) message('已到当前范围最后一项，可导出审核结果。');
  });
}
async function openEvidence() {
  const item = cases[current], button = $('open-board');
  button.disabled = true;
  detailImage = new Image(); detailImage.src = item.board_detail;
  try { await detailImage.decode(); } finally { button.disabled = false; }
  $('detail-region').value = 'target'; $('detail-zoom').value = '1';
  $('detail-title').textContent = `${item.sequence} · #${item.number} · 高清图`;
  drawEvidence(); $('evidence-dialog').showModal();
}
function drawEvidence() {
  const regions = {target:[640,0,320,380], initialization:[0,0,640,380], context:[0,380,960,185], full:[0,0,960,565]};
  const scale = detailImage.naturalWidth / 960;
  const [x,y,width,height] = regions[$('detail-region').value].map(value => value * scale);
  const canvas = $('detail-canvas'), zoom = Number($('detail-zoom').value);
  canvas.width = width; canvas.height = height;
  canvas.style.width = `${width * zoom}px`; canvas.style.height = `${height * zoom}px`;
  canvas.getContext('2d').drawImage(detailImage,x,y,width,height,0,0,width,height);
  $('detail-scroll').scrollTop = 0; $('detail-scroll').scrollLeft = 0;
}
function saveCurrent() {
  if (!reviewer()) { $('saved-state').textContent = '请先填写审核者编号，才能保存。'; return; }
  const item = cases[current], form = new FormData($('review-form'));
  const answer = {
    status:form.get('status') || '', confirmed_category:String(form.get('confirmed_category') || '').trim(),
    stable_attributes:String(form.get('stable_attributes') || '').trim(),
    uncertain_attributes:String(form.get('uncertain_attributes') || '').trim(),
    note:String(form.get('note') || '').trim(), updated_at:new Date().toISOString()
  };
  localStorage.setItem(storageKey(item), JSON.stringify(answer));
  $('saved-state').textContent = answer.status ? '人工判断已保存在当前浏览器' : '草稿已保存，尚未人工确认';
  $('human-status').textContent = answer.status ? '已记录人工判断' : '待人工确认';
  updateProgress(); message('');
}
async function fetchManifest(dataset) {
  if (!cache.has(dataset)) {
    const response = await fetch(`data/${dataset}.json?v=20261007categories`);
    if (!response.ok) throw new Error(`无法读取${dataset}清单：HTTP ${response.status}`);
    cache.set(dataset, await response.json());
  }
  return cache.get(dataset);
}
async function exportCSV() {
  if (!reviewer()) { message('请先填写审核者编号。', true); return; }
  const scope = $('export-scope').value;
  const selected = scope === 'all' ? await Promise.all(datasets.map(fetchManifest)) : [manifest];
  const header = ['reviewer_id','dataset','anchor_number','key','sequence','init_frame','status','confirmed_category','stable_attributes','uncertain_attributes','note','provisional_category','provisional_attributes','updated_at','human_confirmed','model_status','model_category','model_stable_attributes','model_uncertain_attributes','model_context_only_notes','model_evidence','model_reviewer','review_round','review_subject'];
  const rows = [];
  for (const dataset of selected) for (const item of dataset.rows) {
    const answer = saved(item,dataset.dataset), model = item.model_review;
    if (!answer.status && !answer.updated_at) continue;
    rows.push([reviewer(),dataset.dataset,item.number,item.id,item.sequence,item.frame,answer.status,answer.confirmed_category,answer.stable_attributes,answer.uncertain_attributes,answer.note,item.category,item.attributes.join('; '),answer.updated_at,Boolean(answer.status),model.status,model.category,model.stable_attributes,model.uncertain_attributes,model.context_only_notes,model.evidence,model.reviewer,dataset.human_review_round,dataset.review_subject]);
  }
  if (!rows.length) { message('还没有可导出的人工判断或草稿。', true); return; }
  const url = URL.createObjectURL(new Blob([serializeCSV(header,rows)], {type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url;
  link.download = `rgbd_${scope === 'all' ? 'all' : manifest.dataset}_${reviewer()}_${rows.length}_review.csv`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  message(`已导出${rows.length}条记录。草稿的human_confirmed为false，未计为人工确认。`);
}
async function importCSV(file) {
  const rows = parseCSV(await file.text());
  if (!rows.length) throw new Error('CSV没有审核记录。');
  const required = ['reviewer_id','dataset','key','sequence','init_frame','status','confirmed_category','stable_attributes','uncertain_attributes','note','updated_at'];
  if (!required.every(field => field in rows[0])) throw new Error('请导入本网站导出的人工审核CSV；GPT初审已自动导入。');
  const manifests = await Promise.all([...new Set(rows.map(row => row.dataset))].map(dataset => {
    if (!datasets.includes(dataset)) throw new Error(`未知数据集：${dataset}`);
    return fetchManifest(dataset);
  }));
  const lookup = new Map(manifests.flatMap(dataset => dataset.rows.map(item => [`${dataset.dataset}:${item.id}`,item])));
  const plans = [], unique = new Set(), names = new Set();
  for (const row of rows) {
    const item = lookup.get(`${row.dataset}:${row.key}`);
    if (!item || row.sequence !== item.sequence || row.init_frame !== item.frame) throw new Error(`初始化点不匹配：${row.dataset}/${row.key}`);
    if (!row.reviewer_id.trim() || !['','supported','conflicting','uncertain'].includes(row.status)) throw new Error('审核者编号或判断值无效。');
    if (!Number.isFinite(Date.parse(row.updated_at))) throw new Error(`记录缺少有效保存时间：${row.key}`);
    const round = row.review_round || INITIAL_ROUND;
    const key = storageKey(item,row.dataset,row.reviewer_id.trim(),round);
    if (unique.has(key)) throw new Error(`同一审核者的记录重复：${row.key}`);
    unique.add(key); names.add(row.reviewer_id.trim());
    plans.push({key, answer:Object.fromEntries(['status','confirmed_category','stable_attributes','uncertain_attributes','note','updated_at'].map(field => [field,row[field]]))});
  }
  let imported = 0, skipped = 0;
  for (const {key,answer} of plans) {
    const existing = JSON.parse(localStorage.getItem(key) || '{}');
    if (existing.updated_at && Date.parse(existing.updated_at) >= Date.parse(answer.updated_at)) { skipped++; continue; }
    localStorage.setItem(key, JSON.stringify(answer)); imported++;
  }
  for (const name of names) rememberReviewer(name);
  if (names.size === 1) {
    $('reviewer').value = [...names][0]; localStorage.setItem('rgbd-reviewer-id',reviewer());
  }
  applyFilter(cases[current]?.id);
  message(`导入${imported}条；保留已有较新记录${skipped}条。${names.size > 1 ? '可在审核者编号中切换查看。' : ''}`);
}
async function loadDataset(dataset) {
  $('previous').disabled = true; $('next').disabled = true;
  const loaded = await fetchManifest(dataset);
  if ($('dataset').value !== dataset) return;
  manifest = loaded;
  for (const name of manifest.published_reviewers) rememberReviewer(name);
  $('from').value = qs.get('from') || 1; $('to').value = qs.get('to') || manifest.count;
  $('from').max = manifest.count; $('to').max = manifest.count;
  document.querySelectorAll('.dataset-card').forEach(card => {
    card.classList.toggle('active',card.dataset.dataset === dataset);
    card.setAttribute('aria-pressed',String(card.dataset.dataset === dataset));
  });
  applyFilter(qs.get('item'));
}
function selectDataset(dataset) {
  $('dataset').value = dataset; qs.set('dataset',dataset); qs.delete('from'); qs.delete('to'); qs.delete('item');
  history.replaceState(null,'',`?${qs}`); message('');
  loadDataset(dataset).catch(error => message(error.message,true));
}
$('reviewer').value = localStorage.getItem('rgbd-reviewer-id') || '';
rememberReviewer(reviewer());
$('dataset').value = datasets.includes(qs.get('dataset')) ? qs.get('dataset') : 'depthtrack';
$('search').value = qs.get('search') || '';
if ([...$('filter').options].some(option => option.value === qs.get('filter'))) $('filter').value = qs.get('filter');
$('reviewer').addEventListener('change', () => {
  localStorage.setItem('rgbd-reviewer-id',reviewer()); rememberReviewer(reviewer()); applyFilter(cases[current]?.id);
});
$('dataset').addEventListener('change', () => selectDataset($('dataset').value));
document.querySelectorAll('.dataset-card').forEach(card => card.addEventListener('click', () => selectDataset(card.dataset.dataset)));
$('apply').addEventListener('click', () => { applyFilter(); shareState(); });
$('share').addEventListener('click', async () => {
  shareState(); await navigator.clipboard.writeText(location.href); message('已复制当前样本、筛选和范围的审核链接。');
});
$('search').addEventListener('input', () => applyFilter());
$('filter').addEventListener('change', () => applyFilter());
$('previous').addEventListener('click', () => { current--; render(); });
$('next').addEventListener('click', () => { current++; render(); });
$('export').addEventListener('click', () => exportCSV().catch(error => message(error.message,true)));
$('import').addEventListener('click', () => $('import-file').click());
$('import-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (file) await importCSV(file).catch(error => message(error.message,true));
  event.target.value = '';
});
$('close-evidence').addEventListener('click', () => $('evidence-dialog').close());
$('detail-region').addEventListener('change',drawEvidence);
$('detail-zoom').addEventListener('change',drawEvidence);
loadDataset($('dataset').value).catch(error => { $('case').textContent = error.message; });
