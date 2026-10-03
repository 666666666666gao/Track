const qs = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
let manifest, cases = [], current = 0, detailImage;

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function reviewer() { return $('reviewer').value.trim(); }
function storageKey(item) { return `rgbd-review-v1:${manifest.dataset}:${reviewer()}:${item.id}`; }
function saved(item) { return JSON.parse(localStorage.getItem(storageKey(item)) || '{}'); }
function bounds() {
  const first = Math.max(1, Number($('from').value) || 1);
  const last = Math.min(manifest.count, Number($('to').value) || manifest.count);
  return [first, last];
}
function applyFilter() {
  const [first, last] = bounds();
  const query = $('search').value.trim().toLowerCase();
  cases = manifest.rows.filter(item => item.number >= first && item.number <= last &&
    (!query || `${item.number} ${item.sequence} ${item.id}`.toLowerCase().includes(query)));
  current = 0;
  render();
}
function applyAndShare() {
  qs.set('dataset', $('dataset').value);
  qs.set('from', $('from').value || '1');
  qs.set('to', $('to').value || String(manifest.count));
  history.replaceState(null, '', `?${qs}`);
  applyFilter();
}
function render() {
  $('previous').disabled = current <= 0;
  $('next').disabled = current >= cases.length - 1;
  $('position').textContent = cases.length ? `${current + 1} / ${cases.length}` : '没有匹配的样本';
  $('progress').textContent = reviewer() ? `${manifest.rows.filter(item => saved(item).status).length} / ${manifest.count} 已保存` : '先填写审核者编号';
  if (!cases.length) { $('case').innerHTML = ''; return; }
  const item = cases[current], answer = reviewer() ? saved(item) : {};
  $('case').innerHTML = `
    <div class="case-head"><div><h2>${escapeHTML(item.sequence)} · #${item.number}</h2><div class="muted">${escapeHTML(manifest.dataset.toUpperCase())} | 初始化帧 ${escapeHTML(item.frame)} | 序列共 ${item.sequence_frames} 帧</div></div><small class="muted">key ${escapeHTML(item.id)}</small></div>
    <div class="evidence"><div><img class="board" src="${escapeHTML(item.board)}" alt="初始目标红框、放大图和无标记邻近帧" loading="lazy" decoding="async"><div class="board-tools"><button type="button" id="open-board">加载清晰图，查看目标与前后帧</button><a href="${escapeHTML(item.board_detail)}" target="_blank" rel="noopener">在新窗口打开清晰拼图</a></div><div class="video-area"><button type="button" id="load-video">加载 640 像素清晰视频并跳到初始化帧</button><video id="video" class="video" controls preload="none"></video></div></div>
    <div><section class="description"><h3>待审核的自动描述</h3><p>类别：<code>${escapeHTML(item.category)}</code></p><p>属性：${escapeHTML(item.attributes.join(' / ') || '无')}</p><p class="muted">这不是人工真值。红框只在初始化帧出现，邻近帧和视频不使用未来标注。</p></section>
    <form id="review-form" class="review-form"><h3>我的判断</h3><div class="status-options">
      <label><input type="radio" name="status" value="supported" ${answer.status === 'supported' ? 'checked' : ''}>描述支持目标</label>
      <label><input type="radio" name="status" value="conflicting" ${answer.status === 'conflicting' ? 'checked' : ''}>描述指错／冲突</label>
      <label><input type="radio" name="status" value="uncertain" ${answer.status === 'uncertain' ? 'checked' : ''}>无法确认</label>
    </div><div class="form-grid">
      <label>确认的目标类别<input name="confirmed_category" value="${escapeHTML(answer.confirmed_category)}" placeholder="例如 mug；无法确认可留空"></label>
      <label>稳定且可见的属性<input name="stable_attributes" value="${escapeHTML(answer.stable_attributes)}" placeholder="例如 striped, white"></label>
      <label class="wide">不确定或暂时不可见的属性<input name="uncertain_attributes" value="${escapeHTML(answer.uncertain_attributes)}"></label>
      <label class="wide">判断依据与备注<textarea name="note" placeholder="写明哪几帧帮助判断；不确定原因">${escapeHTML(answer.note)}</textarea></label>
    </div><p class="saved" id="saved-state">${answer.status ? '已保存在当前浏览器' : '选择判断后自动保存在当前浏览器'}</p></form></div></div>`;
  $('open-board').addEventListener('click', openEvidence);
  $('load-video').addEventListener('click', () => {
    const player = $('video');
    player.src = item.video;
    player.addEventListener('loadedmetadata', () => { player.currentTime = Math.min(item.video_start_seconds, player.duration); }, {once:true});
    player.load();
  });
  $('review-form').addEventListener('input', saveCurrent);
  $('review-form').addEventListener('change', saveCurrent);
}
async function openEvidence() {
  const item = cases[current], button = $('open-board');
  button.disabled = true;
  detailImage = new Image();
  detailImage.src = item.board_detail;
  await detailImage.decode();
  button.disabled = false;
  $('detail-region').value = 'target';
  $('detail-zoom').value = '1';
  $('detail-title').textContent = `${item.sequence} · #${item.number} · 清晰图`;
  drawEvidence();
  $('evidence-dialog').showModal();
}
function drawEvidence() {
  const regions = {
    target: [640, 0, 320, 380], initialization: [0, 0, 640, 380],
    context: [0, 380, 960, 185], full: [0, 0, 960, 565]
  };
  const scale = detailImage.naturalWidth / 960;
  const [x, y, width, height] = regions[$('detail-region').value].map(value => value * scale);
  const canvas = $('detail-canvas'), zoom = Number($('detail-zoom').value);
  canvas.width = width; canvas.height = height;
  canvas.style.width = `${width * zoom}px`; canvas.style.height = `${height * zoom}px`;
  canvas.getContext('2d').drawImage(detailImage, x, y, width, height, 0, 0, width, height);
  $('detail-scroll').scrollTop = 0; $('detail-scroll').scrollLeft = 0;
}
function saveCurrent() {
  if (!reviewer()) { $('saved-state').textContent = '请先填写审核者编号'; $('reviewer').focus(); return; }
  const item = cases[current], form = new FormData($('review-form'));
  const answer = {
    status: form.get('status') || '', confirmed_category: String(form.get('confirmed_category') || '').trim(),
    stable_attributes: String(form.get('stable_attributes') || '').trim(),
    uncertain_attributes: String(form.get('uncertain_attributes') || '').trim(),
    note: String(form.get('note') || '').trim(), updated_at: new Date().toISOString()
  };
  localStorage.setItem(storageKey(item), JSON.stringify(answer));
  $('saved-state').textContent = '已保存在当前浏览器';
  $('progress').textContent = `${manifest.rows.filter(row => saved(row).status).length} / ${manifest.count} 已保存`;
}
function csvCell(value) { return `"${String(value ?? '').replaceAll('"','""')}"`; }
function exportCSV() {
  if (!reviewer()) { alert('请先填写审核者编号。'); return; }
  const header = ['reviewer_id','dataset','anchor_number','key','sequence','init_frame','status','confirmed_category','stable_attributes','uncertain_attributes','note','provisional_category','provisional_attributes','updated_at'];
  const rows = manifest.rows.map(item => ({item, answer:saved(item)})).filter(pair => pair.answer.status).map(({item,answer}) => [reviewer(),manifest.dataset,item.number,item.id,item.sequence,item.frame,answer.status,answer.confirmed_category,answer.stable_attributes,answer.uncertain_attributes,answer.note,item.category,item.attributes.join('; '),answer.updated_at]);
  const csv = '\ufeff' + [header,...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `rgbd_${manifest.dataset}_${reviewer()}_${rows.length}_review.csv`; link.click();
  URL.revokeObjectURL(url);
}
async function loadDataset(dataset) {
  const response = await fetch(`data/${dataset}.json?v=20261003clear`);
  if (!response.ok) throw new Error(`无法读取 ${dataset} 审核清单：HTTP ${response.status}`);
  manifest = await response.json();
  $('from').value = qs.get('from') || 1;
  $('to').value = qs.get('to') || manifest.count;
  applyFilter();
}
$('reviewer').value = localStorage.getItem('rgbd-reviewer-id') || '';
$('dataset').value = ['depthtrack','cdtb','vot'].includes(qs.get('dataset')) ? qs.get('dataset') : 'depthtrack';
$('reviewer').addEventListener('change', () => { localStorage.setItem('rgbd-reviewer-id', reviewer()); render(); });
$('dataset').addEventListener('change', () => { qs.set('dataset', $('dataset').value); qs.delete('from'); qs.delete('to'); history.replaceState(null,'',`?${qs}`); loadDataset($('dataset').value); });
$('apply').addEventListener('click', applyAndShare);
$('search').addEventListener('input', applyFilter);
$('previous').addEventListener('click', () => { current--; render(); });
$('next').addEventListener('click', () => { current++; render(); });
$('export').addEventListener('click', exportCSV);
$('close-evidence').addEventListener('click', () => $('evidence-dialog').close());
$('detail-region').addEventListener('change', drawEvidence);
$('detail-zoom').addEventListener('change', drawEvidence);
loadDataset($('dataset').value).catch(error => { $('case').textContent = error.message; });
