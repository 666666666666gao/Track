// Fresh reviewer scalar audit: no experiment imports, inference, training or network.
// Run: node reviewer_recompute_20260920.cjs
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = __dirname;
const inputHashes = {};
const errors = [];
const missingLocal = [];
let checkCount = 0;
let maximumAbsoluteDifference = 0;
function bytes(name) {
  const value = fs.readFileSync(path.join(root, name));
  inputHashes[name.replaceAll('\\', '/')] = crypto.createHash('sha256').update(value).digest('hex');
  return value;
}
function read(name) { return JSON.parse(bytes(name).toString('utf8')); }
function hash(name) { bytes(name); return inputHashes[name.replaceAll('\\', '/')]; }
function check(name, condition) { checkCount++; if (!condition) errors.push(name); }
function close(name, observed, expected) {
  const difference = Math.abs(observed - expected);
  maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, difference);
  check(name, difference <= 1e-9);
}
const result = read('recursive_result.json');
const spec = read('training_spec.json');
const recursive = read('recursive_spec.json');
const frozen = read('frozen.json');
const bindings = {
  'training_spec.json': frozen.training_spec_sha256,
  'recursive_spec.json': frozen.recursive_spec_sha256,
  'integration.json': spec.integration_sha256,
  'data_inventory.json': spec.inventory_sha256,
  'run_pair.sh': frozen.queue_sha256,
  'causal_check.json': frozen.causal_check_sha256,
  'native_reference.json': spec.native_result_sha256,
  'M78_reference.json': spec.matched_M78_result_sha256,
  ...frozen.source_sha256,
};
for (const [name, expected] of Object.entries(bindings)) check('hash:' + name, hash(name) === expected);
const integration = read('integration.json');
let integrationPresent = 0;
for (const [name, expected] of Object.entries(integration.source_sha256)) {
  const local = 'code/' + name;
  if (fs.existsSync(path.join(root, local))) {
    integrationPresent++;
    check('integration:' + name, hash(local) === expected);
  } else missingLocal.push(local);
}
check('fit count', spec.sequence_order.length === 130);
check('development count', spec.development_sequences.length === 22);
check('fit unique', new Set(spec.sequence_order.map(x => x.sequence)).size === 130);
check('development unique', new Set(spec.development_sequences).size === 22);
check('fit/development disjoint', !spec.sequence_order.some(x => spec.development_sequences.includes(x.sequence)));
check('train call count', spec.sequence_order.reduce((s, x) => s + x.rgb_frames - 1, 0) === 186694);
const gt = {};
for (const c of recursive.cases) {
  const name = 'dataset_gt/' + c.sequence + '/groundtruth.txt';
  check('GT hash:' + c.sequence, hash(name) === c.gt_sha256);
  gt[c.sequence] = bytes(name).toString('utf8').trim().split(/\r?\n/).map(line => line.split(',').map(Number));
  check('GT length:' + c.sequence, gt[c.sequence].length === c.frames);
  check('GT width:' + c.sequence, gt[c.sequence].every(row => row.length === 4));
  check('GT init:' + c.sequence, JSON.stringify(gt[c.sequence][0]) === JSON.stringify(c.init_bbox));
}
function statistics(rows, truth) {
  let valid = 0, invalid = 0, sum = 0, low = 0, run = 0, episodes = 0;
  for (let i = 1; i < rows.length; i++) {
    const a = rows[i].bbox, b = truth[i];
    if (b.some(v => !Number.isFinite(v)) || b[2] <= 0 || b[3] <= 0) { invalid++; run = 0; continue; }
    const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const overlap = inter / (a[2] * a[3] + b[2] * b[3] - inter);
    valid++; sum += overlap;
    if (overlap <= .1) { low++; run++; if (run === 10) episodes++; } else run = 0;
  }
  return { valid_frames: valid, iou_sum: sum, mean_iou: sum / valid, low_iou_frames: low,
    failure_episodes: episodes, invalid_gt_frames: invalid };
}
function aggregate(perSequence) {
  const totals = { valid_frames: 0, iou_sum: 0, low_iou_frames: 0, failure_episodes: 0 };
  for (const row of Object.values(perSequence)) for (const key of Object.keys(totals)) totals[key] += row[key];
  totals.mean_iou = totals.iou_sum / totals.valid_frames;
  totals.macro_sequence_mean_iou = Object.values(perSequence).reduce((s, x) => s + x.mean_iou, 0) / Object.keys(perSequence).length;
  return totals;
}
const training = {};
for (const arm of ['category', 'empty']) {
  const t = read('training/' + arm + '/result.json'); training[arm] = t;
  check('training complete:' + arm, t.status === 'one_full_causal_fit_pass_complete' && t.sequences === 130 && t.total_track_calls === 186694 && t.optimizer_steps === 5798);
  check('training spec:' + arm, t.training_spec_sha256 === hash('training_spec.json'));
  check('training label sum:' + arm, Object.values(t.training_label_counts).reduce((a, b) => a + b, 0) === t.total_track_calls);
  for (const name of ['final.pth', 'sequence_log.jsonl', 'sampled_state_trace.jsonl']) {
    const local = 'training/' + arm + '/' + name;
    if (!fs.existsSync(path.join(root, local))) missingLocal.push(local);
  }
}
for (const key of ['initial_adapter_state_sha256', 'base_state_before_sha256', 'optimizer_steps', 'total_track_calls']) {
  check('paired training:' + key, training.category[key] === training.empty[key]);
}
const per = {}, aggregates = {}, updates = {}, updateFrames = {};
let positions = 0;
for (const arm of recursive.variants) {
  per[arm] = {}; updates[arm] = 0; updateFrames[arm] = {};
  const receipt = read(arm + '_recursive_receipt.json');
  const trainArm = arm === 'empty' ? 'empty' : 'category';
  check('receipt hash:' + arm, hash(arm + '_recursive_receipt.json') === result.receipts[arm]);
  check('receipt complete:' + arm, receipt.status === 'complete' && receipt.sequences.length === 22);
  check('receipt spec:' + arm, receipt.recursive_spec_sha256 === hash('recursive_spec.json'));
  check('receipt training:' + arm, receipt.training_result_sha256 === hash('training/' + trainArm + '/result.json'));
  check('receipt head:' + arm, receipt.head_sha256 === training[trainArm].final_checkpoint_sha256);
  let total = 0;
  for (const c of recursive.cases) {
    const name = 'recursive/' + arm + '/' + c.sequence + '.json';
    const data = read(name), receiptRow = receipt.sequences.find(x => x.sequence === c.sequence);
    check('trajectory hash:' + arm + '/' + c.sequence, hash(name) === receiptRow.sha256);
    check('trajectory identity:' + arm + '/' + c.sequence, data.sequence === c.sequence && data.arm === arm && data.rows.length === c.frames && receiptRow.frames === c.frames);
    check('trajectory frames/boxes:' + arm + '/' + c.sequence, data.rows.every((r, i) => r.frame === i && r.bbox.length === 4 && r.bbox.every(Number.isFinite) && r.bbox[2] > 0 && r.bbox[3] > 0));
    check('trajectory init:' + arm + '/' + c.sequence, JSON.stringify(data.rows[0].bbox) === JSON.stringify(c.init_bbox));
    total += data.rows.length;
    per[arm][c.sequence] = statistics(data.rows, gt[c.sequence]);
    for (const [key, value] of Object.entries(per[arm][c.sequence])) close('per:' + arm + '/' + c.sequence + '/' + key, value, result.per_sequence[arm][c.sequence][key]);
    updateFrames[arm][c.sequence] = data.rows.filter(x => x.frame > 0 && x.frame % 50 === 0 && x.score > .75).map(x => x.frame);
    updates[arm] += updateFrames[arm][c.sequence].length;
  }
  positions += total;
  check('receipt total:' + arm, total === receipt.total_frames && total === 33130);
  aggregates[arm] = aggregate(per[arm]);
  for (const [key, value] of Object.entries(aggregates[arm])) close('aggregate:' + arm + '/' + key, value, result.aggregates[arm][key]);
}
const native = read('native_reference.json'), m78 = read('M78_reference.json');
aggregates.native = aggregate(native.per_sequence.native);
for (const [key, value] of Object.entries(aggregates.native)) close('native aggregate:' + key, value, result.aggregates.native[key]);
for (const [arm, values] of Object.entries(m78.aggregates)) for (const [key, value] of Object.entries(values)) close('M78 copy:' + arm + '/' + key, value, result.matched_M78_aggregates[arm][key]);
check('M78 spec reference', m78.training_spec_sha256 === spec.parent_training_spec_sha256);
const primary = aggregates.category, baseline = aggregates.native, control = aggregates.empty, rule = spec.promotion_gates;
const broken = other => Object.keys(other).filter(s => other[s].failure_episodes === 0 && per.category[s].failure_episodes > 0);
const gates = {
  mean_vs_native: primary.mean_iou >= baseline.mean_iou + rule.category_pooled_mean_vs_native_minimum,
  mean_vs_control: primary.mean_iou >= control.mean_iou + rule.category_pooled_mean_vs_control_minimum,
  macro_vs_native: primary.macro_sequence_mean_iou >= baseline.macro_sequence_mean_iou,
  macro_vs_control: primary.macro_sequence_mean_iou >= control.macro_sequence_mean_iou,
  low_frames_vs_native: primary.low_iou_frames <= baseline.low_iou_frames,
  low_frames_vs_control: primary.low_iou_frames <= control.low_iou_frames,
  H10_vs_native: primary.failure_episodes <= baseline.failure_episodes,
  H10_vs_control: primary.failure_episodes <= control.failure_episodes,
  native_success_protection: broken(native.per_sequence.native).length === 0,
  control_success_protection: broken(per.empty).length === 0,
};
const content = {}, increment = {}, leaveOneOut = {};
function four(a, b, strictMean) {
  return { mean: strictMean ? a.mean_iou > b.mean_iou : a.mean_iou >= b.mean_iou,
    macro: a.macro_sequence_mean_iou >= b.macro_sequence_mean_iou,
    low: a.low_iou_frames <= b.low_iou_frames, H10: a.failure_episodes <= b.failure_episodes };
}
for (const arm of ['category_empty', 'category_swapped']) {
  content[arm] = four(primary, aggregates[arm], true); leaveOneOut[arm] = {};
  for (const sequence of spec.development_sequences) {
    const a = per.category[sequence], b = per[arm][sequence], other = aggregates[arm];
    const value = (primary.iou_sum - a.iou_sum) / (primary.valid_frames - a.valid_frames) - (other.iou_sum - b.iou_sum) / (other.valid_frames - b.valid_frames);
    leaveOneOut[arm][sequence] = value;
    close('LOO:' + arm + '/' + sequence, value, result.content_leave_one_out[arm][sequence]);
  }
}
for (const arm of ['category', 'empty']) increment[arm] = four(aggregates[arm], m78.aggregates[arm], false);
check('primary gates', JSON.stringify(gates) === JSON.stringify(result.gates));
check('content gates', JSON.stringify(content) === JSON.stringify(result.content_gates));
check('increment gates', JSON.stringify(increment) === JSON.stringify(result.preservation_incremental_gates));
check('primary pass', Object.values(gates).every(Boolean) === result.primary_pass);
check('content pass', Object.values(content).flatMap(Object.values).every(Boolean) === result.content_pass);
check('increment pass', Object.values(increment.category).every(Boolean) === result.preservation_category_increment_pass);
const saved = read('saved_development_audit.json'), posthoc = read('posthoc_descriptive.json');
check('saved audit result', hash('recursive_result.json') === saved.result_sha256);
check('saved audit source', hash('audit_m82_completed_20260909.py') === saved.source_sha256);
check('posthoc source', hash('summarize_completed.py') === posthoc.source_sha256);
check('posthoc result', hash('recursive_result.json') === posthoc.result_sha256);
check('posthoc update counts', JSON.stringify(updates) === JSON.stringify(posthoc.update_count_reconstructed_from_saved_score_and_verified_runtime_rule));
check('posthoc update frames', JSON.stringify(updateFrames) === JSON.stringify(posthoc.update_frames));
const csvRows = {};
const csv = name => bytes(name).toString('utf8').trim().split(/\r?\n/).map(line => line.split(','));
let rows = csv('aggregate_metrics.csv'); csvRows.aggregate = rows.length - 1;
for (const [name, ...values] of rows.slice(1)) {
  const expected = name.startsWith('M78_') ? result.matched_M78_aggregates[name.slice(4)] : result.aggregates[name];
  values.forEach((value, i) => check('CSV aggregate:' + name + '/' + i, Number(value) === expected[rows[0][i + 1]]));
}
rows = csv('sequence_metrics.csv'); csvRows.sequence = rows.length - 1;
for (const [arm, sequence, ...values] of rows.slice(1)) values.forEach((value, i) => check('CSV sequence:' + arm + '/' + sequence + '/' + i, Number(value) === result.per_sequence[arm][sequence][rows[0][i + 2]]));
rows = csv('frozen_gates.csv'); csvRows.gates = rows.length - 1;
for (const [group, arm, criterion, passed] of rows.slice(1)) check('CSV gate:' + group + '/' + arm + '/' + criterion, (passed === 'True') === (group === 'primary' ? result.gates[criterion] : result[group][arm][criterion]));
const exits = {};
for (const name of fs.readdirSync(root).filter(x => x.endsWith('.exit') && !x.startsWith('m83'))) {
  exits[name] = bytes(name).toString('utf8').trim(); check('exit:' + name, exits[name] === '0');
}
const current = read('current_verification_20260920.json');
check('executor current result receipt', current.result_sha256 === hash('recursive_result.json'));
for (const arm of ['category', 'empty']) check('executor current head receipt:' + arm, current.heads[arm] === training[arm].final_checkpoint_sha256);
for (const [name, value] of Object.entries(current.exits)) check('executor current exit receipt:' + name, value === exits[name]);
hash('EXPERIMENT_PLAN.md'); hash('EXPERIMENT_TRACKER.md'); hash('handoff_append.md');
const output = {
  status: errors.length ? 'FAIL' : 'PASS_AVAILABLE_LOCAL_DETERMINISTIC_CHECKS',
  generated_at: new Date().toISOString(), script_sha256: hash('reviewer_recompute_20260920.cjs'),
  reviewer: 'gpt-6-astra', reviewer_reasoning: 'max', review_independence: 'same-family', acceptance_status: 'provisional',
  check_count: checkCount, errors, max_abs_float_difference: maximumAbsoluteDifference,
  families: 4, sequences: 88, positions, valid_positions_per_family: aggregates.category.valid_frames,
  integration_present: integrationPresent, integration_total: Object.keys(integration.source_sha256).length,
  missing_local: missingLocal, additional_unavailable_originals: ['native checkpoint', 'zero initial checkpoints', 'text banks/captions', 'fit GT originals', 'native/M78 original trajectories and M78 training source/spec'],
  aggregates, per_sequence: per, gates, content_gates: content, preservation_incremental_gates: increment,
  content_leave_one_out: leaveOneOut, update_counts_reconstructed: updates, csv_rows: csvRows, exits,
  current_remote_receipt_status: 'executor-provided receipt is internally consistent; reviewer did not connect remotely',
  audited_input_hashes: inputHashes,
};
fs.writeFileSync(path.join(root, 'reviewer_verification_20260920.json'), JSON.stringify(output, null, 2) + '\n');
console.log(JSON.stringify({status: output.status, check_count: checkCount, errors,
  max_abs_float_difference: maximumAbsoluteDifference, positions, csv_rows: csvRows,
  missing_local_count: missingLocal.length, output: 'reviewer_verification_20260920.json'}, null, 2));
process.exitCode = errors.length ? 1 : 0;
