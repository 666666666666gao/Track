"""Apply the user's single-seed amendment without modifying the running training pair."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import torch

B = Path('/root/autodl-tmp')
R = B / 'sttrack_m73_paired_lexical_replication_20260907'
A = R / 'single_seed_amendment'
BACKUP = A / 'previous'
before = {}
changed = []


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def encoded(v): return (json.dumps(v, indent=2, allow_nan=False) + '\n').encode()
def now(): return datetime.now(timezone.utc).isoformat()


def save(p, data):
    p = Path(p)
    assert B.resolve() in p.resolve().parents
    assert p not in changed
    if p.exists():
        q = BACKUP / p.relative_to(B); q.parent.mkdir(parents=True, exist_ok=True)
        assert not q.exists(); shutil.copyfile(p, q); before[str(p)] = sha(p)
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); changed.append(p)


def edit(p, replacements):
    text = p.read_text()
    for old, new in replacements:
        assert old in text, (p, old)
        text = text.replace(old, new)
    if p.suffix == '.py': compile(text, str(p), 'exec')
    save(p, text.encode())


def module(name, path):
    s = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    torch.set_num_threads(1)
    assert not (A / 'plan.json').exists() and not (R / 'result.json').exists()
    assert not (R / 'seed2028/training').exists()
    assert not (R / 'seed2027/training/category/final.pth').exists()
    assert not (R / 'content_counterfactuals/activation.json').exists()
    assert not Path('/proc/484228').exists()
    f = Path('/proc/482884/stat').read_text().rsplit(')', 1)[1].split()
    assert f[0] == 'T' and f[19] == '4579784392'
    f = Path('/proc/482883/stat').read_text().rsplit(')', 1)[1].split()
    assert f[0] == 'T' and f[19] == '4579784392'
    t = read(A / 'transition.json'); t['user_instruction'] = '\u4e0d\u505a\u591aseed\u5b9e\u9a8c'
    save(A / 'transition.json', encoded(t))
    frozen = read(R / 'frozen.json')['seeds']['2027']
    plan = dict(status='user_authorized_single_seed_amendment_before_final_results', observed_utc=now(),
        user_instruction=t['user_instruction'], active_seeds=[2027], cancelled_seeds=[2028], candidate_seed=2027,
        original_spec_sha256=sha(R / 'spec.json'), original_frozen_sha256=sha(R / 'frozen.json'),
        original_queue_sha256=sha(R / 'run_m73.sh'), seed2027_frozen=frozen,
        transition_sha256=sha(A / 'transition.json'), screen_stop_sha256=sha(A / 'screen_stop.json'), training_pair_restarted=False,
        training_data_architecture_budget_and_eleven_gates_unchanged=True,
        interpretation='One fixed-seed paired study. No multi-seed robustness or replication claim.',
        cancelled_seed_prior_work='Preparation and 96-frame causal checks already existed; no full training or development evaluation started.',
        expected_training_track_calls=373388, expected_development_track_calls=66216,
        selection='Retain the originally fixed seed2027 Category final. No selection among seeds or early checkpoints.')
    save(A / 'plan.json', encoded(plan)); amendment_sha = sha(A / 'plan.json')
    save(R / 'seed2028/cancelled.json', encoded(dict(status='cancelled_by_user_before_full_training', amendment_sha256=amendment_sha, observed_utc=now())))

    auditor = B / 'audit_m73_completed_20260907.py'
    assert sha(auditor) == '4fcaf10e73225ea7bf9f57dbd031c1a6d9f58ebed29fa04341277d0d69fb3ef8'
    edit(auditor, [
        ('frozen two-seed M73 study', 'user-amended single-seed M73 study'),
        ('for seed in [2027, 2028]:', 'for seed in [2027]:'),
        ('def checked():\n', "def checked():\n    amendment = read(ROOT / 'single_seed_amendment/plan.json')\n    assert amendment['active_seeds'] == [2027] and amendment['cancelled_seeds'] == [2028]\n    assert amendment['original_spec_sha256'] == SPEC_SHA and amendment['original_frozen_sha256'] == FROZEN_SHA\n"),
        ("ref = read(OUT / 'reference.json'); assert ref['auditor_sha256'] == sha(__file__)", "ref = read(OUT / 'reference.json')\n    assert sha(OUT / 'reference.json') == 'fb387791f1de063990ff2b9f105b7387616893c382dce44642e5b591375c45fa'\n    assert ref['auditor_sha256'] == '4fcaf10e73225ea7bf9f57dbd031c1a6d9f58ebed29fa04341277d0d69fb3ef8'"),
        ('completed_two_seed_paired_lexical_replication', 'completed_single_seed_paired_lexical_experiment'),
        ("assert parent['spec_sha256'] == SPEC_SHA and parent['candidate_seed'] == 2027", "assert parent['spec_sha256'] == SPEC_SHA and parent['candidate_seed'] == 2027\n    assert parent['amendment_sha256'] == sha(ROOT / 'single_seed_amendment/plan.json')\n    assert set(parent['results']) == {'2027'}"),
        ('both_seed_development_gates_pass', 'single_seed_development_gates_pass'),
        ("    assert outputs['2027']['training']['category']['initial_tensor_sha256'] != outputs['2028']['training']['category']['initial_tensor_sha256']\n", ''),
        ('completed_M73_two_seed_checkpoint_and_scalar_audit', 'completed_M73_single_seed_checkpoint_and_scalar_audit'),
        ('historical_reference_sha256=sha(OUT', "amendment_sha256=sha(ROOT / 'single_seed_amendment/plan.json'), cancelled_seeds=[2028],\n        historical_reference_sha256=sha(OUT")])
    content = B / 'm73_content_counterfactuals_20260907.py'
    edit(content, [
        ('conditional on both M73 seeds', 'conditional on the retained M73 seed2027'),
        ('4fcaf10e73225ea7bf9f57dbd031c1a6d9f58ebed29fa04341277d0d69fb3ef8', sha(auditor)),
        ('seed2028 verifies replication and cannot replace the candidate.', 'seed2028 was cancelled by the user before full training.'),
        ('Both seeds individually pass all inherited eleven development requirements', 'The retained seed2027 passes all inherited eleven development requirements'),
        ('completed_M73_two_seed_checkpoint_and_scalar_audit', 'completed_M73_single_seed_checkpoint_and_scalar_audit'),
        ('both_seed_development_gates_pass', 'single_seed_development_gates_pass'),
        ("for seed in ['2027', '2028']:", "for seed in ['2027']:"),
        ("assert audit['candidate_seed'] == 2027", "assert audit['candidate_seed'] == 2027\n    assert audit['amendment_sha256'] == sha(PARENT / 'single_seed_amendment/plan.json')\n    assert set(audit['seeds']) == {'2027'}")])
    cs = read(R / 'content_counterfactuals/spec.json')
    cs.update(status='M73_content_protocol_amended_to_single_seed_before_results', observed_utc=now(),
        source_sha256=sha(content), auditor_sha256=sha(auditor), amendment_sha256=amendment_sha,
        head_selection='Original fixed seed2027 Category final only; seed2028 cancelled by user.',
        prerequisite='Retained seed2027 passes all unchanged eleven development requirements and completed audit.')
    save(R / 'content_counterfactuals/spec.json', encoded(cs))

    candidate = B / 'm73_candidate_evaluation_20260907.py'
    edit(candidate, [
        ('c2b3422b1c5ad19cbb5231d415244431033f238f2da747b11b053eefcd7c686d', sha(content)),
        ('b25a073ad07e2f9e5740a2732f06b8ee4d23472072ccad91bd3a5b37385ee7d5', sha(R / 'content_counterfactuals/spec.json')),
        ('M73 prospectively compares category and empty training in seeds2027/2028;', 'M73 compares category and empty training in fixed seed2027; seed2028 was cancelled by the user;'),
        ('conditional on both-seed development and same-head content checks.', 'conditional on unchanged single-seed development and same-head content checks.'),
        ('Both M73 seed development gates and completed audit', 'Retained M73 seed2027 development gates and completed audit'),
        ("M73/'completion_queue/", "M73/'single_seed_amendment/controller/")])
    croot = R / 'candidate_evaluation'; protocol = read(croot / 'interface/text_protocol.json')
    protocol['selection_provenance'] = 'M73 retains fixed seed2027 Category final after user cancellation of seed2028. Category-versus-empty paired training and unchanged development/content gates remain. No multi-seed claim.'
    protocol['single_seed_amendment_sha256'] = amendment_sha
    save(croot / 'interface/text_protocol.json', encoded(protocol))
    protocol_sha = sha(croot / 'interface/text_protocol.json')
    bank_checks = {}
    for name in ['development', 'low22']:
        p = croot / (name + '_category.pt'); old = torch.load(p, map_location='cpu')
        q = BACKUP / p.relative_to(B); q.parent.mkdir(parents=True, exist_ok=True); assert not q.exists()
        shutil.copyfile(p, q); before[str(p)] = sha(p); changed.append(p)
        updated = dict(old, protocol_sha256=protocol_sha); torch.save(updated, p)
        check = torch.load(p, map_location='cpu')
        assert check['keys'] == old['keys'] and all(torch.equal(check[k], old[k]) for k in ['tokens', 'mask', 'empty'])
        bank_checks[name] = dict(rows=len(check['keys']), tokens_masks_keys_and_empty_unchanged=True)
    edit(B / 'm73_candidate_entry_20260907.sh', [('sttrack_m73_paired_lexical_replication_20260907/completion_queue', 'sttrack_m73_paired_lexical_replication_20260907/single_seed_amendment/controller')])
    spec = read(croot / 'spec.json')
    spec.update(status='M73_candidate_preparation_amended_to_single_seed_before_results', observed_utc=now(),
        source_sha256={n:sha(B / n) for n in spec['source_sha256']}, content_source_sha256=sha(content),
        content_spec_sha256=sha(R / 'content_counterfactuals/spec.json'), text_protocol_sha256=protocol_sha, amendment_sha256=amendment_sha)
    for value in spec['banks'].values(): value['sha256'] = sha(value['path'])
    spec['required_order'][0] = 'Retained seed2027 development gates and completed audit'
    save(croot / 'spec.json', encoded(spec))
    full = B / 'm73_full_preparation_20260907.py'
    edit(full, [('af20b98be8654ab214a71f481fe943c17505a4b1ee1e20c09affcfdd78e4b5c9', sha(croot / 'spec.json'))])
    # Re-execute actual source contracts; preserve the original CPU readiness evidence.
    m = module('m73_amended_full', full); m.contracts()
    readiness = read(croot / 'full_preparation_readiness.json')
    readiness.update(candidate_spec_sha256=sha(croot / 'spec.json'), preparation_source_sha256=sha(full),
        observed_utc=now(), amendment_sha256=amendment_sha)
    readiness['full_source_sha256'] = {p:sha(p) for p in readiness['full_source_sha256']}
    save(croot / 'full_preparation_readiness.json', encoded(readiness))
    module('m73_amended_audit', auditor).checked()
    module('m73_amended_content', content).checked()
    module('m73_amended_candidate', candidate).checked_preparation()
    # Exercise the real entry and full-evaluation prerequisites before any learned final exists.
    negatives = {}
    for name, command in [
        ('content', [str(B / 'envs/sttrack/bin/python'), str(content), 'eligible']),
        ('candidate', [str(B / 'envs/sttrack/bin/python'), str(candidate), 'build']),
        ('full', ['bash', str(B / 'm73_full_evaluation_20260907.sh')])]:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode != 0 and 'M73 completion audit is not yet available' in result.stdout
        negatives[name] = dict(exit_code=result.returncode, expected_missing_completion_audit=True)
        save(A / (name + '_premature_launch.log'), result.stdout.encode())
    assert not (croot / 'bundle.json').exists() and not (croot / 'full_evaluation').exists()
    assert not torch.cuda.is_initialized()
    controller = B / 'm73_single_seed_controller_20260907.py'; compile(controller.read_text(), str(controller), 'exec')
    changed.append(controller)
    receipt = dict(status='single_seed_runtime_contracts_updated_and_CPU_checked', observed_utc=now(),
        amendment_sha256=amendment_sha, before_sha256=before, after_sha256={str(p):sha(p) for p in changed},
        unchanged_seed2027_training_spec=sha(R / 'seed2027/training_spec.json'),
        unchanged_seed2027_recursive_spec=sha(R / 'seed2027/recursive_spec.json'),
        bank_metadata_only_changes=bank_checks, real_premature_launch_rejections=negatives,
        model_forward_calls=0, optimizer_steps=0, new_caption_calls=0,
        development_and_low22_thresholds_unchanged=True, public_evaluation_started=False)
    (A / 'patch_receipt.json').write_bytes(encoded(receipt))
    out = A / 'controller'; out.mkdir()
    (out / 'spec.json').write_bytes(encoded(dict(status='single_seed_adoption_controller_ready', observed_utc=now(),
        source_sha256=sha(controller), amendment_sha256=amendment_sha, patch_receipt_sha256=sha(A / 'patch_receipt.json'),
        suspended_scheduler=dict(pid=482884,start_ticks='4579784392'), suspended_screen=dict(pid=482883,start_ticks='4579784392'),
        adopted_pair=dict(pid=482888,start_ticks='4579784502'), poll_seconds=240)))
    launch = '#!/bin/bash\nset -u\ncd ' + str(out) + ' || exit 1\nexport CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n' + str(B / 'envs/sttrack/bin/python') + ' -u ' + str(controller) + ' > controller.log 2>&1\nstatus=$?\nprintf \'%s\\n\' "$status" > controller.exit\nexit "$status"\n'
    (out / 'run.sh').write_text(launch); subprocess.run(['bash','-n',str(out / 'run.sh')],check=True)
    module('single_seed_controller_check',controller).checked()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['before_sha256','after_sha256']}))


if __name__ == '__main__': main()
