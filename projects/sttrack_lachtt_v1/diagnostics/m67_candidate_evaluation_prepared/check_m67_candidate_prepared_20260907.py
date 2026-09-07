"""CPU checks of the actual prepared entry sources and the absent-completion guard."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907/candidate_evaluation'
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(source, name):
    return next(x for x in ast.parse(source).body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name)


def main():
    path = BASE / 'm67_candidate_evaluation_20260907.py'
    spec = importlib.util.spec_from_file_location('m67_candidate_cpu_check', str(path))
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    prepared = app.checked_preparation()
    for name in prepared['source_sha256']:
        if name.endswith('.py'): py_compile.compile(str(BASE / name), doraise=True)
        else: subprocess.run(['bash', '-n', str(BASE / name)], check=True)
    old = (BASE / 'm62_learned_entry_parity_20260907.py').read_text()
    old = old.replace("M59 / 'predictions/category'", "M67 / 'recursive/support'")
    new = (BASE / 'm67_learned_entry_parity_20260907.py').read_text()
    unchanged = ['wire_function', 'Frame', 'region_box', 'client']
    for name in unchanged:
        assert ast.dump(function(old, name)) == ast.dump(function(new, name)), name
    prior = (BASE / 'm64_vot_low22_20260907.py').read_text().replace('m64_category_low22_analysis', 'm67_support_low22_analysis')
    current = (BASE / 'm67_vot_low22_20260907.py').read_text()
    assert ast.dump(function(prior, 'analyze')) == ast.dump(function(current, 'analyze'))
    old_gate = json.loads((BASE / 'sttrack_m64_category_candidate_20260907/spec.json').read_text())['full_evaluation_gate']
    assert prepared['full_evaluation_gate'] == old_gate
    assert not (ROOT.parent / 'completed_evidence_audit.json').exists()
    assert not (ROOT / 'bundle.json').exists()
    # Invoke the actual builder, without replacing any gate, result, head, or module.
    result = subprocess.run([PYTHON, str(path), 'build'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 1
    assert 'FileNotFoundError' in result.stderr and 'completed_evidence_audit.json' in result.stderr
    assert not result.stdout.strip()
    for name in ['bundle.json', 'binding_result.json', 'development_plan.json', 'low22_plan.json', 'entry_parity', 'low22_run']:
        assert not (ROOT / name).exists(), name
    receipt = dict(status='cpu_conditional_M67_entry_sources_checked', observed_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), preparation_spec_sha256=sha(ROOT / 'spec.json'),
        input_preparation_receipt_sha256=sha(ROOT / 'cpu_preparation_result.json'),
        unchanged_actual_TraX_transport_functions=unchanged, unchanged_low22_metric_and_promotion_function=True,
        same_M64_low22_gate=True, compiled_python_files=3, bash_syntax_checked_files=2,
        real_builder_before_completion_exit=result.returncode, real_builder_rejection='Missing actual M67 completion audit.',
        fabricated_result_or_checkpoint_inputs=0, original_gates_modified=False,
        final_bundle_or_evaluation_outputs_created=False, new_tracking_calls=0, new_optimizer_steps=0,
        actual_gpu_parity_checked=False, public_evaluation_started=False, independent_model_review_pass=False)
    (ROOT / 'source_check_result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
