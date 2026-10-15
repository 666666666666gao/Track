"""Execute extracted source control flow with explicit stdlib fakes; zero torch/NN calls."""
import ast
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SRC = BASE / 'M125_R0_FIXED_SOURCE_20261015'
DEP = SRC / 'frozen_dependency_snapshots'
FIX = OUT / 'SYNTHETIC_CONTROL_FLOW_NO_NN'
FIX.mkdir(exist_ok=False)
checks = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    if isinstance(data, dict):
        data = dict(data, fixture_only=True, NN_calls=0)
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def check(name, actual, expected=True, details=None):
    checks.append(dict(name=name, status='PASS' if actual == expected else 'FAIL',
                       actual=actual, expected=expected, details=details))


def functions(path, globals_):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(module, str(path), 'exec'), globals_)
    return globals_


def frame_loop(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'frame')


old_loop = ast.dump(frame_loop(DEP / 'train_m124_joint_epoch.py'), include_attributes=False)
new_loop = ast.dump(frame_loop(SRC / 'train_m125_R0_fixed_pass.py'), include_attributes=False)
check('M124_frame_prediction_loss_gradient_update_loop_AST_identical', new_loop, old_loop)
# Store large equivalent ASTs once, not twice in the result file.
checks[-1]['actual'] = hashlib.sha256(new_loop.encode()).hexdigest()
checks[-1]['expected'] = hashlib.sha256(old_loop.encode()).hexdigest()


class Count:
    def __init__(self, value):
        self.value = value


class SymbolicNN:
    @staticmethod
    def Linear(a, b, bias=True):
        return Count(a * b + (b if bias else 0))

    @staticmethod
    def LayerNorm(n):
        return Count(2 * n)

    @staticmethod
    def Conv2d(a, b, k, padding=0):
        return Count(a * b * k * k + b)

    @staticmethod
    def MultiheadAttention(embed_dim, num_heads, dropout, batch_first):
        return Count(4 * embed_dim * embed_dim + 4 * embed_dim)

    @staticmethod
    def Sequential(*items):
        return Count(sum(item.value for item in items))

    @staticmethod
    def ModuleDict(items):
        return Count(sum(item.value for item in items.values()))

    @staticmethod
    def GELU():
        return Count(0)

    ReLU = GELU


parameter_groups = {}
for filename, clsname, expected in [('dense_target_decoder.py', 'DenseTargetDecoder', 380167),
                                    ('template_write_C.py', 'TemplateWriteC', 70721)]:
    tree = ast.parse((DEP / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    counts = {}
    for node in constructor.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            value = eval(compile(ast.Expression(node.value), str(DEP / filename), 'eval'),
                         {'nn': SymbolicNN, 'INPUT_DIM': 519})
            counts[node.targets[0].attr] = value.value
    parameter_groups[clsname] = counts
    check('symbolic_parameter_count:' + clsname, sum(counts.values()), expected,
          'Linear/LayerNorm/Conv2d/MHA arithmetic from actual constructor AST; no NN created.')
check('symbolic_total_trainable_parameters', sum(sum(v.values()) for v in parameter_groups.values()), 450888)


def controller_fixture(fail_epoch=None):
    root = FIX / ('controller_success' if fail_epoch is None else 'controller_failure_pass_' + str(fail_epoch))
    root.mkdir()
    candidate = read(SRC / 'CANDIDATE.json')
    gate = dict(candidate, root=str(root), code_root=str(SRC), base_code_root=str(DEP),
                python='NO_PYTHON_CHILD_IS_EXECUTED', metric_python='NO_METRIC_CHILD_IS_EXECUTED',
                status='reviewed_M125_R0_fixed6_source_gate',
                source_sha256={str(p): sha(p) for p in list(SRC.glob('*.py')) + list(DEP.glob('*.py'))},
                input_sha256={str(SRC / 'CANDIDATE.json'): sha(SRC / 'CANDIDATE.json')})
    write(root / 'gate.json', gate)
    calls = []

    def stub_run(folder, supplied_gate, stage, python, arguments, gpu=None):
        row = dict(stage=stage, arguments=arguments, gpu=gpu, fixture_only=True, exit_code=0)
        calls.append(row)
        if stage == 'train_fixed_pass':
            epoch = int(arguments[arguments.index('--epoch') + 1])
            if epoch == fail_epoch:
                raise RuntimeError('INTENTIONAL_STUB_FAILURE; no actual child process')
            fit = Path(arguments[arguments.index('--output') + 1]); fit.mkdir()
            write(fit / 'result.json', dict(epochs=epoch, track_calls=epoch * 219802, sequence_runs=epoch * 152,
                  fixed_final_after_complete_budget=epoch == 6, external_metric_checkpoint_selection=False))
        elif stage == 'prepare':
            evaluation = Path(arguments[arguments.index('--output') + 1]); evaluation.mkdir()
            (evaluation / 'vot/run').mkdir(parents=True)
            write(evaluation / 'vot/run/shard_manifest.json', dict(tracker='fixture_tracker'))
        elif stage == 'collect':
            evaluation = Path(arguments[arguments.index('--root') + 1])
            write(evaluation / 'all_results.json', dict(status='three_M125_R0_fixed6_full_evaluations_complete', any_joint_pass=False))
        return row

    env = functions(SRC / 'run_m125_R0_fixed_suite.py', dict(Path=Path, json=json, sys=SimpleNamespace(argv=['fixture', str(root)], flags=SimpleNamespace(optimize=0)),
            read=read, write=write, sha=sha, datetime=datetime, timezone=timezone, ThreadPoolExecutor=ThreadPoolExecutor))
    env['run'] = stub_run
    error = None
    try:
        env['main']()
    except RuntimeError as exc:
        error = str(exc)
    if fail_epoch is None:
        train = [row for row in calls if row['stage'] == 'train_fixed_pass']
        check('controller_six_training_stages', len(train), 6)
        check('controller_all_six_precede_evaluation', [r['stage'] for r in calls[:6]], ['train_fixed_pass'] * 6)
        check('controller_fixed_epoch_arguments', [int(r['arguments'][r['arguments'].index('--epoch') + 1]) for r in train], list(range(1, 7)))
        check('controller_pass1_has_no_resume', '--resume' not in train[0]['arguments'])
        check('controller_later_resume_exact_previous_latest', all(
              r['arguments'][r['arguments'].index('--resume') + 1] == str(root / 'passes' / ('pass-%02d' % (i - 1)) / 'fit/latest.pt')
              for i, r in enumerate(train[1:], 2)))
        prepare = next(r for r in calls if r['stage'] == 'prepare')
        check('controller_prepares_only_pass6', prepare['arguments'][prepare['arguments'].index('--training') + 1], str(root / 'passes/pass-06/fit'))
        check('controller_no_real64_or_interpass_eval', len(calls), 15)
        check('controller_expected_final_evaluation_stages', sorted(r['stage'] for r in calls[6:]),
              sorted(['prepare', 'depthtrack_test_track', 'cdtb_track', 'depthtrack_test_analyze', 'cdtb_analyze',
                      'vot_track', 'vot_official_analysis', 'vot_seal', 'collect']))
        analysis = next(r for r in calls if r['stage'] == 'vot_official_analysis')
        check('controller_vot_analysis_name_matches_analyzer', analysis['arguments'][analysis['arguments'].index('--name') + 1], 'R0_fixed6_Train152_full127')
        check('controller_success_zero_exit_file', (root / 'controller.exit').read_text().strip(), '0')
    else:
        check('controller_failure_original_exception_preserved', error, 'INTENTIONAL_STUB_FAILURE; no actual child process')
        check('controller_failure_no_retry_or_later_stage', len(calls), fail_epoch)
        check('controller_failure_no_false_success_exit', not (root / 'controller.exit').exists())
    write(root / 'FIXTURE_COMMANDS.json', dict(calls=calls, error=error))


controller_fixture()
controller_fixture(fail_epoch=3)

# Actual stage receipt and supervisor return-code logic, with subprocess.run replaced.
stage = FIX / 'stage_child_exit_7'; stage.mkdir()
subprocess_calls = []


def failed_run(*args, **kwargs):
    subprocess_calls.append(dict(args=args, cwd=kwargs.get('cwd')))
    return SimpleNamespace(returncode=7)


env = functions(SRC / 'run_m125_R0_fixed_suite.py', dict(Path=Path, json=json, os=__import__('os'),
      subprocess=SimpleNamespace(run=failed_run, STDOUT='FAKE_STDOUT'), write=write, datetime=datetime, timezone=timezone))
stage_error = None
try:
    env['run'](stage, dict(code_root='ISOLATED', base_code_root='FROZEN'), 'failure', 'NEVER_EXECUTED', ['script.py'])
except AssertionError as exc:
    stage_error = str(exc)
check('actual_run_function_failure_receipt_retains_7', read(stage / 'failure.receipt.json')['exit_code'], 7)
check('actual_run_function_failure_exit_file_retains_7', (stage / 'failure.exit').read_text().strip(), '7')
check('actual_run_function_failure_invokes_child_once', len(subprocess_calls), 1)
check('actual_run_function_failure_raises_no_retry', stage_error, 'Read this original stage failure; no automatic retry.')

supervisor = FIX / 'supervisor_child_exit_7'; supervisor.mkdir()
write(supervisor / 'gate.json', dict(python='NEVER_EXECUTED', code_root='ISOLATED'))
env = functions(SRC / 'm125_R0_fixed_process_supervisor.py', dict(Path=Path, json=json, os=__import__('os'),
      sys=SimpleNamespace(argv=['fixture', str(supervisor)]), subprocess=SimpleNamespace(run=failed_run), datetime=datetime, timezone=timezone))
returned = env['main']()
check('actual_supervisor_returns_real_nonzero', returned, 7)
check('actual_supervisor_receipt_retains_real_nonzero', read(supervisor / 'controller.supervisor.terminal.json')['exit_code'], 7)
check('actual_supervisor_exit_file_retains_real_nonzero', (supervisor / 'controller.supervised.exit').read_text().strip(), '7')


class FakeTensor:
    def __init__(self, token):
        self.token = token

    def cpu(self):
        return self


class FakeParameter:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class FakeModel:
    def __init__(self, count, events):
        self.count = count
        self.events = events
        self.state = None

    def cuda(self):
        self.events.append(['model_cuda', self.count])
        return self

    def load_state_dict(self, state, strict):
        self.events.append(['model_load', self.count, strict])
        self.state = state

    def state_dict(self):
        return self.state

    def parameters(self):
        return iter([FakeParameter(self.count)])


class FakeOptimizer:
    def __init__(self, groups, events, **kwargs):
        self.events = events
        self.state = {}
        self.events.append(['AdamW_construct', [[p.numel() for p in g['params']] for g in groups], kwargs])

    def load_state_dict(self, state):
        self.state = state
        self.events.append(['AdamW_restore', state])


def digest(model):
    return str(model.count) + ':' + ','.join(v.token for v in model.state.values())


def initialization_fixture(epoch=1, completed=152, supplied_resume=True, format_='M125_R0_fixed_sequence_boundary_v1'):
    events = []
    candidate = read(SRC / 'CANDIDATE.json')
    raw = read(BASE / 'M125_fixed_R0_common_inputs_20261015.json')['files']
    spec = json.loads(raw[candidate['spec']]['text'])
    labels = dict(initial=[None] * 152, human_confirmed=True, dataset='depthtrack')
    bank = dict(sequences=[r['sequence'] for r in spec['sequence_order']], human_confirmed=True, dataset='depthtrack',
                labels_sha256=candidate['input_sha256'][candidate['labels']], encoder_sha256=candidate['input_sha256'][candidate['clip_weight']])
    warm = dict(A_B={'value': FakeTensor('common_AB')}, C={'value': FakeTensor('common_C')})
    frozen = dict(native='fixed_native_digest', clip='fixed_clip_digest')
    resume = dict(format=format_, epoch=epoch - 1, completed_sequences=completed,
                  spec_sha256=candidate['input_sha256'][candidate['spec']], fixed_budget_passes=6,
                  fresh_common_warm_start=True, A_B={'value': FakeTensor('resumed_AB')}, C={'value': FakeTensor('resumed_C')},
                  model_state_sha256=dict(A_B='380167:resumed_AB', C='70721:resumed_C'), frozen_state_sha256=frozen,
                  resume_RNG_saved=True, optimizer={'sentinel': 'all_previous_Adam_states'},
                  RNG=dict(python='python_previous', numpy='numpy_previous', torch_cpu='cpu_previous', torch_cuda=['gpu0_previous','gpu1_previous']))

    class InputPath:
        def __init__(self, name):
            self.name = name

        def read_text(self):
            return json.dumps(labels) if self.name == 'labels' else raw[candidate[self.name]]['text']

    def fake_sha(path):
        return candidate['input_sha256'][candidate[path.name]]

    def fake_load(path, map_location):
        events.append(['torch_load', path.name, map_location])
        return {'bank': bank, 'warm_final': warm, 'resume': resume}[path.name]

    rng = lambda prefix: SimpleNamespace(seed=lambda state: events.append([prefix + '_seed', state]),
                                         setstate=lambda state: events.append([prefix + '_restore', state]),
                                         set_state=lambda state: events.append([prefix + '_restore', state]))
    fake_torch = SimpleNamespace(load=fake_load, equal=lambda a, b: a.token == b.token,
          set_num_threads=lambda count: events.append(['threads', count]),
          manual_seed=lambda seed: events.append(['torch_cpu_seed', seed]),
          set_rng_state=lambda state: events.append(['torch_cpu_restore', state]),
          cuda=SimpleNamespace(device_count=lambda: 2, device=lambda index: nullcontext(),
               manual_seed_all=lambda seed: events.append(['torch_cuda_seed', seed]),
               set_rng_state_all=lambda state: events.append(['torch_cuda_restore', state])),
          optim=SimpleNamespace(AdamW=lambda groups, **kwargs: FakeOptimizer(groups, events, **kwargs)))
    args = SimpleNamespace(**{name: InputPath(name) for name in ['spec', 'repository', 'checkpoint', 'clip_weight', 'bank', 'labels', 'warm_final', 'warm_result']},
                           output=FIX / 'NEVER_CREATED_TRAINING_DIRECTORY', epoch=epoch,
                           resume=InputPath('resume') if supplied_resume else None)
    env = functions(SRC / 'train_m125_R0_fixed_pass.py', dict(json=json, sha=fake_sha, torch=fake_torch, np=SimpleNamespace(random=rng('numpy')),
          random=rng('python'), WARM_SHA=candidate['input_sha256'][candidate['warm_final']],
          DenseTargetDecoder=lambda: FakeModel(380167, events), TemplateWriteC=lambda: FakeModel(70721, events),
          FullDenseTracker=lambda *args: SimpleNamespace(frozen_digest=lambda: frozen), JointFullABCTracker=lambda *args: 'FAKE_TRACKER', state_digest=digest))
    result = env['initialize_training'](args)
    return events, result


events, initialized = initialization_fixture(epoch=1, supplied_resume=False)
check('first_pass_weights_from_common_warm', initialized[8], {'A_B': '380167:common_AB', 'C': '70721:common_C'})
check('first_pass_Adam_state_empty', initialized[9].state, {})
check('first_pass_formal_history_zero', initialized[-1], dict(track_calls=0, optimizer_steps=0, C_optimizer_steps=0,
      supervised_frames=0, C_supervised_frames=0, sequence_records=[]))
check('first_pass_final_rng_reseed_after_module_construction', events[-4:],
      [['python_seed',2027], ['numpy_seed',2027], ['torch_cpu_seed',2027], ['torch_cuda_seed',2027]])
check('first_pass_no_optimizer_restore', not any(e[0] == 'AdamW_restore' for e in events))

events2, initialized2 = initialization_fixture(epoch=2)
check('later_pass_weights_from_complete_resume', initialized2[8], {'A_B': '380167:resumed_AB', 'C': '70721:resumed_C'})
check('later_pass_Adam_state_restore_call', initialized2[9].state, {'sentinel': 'all_previous_Adam_states'})
check('later_pass_all_rng_restored_after_construction', events2[-4:],
      [['python_restore','python_previous'], ['numpy_restore','numpy_previous'], ['torch_cpu_restore','cpu_previous'],
       ['torch_cuda_restore',['gpu0_previous','gpu1_previous']]])
for name, kwargs in [('reject_partial_boundary', dict(epoch=2, completed=151)),
                     ('reject_legacy_M124_checkpoint', dict(epoch=2, format_='M124_joint_sequence_boundary_v1')),
                     ('reject_first_pass_resume', dict(epoch=1)),
                     ('reject_later_pass_without_resume', dict(epoch=2, supplied_resume=False))]:
    rejected = False
    try:
        initialization_fixture(**kwargs)
    except AssertionError:
        rejected = True
    check(name, rejected)

write(OUT / 'deterministic_checks.json', dict(scope='SYNTHETIC control-flow/AST/arithmetic only; no torch import, NN, gradient, serialization, optimizer mathematics, network, SSH, or real metric execution',
      checks=checks, symbolic_parameter_groups=parameter_groups,
      initialization_events=dict(first_pass=events, later_pass=events2),
      count=len(checks), failed=sum(r['status'] != 'PASS' for r in checks)))
print(json.dumps(dict(checks=len(checks), failures=[r['name'] for r in checks if r['status'] != 'PASS'], NN_calls=0)))
