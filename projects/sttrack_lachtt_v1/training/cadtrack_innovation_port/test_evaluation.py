"""CPU control-plane tests; no neural tracking or formal benchmark execution."""
from contextlib import nullcontext, redirect_stdout
import io
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluation as ev
import run_evaluation as runner


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def make_plan(self):
        plan = dict(schema="CADTrack_M125_port_official_eval_v1", dataset="depthtrack_test")
        for field in ("final", "bank", "binding", "checkpoint", "config",
                      "clip_weight", "evaluation_source", "evaluation_preparer",
                      "evaluation_runner", "failure_source", "failure_common",
                      "vot_identity_tool", "vot_reference", "launch", "terminal",
                      "training_rows", "cases", "metric_source"):
            path = self.root / field
            path.write_bytes(field.encode())
            if field == "metric_source":
                plan[field] = str(path)
            else:
                plan[field + "_path"] = str(path)
            plan[field + "_sha256"] = ev.sha(path)
        result_path = self.root / "training_result.json"
        ev.write(result_path, dict(status="complete_train", calls=1318812,
            final_sha256=plan["final_sha256"], frozen_before_after_exact=True,
            module_roundtrip_exact=True))
        plan.update(training_result_path=str(result_path),
                    training_result_sha256=ev.sha(result_path))
        source = self.root / "runtime.py"
        source.write_bytes(b"frozen source fixture")
        plan["source_sha256"] = {str(source): ev.sha(source)}
        bundle_path = self.root / "bundle.json"
        ev.write(bundle_path, dict(schema="CADTrack_port_eval_binding_v1", bindings=dict(plan)))
        plan.update(bundle_path=str(bundle_path), bundle_sha256=ev.sha(bundle_path))
        path = self.root / "plan.json"
        ev.write(path, plan)
        return path, plan

    def test_plan_rejects_changed_frozen_source_and_final(self):
        path, plan = self.make_plan()
        ev.checked_plan(path)
        source = Path(next(iter(plan["source_sha256"])))
        source.write_bytes(b"changed")
        with self.assertRaises(AssertionError):
            ev.checked_plan(path)
        source.write_bytes(b"frozen source fixture")
        Path(plan["final_path"]).write_bytes(b"other final")
        with self.assertRaises(AssertionError):
            ev.checked_plan(path)

    def test_plan_rejects_rebound_final_even_with_new_plan_hash(self):
        path, plan = self.make_plan()
        plan["final_path"] = str(self.root / "other_final")
        ev.write(path, plan)
        with self.assertRaises(AssertionError):
            ev.checked_plan(path)

    def test_stage_preserves_success_and_original_failure(self):
        with redirect_stdout(io.StringIO()):
            row = runner.stage(self.root, "ok",
                [sys.executable, "-c", "print('child output')"],
                self.root, dict(os.environ))
        self.assertEqual(row["exit_code"], 0)
        self.assertIn("child output", (self.root / "ok.log").read_text())
        with self.assertRaises(AssertionError):
            runner.stage(self.root, "failed", [sys.executable, "-c", "raise SystemExit(7)"],
                         self.root, dict(os.environ))
        self.assertEqual((self.root / "failed.exit").read_text().strip(), "7")
        self.assertEqual(ev.read(self.root / "failed.receipt.json")["exit_code"], 7)
        with self.assertRaises(AssertionError):
            runner.stage(self.root, "failed", [sys.executable, "-c", "print('retry')"],
                         self.root, dict(os.environ))

    def test_manifest_uses_flat_expected_trajectory_list(self):
        shard = dict(root=str(self.root), expected_trajectories=["sequence_00000000"])
        paths = runner.tracker_files(self.root, "tracker", shard["expected_trajectories"][0])
        paths[0].parent.mkdir(parents=True)
        for path in paths:
            path.write_bytes(b"nonempty trajectory fixture")
        self.assertEqual(runner.complete_anchors(shard, "tracker"), 1)
        paths[-1].write_bytes(b"")
        self.assertEqual(runner.complete_anchors(shard, "tracker"), 0)

    def test_nine_gate_boundaries_and_nonfinite(self):
        metrics = dict(depthtrack_test=dict(P=65.2, R=64.9, F=65.1),
                       cdtb=dict(P=72.9, R=75.6, F=74.2),
                       vot=dict(EAO=77.9, ACC=82.1, ROB=93.7))
        outcomes = runner.gate_outcomes(metrics)
        self.assertTrue(all(row["passed"] for dataset in
                        ("depthtrack_test", "cdtb") for row in outcomes[dataset].values()))
        self.assertFalse(any(row["passed"] for row in outcomes["vot"].values()))
        metrics["vot"] = dict(EAO=77.900001, ACC=82.100001, ROB=93.700001)
        self.assertTrue(all(row["passed"] for dataset in
                            runner.gate_outcomes(metrics).values() for row in dataset.values()))
        metrics["vot"]["EAO"] = float("nan")
        with self.assertRaises(AssertionError):
            runner.gate_outcomes(metrics)

    def test_vot_changed_execution_ledger_rejected_before_launch(self):
        root = self.root / "vot"
        root.mkdir()
        path = root / "execution.json"
        ev.write(path, dict(source_sha256={"original": "fixture"}))
        selected = ev.sha(path)
        ev.write(path, dict(source_sha256={"changed": "fixture"}))
        with patch.object(runner.subprocess, "Popen") as launch, \
                self.assertRaises(AssertionError):
            runner.run_vot(self.root, {}, sys.executable, selected)
        launch.assert_not_called()
        self.assertFalse((root / "run/controller_logs").exists())

    def test_failure_auditor_is_passed_explicitly_and_rejects_missing_anchors(self):
        sequences = ["seq%03d" % index for index in range(127)]
        analysis = self.root / "master/analysis/tracker_full127.json"
        analysis.parent.mkdir(parents=True)
        ev.write(analysis, dict(sequences={name: dict(length=30) for name in sequences}))
        def complete(master, tracker, expected_anchors):
            self.assertEqual(expected_anchors, 1765)
            return {str(i): {} for i in range(1765)}, 0, {
                name: dict(anchors=1) for name in sequences}, {}
        result = runner.checked_trajectory_coverage(self.root, "tracker", sequences, complete)
        self.assertEqual(len(result[2]), 1765)
        def incomplete(*args, **kwargs):
            return {}, 0, {}, {}
        with self.assertRaises(AssertionError):
            runner.checked_trajectory_coverage(self.root, "tracker", sequences, incomplete)

    def tracker_fixture(self):
        tracker = SimpleNamespace(
            actor=SimpleNamespace(frozen_digest=lambda: "native", decoder="decoder"),
            writer="writer", initialize=lambda *args: None,
            track=lambda image: dict(target_bbox=[0.25, 0.5, 4., 5.], best_score=0.75))
        initial = dict(native_clip="native", A_B="decoder", C="writer")
        return tracker, initial

    def test_vot_protocol_rejects_changed_stack_before_predictions(self):
        reference = self.root / "reference.json"
        parameters = dict(anchor="anchor", analyses=[dict(burnin=10)])
        ev.write(reference, dict(toolkit="0.7.1",
            results=dict(baseline=dict(parameters=parameters))))
        (self.root / "run").mkdir()
        ev.write(self.root / "run/shard_manifest.json",
                 dict(shards=[dict(root="fixture") for _ in range(4)]))
        experiment_type = type("MultiStartExperiment", (), {
            "__module__": "vot.experiment.multistart", "identifier": "baseline",
            "dump": lambda self: parameters})
        workspace = SimpleNamespace(stack=[experiment_type()])
        vot = SimpleNamespace(__version__="0.7.1")
        module = SimpleNamespace(Workspace=SimpleNamespace(load=lambda root: workspace))
        plan = dict(dataset="vot", vot_reference_path=str(reference))
        with patch.object(ev, "checked_plan", return_value=(plan, {})), \
                patch.dict(sys.modules, {"vot": vot, "vot.workspace": module}), \
                redirect_stdout(io.StringIO()):
            ev.vot_protocol(self.root / "plan.json")
            parameters["analyses"][0]["burnin"] = 11
            with self.assertRaises(AssertionError):
                ev.vot_protocol(self.root / "plan.json")

    def test_ope_full_frame_output_without_subsequent_gt(self):
        tracker, initial = self.tracker_fixture()
        cases, indices = [], {}
        for index in range(50):
            name = "seq%03d" % index
            image = self.root / name / "color/00000001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(name.encode())
            bbox = [0., 0., 4., 5.]
            indices["depthtrack_test:" + ev.initialization_key(ev.sha(image), bbox)] = index
            cases.append(dict(sequence=name, init_bbox=bbox,
                              frames=76324 if index == 0 else 1))
        cases_path = self.root / "cases.json"
        ev.write(cases_path, cases)
        final = self.root / "final.pt"
        final.write_bytes(b"fixture not neural weights")
        plan = dict(dataset="depthtrack_test", cases_path=str(cases_path),
            dataset_root=str(self.root), output=str(self.root / "predictions"),
            final_path=str(final), final_sha256=ev.sha(final),
            training_result_sha256="fixture", bank_sha256="fixture", binding_sha256="fixture")
        path = self.root / "plan.json"
        ev.write(path, plan)
        depth = SimpleNamespace(get_rgbd_frame=lambda *args, **kwargs: None)
        loaded = (SimpleNamespace(no_grad=nullcontext), tracker, {}, indices, initial, lambda x: x)
        with patch.object(ev, "checked_plan", return_value=(plan, plan)), \
                patch.object(ev, "load_tracker", return_value=loaded), \
                patch.dict(sys.modules, {"lib.train.dataset.depth_utils": depth}), \
                redirect_stdout(io.StringIO()):
            ev.ope_track(path)
        receipt = ev.read(self.root / "predictions/receipt.json")
        self.assertEqual(receipt["frames"], 76373)
        self.assertEqual(len(receipt["sequences"]), 50)
        self.assertFalse(receipt["subsequent_GT_opened"])
        rows = ev.np.loadtxt(self.root / "predictions/seq000.txt", delimiter=",")
        self.assertEqual(rows.shape, (76324, 4))
        self.assertEqual(rows[0].tolist(), cases[0]["init_bbox"])
        self.assertEqual(rows[-1].tolist(), [0.25, 0.5, 4., 5.])

    def test_trax_initialization_and_reports_without_gt(self):
        tracker, initial = self.tracker_fixture()
        image = self.root / "first.jpg"
        image.write_bytes(b"legal initialization fixture")
        bbox = [0., 0., 4., 5.]
        frames = iter([[str(image), "depth.png"], ["next.jpg", "next.png"], None])
        reports, initialized = [], []
        tracker.initialize = lambda *args: initialized.append(args)
        handle = SimpleNamespace(region=lambda: bbox, frame=lambda: next(frames),
                                 report=lambda *args: reports.append(args))
        bridge = SimpleNamespace(VOT=lambda *args, **kwargs: handle,
                                 Rectangle=lambda *args: list(args))
        loader = SimpleNamespace(exec_module=lambda module: None)
        plan = dict(dataset="vot", vot_bridge_path="fixture.py")
        indices = {"vot:" + ev.initialization_key(ev.sha(image), bbox): 3}
        loaded = (SimpleNamespace(no_grad=nullcontext), tracker, {}, indices, initial, lambda x: x)
        depth = SimpleNamespace(get_rgbd_frame=lambda *args, **kwargs: None)
        with patch.object(ev, "checked_plan", return_value=(plan, {})), \
                patch.object(ev, "load_tracker", return_value=loaded), \
                patch.object(ev.importlib.util, "spec_from_file_location", return_value=SimpleNamespace(loader=loader)), \
                patch.object(ev.importlib.util, "module_from_spec", return_value=bridge), \
                patch.dict(sys.modules, {"lib.train.dataset.depth_utils": depth}):
            ev.vot_track(self.root / "plan.json")
        self.assertEqual(initialized, [(None, bbox, 3)])
        self.assertEqual(reports, [([0.25, 0.5, 4., 5.], 0.75)])

    def test_full_preparer_control_plane_and_terminal_gate(self):
        # Serialized tensor reads are mocked; filesystem, counts and hashes are real.
        port = Path(__file__).resolve().parent
        torch_fixture = SimpleNamespace()
        spec = importlib.util.spec_from_file_location(
            "fixture_preparer", port / "prepare_evaluation.py")
        prep = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch_fixture}):
            spec.loader.exec_module(prep)
        root = self.root / "training"
        fit = root / "fit"
        fit.mkdir(parents=True)
        source = {}
        for folder in ("code", "legacy-code", "native-source"):
            path = root / folder / "fixture.py"
            path.parent.mkdir()
            path.write_text("# CPU control-plane fixture\n")
            source[path.relative_to(root).as_posix()] = ev.sha(path)
        inputs = {}
        for field in ("checkpoint", "config", "clip_weight"):
            inputs[field] = self.root / field
            inputs[field].write_text(field)
        sequence_order = [dict(sequence="train%03d" % index,
            rgb_frames=219652 if index == 0 else 2) for index in range(152)]
        inventory = root / "spec.json"
        ev.write(inventory, dict(seed=2027, sequence_order=sequence_order))
        protocol = dict(mode="train", seed=2027, passes=6,
            final_selection="fixed pass06 after all152 sequences; no external test selection",
            baseline_sha256=ev.sha(inputs["checkpoint"]),
            inputs=dict(config=ev.sha(inputs["config"]),
                        clip_weight=ev.sha(inputs["clip_weight"]), spec=ev.sha(inventory)),
            source={"fixture.py": source["code/fixture.py"]},
            legacy_source={"fixture.py": source["legacy-code/fixture.py"]})
        command = ["fixture_python", "--mode", "train", "--output", str(fit),
                   "--spec", str(inventory)]
        ev.write(root / "launch.json", dict(command=command, source_sha256=source))
        child = dict(supervisor_pid=1, child_pid=2, command=command)
        ev.write(root / "child.json", child)
        ev.write(root / "terminal.json", dict(child, exit_code=0))
        (root / "train.exit").write_text("0\n")
        rows = []
        calls = 0
        for epoch in range(6):
            for case in sequence_order:
                calls += case["rgb_frames"] - 1
                rows.append(dict(epoch=epoch + 1, sequence=case["sequence"],
                    frames=case["rgb_frames"] - 1, calls=calls,
                    loss_sum=1.0, steps=len(rows) + 1, C_steps=len(rows) + 1))
        (fit / "sequences.jsonl").write_text(
            "".join(__import__("json").dumps(row) + "\n" for row in rows))
        final_path = fit / "final.pt"
        final_path.write_bytes(b"mocked serialized tensor fixture")
        ev.write(fit / "result.json", dict(status="complete_train", calls=calls,
            steps=912, C_steps=912, final_sha256=ev.sha(final_path), formal_metrics=False,
            frozen_before_after_exact=True, module_roundtrip_exact=True))
        base = self.root / "base"
        base.mkdir()
        binding = []
        all_keys = []
        cases_by_dataset = {}
        for dataset, count, frames in (("depthtrack_test", 50, 76373), ("cdtb", 80, 101956)):
            cases = []
            for index in range(count):
                sequence = "seq%03d" % index
                image = self.root / dataset / sequence / "color/00000001.jpg"
                image.parent.mkdir(parents=True)
                image.write_bytes((dataset + sequence).encode())
                bbox = [0., 0., 1., 1.]
                key = dataset + ":" + ev.initialization_key(ev.sha(image), bbox)
                all_keys.append(key)
                binding.append(dict(dataset=dataset, key=key, human_confirmed=True,
                    image=str(image), image_sha256=ev.sha(image), init_bbox=bbox))
                cases.append(dict(sequence=sequence, init_bbox=bbox, gt_sha256="unused",
                                  frames=frames - count + 1 if index == 0 else 1))
            cases_by_dataset[dataset] = cases
        vot_image = self.root / "vot_first_frame.jpg"
        vot_image.write_bytes(b"fixture legal initialization image")
        for index in range(1765):
            bbox = [float(index), 0., 1., 1.]
            key = "vot:" + ev.initialization_key(ev.sha(vot_image), bbox)
            all_keys.append(key)
            binding.append(dict(dataset="vot", key=key, human_confirmed=True,
                image=str(vot_image), image_sha256=ev.sha(vot_image), init_bbox=bbox))
        binding_path = self.root / "binding.json"
        ev.write(binding_path, dict(rows=binding))
        bank_path = self.root / "bank.pt"
        bank_path.write_bytes(b"mocked bank tensor fixture")
        bank = dict(format="M122_human_official_initialization_v1", human_confirmed=True,
            binding_sha256=ev.sha(binding_path), encoder_sha256=ev.sha(inputs["clip_weight"]),
            keys=all_keys, tokens=SimpleNamespace(shape=(1895, 5, 768)),
            mask=SimpleNamespace(shape=(1895, 5)))
        torch_fixture.load = lambda path, **kwargs: (
            dict(protocol=protocol) if Path(path) == final_path else bank)
        bridge = self.root / "bridge.py"
        bridge.write_text("# fixture\n")
        bundle_path = base / "bundle.json"
        ev.write(bundle_path, dict(schema="M125_region_fixed6_Train152_official_v1",
            vot_bridge_path=str(bridge), failure_source_path=str(bridge),
            failure_common_path=str(bridge)))
        metric = self.root / "metric.py"
        metric.write_text("# fixture metric; not executed\n")
        reference = self.root / "reference.json"
        ev.write(reference, dict(fixture=True))
        for dataset in ("depthtrack_test", "cdtb", "vot"):
            folder = base / dataset
            folder.mkdir()
            original = folder / "original.json"
            ev.write(original, {})
            plan = dict(dataset=dataset, base_plan_path=str(original),
                base_plan_sha256=ev.sha(original), bundle_sha256=ev.sha(bundle_path),
                bank_path=str(bank_path), bank_sha256=ev.sha(bank_path),
                binding_path=str(binding_path), binding_sha256=ev.sha(binding_path))
            if dataset != "vot":
                cases_path = folder / "cases.json"
                ev.write(cases_path, cases_by_dataset[dataset])
                plan.update(cases_path=str(cases_path), cases_sha256=ev.sha(cases_path),
                    dataset_root=str(self.root / dataset), metric_source=str(metric),
                    metric_source_sha256=ev.sha(metric))
            ev.write(folder / "plan.json", plan)
        run = base / "vot/run"
        run.mkdir()
        shards = []
        offset = 0
        for index, anchors in enumerate((441, 441, 441, 442)):
            folder = run / ("shard-%02d" % index)
            (folder / "sequences").mkdir(parents=True)
            (folder / "sequences/list.txt").write_text("fixture\n")
            (folder / "config.yaml").write_text("stack: vot2022/rgbd\n")
            shards.append(dict(index=index, root=str(folder), gpu=index % 2,
                anchor_count=anchors, expected_trajectories=[
                    "seq%03d_%08d" % (i % 127, i) for i in range(offset, offset + anchors)]))
            offset += anchors
        manifest_source = {}
        for i in range(127):
            name = "seq%03d" % i
            folder = self.root / "vot_sequences" / name
            folder.mkdir(parents=True)
            anchor = folder / "anchor.value"
            anchor.write_text("0\n")
            manifest_source[name] = dict(root=str(folder), anchor_sha256=ev.sha(anchor))
        ev.write(run / "shard_manifest.json", dict(
            schema="M122_full127_frozen_shards_v1", total_anchor_count=1765,
            sequences=["seq%03d" % i for i in range(127)], shards=shards,
            source=manifest_source))
        ev.write(base / "vot/execution.json", dict(source_sha256={}))
        output = self.root / "evaluation-output"
        argv = ["prepare_evaluation.py", "--training-root", str(root),
            "--base-evaluation", str(base), "--checkpoint", str(inputs["checkpoint"]),
            "--config", str(inputs["config"]), "--clip-weight", str(inputs["clip_weight"]),
            "--evaluation-source", str(port / "evaluation.py"),
            "--evaluation-runner", str(port / "run_evaluation.py"),
            "--vot-identity-tool", str(bridge), "--vot-reference", str(reference),
            "--output", str(output)]
        (root / "train.exit").write_text("7\n")
        with patch.object(sys, "argv", argv), self.assertRaises(AssertionError):
            prep.main()
        self.assertFalse(output.exists())
        (root / "train.exit").write_text("0\n")
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                patch.object(prep, "LOCKED_METRIC_SHA256", ev.sha(metric)), \
                patch.object(prep, "REFERENCE_SHA256", ev.sha(reference)):
            prep.main()
        plans = [ev.checked_plan(output / dataset / "plan.json")[0]
                 for dataset in ("depthtrack_test", "cdtb", "vot")]
        self.assertEqual(len({plan["final_sha256"] for plan in plans}), 1)
        self.assertEqual(len({plan["bundle_sha256"] for plan in plans}), 1)
        self.assertTrue(all(Path(plan["final_path"]).is_absolute() for plan in plans))
        self.assertFalse((output / "depthtrack_test/predictions").exists())


if __name__ == "__main__":
    unittest.main()
