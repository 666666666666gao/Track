import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module("cadtrack_pure_evaluation", "evaluation.py")
preparation = load_module("cadtrack_pure_prepare", "prepare_evaluation.py")


class EvaluationTests(unittest.TestCase):
    def test_final_five_selection_binds_all_checkpoints_and_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "final_five").mkdir()
            (root / "evaluations").mkdir()
            rows = []
            for epoch in range(31, 36):
                checkpoint = root / "final_five" / ("epoch_%02d.pth.tar" % epoch)
                checkpoint.write_bytes(("weights-%d" % epoch).encode("ascii"))
                digest = preparation.sha(checkpoint)
                metrics = dict(
                    precision_percent=60.0,
                    recall_percent=55.0,
                    f_score=50.0 + epoch - 30,
                )
                rows.append(
                    dict(epoch=epoch, checkpoint_sha256=digest, metrics=metrics)
                )
                folder = root / "evaluations" / ("epoch_%02d" % epoch)
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(
                    json.dumps(
                        dict(checkpoint_sha256=digest, metrics=metrics)
                    ),
                    encoding="utf-8",
                )
            (root / "launch.json").write_text(
                json.dumps(dict(status="complete")), encoding="utf-8"
            )
            (root / "epoch_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected = rows[4]
            path = root / "final_five" / "epoch_35.pth.tar"
            (root / "best.json").write_text(
                json.dumps(
                    dict(
                        selection="DepthTrack Test50 F-score, epochs 31-35",
                        epoch=35,
                        f_score=selected["metrics"]["f_score"],
                        checkpoint_path=str(path),
                        checkpoint_sha256=selected["checkpoint_sha256"],
                    )
                ),
                encoding="utf-8",
            )
            result, retained, metrics = preparation.verify_final_five(root, path)
            self.assertEqual(result["epoch"], 35)
            self.assertEqual([row["epoch"] for row in retained], list(range(31, 36)))
            self.assertEqual(len(metrics), 5)

    def test_final_five_rejects_changed_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "final_five").mkdir()
            (root / "evaluations").mkdir()
            rows = []
            for epoch in range(31, 36):
                path = root / "final_five" / ("epoch_%02d.pth.tar" % epoch)
                path.write_bytes(("weights-%d" % epoch).encode("ascii"))
                digest = preparation.sha(path)
                rows.append(
                    dict(
                        epoch=epoch,
                        checkpoint_sha256=digest,
                        metrics=dict(f_score=float(epoch)),
                    )
                )
                folder = root / "evaluations" / ("epoch_%02d" % epoch)
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(
                    json.dumps(dict(checkpoint_sha256=digest, metrics=rows[-1]["metrics"])),
                    encoding="utf-8",
                )
            rows[0]["checkpoint_sha256"] = "0" * 64
            (root / "launch.json").write_text('{"status":"complete"}', encoding="utf-8")
            (root / "epoch_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            best_path = root / "final_five" / "epoch_35.pth.tar"
            (root / "best.json").write_text(
                json.dumps(
                    dict(
                        selection="DepthTrack Test50 F-score, epochs 31-35",
                        epoch=35,
                        f_score=35.0,
                        checkpoint_path=str(best_path),
                        checkpoint_sha256=preparation.sha(best_path),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaises(AssertionError):
                preparation.verify_final_five(root, best_path)

    def test_vot_protocol_requires_exact_workspace_stack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name, content in (
                ("checkpoint", b"weights"),
                ("selection", b"selection"),
                ("source", b"source"),
                ("identity", b"identity"),
                ("coverage", b"coverage"),
                ("config", b"config"),
                ("reference", b"reference"),
                ("bridge", b"bridge"),
                ("evaluator", b"evaluator"),
            ):
                path = root / name
                path.write_bytes(content)
                paths[name] = path
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    dict(
                        source_sha256={},
                        evaluator_path=str(paths["evaluator"]),
                        evaluator_sha256=evaluation.sha(paths["evaluator"]),
                    )
                ),
                encoding="utf-8",
            )
            execution = root / "execution.json"
            execution.write_text(
                json.dumps(dict(source_sha256={})), encoding="utf-8"
            )
            protocol = dict(
                analyses=[dict(type="locked-analysis", burnin=10)],
                anchor="locked-anchor",
            )
            (root / "bundle.json").write_text(
                json.dumps(
                    dict(
                        schema="CADTrack_pure_finetune_bundle_v1",
                        checkpoint_sha256=evaluation.sha(paths["checkpoint"]),
                        selected_epoch=33,
                        selection_file_sha256=evaluation.sha(paths["selection"]),
                        evaluation_source_sha256=evaluation.sha(paths["source"]),
                        source_manifest_sha256=evaluation.sha(source_manifest),
                        evaluator_sha256=evaluation.sha(paths["evaluator"]),
                        vot_execution_sha256=evaluation.sha(execution),
                        inference_profile=dict(
                            config_path=str(paths["config"]),
                            config_sha256=evaluation.sha(paths["config"]),
                        ),
                        vot_reference_sha256=evaluation.sha(paths["reference"]),
                        vot_identity_tool_sha256=evaluation.sha(paths["identity"]),
                        vot_coverage_tool_sha256=evaluation.sha(paths["coverage"]),
                    )
                ),
                encoding="utf-8",
            )
            plan = dict(
                schema="CADTrack_pure_finetune_evaluation_v1",
                checkpoint_path=str(paths["checkpoint"]),
                checkpoint_sha256=evaluation.sha(paths["checkpoint"]),
                selection_path=str(paths["selection"]),
                selection_file_sha256=evaluation.sha(paths["selection"]),
                evaluation_source_path=str(paths["source"]),
                evaluation_source_sha256=evaluation.sha(paths["source"]),
                vot_identity_tool_path=str(paths["identity"]),
                vot_identity_tool_sha256=evaluation.sha(paths["identity"]),
                vot_coverage_tool_path=str(paths["coverage"]),
                vot_coverage_tool_sha256=evaluation.sha(paths["coverage"]),
                source_manifest_path=str(source_manifest),
                source_manifest_sha256=evaluation.sha(source_manifest),
                evaluator_path=str(paths["evaluator"]),
                evaluator_sha256=evaluation.sha(paths["evaluator"]),
                inference_profile=dict(
                    config_path=str(paths["config"]),
                    config_sha256=evaluation.sha(paths["config"]),
                ),
                selection="test-set-selected",
                selection_metric="DepthTrack Test50 F-score",
                selected_epoch=33,
                retained_checkpoints=[
                    dict(epoch=i, path=str(paths["checkpoint"]), sha256=evaluation.sha(paths["checkpoint"]))
                    for i in range(31, 36)
                ],
                bundle_path=str(root / "bundle.json"),
                bundle_sha256=evaluation.sha(root / "bundle.json"),
                dataset="vot",
                vot_bridge_path=str(paths["bridge"]),
                vot_bridge_sha256=evaluation.sha(paths["bridge"]),
                vot_reference_path=str(paths["reference"]),
                vot_reference_sha256=evaluation.sha(paths["reference"]),
                vot_execution_path=str(execution),
                vot_execution_sha256=evaluation.sha(execution),
                protocol_receipt=str(root / "protocol_receipt.json"),
                protocol_workspaces=[],
            )
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            module = type("FakeVot", (), {"__version__": "0.7.1"})
            workspace_module = type(
                "FakeWorkspaceModule",
                (),
                {
                    "Workspace": type(
                        "FakeWorkspace",
                        (),
                        {
                            "load": staticmethod(
                                lambda path: type(
                                    "WorkspaceObj",
                                    (),
                                    {
                                        "stack": [
                                            type(
                                            "MultiStartExperiment",
                                                (),
                                                {
                                                "__module__": "vot.experiment.multistart",
                                                    "identifier": "baseline",
                                                    "dump": lambda self: protocol,
                                                },
                                            )()
                                        ]
                                    },
                                )()
                            )
                        },
                    )
                },
            )
            import sys
            with mock.patch.dict(
                sys.modules,
                {
                    "vot": module,
                    "vot.workspace": workspace_module,
                },
            ):
                plan["vot_reference_path"] = str(root / "reference.json")
                plan["vot_reference_sha256"] = evaluation.sha(paths["reference"])
                plan["protocol_workspaces"] = [str(root / ("workspace-%d" % i)) for i in range(4)]
                reference_data = dict(
                    toolkit="0.7.1",
                    results=dict(
                        baseline=dict(
                            type="vot.experiment.multistart.MultiStartExperiment",
                            parameters=protocol,
                        )
                    ),
                )
                reference_file = root / "reference.json"
                reference_file.write_text(json.dumps(reference_data), encoding="utf-8")
                plan["vot_reference_sha256"] = evaluation.sha(reference_file)
                bundle_path = root / "bundle.json"
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                bundle["vot_reference_sha256"] = plan["vot_reference_sha256"]
                bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
                plan["bundle_sha256"] = evaluation.sha(bundle_path)
                for workspace in plan["protocol_workspaces"]:
                    folder = Path(workspace)
                    folder.mkdir()
                    (folder / "config.yaml").write_text("config", encoding="utf-8")
                    (folder / "trackers.ini").write_text("tracker", encoding="utf-8")
                evaluation.write(plan_path, plan)
                evaluation.vot_protocol(plan_path)
                receipt = json.loads(
                    (root / "protocol_receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["status"], "VOT_protocol_reference_exact")
                protocol["anchor"] = "changed"
                with self.assertRaises(AssertionError):
                    evaluation.vot_protocol(plan_path)

    def test_complete_anchor_requires_all_three_vot_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_root = root / "shard"
            folder = shard_root / "results" / "tracker" / "baseline" / "seq"
            folder.mkdir(parents=True)
            name = "seq_00000000"
            for suffix in (".bin", "_confidence.value", "_time.value"):
                (folder / (name + suffix)).write_bytes(b"x")
            shard = dict(root=str(shard_root), expected_trajectories=[name])
            self.assertEqual(evaluation.complete_anchors(shard, "tracker"), 1)
            (folder / (name + "_time.value")).unlink()
            self.assertEqual(evaluation.complete_anchors(shard, "tracker"), 0)

    def test_prepared_plans_validate_and_reject_changed_source(self):
        import sys

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            (training / "final_five").mkdir(parents=True)
            rows = []
            for epoch in range(31, 36):
                checkpoint = training / "final_five" / ("epoch_%02d.pth.tar" % epoch)
                checkpoint.write_bytes(("weights-%d" % epoch).encode("ascii"))
                row = dict(epoch=epoch, checkpoint_sha256=preparation.sha(checkpoint),
                           metrics=dict(f_score=float(epoch)))
                rows.append(row)
                folder = training / "evaluations" / ("epoch_%02d" % epoch)
                folder.mkdir(parents=True)
                preparation.write(folder / "metrics.json", row)
            preparation.write(training / "launch.json", dict(status="complete"))
            (training / "epoch_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            preparation.write(training / "best.json", dict(
                selection="DepthTrack Test50 F-score, epochs 31-35",
                epoch=35, f_score=35.0, checkpoint_path=str(checkpoint),
                checkpoint_sha256=rows[-1]["checkpoint_sha256"]))
            probe = root / "probe"
            native = probe / "source"
            config = native / "experiments/cadtrack/cadtrack.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("fixture-config", encoding="utf-8")
            (probe / "evaluate_ope.py").write_text("fixture-evaluator", encoding="utf-8")
            base = root / "base"
            base.mkdir()
            preparation.write(base / "bundle.json", dict(status="fixture"))
            metric = root / "metric.py"
            metric.write_text("fixture-metric", encoding="utf-8")
            for name, count, frames in (("depthtrack_test", 50, 76373),
                                        ("cdtb", 80, 101956)):
                folder = base / name
                folder.mkdir()
                cases = [dict(sequence="seq%d" % i, frames=1) for i in range(count)]
                cases[0]["frames"] += frames - count
                preparation.write(folder / "cases.json", cases)
                preparation.write(folder / "plan.json", dict(
                    dataset=name, bundle_sha256=preparation.sha(base / "bundle.json"),
                    cases_path=str(folder / "cases.json"),
                    cases_sha256=preparation.sha(folder / "cases.json"),
                    metric_source=str(metric), metric_source_sha256=preparation.sha(metric),
                    dataset_root=str(root / name)))
            sequences = ["seq%d" % i for i in range(127)]
            data = root / "vot-data"
            data.mkdir()
            source = {}
            for name in sequences:
                folder = data / name
                folder.mkdir()
                (folder / "anchor.value").write_text("1", encoding="utf-8")
                source[name] = dict(root=str(folder))
            shards = []
            start = 0
            for index, count in enumerate((441, 441, 441, 442)):
                folder = root / ("old-shard-%d" % index)
                (folder / "sequences").mkdir(parents=True)
                (folder / "sequences/list.txt").write_text(
                    "\n".join(sequences), encoding="utf-8")
                (folder / "config.yaml").write_text("fixture-stack", encoding="utf-8")
                shards.append(dict(
                    index=index, root=str(folder), anchor_count=count,
                    list_sha256=preparation.sha(folder / "sequences/list.txt"),
                    expected_trajectories=["seq0_%08d" % i for i in range(start, start + count)]))
                start += count
            (base / "vot/run").mkdir(parents=True)
            preparation.write(base / "vot/run/shard_manifest.json", dict(
                schema="M122_full127_frozen_shards_v1", sequences=sequences,
                total_anchor_count=1765, shards=shards, source=source))
            reference = root / "reference.json"
            preparation.write(reference, dict(toolkit="0.7.1"))
            identity = root / "identity.py"
            coverage = root / "coverage.py"
            identity.write_text("fixture-identity", encoding="utf-8")
            coverage.write_text("fixture-coverage", encoding="utf-8")
            output = root / "evaluation"
            arguments = ["prepare_evaluation.py"]
            for key, value in dict(
                training_root=training, base_evaluation=base, checkpoint=checkpoint,
                evaluation_source=HERE / "evaluation.py",
                evaluation_runner=HERE / "evaluation.py", vot_identity_tool=identity,
                vot_coverage_tool=coverage, vot_reference=reference, output=output
            ).items():
                arguments.extend(["--" + key.replace("_", "-"), str(value)])
            config_data = dict(TEST=dict(
                TEMPLATE_FACTOR=2.0, TEMPLATE_SIZE=128, SEARCH_FACTOR=4.0,
                SEARCH_SIZE=256, UPDATE_INTERVALS=5, UPDATE_THRESHOLD=0.65))
            fake_torch = SimpleNamespace(load=lambda *a, **k: dict(
                epoch=35, net_type="mock", net={}))
            fake_yaml = SimpleNamespace(safe_load=lambda text: config_data)
            with mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(preparation, "PROBE_ROOT", probe), \
                    mock.patch.object(preparation, "LOCKED_METRIC_SHA256", preparation.sha(metric)), \
                    mock.patch.object(preparation, "VOT_REFERENCE_SHA256", preparation.sha(reference)), \
                    mock.patch.dict(sys.modules, {"torch": fake_torch, "yaml": fake_yaml}):
                preparation.main()
            for name in ("depthtrack_test", "cdtb", "vot"):
                plan = evaluation.checked_plan(output / name / "plan.json")
                self.assertEqual(plan["selected_epoch"], 35)
            selection = preparation.read(output / "selection.json")
            self.assertEqual(selection["runner_sha256"], preparation.sha(output / "run_evaluation.py"))
            self.assertEqual(selection["vot_plan_sha256"], preparation.sha(output / "vot/plan.json"))
            config.write_text("changed-config", encoding="utf-8")
            with self.assertRaises(AssertionError):
                evaluation.checked_plan(output / "depthtrack_test/plan.json")


if __name__ == "__main__":
    unittest.main()
