"""Behavioral regression tests; no model downloads or GPU required.
Run from the package directory: python -m unittest discover -s tests -v
"""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from speechevalx.alignment import word_alignment
from speechevalx.corruption import generate_matched_substitutions
from speechevalx.datasets import ReferenceAction, SpeechEvalSample
from speechevalx.datasets.bfcl_audio import BFCLAudioAdapter
from speechevalx.datasets.manifest import normalize_record
from speechevalx.experiment import run_paired_variant
from speechevalx.models import MockAgent
from speechevalx.parsing import ToolCallParseError, parse_tool_call
from speechevalx.prediction import parse_output, error_category
from speechevalx.scoring import score_action, ScoringConfig
from speechevalx.metrics.lexical import wer, corpus_wer
from speechevalx.reporting import paired_report, aggregate_actions
from speechevalx.runner import run


class CoreTests(unittest.TestCase):
    def test_empty_output_is_failure_on_no_call_reference(self):
        parsed = parse_output("  ")
        score = score_action(parsed.action, ReferenceAction(None))
        self.assertFalse(score.jca)
        self.assertEqual(error_category(parsed, ReferenceAction(None), score), "malformed_output")
        self.assertTrue(score_action(parse_output("NO_CALL").action, ReferenceAction(None)).jca)

    def test_malformed_call_can_contribute_to_tool_denominator(self):
        parsed = parse_output("search(query=)")
        score = score_action(parsed.action, ReferenceAction("search", {"query": "x"}), recovered_tool=parsed.recovered_tool)
        self.assertTrue(score.tool_correct)
        self.assertFalse(score.jca)
        self.assertEqual(error_category(parsed, ReferenceAction("search"), score), "malformed_output")

    def test_duplicate_keywords_are_malformed(self):
        with self.assertRaises(ToolCallParseError): parse_tool_call("search(query='x', query='y')")

    def test_bool_is_not_a_numeric_value(self):
        self.assertFalse(score_action(ReferenceAction("set", {"n": True}), ReferenceAction("set", {"n": 1})).jca)

    def test_failure_precedence(self):
        expected = ReferenceAction("search", {"query": "x"})
        cases = {"broken output": "malformed_output", "NO_CALL": "call_decision", "other(query='x')": "wrong_tool",
                 "search()": "argument_structure", "search(query='y')": "argument_value"}
        for output, category in cases.items():
            parsed = parse_output(output)
            score = score_action(parsed.action, expected, recovered_tool=parsed.recovered_tool)
            self.assertEqual(error_category(parsed, expected, score), category)

    def test_extra_arguments_are_configurable(self):
        expected = ReferenceAction("search", {"query": "x"})
        prediction = ReferenceAction("search", {"query": "x", "unexpected": 1})
        self.assertFalse(score_action(prediction, expected).jca)
        self.assertTrue(score_action(prediction, expected, ScoringConfig(allow_extra_arguments=True)).jca)

    def test_numeric_alias_option(self):
        expected = ReferenceAction("set", {"number": 22})
        prediction = ReferenceAction("set", {"number": "22"})
        self.assertTrue(score_action(prediction, expected).jca)
        self.assertFalse(score_action(prediction, expected, ScoringConfig(normalize_numbers=False)).jca)

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_schema_types_before_numeric_aliasing(self):
        tools = [{"name": "set", "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}]
        expected = ReferenceAction("set", {"number": 22})
        score = score_action(ReferenceAction("set", {"number": "22"}), expected, tool_schema=tools)
        self.assertFalse(score.argument_structure_valid)
        self.assertFalse(score.jca)

    def test_matched_generator_checks_actual_edits(self):
        invalid = [("22", "22"), ("22", ""), ("22", "32 42"), ("to Alice", "to Bob")]
        for rule in invalid:
            self.assertEqual(generate_matched_substitutions("s", "Transfer 22 to Alice", {"number": rule}), [])
        record, changed = generate_matched_substitutions("s", "Transfer 22 to Alice", {"number": ("22", "32")})[0]
        self.assertEqual(len(word_alignment("Transfer 22 to Alice", changed)), 1)
        self.assertEqual(record.severity, .25)
        self.assertEqual(generate_matched_substitutions("s", "22 22", {"number": ("22", "32")}), [])
        self.assertEqual(generate_matched_substitutions("s", "22.5", {"number": ("22", "32")}), [])

    def test_corpus_wer_is_not_utterance_average(self):
        pairs = [("x", "y"), ("a b c d e f g h i", "a b c d e f g h i")]
        self.assertEqual(corpus_wer(pairs)["wer"], .1)
        self.assertEqual(wer("ＦＯＯ, BAR!", "foo bar"), 0)

    def test_bfcl_missing_action_is_not_no_call(self):
        adapter = BFCLAudioAdapter(Path("."))
        raw = {"id": "x", "instruction": "Transfer 22", "path": "x.wav", "tools": []}
        with self.assertRaises(ValueError): adapter.normalize(raw)
        sample = adapter.normalize({**raw, "reference_action": None})
        self.assertEqual(sample.reference_transcript, "Transfer 22")
        self.assertEqual(sample.instruction, "")

    def test_manifest_rejects_missing_or_leaked_reference(self):
        raw = {"id": "s", "reference_transcript": "Transfer 22", "tool_schema": [], "reference_action": None}
        with self.assertRaises(ValueError): normalize_record({k: v for k, v in raw.items() if k != "reference_action"}, Path("."))
        with self.assertRaises(ValueError): normalize_record({**raw, "context": "Original: Transfer 22"}, Path("."))
        self.assertTrue(normalize_record(raw, Path(".")).reference_action.no_call)

    def test_pair_rejects_wrong_source_and_edit_metadata(self):
        sample = SpeechEvalSample("s", "demo", Path("x.wav"), "Transfer 22", [], ReferenceAction(None))
        corruption, text = generate_matched_substitutions("other", sample.reference_transcript, {"number": ("22", "32")})[0]
        with self.assertRaises(ValueError): run_paired_variant(sample, corruption, text, MockAgent("NO_CALL"))
        corruption, text = generate_matched_substitutions("s", sample.reference_transcript, {"number": ("22", "32")})[0]
        with self.assertRaises(ValueError): run_paired_variant(sample, corruption, "Transfer 22", MockAgent("NO_CALL"))

    def test_common_cohort_excludes_partial_source(self):
        def row(source, role, changed):
            return {"source_sample_id": source, "benchmark": "demo", "agent_id": "a", "analysis_type": "controlled",
                    "error_class": role, "clean_jca": True, "corrupted_jca": changed,
                    "paired_outcome": "preserved_success" if changed else "harmed", "first_changed_failure": "argument_value"}
        rows = [row("one", "number", False), row("one", "entity", True), row("two", "number", False)]
        report = paired_report(rows, ["number", "entity"], 50)["a"]
        self.assertEqual(report["common_sources"], 1)
        self.assertEqual({r["n"] for r in report["roles"]}, {1})
        self.assertEqual(report["harmed_pairs"], 1)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data = [json.loads(line) for line in (PACKAGE / "examples/demo.jsonl").read_text().splitlines()]
    def tearDown(self): self.temp.cleanup()
    def config(self, data=None, **options):
        (self.directory / "data.jsonl").write_text("\n".join(json.dumps(r) for r in (data if data is not None else self.data)), encoding="utf-8")
        config = {"data": "data.jsonl", "output": "output", "roles": ["number", "entity", "function_word"],
                  "agents": {"replay": {"type": "replay"}}, "bootstrap_iterations": 50, **options}
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_offline_end_to_end_and_reports(self):
        config = self.config()
        self.assertEqual(run(config, validate_only=True)["controlled_eligible_sources"], 3)
        result = run(config)
        metrics = {r["condition"]: r for r in result["actions"]}
        self.assertEqual(metrics["reference"]["jca"], 1)
        self.assertEqual(metrics["asr:whisper"]["jca"], 0)
        self.assertEqual(result["paired"]["replay"]["common_sources"], 3)
        self.assertEqual(result["instrumentation_failures"], 0)
        output = self.directory / "output"
        for filename in ("actions.jsonl", "paired_traces.jsonl", "metrics.csv", "report.md", "summary.json", "environment.json"):
            self.assertTrue((output / filename).is_file())
        with self.assertRaises(ValueError): run(config)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional")
    def test_batch_diagnostics_saves_oof_predictions_and_transfer(self):
        rows = []
        for i in range(12):
            row = copy.deepcopy(self.data[0])
            row["id"] = "source-" + str(i)
            row["features"] = {role: {"bert_cosine": .1, "semantic_wer": .2} for role in row["substitutions"]}
            if i % 2: row["predictions"]["replay"]["entity"] = row["predictions"]["replay"]["reference"]
            row["predictions"]["second"] = copy.deepcopy(row["predictions"]["replay"])
            rows.append(row)
        result = run(self.config(rows, agents={"replay": {"type": "replay"}, "second": {"type": "replay"}},
            observed_conditions=[], diagnostics={"enabled": True, "primary_agent": "replay", "folds": 3, "bootstrap_iterations": 20}))
        self.assertEqual(result["diagnostics"]["status"], "complete")
        self.assertEqual(result["diagnostics"]["transfer"]["second"]["B4"]["n"], 36)
        entries = (self.directory / "output/diagnostic_predictions.jsonl").read_text().splitlines()
        self.assertEqual(len(entries), 12 * 3 * 2 * 5)
        self.assertTrue((self.directory / "output/input_manifest.jsonl").is_file())

    def test_missing_model_result_is_instrumentation_not_action_failure(self):
        del self.data[0]["predictions"]["replay"]["number"]
        result = run(self.config())
        metrics = {r["condition"]: r for r in result["actions"]}
        self.assertEqual(metrics["number"]["scored"], 2)
        self.assertEqual(metrics["number"]["eligible"], 3)
        self.assertEqual(metrics["number"]["instrumentation_failures"], 1)
        self.assertEqual(result["paired"]["replay"]["common_sources"], 2)

    def test_missing_observed_transcript_is_reported_in_coverage(self):
        del self.data[0]["hypotheses"]["whisper"]
        result = run(self.config())
        metric = next(r for r in result["actions"] if r["condition"] == "asr:whisper")
        self.assertEqual(metric["eligible"], 4)
        self.assertEqual(metric["scored"], 3)
        self.assertEqual(result["instrumentation_failures"], 1)

    def test_duplicates_rejected_before_inference(self):
        with self.assertRaises(ValueError): run(self.config(self.data + self.data[:1]))

    def test_audio_backend_is_used_when_hypothesis_is_absent(self):
        from speechevalx.models.asr.base import ASRResult
        for row in self.data:
            row["hypotheses"] = {}
            row["audio_path"] = "audio.wav"
        class Backend:
            def transcribe(self, filename): return ASRResult("Transfer 32 dollars to Alice", "test-asr")
        with patch("speechevalx.runner.make_asr", return_value=Backend()):
            result = run(self.config(asr={"whisper": {"type": "test"}}))
        self.assertEqual(result["instrumentation_failures"], 0)
        self.assertEqual(next(r for r in result["actions"] if r["condition"] == "asr:whisper")["scored"], 4)

    def test_unknown_agent_config_rejected_by_validate(self):
        with self.assertRaises(ValueError): run(self.config(agents={"bad": {"type": "unknown"}}), validate_only=True)

    def test_converter_preserves_annotations_and_removes_user_context(self):
        from speechevalx.datasets.convert import convert_dataset
        directory = self.directory / "bfcl"
        directory.mkdir()
        raw = {"id": "x", "instruction": "Transfer 22", "path": "x.wav", "tools": [], "reference_action": None,
               "substitutions": {"number": ["22", "32"]}}
        (directory / "data.json").write_text(json.dumps([raw]), encoding="utf-8")
        output = self.directory / "converted.jsonl"
        result = convert_dataset(directory, output, "bfcl_audio")
        row = json.loads(output.read_text())
        self.assertEqual(result["sources"], 1)
        self.assertEqual(row["context"], "")
        self.assertEqual(row["reference_transcript"], "Transfer 22")
        self.assertEqual(row["substitutions"], raw["substitutions"])
        self.assertIsNone(row["reference_action"])

    def test_malformed_json_is_not_silently_skipped_by_converter(self):
        from speechevalx.datasets.convert import convert_dataset
        directory = self.directory / "bad"
        directory.mkdir()
        (directory / "bad.json").write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError): convert_dataset(directory, self.directory / "out.jsonl", "bfcl_audio")

    def test_partial_roles_are_reported_before_inference(self):
        del self.data[0]["substitutions"]["entity"]
        result = run(self.config(), validate_only=True)
        self.assertEqual(result["controlled_eligible_sources"], 2)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional")
class DiagnosticTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(18):
            for agent in ("primary", "second"):
                for role in ("number", "entity", "function_word"):
                    failed = role == "number" or (role == "entity" and i % 2 == 0)
                    rows.append({"analysis_type": "controlled", "agent_id": agent, "clean_jca": True,
                        "corrupted_jca": not failed, "group_id": f"group-{i}", "source_sample_id": str(i),
                        "benchmark": "demo", "error_class": role, "edit_operation": "substitution",
                        "edit_count": 1, "severity": .2, "reference_transcript": "Transfer 22 dollars to Alice",
                        "corrupted_transcript": "Transfer 32 dollars to Alice", "features": {"bert_cosine": .1, "semantic_wer": .2}})
        return rows

    def test_all_ablation_levels_and_transfer_are_held_out_by_group(self):
        from speechevalx.diagnostics import run_diagnostics
        result, predictions = run_diagnostics(self.rows(), {"primary_agent": "primary", "folds": 3, "bootstrap_iterations": 30})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["models"]), {"B0", "B1", "B2", "B3", "B4"})
        self.assertEqual(result["transfer"]["second"]["B4"]["n"], 54)
        self.assertGreater(result["models"]["B4"]["roc_auc"], .9)
        assignments = {}
        for row in predictions:
            assignments.setdefault(row["group_id"], set()).add(row["fold"])
        self.assertTrue(all(len(folds) == 1 for folds in assignments.values()))
        self.assertIn("B4_minus_B3", result)

    def test_missing_semantic_controls_are_not_fabricated(self):
        from speechevalx.diagnostics import run_diagnostics
        rows = self.rows()
        for row in rows: row["features"] = {}
        result, _ = run_diagnostics(rows, {"primary_agent": "primary", "folds": 3})
        self.assertEqual(set(result["models"]), {"B0"})
        self.assertTrue(result["status"].startswith("partial"))

    def test_too_few_groups_has_explicit_status(self):
        from speechevalx.diagnostics import run_diagnostics
        rows = self.rows()[:6]
        result, _ = run_diagnostics(rows, {"primary_agent": "primary", "folds": 5})
        self.assertTrue(result["status"].startswith("unavailable"))


if __name__ == "__main__": unittest.main()
