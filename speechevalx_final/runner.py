"""Batch runner. All paths in config resolve relative to the config file."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

from speechevalx.alignment import word_alignment
from speechevalx.corruption import Corruption, generate_matched_substitutions
from speechevalx.corruption.matched import TASK_ROLES
from speechevalx.datasets.manifest import load_manifest
from speechevalx.experiment import run_paired_variant
from speechevalx.models.agent.base import AgentResult
from speechevalx.prediction import parse_output, error_category
from speechevalx.scoring import ScoringConfig, score_action
from speechevalx.reporting import aggregate_actions, paired_report, observed_report, markdown_report


class ReplayAgent:
    model_name = "replay"
    def generate(self, *args):
        raise RuntimeError("Replay outputs must be supplied for every requested condition")


def make_agent(config, seed):
    options = dict(config)
    kind = options.pop("type", "hf")
    if kind == "replay":
        if options: raise ValueError("Replay agent accepts only type")
        return ReplayAgent()
    if kind == "template":
        if options: raise ValueError("Template agent accepts only type")
        from speechevalx.models import TemplateAgent
        return TemplateAgent()
    options.setdefault("seed", seed)
    if kind == "hf":
        from speechevalx.models import QwenAgent
        return QwenAgent(**options)
    if kind == "compatible":
        from speechevalx.models.agent.remote import CompatibleAgent
        return CompatibleAgent(**options)
    raise ValueError(f"Unknown agent type: {kind}")


def make_asr(config):
    options = dict(config)
    kind = options.pop("type")
    if kind == "whisper":
        from speechevalx.models import WhisperASR
        return WhisperASR(**options)
    if kind == "funasr":
        from speechevalx.models.asr.funasr import FunASR
        return FunASR(**options)
    raise ValueError(f"Unknown ASR type: {kind}")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _variants(sample, raw):
    rules = raw.get("substitutions", {})
    if not isinstance(rules, dict): raise ValueError(f"{sample.sample_id}: substitutions must be an object")
    for role, rule in rules.items():
        if role not in TASK_ROLES: raise ValueError(f"{sample.sample_id}: unknown role {role}")
        if not isinstance(rule, (list, tuple)) or len(rule) != 2 or not all(isinstance(s, str) for s in rule):
            raise ValueError(f"{sample.sample_id}: substitution must be [old_token, new_token]")
    return {record.error_class: (record, text) for record, text in generate_matched_substitutions(sample.sample_id, sample.reference_transcript, rules)}


def prepare(config, base):
    if "data" not in config: raise ValueError("Config requires data")
    data = (base / config["data"]).resolve()
    sources = load_manifest(data)
    roles = config.get("roles", list(TASK_ROLES))
    if not isinstance(roles, list) or len(roles) != len(set(roles)) or any(r not in TASK_ROLES for r in roles):
        raise ValueError("roles must be a list of unique task roles; [] disables controlled edits")
    asr_configs = config.get("asr", {})
    if not isinstance(asr_configs, dict): raise ValueError("asr must map condition names to model configs")
    requested = config.get("observed_conditions")
    observed = set(asr_configs)
    prepared, eligibility = [], []
    for sample, raw in sources:
        variants = _variants(sample, raw)
        hypotheses = raw.get("hypotheses", {})
        if not isinstance(hypotheses, dict): raise ValueError("hypotheses must be an object")
        observed.update(hypotheses)
        for name, value in hypotheses.items():
            if not isinstance(name, str) or not name: raise ValueError("Hypothesis names must be nonempty strings")
            if not isinstance(value, (str, dict)) or isinstance(value, dict) and not isinstance(value.get("text"), str):
                raise ValueError("Hypothesis must be text or an object with text")
            labels = value.get("roles", []) if isinstance(value, dict) else []
            if not isinstance(labels, list) or any(label not in TASK_ROLES for label in labels):
                raise ValueError("Hypothesis roles must be a list of known task roles")
        complete = bool(roles) and all(r in variants for r in roles)
        eligibility.append({"benchmark": sample.benchmark, "source_sample_id": sample.sample_id,
                            "controlled_eligible": complete, "missing_roles": [r for r in roles if r not in variants]})
        prepared.append((sample, raw, variants if complete else {}))
    if requested is not None:
        if not isinstance(requested, list) or not all(isinstance(x, str) for x in requested):
            raise ValueError("observed_conditions must be a list of names")
        observed = set(requested)
    return data, prepared, eligibility, roles, sorted(observed)


def run(config_path: Path, validate_only=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict): raise ValueError("Config must be an object")
    base = config_path.parent
    seed = config.get("seed", 2027)
    iterations = config.get("bootstrap_iterations", 2000)
    if not isinstance(iterations, int) or iterations < 1: raise ValueError("bootstrap_iterations must be >= 1")
    agent_configs = config.get("agents")
    if not isinstance(agent_configs, dict) or not agent_configs: raise ValueError("Config needs at least one named agent")
    if config.get("on_error", "continue") not in {"continue", "raise"}: raise ValueError("on_error must be continue or raise")
    data_path, prepared, eligibility, roles, observed = prepare(config, base)
    scoring_options = dict(config.get("scoring", {}))
    scoring_options["case_sensitive_fields"] = set(scoring_options.get("case_sensitive_fields", []))
    scoring_options["required_arguments"] = {k: tuple(v) for k, v in scoring_options.get("required_arguments", {}).items()}
    scoring = ScoringConfig(**scoring_options)
    # Validate scoring rules against references before any inference.
    for sample, raw, variants in prepared:
        if not score_action(sample.reference_action, sample.reference_action, scoring, sample.tool_schema).jca:
            raise ValueError(f"Reference action does not score as correct: {sample.sample_id}")
    validation = {"sources": len(prepared), "controlled_eligible_sources": sum(r["controlled_eligible"] for r in eligibility),
                  "roles": roles, "observed_conditions": observed, "agents": list(agent_configs)}
    agents = {name: make_agent(options, seed) for name, options in agent_configs.items()}
    configured_asr = {name: make_asr(options) for name, options in config.get("asr", {}).items()}
    if validate_only: return validation
    output = (base / config.get("output", "results")).resolve()
    # Prevent accidental replacement of any previous run or the input dataset.
    if output == data_path.parent or output in data_path.parents:
        raise ValueError("Output must be a dedicated directory separate from the dataset directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    _write_jsonl(output / "eligibility.jsonl", eligibility)
    _write_jsonl(output / "input_manifest.jsonl", [raw for _, raw, _ in prepared])
    print(f"Sources: {len(prepared)}; controlled eligible: {validation['controlled_eligible_sources']}; roles: {roles}", flush=True)
    def append_record(filename, record):
        with (output / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    asr_models, actions, pairs, errors, hypothesis_log = configured_asr, [], [], [], []
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    def record_error(stage, sample, condition, exc, agent=None):
        error = {"stage": stage, "benchmark": sample.benchmark, "source_sample_id": sample.sample_id,
                 "condition": condition, "agent": agent, "error_type": type(exc).__name__, "error": str(exc)}
        errors.append(error)
        append_record("errors.jsonl", error)
        if config.get("on_error", "continue") == "raise": raise exc
        return str(exc)
    def evaluate(sample, raw, agent_id, condition, text, upstream_error=None):
        agent = agents[agent_id]
        row = {"agent_id": agent_id, "benchmark": sample.benchmark, "source_sample_id": sample.sample_id,
               "group_id": sample.metadata["group_id"], "condition": condition,
               "reference_no_call": sample.reference_action.no_call,
               "reference_action": asdict(sample.reference_action), "reference_transcript": sample.reference_transcript,
               "transcript": text, "transcript_available": upstream_error is None, "instrumentation_error": upstream_error}
        result = None
        try:
            if upstream_error is not None: return row, None
            if isinstance(agent, ReplayAgent):
                outputs = raw.get("predictions", {}).get(agent_id, {})
                if condition not in outputs: raise ValueError(f"Missing replay prediction: {agent_id}/{condition}")
                if not isinstance(outputs[condition], str): raise ValueError("Replay prediction must be text")
                result = AgentResult(outputs[condition], "replay:" + agent_id)
            else:
                result = agent.generate(text, sample.tool_schema, sample.instruction)
            parsed = parse_output(result.raw_output)
            score = score_action(parsed.action, sample.reference_action, scoring, sample.tool_schema, parsed.recovered_tool)
            row.update({"raw_output": result.raw_output, "prediction": asdict(parsed.action) if parsed.action else None,
                        "parser_error": parsed.error, "recovered_tool": parsed.recovered_tool,
                        "score": asdict(score), "failure_category": error_category(parsed, sample.reference_action, score),
                        "model_name": result.model_name, "model_config": result.model_config})
        except Exception as exc:
            row["instrumentation_error"] = record_error("agent", sample, condition, exc, agent_id)
            result = None
        finally:
            actions.append(row)
            append_record("actions.jsonl", row)
        return row, result
    for source_index, (sample, raw, variants) in enumerate(prepared, 1):
        hypotheses = {}
        for name in observed:
            supplied = raw.get("hypotheses", {}).get(name)
            try:
                if supplied is not None:
                    value = {"text": supplied} if isinstance(supplied, str) else dict(supplied)
                else:
                    if name not in config.get("asr", {}): raise ValueError(f"Missing hypothesis and no ASR configured: {name}")
                    if not sample.metadata["has_audio"]: raise ValueError("Missing audio_path")
                    if name not in asr_models: asr_models[name] = make_asr(config["asr"][name])
                    result = asr_models[name].transcribe(sample.audio_path)
                    value = {"text": result.text, "asr_config": asdict(result)}
                hypotheses[name] = value
                entry = {"benchmark": sample.benchmark, "source_sample_id": sample.sample_id, "condition": name, **value}
                hypothesis_log.append(entry)
                append_record("hypotheses.jsonl", entry)
            except Exception as exc:
                hypotheses[name] = {"text": "", "instrumentation_error": record_error("asr", sample, name, exc)}
        for agent_id, agent in agents.items():
            clean_row, clean_result = evaluate(sample, raw, agent_id, "reference", sample.reference_transcript)
            conditions = [(role, text, corruption, "controlled", {}) for role, (corruption, text) in variants.items() if role in roles]
            for name, value in hypotheses.items():
                text = value["text"]
                edits = word_alignment(sample.reference_transcript, text)
                corruption = Corruption(sample.sample_id + ":asr:" + name, sample.sample_id,
                    sample.reference_transcript, text, "observed_asr", "mixed", len(edits),
                    len(edits) / len(sample.reference_transcript.split()), "observed_asr", "benchmark_audio", "all eligible sources")
                conditions.append(("asr:" + name, text, corruption, "observed_asr", value))
            for condition, text, corruption, analysis_type, value in conditions:
                row, changed_result = evaluate(sample, raw, agent_id, condition, text, value.get("instrumentation_error"))
                if clean_result is None or changed_result is None: continue
                trace = run_paired_variant(sample, corruption, text, agent, scoring, seed,
                    {"run": config_id, "agent": agent_id, "scoring": "strict-v2"}, clean_result, changed_result).to_dict()
                trace.update({"agent_id": agent_id, "condition": condition, "analysis_type": analysis_type,
                              "group_id": sample.metadata["group_id"], "role_labels": sorted(set(value.get("roles", []))),
                              "features": dict(raw.get("features", {}).get(condition, {}))})
                pairs.append(trace)
                append_record("paired_traces.jsonl", trace)
        print(f"[{source_index}/{len(prepared)}] {sample.benchmark}/{sample.sample_id}", flush=True)
    # Save primary evidence before optional analysis so expensive inference is retained.
    _write_jsonl(output / "actions.jsonl", actions)
    _write_jsonl(output / "paired_traces.jsonl", pairs)
    _write_jsonl(output / "hypotheses.jsonl", hypothesis_log)
    _write_jsonl(output / "errors.jsonl", errors)
    summary = {"validation": validation, "config_sha256": config_id,
               "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "actions": aggregate_actions(actions), "paired": paired_report(pairs, roles, iterations, seed),
               "observed_asr": observed_report(pairs), "instrumentation_failures": len(errors),
               "diagnostics": None}
    options = config.get("diagnostics", {})
    if options.get("enabled"):
        from speechevalx.diagnostics import run_diagnostics
        try:
            diagnostic, predictions = run_diagnostics(pairs, options, seed)
            summary["diagnostics"] = diagnostic
            _write_jsonl(output / "diagnostic_predictions.jsonl", predictions)
            # Includes any computed BERT features for subsequent audit.
            _write_jsonl(output / "paired_traces.jsonl", pairs)
        except Exception as exc:
            summary["diagnostics"] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    if summary["actions"]:
        with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["actions"][0]))
            writer.writeheader()
            writer.writerows(summary["actions"])
    versions = {}
    for name in ("numpy", "scikit-learn", "jsonschema", "torch", "transformers", "funasr"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    _write_json(output / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "packages": versions})
    summary["output"] = str(output)
    return summary
