"""Shared paired execution and scoring against the original reference action."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from speechevalx.alignment import word_alignment
from speechevalx.corruption import Corruption
from speechevalx.datasets import SpeechEvalSample
from speechevalx.models import AgentModel
from speechevalx.models.agent.base import AgentResult
from speechevalx.prediction import parse_output, error_category
from speechevalx.scoring import ScoringConfig, score_action
from speechevalx.trace import PairedTrace


def _prediction(raw_output):
    return parse_output(raw_output).action


def _paired_outcome(clean: bool, corrupted: bool) -> str:
    if clean and corrupted: return "preserved_success"
    if clean: return "harmed"
    if corrupted: return "rescued"
    return "persistent_failure"


def run_paired_variant(sample: SpeechEvalSample, corruption: Corruption, corrupted_transcript: str,
                       agent: AgentModel, scoring: ScoringConfig = ScoringConfig(), seed: int = 2027,
                       config_ids: dict[str, str] | None = None,
                       clean_result: AgentResult | None = None,
                       corrupted_result: AgentResult | None = None) -> PairedTrace:
    if corruption.source_sample_id != sample.sample_id:
        raise ValueError("Corruption source does not match sample")
    edits = word_alignment(sample.reference_transcript, corrupted_transcript)
    if len(edits) != corruption.edit_count:
        raise ValueError("Declared edit count differs from the actual transcript alignment")
    if corruption.generation_method != "observed_asr":
        if not edits or any(e.operation != corruption.edit_operation for e in edits):
            raise ValueError("Declared edit operation differs from the actual transcript alignment")
    clean_result = clean_result or agent.generate(sample.reference_transcript, sample.tool_schema, sample.instruction)
    changed_result = corrupted_result or agent.generate(corrupted_transcript, sample.tool_schema, sample.instruction)
    clean, changed = parse_output(clean_result.raw_output), parse_output(changed_result.raw_output)
    clean_score = score_action(clean.action, sample.reference_action, scoring, sample.tool_schema, clean.recovered_tool)
    changed_score = score_action(changed.action, sample.reference_action, scoring, sample.tool_schema, changed.recovered_tool)
    failure = error_category(changed, sample.reference_action, changed_score)
    serialize = lambda action: None if action is None else {"tool_name": action.tool_name, "arguments": action.arguments, "no_call": action.no_call}
    return PairedTrace(
        sample_id=corruption.corruption_id, benchmark=sample.benchmark, source_sample_id=sample.sample_id,
        reference_transcript=sample.reference_transcript, corrupted_transcript=corrupted_transcript,
        reference_action=serialize(sample.reference_action), clean_raw_output=clean_result.raw_output,
        clean_prediction=serialize(clean.action), corrupted_raw_output=changed_result.raw_output,
        corrupted_prediction=serialize(changed.action), alignment=[e.to_dict() for e in edits],
        span_map=[] if corruption.generation_method == "observed_asr" else [{"text": corruption.original_span, "role": corruption.error_class, "source": "provided_annotation"}],
        error_class=corruption.error_class, edit_operation=corruption.edit_operation, edit_count=len(edits),
        severity=len(edits) / len(sample.reference_transcript.split()),
        clean_tool_correct=clean_score.tool_correct, clean_arg_correct=clean_score.argument_exact_match,
        clean_jca=clean_score.jca, corrupted_tool_correct=changed_score.tool_correct,
        corrupted_arg_correct=changed_score.argument_exact_match, corrupted_jca=changed_score.jca,
        paired_outcome=_paired_outcome(clean_score.jca, changed_score.jca), first_failure=failure,
        first_changed_failure=failure if clean_score.jca and not changed_score.jca else None,
        config_ids=config_ids or {"agent": clean_result.model_name, "scoring": "strict-v2"}, random_seed=seed,
        provenance=asdict(corruption), clean_parser_error=clean.error, corrupted_parser_error=changed.error,
        clean_recovered_tool=clean.recovered_tool, corrupted_recovered_tool=changed.recovered_tool,
    )
