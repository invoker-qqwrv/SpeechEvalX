"""Paired execution: the only intended path for aggregate experiment results."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from speechevalx.alignment import word_alignment
from speechevalx.corruption import Corruption
from speechevalx.datasets import ReferenceAction, SpeechEvalSample
from speechevalx.models import AgentModel
from speechevalx.parsing import ToolCallParseError, parse_tool_call
from speechevalx.scoring import ActionScore, ScoringConfig, score_action
from speechevalx.trace import PairedTrace


def _prediction(raw_output: str) -> ReferenceAction | None:
    cleaned = raw_output.strip().casefold()
    if cleaned in {"no_call", "none", ""}:
        return ReferenceAction(None)
    try:
        call = parse_tool_call(raw_output)
        return ReferenceAction(call.tool, call.arguments)
    except ToolCallParseError:
        return None


def _first_fail(score: ActionScore) -> str | None:
    """Localize from the tool interface; lexical edits are stored separately."""
    if not score.tool_correct:
        return "tool_selection"
    if not score.argument_exact_match:
        return "argument_correctness"
    if not score.jca:
        return "terminal_jca"
    return None


def _paired_outcome(clean: bool, corrupted: bool) -> str:
    if clean and corrupted: return "preserved_success"
    if clean: return "harmed"
    if corrupted: return "rescued"
    return "persistent_failure"


def run_paired_variant(sample: SpeechEvalSample, corruption: Corruption, corrupted_transcript: str, agent: AgentModel, scoring: ScoringConfig = ScoringConfig(), seed: int = 7, config_ids: dict[str, str] | None = None) -> PairedTrace:
    """Run exactly matched agent settings over clean and corrupted transcripts."""
    clean_result = agent.generate(sample.reference_transcript, sample.tool_schema, sample.instruction)
    corrupted_result = agent.generate(corrupted_transcript, sample.tool_schema, sample.instruction)
    clean_prediction, corrupted_prediction = _prediction(clean_result.raw_output), _prediction(corrupted_result.raw_output)
    clean_score = score_action(clean_prediction, sample.reference_action, scoring)
    corrupted_score = score_action(corrupted_prediction, sample.reference_action, scoring)
    alignment = [item.to_dict() for item in word_alignment(sample.reference_transcript, corrupted_transcript)]
    clean_failure, corrupted_failure = _first_fail(clean_score), _first_fail(corrupted_score)
    changed_failure = corrupted_failure if clean_score.jca and not corrupted_score.jca else None
    serialize = lambda action: None if action is None else {"tool_name": action.tool_name, "arguments": action.arguments, "no_call": action.no_call}
    return PairedTrace(
        sample_id=corruption.corruption_id, benchmark=sample.benchmark, source_sample_id=sample.sample_id,
        reference_transcript=sample.reference_transcript, corrupted_transcript=corrupted_transcript,
        reference_action=serialize(sample.reference_action) or {}, clean_raw_output=clean_result.raw_output,
        clean_prediction=serialize(clean_prediction), corrupted_raw_output=corrupted_result.raw_output,
        corrupted_prediction=serialize(corrupted_prediction), alignment=alignment,
        span_map=[{"text": corruption.original_span, "normalized_value": corruption.original_span, "role": corruption.error_class, "source": "controlled_corruption", "confidence": 1.0, "audit_required": True}],
        error_class=corruption.error_class, edit_operation=corruption.edit_operation, edit_count=corruption.edit_count,
        severity=corruption.severity, clean_tool_correct=clean_score.tool_correct,
        clean_arg_correct=clean_score.argument_exact_match, clean_jca=clean_score.jca,
        corrupted_tool_correct=corrupted_score.tool_correct, corrupted_arg_correct=corrupted_score.argument_exact_match,
        corrupted_jca=corrupted_score.jca, paired_outcome=_paired_outcome(clean_score.jca, corrupted_score.jca),
        first_failure=corrupted_failure, first_changed_failure=changed_failure,
        config_ids=config_ids or {"agent": clean_result.model_name, "scoring": "default-v1"}, random_seed=seed,
        provenance=asdict(corruption),
    )
