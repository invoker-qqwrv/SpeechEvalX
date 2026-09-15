"""Fixed, configuration-driven action canonicalization and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from speechevalx.datasets import ReferenceAction
from speechevalx.metrics.tool import normalize_value


@dataclass(frozen=True)
class ScoringConfig:
    string_aliases: dict[str, str] = field(default_factory=dict)
    case_sensitive_fields: set[str] = field(default_factory=set)
    required_arguments: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionScore:
    valid_call: bool
    no_call: bool
    tool_correct: bool
    argument_exact_match: bool
    jca: bool


def _canonical(value: Any, field_name: str | None, config: ScoringConfig) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if field_name not in config.case_sensitive_fields:
            value = value.casefold()
        value = config.string_aliases.get(value, value)
    if isinstance(value, dict):
        return {key: _canonical(item, key, config) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item, field_name, config) for item in value]
    return normalize_value(value)


def canonicalize_action(action: ReferenceAction, config: ScoringConfig) -> ReferenceAction:
    return ReferenceAction(action.tool_name, _canonical(action.arguments, None, config))


def score_action(prediction: ReferenceAction | None, expected: ReferenceAction, config: ScoringConfig = ScoringConfig()) -> ActionScore:
    if prediction is None:
        return ActionScore(False, False, False, False, False)
    if expected.no_call:
        correct = prediction.no_call
        return ActionScore(True, prediction.no_call, correct, correct, correct)
    tool_correct = prediction.tool_name == expected.tool_name
    required = config.required_arguments.get(expected.tool_name or "", tuple(expected.arguments))
    normalized_prediction = canonicalize_action(prediction, config).arguments
    normalized_expected = canonicalize_action(expected, config).arguments
    argument_exact = tool_correct and all(key in normalized_prediction and normalized_prediction[key] == normalized_expected.get(key) for key in required)
    return ActionScore(True, prediction.no_call, tool_correct, argument_exact, tool_correct and argument_exact)
