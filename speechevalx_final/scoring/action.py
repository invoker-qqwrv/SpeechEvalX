"""Shared scoring for audio, reference text and controlled variants."""
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
    allow_extra_arguments: bool = False
    normalize_numbers: bool = True


@dataclass(frozen=True)
class ActionScore:
    valid_call: bool
    no_call: bool
    tool_correct: bool
    argument_exact_match: bool
    jca: bool
    argument_structure_valid: bool = False


def _canonical(value: Any, field_name: str | None, config: ScoringConfig) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if field_name not in config.case_sensitive_fields:
            value = value.casefold()
        value = config.string_aliases.get(value, value)
        return normalize_value(value) if config.normalize_numbers else value
    if isinstance(value, dict):
        return {key: _canonical(item, key, config) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item, field_name, config) for item in value]
    return value


def canonicalize_action(action: ReferenceAction, config: ScoringConfig) -> ReferenceAction:
    return ReferenceAction(action.tool_name, _canonical(action.arguments, None, config))


def argument_schema(tool_name: str, tools: list[dict] | None) -> dict | None:
    for tool in tools or []:
        spec = tool.get("function", tool)
        if spec.get("name") == tool_name:
            return spec.get("parameters")
    return None


def schema_error(arguments: dict, schema: dict | None) -> str | None:
    if schema is None:
        return None
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("Tool schema validation requires: pip install jsonschema") from exc
    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)
    errors = sorted(validator(schema).iter_errors(arguments), key=lambda e: str(list(e.path)))
    return errors[0].message if errors else None


def _equivalent(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    return left == right


def score_action(prediction: ReferenceAction | None, expected: ReferenceAction,
                 config: ScoringConfig = ScoringConfig(), tool_schema: list[dict] | None = None,
                 recovered_tool: str | None = None) -> ActionScore:
    if prediction is None:
        return ActionScore(False, False, not expected.no_call and recovered_tool == expected.tool_name, False, False)
    if prediction.no_call:
        return ActionScore(False, True, expected.no_call, expected.no_call, expected.no_call, expected.no_call)
    if expected.no_call:
        return ActionScore(True, False, False, False, False)
    tool_correct = prediction.tool_name == expected.tool_name
    required = config.required_arguments.get(expected.tool_name or "", tuple(expected.arguments))
    if any(key not in expected.arguments for key in required):
        raise ValueError("Scoring required_arguments contains keys absent from the reference action")
    schema = argument_schema(prediction.tool_name, tool_schema)
    missing = any(key not in prediction.arguments for key in required)
    extra = not config.allow_extra_arguments and bool(set(prediction.arguments) - set(expected.arguments))
    # Schema types are checked BEFORE value normalization: a string does not
    # become a valid JSON number merely because numeric aliasing is enabled.
    structure = not missing and not extra and schema_error(prediction.arguments, schema) is None
    left = canonicalize_action(prediction, config).arguments
    right = canonicalize_action(expected, config).arguments
    equal = tool_correct and structure and all(_equivalent(left.get(key), right[key]) for key in required)
    return ActionScore(True, False, tool_correct, equal, equal, structure)
