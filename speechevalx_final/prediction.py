"""Strict output parsing, with recoverable tool names retained for Tool accuracy."""
from __future__ import annotations
from dataclasses import dataclass
import re
from speechevalx.datasets import ReferenceAction
from speechevalx.parsing import ToolCallParseError, parse_tool_call


@dataclass(frozen=True)
class ParsedOutput:
    action: ReferenceAction | None
    recovered_tool: str | None
    error: str | None


def parse_output(raw: str) -> ParsedOutput:
    if not isinstance(raw, str):
        raise TypeError("Agent raw output must be a string")
    if raw.strip().casefold() in {"no_call", "none", "null"}:
        return ParsedOutput(ReferenceAction(None), None, None)
    try:
        call = parse_tool_call(raw)
        return ParsedOutput(ReferenceAction(call.tool, call.arguments), call.tool, None)
    except ToolCallParseError as exc:
        # Only recover a leading call name, never a tool mentioned in prose.
        match = re.match(r"^\s*\[?\s*([A-Za-z_]\w*)\s*\(", raw)
        return ParsedOutput(None, match.group(1) if match else None, str(exc))


def error_category(parsed: ParsedOutput, expected: ReferenceAction, score) -> str | None:
    if parsed.error is not None:
        return "malformed_output"
    if parsed.action.no_call != expected.no_call:
        return "call_decision"
    if not score.tool_correct:
        return "wrong_tool"
    if not score.argument_structure_valid:
        return "argument_structure"
    if not score.argument_exact_match:
        return "argument_value"
    return None
