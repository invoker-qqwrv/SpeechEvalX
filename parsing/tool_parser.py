from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalToolCall:
    tool: str
    arguments: dict[str, Any]


class ToolCallParseError(ValueError):
    pass


def _extract_expression(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if not cleaned:
        raise ToolCallParseError("Empty agent output")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1].strip()
    return cleaned


def parse_tool_call(raw_output: str) -> CanonicalToolCall:
    """Parse a single Python-like call safely; never evaluate arbitrary code."""
    expression = _extract_expression(raw_output)
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ToolCallParseError(f"Malformed tool call: {exc.msg}") from exc
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ToolCallParseError("Expected one simple tool call, e.g. search(query='x')")
    if node.args:
        raise ToolCallParseError("Positional arguments are not supported; use keyword arguments")
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ToolCallParseError("**kwargs are not supported")
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ToolCallParseError(f"Argument {keyword.arg!r} must be a literal") from exc
    return CanonicalToolCall(tool=node.func.id, arguments=arguments)
