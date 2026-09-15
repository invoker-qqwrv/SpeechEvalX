from dataclasses import dataclass

from speechevalx.metrics.tool import normalize_value
from speechevalx.parsing.tool_parser import CanonicalToolCall


@dataclass(frozen=True)
class SingleToolEvaluation:
    tool_correct: bool
    argument_exact_match: bool
    task_success: None = None


def evaluate_single_tool(prediction: CanonicalToolCall, expected: CanonicalToolCall) -> SingleToolEvaluation:
    return SingleToolEvaluation(
        tool_correct=prediction.tool == expected.tool,
        argument_exact_match=normalize_value(prediction.arguments) == normalize_value(expected.arguments),
    )
