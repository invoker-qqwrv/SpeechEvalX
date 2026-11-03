"""Compatibility wrapper around the shared action scorer."""
from dataclasses import dataclass
from speechevalx.datasets.core import ReferenceAction
from speechevalx.parsing.tool_parser import CanonicalToolCall
from speechevalx.scoring import score_action


@dataclass(frozen=True)
class SingleToolEvaluation:
    tool_correct: bool
    argument_exact_match: bool
    task_success: None = None


def evaluate_single_tool(prediction: CanonicalToolCall, expected: CanonicalToolCall) -> SingleToolEvaluation:
    score = score_action(ReferenceAction(prediction.tool, prediction.arguments),
                         ReferenceAction(expected.tool, expected.arguments))
    return SingleToolEvaluation(score.tool_correct, score.argument_exact_match)
