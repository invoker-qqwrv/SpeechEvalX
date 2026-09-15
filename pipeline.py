from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from speechevalx.datasets import SpeechEvalCase
from speechevalx.evaluation import SingleToolEvaluation, evaluate_single_tool
from speechevalx.metrics import cer, wer
from speechevalx.models import ASRModel, AgentModel
from speechevalx.parsing import CanonicalToolCall, ToolCallParseError, parse_tool_call
from speechevalx.trace import CaseTrace


@dataclass(frozen=True)
class RunResult:
    trace: CaseTrace
    trace_path: Path
    lexical_wer: float
    lexical_cer: float


def run_single_case(case: SpeechEvalCase, asr_model: ASRModel, agent_model: AgentModel, trace_directory: Path) -> RunResult:
    asr = asr_model.transcribe(case.audio_path)
    agent = agent_model.generate(asr.text, case.available_tools, case.instruction)
    expected = parse_tool_call(case.expected_tool_call)
    prediction: CanonicalToolCall | None = None
    parser_error: str | None = None
    evaluation: SingleToolEvaluation | None = None
    try:
        prediction = parse_tool_call(agent.raw_output)
        evaluation = evaluate_single_tool(prediction, expected)
    except ToolCallParseError as exc:
        parser_error = str(exc)
    trace = CaseTrace(
        case_id=case.case_id,
        reference=case.reference,
        audio_path=str(case.audio_path),
        asr={"model": asr.model_name, "hypothesis": asr.text},
        tool_prediction={"raw_output": agent.raw_output, "tool": prediction.tool if prediction else None, "arguments": prediction.arguments if prediction else {}},
        ground_truth={"tool": expected.tool, "arguments": expected.arguments},
        metrics={"wer": wer(case.reference, asr.text), "cer": cer(case.reference, asr.text), "tool_correct": evaluation.tool_correct if evaluation else None, "argument_exact_match": evaluation.argument_exact_match if evaluation else None, "task_success": None},
        parser_error=parser_error,
    )
    path = trace.save(trace_directory)
    return RunResult(trace=trace, trace_path=path, lexical_wer=trace.metrics["wer"], lexical_cer=trace.metrics["cer"])
