"""Single audio case using the same parser and action scorer as batch runs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from speechevalx.datasets import SpeechEvalCase
from speechevalx.metrics import cer, wer
from speechevalx.models import ASRModel, AgentModel
from speechevalx.prediction import parse_output, error_category
from speechevalx.scoring import ScoringConfig, score_action
from speechevalx.trace import CaseTrace


@dataclass(frozen=True)
class RunResult:
    trace: CaseTrace
    trace_path: Path
    lexical_wer: float
    lexical_cer: float


def run_single_case(case: SpeechEvalCase, asr_model: ASRModel, agent_model: AgentModel,
                    trace_directory: Path, scoring: ScoringConfig = ScoringConfig()) -> RunResult:
    asr = asr_model.transcribe(case.audio_path)
    agent = agent_model.generate(asr.text, case.available_tools, case.instruction)
    expected = parse_output(case.expected_tool_call)
    if expected.action is None: raise ValueError("Invalid reference action: " + str(expected.error))
    parsed = parse_output(agent.raw_output)
    score = score_action(parsed.action, expected.action, scoring, case.available_tools, parsed.recovered_tool)
    prediction = parsed.action
    trace = CaseTrace(
        case_id=case.case_id, reference=case.reference, audio_path=str(case.audio_path),
        asr={"model": asr.model_name, "hypothesis": asr.text},
        tool_prediction={"raw_output": agent.raw_output, "tool": prediction.tool_name if prediction else None,
                         "arguments": prediction.arguments if prediction else {}},
        ground_truth={"tool": expected.action.tool_name, "arguments": expected.action.arguments},
        metrics={"wer": wer(case.reference, asr.text), "cer": cer(case.reference, asr.text),
                 **asdict(score), "task_success": None,
                 "failure_category": error_category(parsed, expected.action, score)}, parser_error=parsed.error)
    trace_path = trace.save(trace_directory)
    return RunResult(trace, trace_path, trace.metrics["wer"], trace.metrics["cer"])
