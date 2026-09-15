from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .core import ReferenceAction, SpeechEvalSample


@dataclass(frozen=True)
class SpeechEvalCase:
    case_id: str
    audio_path: Path
    reference: str
    instruction: str
    available_tools: list[dict[str, Any]]
    expected_tool_call: str


class VoiceAgentBenchAdapter:
    """Normalize raw VoiceAgentBench records without coupling the pipeline to it."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    def load_case(self, case_id: str) -> SpeechEvalCase:
        for raw in self.iter_raw_cases():
            if str(raw.get("id", raw.get("case_id", ""))) == case_id:
                return self.normalize(raw)
        raise KeyError(f"Case {case_id!r} was not found beneath {self.dataset_root}")

    def iter_raw_cases(self) -> Iterator[dict[str, Any]]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        for path in self.dataset_root.rglob("*.json*"):
            if path.name.startswith("."):
                continue
            yield from self._read_records(path)

    def _read_records(self, path: Path) -> Iterator[dict[str, Any]]:
        try:
            if path.suffix == ".jsonl":
                data: Any = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = data.get("data", data.get("cases", [])) if isinstance(data, dict) else data
        if isinstance(records, dict):
            records = [records]
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    yield record

    def normalize(self, raw: dict[str, Any]) -> SpeechEvalCase:
        case_id = str(raw.get("id", raw.get("case_id", "")))
        if not case_id:
            raise ValueError("VoiceAgentBench case is missing id")
        relative_audio = raw.get("path", raw.get("audio_path", ""))
        if not relative_audio:
            raise ValueError(f"Case {case_id!r} is missing path/audio_path")
        tools = raw.get("functions", raw.get("available_tools", []))
        if isinstance(tools, str):
            tools = json.loads(tools)
        expected = raw.get("expected_tool_call", raw.get("expected_call", ""))
        if isinstance(expected, dict):
            tool = expected.get("tool", expected.get("name", ""))
            arguments = expected.get("arguments", expected.get("parameters", {}))
            if tool and isinstance(arguments, dict):
                expected = f"{tool}(" + ", ".join(f"{key}={value!r}" for key, value in arguments.items()) + ")"
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Case {case_id!r} is missing a usable expected tool call")
        return SpeechEvalCase(
            case_id=case_id,
            audio_path=(self.dataset_root / relative_audio).resolve(),
            reference=str(raw.get("query", raw.get("reference", ""))),
            instruction=str(raw.get("instruction", "")),
            available_tools=list(tools),
            expected_tool_call=expected,
        )

    def iter_samples(self) -> Iterator[SpeechEvalSample]:
        for raw in self.iter_raw_cases():
            case = self.normalize(raw)
            from speechevalx.parsing import parse_tool_call
            expected = parse_tool_call(case.expected_tool_call)
            yield SpeechEvalSample(case.case_id, "voiceagentbench", case.audio_path, case.reference, case.available_tools, ReferenceAction(expected.tool, expected.arguments), case.instruction)
