"""Adapter for BFCL Audio style JSON/JSONL exports.

The public releases use slightly different field names, so normalization is
deliberately permissive while the resulting :class:`SpeechEvalSample` is not.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .core import ReferenceAction, SpeechEvalSample


class BFCLAudioAdapter:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    def iter_samples(self) -> Iterator[SpeechEvalSample]:
        for path in self.dataset_root.rglob("*.json*"):
            try:
                payload: Any = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
                records = list(payload) if path.suffix == ".jsonl" else payload.get("data", payload.get("cases", [])) if isinstance(payload, dict) else payload
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for raw in records if isinstance(records, list) else []:
                if isinstance(raw, dict):
                    yield self.normalize(raw)

    def normalize(self, raw: dict[str, Any]) -> SpeechEvalSample:
        sample_id = str(raw.get("id", raw.get("case_id", raw.get("sample_id", ""))))
        transcript = str(raw.get("transcript", raw.get("query", raw.get("instruction", ""))))
        audio = raw.get("audio_path", raw.get("path", ""))
        tools = raw.get("tools", raw.get("functions", raw.get("tool_schema", [])))
        action = raw.get("reference_action", raw.get("expected_tool_call", raw.get("expected_call")))
        if isinstance(tools, str):
            tools = json.loads(tools)
        if not sample_id or not transcript or not audio or not isinstance(tools, list):
            raise ValueError("BFCL Audio record is missing id, transcript, audio, or tools")
        if action in (None, "no_call"):
            expected = ReferenceAction(None)
        elif isinstance(action, dict):
            expected = ReferenceAction(action.get("tool_name", action.get("tool", action.get("name"))), dict(action.get("arguments", action.get("parameters", {}))))
        else:
            from speechevalx.parsing import parse_tool_call
            call = parse_tool_call(str(action))
            expected = ReferenceAction(call.tool, call.arguments)
        return SpeechEvalSample(sample_id, "bfcl_audio", (self.dataset_root / str(audio)).resolve(), transcript, tools, expected, str(raw.get("instruction", "")), {k: v for k, v in raw.items() if k not in {"id", "case_id", "sample_id", "audio_path", "path", "transcript", "query", "instruction", "tools", "functions", "tool_schema", "reference_action", "expected_tool_call", "expected_call"}})
