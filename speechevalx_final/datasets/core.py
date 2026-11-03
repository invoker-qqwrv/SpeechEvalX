"""Benchmark-neutral records used by every experimental stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceAction:
    """The terminal action expected by a benchmark; ``tool_name=None`` is no-call."""

    tool_name: str | None
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def no_call(self) -> bool:
        return self.tool_name is None


@dataclass(frozen=True)
class SpeechEvalSample:
    sample_id: str
    benchmark: str
    audio_path: Path
    reference_transcript: str
    tool_schema: list[dict[str, Any]]
    reference_action: ReferenceAction
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
