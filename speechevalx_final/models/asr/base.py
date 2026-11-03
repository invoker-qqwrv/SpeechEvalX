from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ASRResult:
    text: str
    model_name: str
    checkpoint_revision: str | None = None
    language: str | None = None
    vad_config: dict[str, Any] | None = None
    decoding_config: dict[str, Any] | None = None
    preprocessing_config: dict[str, Any] | None = None
    latency_seconds: float | None = None


class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> ASRResult:
        """Return a transcription for one audio file."""
