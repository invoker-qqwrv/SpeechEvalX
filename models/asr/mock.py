from pathlib import Path

from .base import ASRModel, ASRResult


class MockASR(ASRModel):
    """Deterministic ASR used for tests and pipeline smoke checks."""

    def __init__(self, transcript: str, model_name: str = "mock-asr") -> None:
        self.transcript = transcript
        self.model_name = model_name

    def transcribe(self, audio_path: Path) -> ASRResult:
        return ASRResult(text=self.transcript, model_name=self.model_name)
