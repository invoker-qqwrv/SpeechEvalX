from pathlib import Path

from .base import ASRModel, ASRResult


class WhisperASR(ASRModel):
    """Lazy Hugging Face Whisper adapter; imports model dependencies on use."""

    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("WhisperASR requires optional dependencies: pip install transformers torch") from exc
            device = self.device if self.device is not None else -1
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_name, device=device)
        return self._pipe

    def transcribe(self, audio_path: Path) -> ASRResult:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        result = self._pipeline()(str(audio_path))
        return ASRResult(text=str(result["text"]).strip(), model_name=self.model_name)
