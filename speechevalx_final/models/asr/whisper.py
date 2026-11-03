from __future__ import annotations
from pathlib import Path
from .base import ASRModel, ASRResult
from .audio import read_audio


class WhisperASR(ASRModel):
    def __init__(self, model_name="openai/whisper-large-v3", device=None, language="english", revision=None):
        self.model_name, self.device, self.language, self.revision = model_name, device, language, revision
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Whisper requires: pip install transformers torch") from exc
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_name,
                                  device=self.device if self.device is not None else -1,
                                  revision=self.revision)
        return self._pipe

    def transcribe(self, audio_path: Path):
        audio = read_audio(audio_path)
        result = self._pipeline()({"raw": audio, "sampling_rate": 16000},
                                  generate_kwargs={"language": self.language, "task": "transcribe", "do_sample": False})
        return ASRResult(str(result["text"]).strip(), self.model_name, checkpoint_revision=self.revision,
                         language=self.language, preprocessing_config={"sample_rate": 16000, "mono": True, "dtype": "float32"})
