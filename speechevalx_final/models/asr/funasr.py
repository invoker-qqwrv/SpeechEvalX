"""Optional SenseVoice-Small / Paraformer adapter via FunASR."""
from .base import ASRModel, ASRResult
from .audio import read_audio


class FunASR(ASRModel):
    def __init__(self, model_name, device="cpu", language="en", revision=None):
        self.model_name, self.device, self.language, self.revision = model_name, device, language, revision
        self._model = None

    def transcribe(self, audio_path):
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("SenseVoice/Paraformer requires: pip install funasr modelscope") from exc
            options = {"model": self.model_name, "device": self.device, "disable_update": True}
            if self.revision: options["model_revision"] = self.revision
            self._model = AutoModel(**options)
        audio = read_audio(audio_path)
        result = self._model.generate(input=audio, cache={}, language=self.language, use_itn=False)
        if not result: raise ValueError("ASR returned no result")
        text = str(result[0]["text"])
        # SenseVoice language/emotion/event tags are metadata, not transcript words.
        import re
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        return ASRResult(text, self.model_name, checkpoint_revision=self.revision, language=self.language,
                         preprocessing_config={"sample_rate": 16000, "mono": True, "dtype": "float32"})
