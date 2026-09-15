from __future__ import annotations

import json
from typing import Any

from .base import AgentModel, AgentResult


class QwenAgent(AgentModel):
    """Optional local Qwen adapter, loaded lazily to keep the MVP runnable."""

    def __init__(self, model_name: str, device_map: str | None = "auto") -> None:
        self.model_name = model_name
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("QwenAgent requires optional dependencies: pip install transformers torch") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map, trust_remote_code=True)

    def generate(self, transcript: str, tools: list[dict[str, Any]], instruction: str) -> AgentResult:
        self._load()
        prompt = (
            "Return exactly one Python-like tool call using keyword arguments.\n"
            f"Instruction: {instruction}\nTools: {json.dumps(tools, ensure_ascii=False)}\n"
            f"User transcript: {transcript}\nTool call:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = self._tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return AgentResult(raw_output=text, model_name=self.model_name)
