"""Frozen local causal-LM adapter for Qwen or Llama instruction models."""
from __future__ import annotations
import json
from .base import AgentModel, AgentResult


def build_messages(transcript, tools, instruction):
    system = ("Return exactly one Python-like tool call with keyword arguments and literal values. "
              "Return exactly NO_CALL when no tool should be called. Do not explain your answer.\n"
              + ("System context: " + instruction + "\n" if instruction else "")
              + "Available tools: " + json.dumps(tools, ensure_ascii=False, sort_keys=True))
    return [{"role": "system", "content": system}, {"role": "user", "content": transcript}]


class QwenAgent(AgentModel):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device_map="auto",
                 dtype: str = "bfloat16", seed: int = 2027, revision: str | None = None,
                 max_new_tokens: int = 256):
        self.model_name, self.device_map = model_name, device_map
        self.dtype, self.seed, self.revision = dtype, seed, revision
        self.max_new_tokens = max_new_tokens
        self._tokenizer = self._model = None

    def _load(self):
        if self._model is not None: return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Local agents require: pip install torch transformers accelerate") from exc
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16 or float32")
        torch.manual_seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, device_map=self.device_map,
            torch_dtype=getattr(torch, self.dtype))
        self._model.eval()

    def generate(self, transcript, tools, instruction):
        self._load()
        import torch
        torch.manual_seed(self.seed)
        messages = build_messages(transcript, tools, instruction)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                              do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
        text = self._tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        config = {"dtype": self.dtype, "seed": self.seed, "revision": self.revision,
                  "resolved_revision": getattr(self._model.config, "_commit_hash", None),
                  "max_new_tokens": self.max_new_tokens, "do_sample": False, "prompt_version": "python-call-v2"}
        return AgentResult(text, self.model_name, model_config=config)
