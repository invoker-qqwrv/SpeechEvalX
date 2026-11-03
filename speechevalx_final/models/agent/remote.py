"""OpenAI-compatible local/remote inference server, including vLLM and Ollama."""
from __future__ import annotations
import json
import os
from urllib.request import Request, urlopen
from .base import AgentModel, AgentResult
from .qwen import build_messages


class CompatibleAgent(AgentModel):
    def __init__(self, model_name, base_url="http://localhost:8000/v1", api_key_env=None,
                 seed=2027, max_new_tokens=256, timeout=180):
        self.model_name, self.base_url, self.api_key_env = model_name, base_url.rstrip("/"), api_key_env
        self.seed, self.max_new_tokens, self.timeout = seed, max_new_tokens, timeout

    def generate(self, transcript, tools, instruction):
        payload = {"model": self.model_name, "messages": build_messages(transcript, tools, instruction),
                   "temperature": 0, "top_p": 1, "max_tokens": self.max_new_tokens, "seed": self.seed}
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key: raise ValueError(f"Missing API key environment variable {self.api_key_env}")
            headers["Authorization"] = "Bearer " + key
        request = Request(self.base_url + "/chat/completions", json.dumps(payload).encode("utf-8"), headers)
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("Server must return text content in the configured Python-call format")
        return AgentResult(content, self.model_name, model_config={"base_url": self.base_url,
                           "seed": self.seed, "prompt_version": "python-call-v2"})
