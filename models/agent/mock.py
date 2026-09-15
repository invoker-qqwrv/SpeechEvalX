from typing import Any

from .base import AgentModel, AgentResult


class MockAgent(AgentModel):
    def __init__(self, raw_output: str, model_name: str = "mock-agent") -> None:
        self.raw_output = raw_output
        self.model_name = model_name

    def generate(self, transcript: str, tools: list[dict[str, Any]], instruction: str) -> AgentResult:
        return AgentResult(raw_output=self.raw_output, model_name=self.model_name)
