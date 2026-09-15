from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentResult:
    raw_output: str
    model_name: str
    latency_seconds: float | None = None
    model_config: dict[str, Any] | None = None


class AgentModel(ABC):
    @abstractmethod
    def generate(self, transcript: str, tools: list[dict[str, Any]], instruction: str) -> AgentResult:
        """Generate a raw tool-call expression from the transcript."""
