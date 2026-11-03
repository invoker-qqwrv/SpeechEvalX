"""Deterministic synthetic-only agent for pipeline and statistic smoke tests.

It intentionally understands a compact command grammar, not hidden reference
actions, so corrupting a number/entity changes the generated call.
"""
from __future__ import annotations

import re
from typing import Any

from .base import AgentModel, AgentResult


class TemplateAgent(AgentModel):
    def generate(self, transcript: str, tools: list[dict[str, Any]], instruction: str) -> AgentResult:
        text = transcript.casefold()
        numbers = re.findall(r"\b\d+\b", text)
        if "do not transfer" in text:
            return AgentResult("no_call", "template-agent-v1", model_config={"purpose": "synthetic-smoke"})
        if "restaurant" in text:
            tool = "find_restaurants" if any(str(t.get("name")) == "find_restaurants" for t in tools) else "search"
            city = "mumbai" if "mumbai" in text else "delhi" if "delhi" in text else "unknown"
            return AgentResult(f"{tool}(city='{city}', number={numbers[0] if numbers else 0})", "template-agent-v1", model_config={"purpose": "synthetic-smoke"})
        if "transfer" in text:
            tool = "transfer_money" if "transfer_money" in {str(t.get("name")) for t in tools} else "unknown"
            recipient = "alice" if "alice" in text else "bob" if "bob" in text else "unknown"
            return AgentResult(f"{tool}(recipient='{recipient}', amount={numbers[0] if numbers else 0})", "template-agent-v1", model_config={"purpose": "synthetic-smoke"})
        return AgentResult("no_call", "template-agent-v1", model_config={"purpose": "synthetic-smoke"})
