import re
from typing import Any

from .base import AgentModel, AgentResult


class RuleBasedAgent(AgentModel):
    """Tiny offline baseline, deliberately not a substitute for a production LLM."""

    def generate(self, transcript: str, tools: list[dict[str, Any]], instruction: str) -> AgentResult:
        names = {str(tool.get("name", "")) for tool in tools}
        if "find_restaurants" in names:
            count = re.search(r"\b(\d+)\b", transcript)
            food = re.search(r"good\s+(.+?)\s+restaurants?", transcript, re.I)
            place = re.search(r"\bin\s+(.+?)[?.!]*$", transcript, re.I)
            args = []
            if place:
                args.append(f"location={place.group(1).strip()!r}")
            if food:
                args.append(f"food_type={food.group(1).strip()!r}")
            if count:
                args.append(f"number={int(count.group(1))}")
            return AgentResult(f"find_restaurants({', '.join(args)})", "rule-based-agent")
        return AgentResult("", "rule-based-agent")
