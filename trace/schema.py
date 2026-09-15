from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseTrace:
    case_id: str
    reference: str
    audio_path: str
    asr: dict[str, Any]
    tool_prediction: dict[str, Any]
    ground_truth: dict[str, Any]
    metrics: dict[str, Any]
    parser_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{self.case_id}.json"
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
