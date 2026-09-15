"""Trace schema for an auditable clean/corrupted experimental pair."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairedTrace:
    sample_id: str
    benchmark: str
    source_sample_id: str
    reference_transcript: str
    corrupted_transcript: str
    reference_action: dict[str, Any]
    clean_raw_output: str
    clean_prediction: dict[str, Any] | None
    corrupted_raw_output: str
    corrupted_prediction: dict[str, Any] | None
    alignment: list[dict[str, Any]]
    span_map: list[dict[str, Any]]
    error_class: str
    edit_operation: str
    edit_count: int
    severity: int
    clean_tool_correct: bool
    clean_arg_correct: bool
    clean_jca: bool
    corrupted_tool_correct: bool
    corrupted_arg_correct: bool
    corrupted_jca: bool
    paired_outcome: str
    first_failure: str | None
    first_changed_failure: str | None
    config_ids: dict[str, str]
    random_seed: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: return asdict(self)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
