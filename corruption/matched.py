"""Explicit, single-edit corruption records for the primary matched analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class Corruption:
    corruption_id: str
    source_sample_id: str
    original_span: str
    corrupted_span: str
    error_class: str
    edit_operation: str = "substitution"
    edit_count: int = 1
    severity: int = 1
    generation_method: str = "deterministic"
    provenance: str = "rule:v1"
    eligibility_rule: str = "single-token exact substitution"

    def to_dict(self) -> dict[str, object]: return asdict(self)


def generate_matched_substitutions(sample_id: str, transcript: str, substitutions: dict[str, tuple[str, str]]) -> list[tuple[Corruption, str]]:
    """Create only eligible one-token replacements; never silently alter a sample."""
    generated: list[tuple[Corruption, str]] = []
    for error_class, (old, new) in substitutions.items():
        # Count whole token/span matches, not substrings (e.g. ``in`` in Find).
        pattern = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
        if len(pattern.findall(transcript)) != 1:
            continue
        record = Corruption(f"{sample_id}:{error_class}:s1", sample_id, old, new, error_class)
        generated.append((record, pattern.sub(new, transcript, count=1)))
    return generated
