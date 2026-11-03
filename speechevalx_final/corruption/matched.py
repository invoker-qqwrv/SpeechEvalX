"""Explicit, single-edit corruption records for the primary matched analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from speechevalx.alignment import word_alignment

TASK_ROLES = ("number", "entity", "negation", "intent_cue", "other_slot", "function_word", "other_noncritical")


@dataclass(frozen=True)
class Corruption:
    corruption_id: str
    source_sample_id: str
    original_span: str
    corrupted_span: str
    error_class: str
    edit_operation: str = "substitution"
    edit_count: int = 1
    severity: float = 1.0
    generation_method: str = "deterministic"
    provenance: str = "rule:v1"
    eligibility_rule: str = "single-token exact substitution"

    def to_dict(self) -> dict[str, object]: return asdict(self)


def generate_matched_substitutions(sample_id: str, transcript: str, substitutions: dict[str, tuple[str, str]]) -> list[tuple[Corruption, str]]:
    """Create only eligible one-token replacements; never silently alter a sample."""
    generated: list[tuple[Corruption, str]] = []
    for error_class, (old, new) in substitutions.items():
        if len(old.split()) != 1 or len(new.split()) != 1 or old == new:
            continue
        # Count whole token/span matches, not substrings (e.g. ``in`` in Find).
        pattern = re.compile(r"(?<!\S)" + re.escape(old) + r"(?!\S)")
        if len(pattern.findall(transcript)) != 1:
            continue
        if error_class not in TASK_ROLES:
            raise ValueError(f"Unknown task role: {error_class}")
        changed = pattern.sub(lambda _: new, transcript, count=1)
        edits = word_alignment(transcript, changed)
        if len(edits) != 1 or edits[0].operation != "substitution":
            continue
        record = Corruption(f"{sample_id}:{error_class}:s1", sample_id, old, new, error_class,
                            severity=1 / len(transcript.split()))
        generated.append((record, changed))
    return generated
