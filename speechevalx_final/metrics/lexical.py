"""Frozen lexical normalization and corpus-level WER."""
from collections.abc import Sequence
import unicodedata


def word_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if not unicodedata.category(c).startswith("P")).split()


def _distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, 1):
        current = [i]
        for j, target in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1]


def corpus_wer(pairs):
    edits = words = 0
    for reference, hypothesis in pairs:
        left, right = word_tokens(reference), word_tokens(hypothesis)
        edits += _distance(left, right)
        words += len(left)
    return {"edits": edits, "reference_words": words, "wer": edits / words if words else None}


def wer(reference: str, hypothesis: str) -> float:
    left, right = word_tokens(reference), word_tokens(hypothesis)
    return _distance(left, right) / len(left) if left else float(bool(right))


def cer(reference: str, hypothesis: str) -> float:
    return _distance(list(reference), list(hypothesis)) / len(reference) if reference else float(bool(hypothesis))
