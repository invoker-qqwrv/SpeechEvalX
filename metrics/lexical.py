from collections.abc import Sequence


def _distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, 1):
        current = [i]
        for j, target in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    tokens = reference.lower().split()
    return _distance(tokens, hypothesis.lower().split()) / len(tokens) if tokens else float(bool(hypothesis.strip()))


def cer(reference: str, hypothesis: str) -> float:
    return _distance(list(reference), list(hypothesis)) / len(reference) if reference else float(bool(hypothesis))
