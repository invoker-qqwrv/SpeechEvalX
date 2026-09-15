from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Edit:
    operation: str
    reference: str | None
    hypothesis: str | None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def word_alignment(reference: str, hypothesis: str) -> list[Edit]:
    """Deterministic Levenshtein backtrace retaining S/I/D operations."""
    left, right = reference.split(), hypothesis.split()
    grid = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1): grid[i][0] = i
    for j in range(len(right) + 1): grid[0][j] = j
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1): grid[i][j] = min(grid[i - 1][j] + 1, grid[i][j - 1] + 1, grid[i - 1][j - 1] + (a != b))
    edits: list[Edit] = []
    i, j = len(left), len(right)
    while i or j:
        if i and j and left[i - 1] == right[j - 1]: i, j = i - 1, j - 1
        elif i and j and grid[i][j] == grid[i - 1][j - 1] + 1: edits.append(Edit("substitution", left[i - 1], right[j - 1])); i, j = i - 1, j - 1
        elif j and grid[i][j] == grid[i][j - 1] + 1: edits.append(Edit("insertion", None, right[j - 1])); j -= 1
        else: edits.append(Edit("deletion", left[i - 1], None)); i -= 1
    return list(reversed(edits))
