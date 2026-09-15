from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def _outcome(row: dict[str, Any]) -> str:
    return "harmed" if row["clean_jca"] and not row["corrupted_jca"] else "rescued" if not row["clean_jca"] and row["corrupted_jca"] else "preserved_success" if row["clean_jca"] else "persistent_failure"


def cei_by_class(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["error_class"]].append(row)
    return [{"error_class": key, "n": len(group), "harmed": sum(_outcome(r) == "harmed" for r in group), "rescued": sum(_outcome(r) == "rescued" for r in group), "cei": sum(int(r["clean_jca"]) - int(r["corrupted_jca"]) for r in group) / len(group)} for key, group in sorted(groups.items())]


def cluster_bootstrap_cei(rows: list[dict[str, Any]], iterations: int = 1000, seed: int = 7) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row["source_sample_id"]].append(row)
    clusters = list(groups)
    rng = random.Random(seed)
    by_class: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled = [item for source in (rng.choice(clusters) for _ in clusters) for item in groups[source]]
        for value in cei_by_class(sampled): by_class[value["error_class"]].append(value["cei"])
    return [{"error_class": key, "ci_low": sorted(values)[int(.025 * (len(values) - 1))], "ci_high": sorted(values)[int(.975 * (len(values) - 1))], "iterations": iterations} for key, values in sorted(by_class.items())]


def epr_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # lexical error is observed from the stored edit; B is JCA failure.
    result = []
    for label, selected in (("all", rows), *[(name, [r for r in rows if r["error_class"] == name]) for name in sorted({r["error_class"] for r in rows})]):
        result.append({"stratum": label, "stage_a": "lexical_corruption", "stage_b": "terminal_jca_failure", "n": len(selected), "epr": sum(not r["corrupted_jca"] for r in selected) / len(selected) if selected else None})
    return result


def mcnemar_exact(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    b = sum(r["clean_jca"] and not r["corrupted_jca"] for r in rows)
    c = sum(not r["clean_jca"] and r["corrupted_jca"] for r in rows)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)) if n else 1.0
    return {"harmed_discordant": b, "rescued_discordant": c, "p_value": p}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result: dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        adjusted = max(previous, min(1.0, (total - rank) * p_value))
        result[name], previous = adjusted, adjusted
    return result
