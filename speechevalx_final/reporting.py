"""Explicit denominators, paired source support and role-vs-role contrasts."""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from speechevalx.metrics.lexical import corpus_wer
from speechevalx.statistics import cei_by_class, cluster_bootstrap_cei, mcnemar_exact, holm_adjust


def aggregate_actions(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row["agent_id"], row["condition"])].append(row)
    result = []
    for (agent, condition), group in sorted(groups.items()):
        ok = [r for r in group if r.get("instrumentation_error") is None]
        calls = [r for r in ok if not r["reference_no_call"]]
        no_calls = [r for r in ok if r["reference_no_call"]]
        correct = sum(r["score"]["jca"] for r in calls)
        tools = sum(r["score"]["tool_correct"] for r in calls)
        correct_no_call = sum(r["score"]["jca"] for r in no_calls)
        lexical = corpus_wer((r["reference_transcript"], r["transcript"]) for r in group if r.get("transcript_available", True))
        result.append({"agent": agent, "condition": condition, "eligible": len(group),
                       "scored": len(ok), "instrumentation_failures": len(group) - len(ok),
                       "coverage": len(ok) / len(group), "call_references": len(calls),
                       "no_call_references": len(no_calls), "correct_calls": correct,
                       "correct_no_calls": correct_no_call, "recoverable_correct_tools": tools,
                       "jca": (correct + correct_no_call) / len(ok) if ok else None,
                       "tool_accuracy": tools / len(calls) if calls else None,
                       "conditional_arg_em": correct / tools if tools else None,
                       "no_call_accuracy": correct_no_call / len(no_calls) if no_calls else None,
                       "corpus_wer": lexical["wer"], "word_edits": lexical["edits"],
                       "reference_words": lexical["reference_words"]})
    return result


def paired_report(rows, roles, iterations=2000, seed=2027):
    controlled = [r for r in rows if r["analysis_type"] == "controlled"]
    by_agent = defaultdict(list)
    for row in controlled: by_agent[row["agent_id"]].append(row)
    report = {}
    for agent, group in sorted(by_agent.items()):
        indexed = defaultdict(dict)
        for row in group:
            key = (row["benchmark"], row["source_sample_id"])
            if row["error_class"] in indexed[key]: raise ValueError("Multiple variants per source-role")
            indexed[key][row["error_class"]] = row
        # Inference errors must not silently turn role comparisons into different cohorts.
        common = {key: values for key, values in indexed.items() if set(roles) <= set(values)}
        selected = [values[role] for values in common.values() for role in roles]
        losses = cei_by_class(selected)
        cis = {r["error_class"]: r for r in cluster_bootstrap_cei(selected, iterations, seed)}
        for loss in losses:
            loss.update(cis[loss["error_class"]])
            loss["loss_pp"] = 100 * loss["cei"]
        contrasts, pvalues = [], {}
        for a, b in combinations(roles, 2):
            pairs = [{"clean_jca": v[a]["corrupted_jca"], "corrupted_jca": v[b]["corrupted_jca"]} for v in common.values()]
            test = mcnemar_exact(pairs)
            name = a + "_vs_" + b
            pvalues[name] = test["p_value"]
            contrasts.append({"contrast": name, "sources": len(pairs), **test})
        adjusted = holm_adjust(pvalues)
        for contrast in contrasts: contrast["holm_p"] = adjusted[contrast["contrast"]]
        harmed = [r for r in selected if r["paired_outcome"] == "harmed"]
        counts = Counter(r["first_changed_failure"] or "unclassified" for r in harmed)
        report[agent] = {"common_sources": len(common), "sources_with_some_scored_variants": len(indexed),
                         "roles": losses, "role_contrasts": contrasts, "harmed_pairs": len(harmed),
                         "failure_categories": [{"category": k, "count": v, "share": v / len(harmed)} for k, v in sorted(counts.items())]}
    return report


def observed_report(rows):
    """Multi-label occurrences are descriptive; do not attribute harm to each label."""
    groups = defaultdict(list)
    for row in rows:
        if row["analysis_type"] == "observed_asr":
            groups[(row["agent_id"], row["condition"])].append(row)
    result = []
    for (agent, condition), group in sorted(groups.items()):
        labels = sorted({label for r in group for label in r.get("role_labels", [])})
        occurrences = []
        for label in labels:
            subset = [r for r in group if r["edit_count"] > 0 and label in r.get("role_labels", [])]
            occurrences.append({"role": label, "with_role_error": len(subset),
                                "action_failures": sum(not r["corrupted_jca"] for r in subset),
                                "harmed": sum(r["paired_outcome"] == "harmed" for r in subset)})
        result.append({"agent": agent, "condition": condition, "pairs": len(group),
                       "distinct_harmed_pairs": sum(r["paired_outcome"] == "harmed" for r in group),
                       "distinct_failed_hypotheses": sum(not r["corrupted_jca"] for r in group),
                       "error_hypotheses_without_role_labels": sum(r["edit_count"] > 0 and not r.get("role_labels") for r in group),
                       "role_occurrences": occurrences})
    return result


def markdown_report(summary):
    lines = ["# SpeechEvalX evaluation", "", "All rates are proportions; paired loss_pp is in percentage points.", "",
             "Instrumentation failures are excluded from scored denominators and reported as coverage.", "",
             "| Agent | Input | Scored / eligible | JCA | Tool | Arg-EM | No-call | Corpus WER |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "N/A" if v is None else f"{v:.4f}"
    for r in summary["actions"]:
        lines.append(f'| {r["agent"]} | {r["condition"]} | {r["scored"]}/{r["eligible"]} | ' + " | ".join(fmt(r[k]) for k in ("jca", "tool_accuracy", "conditional_arg_em", "no_call_accuracy", "corpus_wer")) + " |")
    for agent, r in summary["paired"].items():
        lines += ["", f"## Matched roles: {agent}", "", f'Common scored sources: {r["common_sources"]}', "",
                  "| Role | N | H | R | Loss (pp) | 95% CI (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for x in r["roles"]:
            lines.append(f'| {x["error_class"]} | {x["n"]} | {x["harmed"]} | {x["rescued"]} | {x["loss_pp"]:.2f} | [{100*x["ci_low"]:.2f}, {100*x["ci_high"]:.2f}] |')
        if r["failure_categories"]:
            lines += ["", "| Harmed-pair category | Count | Share |", "|---|---:|---:|"]
            for item in r["failure_categories"]:
                lines.append(f'| {item["category"]} | {item["count"]} | {item["share"]:.4f} |')
        if r["role_contrasts"]:
            lines += ["", "| Role contrast | Sources | Exact p | Holm p |", "|---|---:|---:|---:|"]
            for item in r["role_contrasts"]:
                lines.append(f'| {item["contrast"]} | {item["sources"]} | {item["p_value"]:.4g} | {item["holm_p"]:.4g} |')
    for observed in summary["observed_asr"]:
        lines += ["", f'## Observed ASR: {observed["agent"]} / {observed["condition"]}', "",
                  f'Distinct harmed pairs: {observed["distinct_harmed_pairs"]}. Role counts below overlap.', "",
                  "| Role | Hypotheses with role error | Action failures | Harmed |", "|---|---:|---:|---:|"]
        for item in observed["role_occurrences"]:
            lines.append(f'| {item["role"]} | {item["with_role_error"]} | {item["action_failures"]} | {item["harmed"]} |')
    if summary.get("diagnostics"):
        lines += ["", "## Failure prediction", "", "See summary.json for grouped OOF scores, transfer scores and bootstrap intervals.", "", json_text(summary["diagnostics"])]
    return "\n".join(lines) + "\n"


def json_text(value):
    import json
    return "\u0060\u0060\u0060json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n\u0060\u0060\u0060"
