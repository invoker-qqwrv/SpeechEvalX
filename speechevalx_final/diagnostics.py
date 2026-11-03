"""Source-grouped cumulative diagnostic ablation and no-refit model transfer.

Semantic-WER must be supplied by its reference implementation; it is never
silently replaced by an approximate metric. Optional BERT uses masked mean
pooling including special tokens, with a pinned model revision if provided.
"""
from __future__ import annotations
from collections import Counter
import math


def bert_distances(rows, model_name="bert-base-uncased", revision=None, device="cpu"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    cache = {}
    def embed(text):
        if text not in cache:
            inputs = tokenizer(text, return_tensors="pt", truncation=False).to(device)
            if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError("BERT input exceeds maximum length; refusing silent truncation")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                cache[text] = ((hidden * mask).sum(1) / mask.sum(1)).cpu()
        return cache[text]
    for row in rows:
        left, right = embed(row["reference_transcript"]), embed(row["corrupted_transcript"])
        value = float(1 - torch.nn.functional.cosine_similarity(left, right).item())
        row.setdefault("features", {})["bert_cosine"] = max(0.0, min(2.0, value))


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(set(y)) < 2: return {"roc_auc": None, "average_precision": None, "n": len(y)}
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)), "n": len(y)}


def _paired_bootstrap(y, base, full, groups, iterations, seed):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    unique = sorted(set(groups))
    indexes = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(seed)
    yy, bb, ff = np.asarray(y), np.asarray(base), np.asarray(full)
    values = []
    for _ in range(iterations):
        take = np.concatenate([indexes[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(set(yy[take])) < 2: continue
        values.append(float(roc_auc_score(yy[take], ff[take]) - roc_auc_score(yy[take], bb[take])))
    if not values: return {"ci_low": None, "ci_high": None, "valid_bootstrap_iterations": 0}
    low, high = np.quantile(values, [.025, .975])
    # Two-sided bootstrap tail probability, with finite-sample correction.
    p = min(1.0, 2 * min((1 + sum(v <= 0 for v in values)) / (len(values) + 1),
                        (1 + sum(v >= 0 for v in values)) / (len(values) + 1)))
    return {"roc_auc_delta": float(roc_auc_score(yy, ff) - roc_auc_score(yy, bb)),
            "ci_low": float(low), "ci_high": float(high), "bootstrap_p": p,
            "valid_bootstrap_iterations": len(values)}


def run_diagnostics(rows, options, seed=2027):
    try:
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise RuntimeError("Diagnostics require: pip install numpy scikit-learn") from exc
    selected = [r for r in rows if r["analysis_type"] == "controlled" and r["clean_jca"]]
    primary = options.get("primary_agent")
    if not primary:
        primary = next((r["agent_id"] for r in selected), None)
    source = [r for r in selected if r["agent_id"] == primary]
    report = {"primary_agent": primary, "endpoint": "corrupted action failure among clean-correct sources",
              "models": {}, "transfer": {}, "warnings": []}
    if not source:
        report["status"] = "unavailable: no clean-correct controlled pairs"
        return report, []
    folds = options.get("folds", 5)
    if not isinstance(folds, int) or folds < 2: raise ValueError("diagnostics.folds must be >= 2")
    if options.get("compute_bert"):
        bert_distances(selected, options.get("bert_model", "bert-base-uncased"),
                       options.get("bert_revision"), options.get("bert_device", "cpu"))
    def has_feature(name):
        for r in source:
            value = r.get("features", {}).get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): return False
            if value < 0 or (name == "bert_cosine" and value > 2): return False
        return True
    levels = [("B0", ["severity", "edit_count", "length"], ["operation", "benchmark"])]
    if has_feature("bert_cosine"):
        levels.append(("B1", levels[-1][1] + ["bert_cosine"], levels[-1][2]))
        if has_feature("semantic_wer"):
            levels.append(("B2", levels[-1][1] + ["semantic_wer"], levels[-1][2]))
            levels.append(("B3", levels[-1][1] + ["critical"], levels[-1][2]))
            levels.append(("B4", levels[-1][1], levels[-1][2] + ["role"]))
    if len(levels) < 5:
        report["warnings"].append("Full B0-B4 unavailable: provide finite bert_cosine and reference Semantic-WER for EVERY primary controlled variant. Missing controls are never replaced with zero.")
    def features(row, numeric, categorical):
        values = {"severity": row["severity"], "edit_count": row["edit_count"],
                  "length": len(row["reference_transcript"].split()), "operation": row["edit_operation"],
                  "benchmark": row["benchmark"], "role": row["error_class"],
                  "critical": int(row["error_class"] not in {"function_word", "other_noncritical"}),
                  **{k: row.get("features", {}).get(k) for k in ("bert_cosine", "semantic_wer")}}
        return [values[k] for k in numeric + categorical]
    y = np.asarray([int(not r["corrupted_jca"]) for r in source])
    groups = np.asarray([r["group_id"] for r in source])
    report.update({"samples": len(source), "source_groups": len(set(groups)), "class_counts": dict(Counter(int(v) for v in y))})
    if len(set(groups)) < folds or len(set(y)) < 2:
        report["status"] = "unavailable: need enough independent groups and both outcome classes"
        return report, []
    split = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    if any(len(set(y[test])) < 2 or len(set(y[train])) < 2 for train, test in split):
        report["status"] = "unavailable: a grouped fold lacks an outcome class; use more data or fewer folds"
        return report, []
    targets = {agent: [r for r in selected if r["agent_id"] == agent] for agent in sorted({r["agent_id"] for r in selected} - {primary})}
    oof, predictions = {}, []
    report["transfer_eligible_samples"] = {agent: len(target) for agent, target in targets.items()}
    for name, numeric, categorical in levels:
        x = np.asarray([features(r, numeric, categorical) for r in source], dtype=object)
        probs = np.empty(len(y), dtype=float)
        transfer_records = {agent: [] for agent in targets}
        for fold, (train, test) in enumerate(split):
            transform = ColumnTransformer([("numeric", StandardScaler(), list(range(len(numeric)))),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(len(numeric), len(numeric) + len(categorical))))])
            estimator = Pipeline([("features", transform), ("logistic", LogisticRegression(
                C=float(options.get("C", 1.0)), max_iter=2000, random_state=seed))])
            estimator.fit(x[train], y[train])
            probs[test] = estimator.predict_proba(x[test])[:, 1]
            heldout = set(groups[test])
            for index in test:
                row = source[index]
                predictions.append({"agent": primary, "feature_set": name, "fold": fold,
                                    "benchmark": row["benchmark"], "source_sample_id": row["source_sample_id"],
                                    "role": row["error_class"], "group_id": row["group_id"], "y": int(y[index]),
                                    "probability": float(probs[index]), "mode": "out_of_fold"})
            for agent, target in targets.items():
                test_rows = [r for r in target if r["group_id"] in heldout]
                if not test_rows: continue
                try:
                    tx = np.asarray([features(r, numeric, categorical) for r in test_rows], dtype=object)
                    tp = estimator.predict_proba(tx)[:, 1]
                except (KeyError, ValueError):
                    transfer_records[agent] = None
                    continue
                if transfer_records[agent] is None: continue
                for row, probability in zip(test_rows, tp):
                    entry = {"agent": agent, "feature_set": name, "fold": fold,
                             "benchmark": row["benchmark"], "source_sample_id": row["source_sample_id"],
                             "role": row["error_class"], "group_id": row["group_id"],
                             "y": int(not row["corrupted_jca"]), "probability": float(probability), "mode": "transfer_no_refit"}
                    transfer_records[agent].append(entry)
        oof[name] = probs
        report["models"][name] = _metrics(y, probs)
        for agent, entries in transfer_records.items():
            report["transfer"].setdefault(agent, {})[name] = _metrics([e["y"] for e in entries], [e["probability"] for e in entries]) if entries else {"status": "unavailable: missing features or held-out support"}
            if entries: predictions.extend(entries)
    if "B4" in oof:
        report["B4_minus_B3"] = _paired_bootstrap(y, oof["B3"], oof["B4"], groups,
                                                   int(options.get("bootstrap_iterations", 2000)), seed)
    report["transfer_note"] = "Transfer uses the primary model fold estimator without refitting, and only target rows in its held-out source groups. Compare predicted n with transfer_eligible_samples."
    report["status"] = "complete" if len(levels) == 5 else "partial: semantic controls missing"
    return report, predictions
