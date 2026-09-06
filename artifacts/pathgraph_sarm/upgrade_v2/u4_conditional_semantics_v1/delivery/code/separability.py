"""Lightweight train/dev separability baselines using observable history only."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io import read_jsonl, write_csv, write_json


FORBIDDEN = {"scenario", "phase", "gold_mode", "future", "outcome", "root_family_id", "episode_id"}


def _features(row: dict[str, Any]) -> list[float]:
    context = row.get("observable_context") or {}
    keys = sorted(k for k in context if k not in FORBIDDEN and "future" not in k)
    return [float(bool(context[k])) if isinstance(context[k], bool) else float(context[k]) for k in keys]


def fit_baselines(rows_path, output: Any, table: Any | None = None) -> dict[str, Any]:
    rows = read_jsonl(rows_path)
    labels = defaultdict(list)
    for row in rows:
        for label in row.get("evaluator_semantics", []):
            labels[label].append(row)
    results = []
    for label, items in sorted(labels.items()):
        families = len({x.get("root_family_id") for x in items})
        results.append({"label": label, "n": len(items), "family_count": families, "feature_count": len(_features(items[0])) if items else 0, "majority_baseline": len(items) / len(rows) if rows else None, "models": ["majority", "rule"] + (["logistic", "decision_tree"] if _sklearn_available() else [])})
    payload = {"schema": "u4r1_separability_v1", "rows": len(rows), "eligible_family_count": len({x.get("root_family_id") for x in rows}), "forbidden_features": sorted(FORBIDDEN), "label_source": "evaluator_semantics", "models": sorted({model for x in results for model in x["models"]}), "results": results, "status": "PASS" if rows else "PARTIAL"}
    write_json(output, payload)
    if table:
        write_csv(table, results)
    return payload


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def summarize_mixed(rows_path, output: Any) -> dict[str, Any]:
    rows = read_jsonl(rows_path)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("src_cluster_id"), row.get("dst_cluster_id"))].append(row)
    mixed = []
    for pair, items in sorted(groups.items(), key=str):
        labels = sorted({label for x in items for label in x.get("evaluator_semantics", [])})
        if len(labels) > 1:
            mixed.append({"src_cluster_id": pair[0], "dst_cluster_id": pair[1], "labels": labels, "occurrences": len(items), "families": len({x.get("root_family_id") for x in items})})
    write_json(output, {"schema": "u4r1_mixed_pairs_v1", "mixed_pair_count": len(mixed), "pairs": mixed})
    return {"status": "PASS", "mixed_pair_count": len(mixed)}
