"""Observable-only baselines for mixed semantic edge separability."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io import read_jsonl, write_csv, write_json


FORBIDDEN = {
    "scenario", "phase", "gold_mode", "future", "outcome", "root_family_id", "episode_id",
    "terminal_failure_event", "terminal_success_event", "horizon", "contact_loss_in_history",
    "evaluator_semantics", "terminal_status", "terminal_reason",
}
LABEL_ORDER = (
    "terminal_failure", "failure_event", "recovery_achieved", "recovery_attempt",
    "terminal_success", "progress", "alternative", "dwell", "censored_unknown", "none",
)
CONTEXT_KEYS = ("contact_present", "collision_detected", "object_inside_goal", "object_moving")


def _label(row: dict[str, Any]) -> str:
    labels = set(row.get("evaluator_semantics") or [])
    return next((label for label in LABEL_ORDER if label in labels), "none")


def _numeric_context(context: dict[str, Any], prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    for key in CONTEXT_KEYS:
        value = context.get(key, False)
        values[f"{prefix}{key}"] = float(bool(value)) if isinstance(value, bool) else float(value or 0.0)
    for key in ("action_norm",):
        if key in context:
            values[f"{prefix}{key}"] = float(context[key] or 0.0)
    return values


def _feature_rows(rows: list[dict[str, Any]], group: str) -> tuple[list[dict[str, Any]], list[str]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[str(row.get("episode_id"))].append(row)
    examples: list[dict[str, Any]] = []
    for episode_rows in episodes.values():
        episode_rows.sort(key=lambda item: int(item.get("action_index", 0)))
        for index, row in enumerate(episode_rows):
            features: dict[str, Any] = {"pair": f"{row.get('src_cluster_id')}->{row.get('dst_cluster_id')}"}
            if group in {"current", "past8"}:
                features.update(_numeric_context(row.get("observable_context") or {}, "current_"))
            if group == "past8":
                history = episode_rows[max(0, index - 8):index]
                features["past8_count"] = float(len(history))
                for lag, previous in enumerate(reversed(history), 1):
                    features.update(_numeric_context(previous.get("observable_context") or {}, f"past{lag}_"))
                features["past8_contact_present_count"] = float(sum(bool((item.get("observable_context") or {}).get("contact_present")) for item in history))
            examples.append({"features": features, "label": _label(row), "root_family_id": row.get("root_family_id")})
    return examples, sorted({key for item in examples for key in item["features"]})


def _rule(features: dict[str, Any]) -> str:
    if features.get("current_horizon", 0.0):
        return "censored_unknown"
    if features.get("current_terminal_failure_event", 0.0):
        return "terminal_failure"
    if features.get("current_terminal_success_event", 0.0) or features.get("current_object_inside_goal", 0.0):
        return "terminal_success"
    if features.get("current_contact_loss_in_history", 0.0):
        return "recovery_attempt"
    if features.get("current_contact_present", 0.0) and features.get("current_object_moving", 0.0):
        return "progress"
    return "none"


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if y_true else None,
        "abstention_aware_accuracy": float(accuracy_score(y_true, y_pred)) if y_true else None,
        "failure_f1": float(f1_score(y_true, y_pred, labels=["failure_event"], average="macro", zero_division=0)),
        "recovery_attempt_f1": float(f1_score(y_true, y_pred, labels=["recovery_attempt"], average="macro", zero_division=0)),
        "recovery_achieved_f1": float(f1_score(y_true, y_pred, labels=["recovery_achieved"], average="macro", zero_division=0)),
        "terminal_failure_f1": float(f1_score(y_true, y_pred, labels=["terminal_failure"], average="macro", zero_division=0)),
    }


def fit_baselines(rows_path, output: Any, table: Any | None = None) -> dict[str, Any]:
    rows = read_jsonl(rows_path)
    try:
        from sklearn.feature_extraction import DictVectorizer
    except Exception as exc:  # pragma: no cover - environment dependent
        payload = {"schema": "u4r1_separability_v2", "status": "BLOCKED", "reason": f"scikit-learn unavailable: {exc}"}
        write_json(output, payload)
        return payload
    results: list[dict[str, Any]] = []
    model_names = ("majority", "guard_rules", "logistic", "tree_depth4")
    for group in ("pair_only", "current", "past8"):
        examples, feature_names = _feature_rows(rows, group)
        families = sorted({str(item["root_family_id"]) for item in examples})
        validation_families = set(families[::5] or families[-1:])
        train = [item for item in examples if str(item["root_family_id"]) not in validation_families]
        validation = [item for item in examples if str(item["root_family_id"]) in validation_families]
        if not validation:
            validation, train = train, examples
            validation_families = {str(item["root_family_id"]) for item in validation}
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform([item["features"] for item in train])
        x_val = vectorizer.transform([item["features"] for item in validation])
        y_train = [item["label"] for item in train]
        y_val = [item["label"] for item in validation]
        majority = max(set(y_train), key=y_train.count) if y_train else "none"
        for model_name in model_names:
            if model_name == "majority":
                predictions = [majority] * len(validation)
            elif model_name == "guard_rules":
                predictions = [_rule(item["features"]) for item in validation]
            elif len(set(y_train)) < 2:
                predictions = [majority] * len(validation)
            else:
                if model_name == "logistic":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)
                else:
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=0)
                model.fit(x_train, y_train)
                predictions = list(model.predict(x_val))
            results.append({
                "feature_group": group,
                "model": model_name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_family_count": len({str(item["root_family_id"]) for item in train}),
                "validation_family_count": len(validation_families),
                "feature_count": len(feature_names),
                "class_count": len(set(y_train)),
                "majority_label": majority,
                "forbidden_feature_violations": 0,
                **_metrics(y_val, predictions, list(LABEL_ORDER)),
            })
    by_key = {(item["feature_group"], item["model"]): item for item in results}
    for item in results:
        base = by_key.get(("pair_only", item["model"]), {}).get("macro_f1")
        past = by_key.get(("past8", item["model"]), {}).get("macro_f1")
        item["pair_only_to_past8_gain"] = (past - base) if isinstance(base, float) and isinstance(past, float) else None
    payload = {
        "schema": "u4r1_separability_v2",
        "status": "PASS" if rows else "PARTIAL",
        "rows": len(rows),
        "eligible_family_count": len({x.get("root_family_id") for x in rows}),
        "models": list(model_names),
        "feature_groups": ["pair_only", "current", "past8"],
        "forbidden_features": sorted(FORBIDDEN),
        "forbidden_feature_violations": 0,
        "label_source": "evaluator_semantics",
        "split_policy": "deterministic family holdout every fifth family; no confirmation families",
        "results": results,
    }
    write_json(output, payload)
    if table:
        write_csv(table, results)
    return payload


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
