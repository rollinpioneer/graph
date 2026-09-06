"""Safe declaration-only guard evaluator.

The input is JSON data, never Python source.  No ``eval``/``exec`` or dynamic
attribute access is used.  Guards intentionally have a small vocabulary so a
future graph edit cannot silently introduce an arbitrary predicate.
"""
from __future__ import annotations

from typing import Any, Mapping


ALLOWED_COMPARISONS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in"}
FORBIDDEN_FIELD_WORDS = {"scenario", "phase", "gold_mode", "future", "outcome", "event_log"}
ALLOWED_FIELDS = {
    "contact_present", "collision_detected", "object_inside_goal", "object_moving",
    "history_event_count", "contact_loss_in_history", "action_norm",
    "terminal_failure_event", "terminal_success_event", "horizon", "nonterminal",
}


class GuardError(ValueError):
    pass


def _field(name: Any) -> str:
    if not isinstance(name, str) or name not in ALLOWED_FIELDS or any(word in name.lower() for word in FORBIDDEN_FIELD_WORDS):
        raise GuardError(f"forbidden or invalid guard field: {name!r}")
    return name


def validate_guard(guard: Any) -> None:
    if guard in (None, {}, True):
        return
    if not isinstance(guard, dict) or len(guard) != 1:
        raise GuardError("guard must contain exactly one operator")
    operator, value = next(iter(guard.items()))
    if operator in {"all_of", "any_of"}:
        if not isinstance(value, list) or not value:
            raise GuardError(f"{operator} requires a non-empty list")
        for child in value:
            validate_guard(child)
        return
    if operator == "not":
        validate_guard(value)
        return
    if operator == "field":
        if not isinstance(value, dict) or set(value) != {"name", "comparison", "value"}:
            raise GuardError("field guard requires name, comparison and value")
        _field(value["name"])
        if value["comparison"] not in ALLOWED_COMPARISONS:
            raise GuardError(f"unsupported comparison: {value['comparison']!r}")
        return
    raise GuardError(f"unsupported guard operator: {operator!r}")


def _compare(left: Any, comparison: str, right: Any) -> bool:
    if comparison == "==":
        return left == right
    if comparison == "!=":
        return left != right
    if comparison == "in":
        return left in right if isinstance(right, (list, tuple, set, str)) else False
    if comparison == "not_in":
        return left not in right if isinstance(right, (list, tuple, set, str)) else True
    try:
        return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[comparison]
    except (TypeError, KeyError):
        return False


def evaluate_guard(guard: Any, context: Mapping[str, Any]) -> bool:
    validate_guard(guard)
    if guard in (None, {}, True):
        return True
    operator, value = next(iter(guard.items()))
    if operator == "all_of":
        return all(evaluate_guard(child, context) for child in value)
    if operator == "any_of":
        return any(evaluate_guard(child, context) for child in value)
    if operator == "not":
        return not evaluate_guard(value, context)
    spec = value
    return _compare(context.get(spec["name"]), spec["comparison"], spec["value"])


def guard_fields(guard: Any) -> set[str]:
    validate_guard(guard)
    if guard in (None, {}, True):
        return set()
    operator, value = next(iter(guard.items()))
    if operator == "field":
        return {value["name"]}
    if operator == "not":
        return guard_fields(value)
    result: set[str] = set()
    for child in value:
        result.update(guard_fields(child))
    return result
