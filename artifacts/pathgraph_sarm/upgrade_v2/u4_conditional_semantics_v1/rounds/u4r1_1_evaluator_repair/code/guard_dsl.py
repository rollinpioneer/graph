"""Safe declaration-only guard evaluator.

The input is JSON data, never Python source.  No ``eval``/``exec`` or dynamic
attribute access is used.  Guards intentionally have a small vocabulary so a
future graph edit cannot silently introduce an arbitrary predicate.
"""
from __future__ import annotations

from typing import Any, Mapping


ALLOWED_COMPARISONS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in"}
FORBIDDEN_FIELD_WORDS = {"scenario", "phase", "gold_mode", "future", "outcome", "event_log", "eval", "exec"}
# This is deliberately a closed vocabulary.  The evaluator can therefore
# distinguish an absent observable from an observable whose value is false.
ALLOWED_FIELDS = {
    "terminal_failure_event", "stable_success_event", "horizon_censored",
    "contact_before", "contact_after", "contact_recently_lost",
    "collision_detected", "object_inside_goal", "stagnation_detected",
    "goal_distance_delta_sign", "object_speed_bin", "recent_recovery_attempt",
    "contact_present", "object_moving", "history_event_count", "action_norm",
    # Compatibility aliases used by the recovered U4 B+ graph.
    "terminal_success_event", "horizon", "nonterminal", "contact_loss_in_history",
}

MISSING = object()


class GuardError(ValueError):
    pass


def _field(name: Any) -> str:
    if not isinstance(name, str) or name not in ALLOWED_FIELDS or any(word in name.lower() for word in FORBIDDEN_FIELD_WORDS):
        raise GuardError(f"forbidden or invalid guard field: {name!r}")
    return name


def validate_guard(guard: Any) -> None:
    if guard in (None, {}, True):
        return
    if isinstance(guard, dict) and {"field", "comparison"}.issubset(guard) and ("constant" in guard or "value" in guard):
        _field(guard["field"])
        if guard["comparison"] not in ALLOWED_COMPARISONS:
            raise GuardError(f"unsupported comparison: {guard['comparison']!r}")
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
    if operator in ALLOWED_COMPARISONS and isinstance(value, dict):
        _field(value.get("field", value.get("name")))
        if "constant" not in value and "value" not in value:
            raise GuardError("compact field guard requires constant")
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


def guard_status(guard: Any, context: Mapping[str, Any]) -> str:
    """Return ``true``, ``false`` or ``ambiguous`` for a guard.

    ``ambiguous`` is important for safety: a missing causal observable must
    not silently route an occurrence to the negative branch.
    """
    validate_guard(guard)
    if guard in (None, {}, True):
        return "true"
    operator, value = next(iter(guard.items()))
    if operator == "all_of":
        statuses = [guard_status(child, context) for child in value]
        if "false" in statuses:
            return "false"
        return "ambiguous" if "ambiguous" in statuses else "true"
    if operator == "any_of":
        statuses = [guard_status(child, context) for child in value]
        if "true" in statuses:
            return "true"
        return "ambiguous" if "ambiguous" in statuses else "false"
    if operator == "not":
        status = guard_status(value, context)
        return {"true": "false", "false": "true", "ambiguous": "ambiguous"}[status]
    if isinstance(guard, dict) and {"field", "comparison"}.issubset(guard):
        name = guard["field"]; comparison = guard["comparison"]; right = guard.get("constant", guard.get("value"))
        left = context.get(name, MISSING)
        if left is MISSING:
            return "ambiguous"
        return "true" if _compare(left, comparison, right) else "false"
    spec = value
    if operator in ALLOWED_COMPARISONS:
        name = spec.get("field", spec.get("name"))
        right = spec.get("constant", spec.get("value"))
        left = context.get(name, MISSING)
        if left is MISSING:
            return "ambiguous"
        return "true" if _compare(left, operator, right) else "false"
    left = context.get(spec["name"], MISSING)
    if left is MISSING:
        return "ambiguous"
    return "true" if _compare(left, spec["comparison"], spec["value"]) else "false"


def evaluate_guard(guard: Any, context: Mapping[str, Any]) -> bool:
    """Boolean compatibility API; ambiguous guards do not activate."""
    return guard_status(guard, context) == "true"


def guard_fields(guard: Any) -> set[str]:
    validate_guard(guard)
    if guard in (None, {}, True):
        return set()
    if isinstance(guard, dict) and {"field", "comparison"}.issubset(guard):
        return {guard["field"]}
    operator, value = next(iter(guard.items()))
    if operator == "field":
        return {value["name"]}
    if operator in ALLOWED_COMPARISONS:
        return {value.get("field", value.get("name"))}
    if operator == "not":
        return guard_fields(value)
    result: set[str] = set()
    for child in value:
        result.update(guard_fields(child))
    return result
