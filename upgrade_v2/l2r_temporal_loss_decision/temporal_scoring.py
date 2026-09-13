from __future__ import annotations

from typing import Any


def score_event(*, truth: str, onset_ns: int | None, end_ns: int, first_evidence_ns: int | None, deadline_ns: int = 750_000_000) -> str:
    if first_evidence_ns is not None and first_evidence_ns > end_ns:
        raise ValueError("evidence after observed end")
    if truth == "UNRESOLVED":
        return "REFERENCE_UNRESOLVED"
    if truth == "NO_LOSS":
        return "CORRECT_NO_EVENT" if first_evidence_ns is None else "FALSE_LOSS_EVENT"
    if truth != "LOSS" or onset_ns is None:
        raise ValueError("LOSS requires onset")
    if first_evidence_ns is None:
        return "RIGHT_CENSORED" if end_ns < onset_ns + deadline_ns else "MISSED"
    if first_evidence_ns < onset_ns:
        return "EARLY_LOSS_EVENT"
    return "ON_TIME" if first_evidence_ns <= onset_ns + deadline_ns else "LATE"


def truth_for(record: dict[str, Any]) -> str:
    return {"LOSS": "LOSS", "NO_LOSS": "NO_LOSS", "COMMANDED_RELEASE": "NO_LOSS"}.get(record.get("reference_state", "UNRESOLVED"), "UNRESOLVED")
