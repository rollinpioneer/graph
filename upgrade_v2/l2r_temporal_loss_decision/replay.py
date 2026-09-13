from __future__ import annotations

from typing import Any

from .baseline import fit
from .sequential_decision import TemporalDecision


def _baseline_prefix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for index in range(1, len(rows) + 1):
        result = fit(rows[:index])
        if result.get("ready"):
            return result | {"prefix_index": index}
    return {"ready": False, "prefix_index": None, "anchor_xy": None, "scale_px": None}


def replay(record: dict[str, Any], method: str, *, window_ns: int = 500_000_000, theta_motion: float = 0.35, theta_sep: float = 0.10) -> dict[str, Any]:
    rows = [row for row in record.get("observations", []) if int(row.get("physical_time_ns", 0)) <= int(record.get("evidence_end_ns") or 0)]
    baseline = {"ready": False, "prefix_index": None, "anchor_xy": None, "scale_px": None}
    decision = TemporalDecision(method, window_ns, theta_motion, theta_sep, True, None, None)
    outputs: list[dict[str, Any]] = []
    first = None
    for index, row in enumerate(rows):
        # Establish the online-only baseline at the first prefix where it is
        # actually ready; never backfill it into earlier observations.
        if not baseline["ready"]:
            candidate = fit(rows[: index + 1])
            if candidate.get("ready"):
                baseline = candidate | {"prefix_index": index + 1}
                decision.anchor_xy = baseline.get("anchor_xy")
                decision.scale_px = baseline.get("scale_px")
        output = decision.step(rows, index)
        outputs.append(output)
        if first is None and output.get("state") == "LOSS_EVIDENCE":
            first = int(row["physical_time_ns"])
    return {"method": method, "outputs": outputs, "first_evidence_ns": first, "baseline": baseline, "evidence_end_ns": int(record.get("evidence_end_ns") or 0)}
