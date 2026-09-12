from __future__ import annotations

from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.temporal_scoring import score_temporal

RECOVERY_DEADLINE_S = 0.75
EARLY_ACTION_OUTCOME = "EARLY_ACTION"
PHYSICAL_LOSS_ONSET_FIELD = "physical_loss_onset"


def score_records(reference: dict[str, Any], online: list[dict[str, Any]],
                  actions: list[dict[str, Any]]) -> dict[str, Any]:
    truth = reference.get("reference_action", "none")
    onset = reference.get("loss_onset_time_abs") if truth == "recover_object" else None
    if truth == "retry_grasp":
        edge = next((row for row in online if row.get("attempt_end") is True), None)
        onset = None if edge is None else float(edge["time"])
        deadline = None if onset is None else min(onset + 0.50, float(online[-1]["time"]))
    elif truth == "recover_object":
        onset = None if onset is None else float(onset)
        deadline = None if onset is None else min(onset + RECOVERY_DEADLINE_S, float(online[-1]["time"]))
    else:
        deadline = None
    return score_temporal(truth_action=truth, event_onset=onset, deadline=deadline,
                          actions=actions, resolvable=bool(reference.get("resolvable", True)))
