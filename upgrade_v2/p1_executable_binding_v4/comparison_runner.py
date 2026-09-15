"""Build scorer rows with real intermediate linear/unordered progress."""
from __future__ import annotations
from typing import Any
from .reference_binding import NODE_INDEX, NONE_EDGE

def to_score_row(tr: dict[str, Any], *, phi_before: float, phi_after: float,
                 evidence_tier: str, contract: str, group: str) -> dict[str, Any]:
    a, b = tr["node_before"], tr["node_after"]
    success_a = a == "success"
    success_b = b == "success"
    term_a = a in ("success", "terminal_failure")
    term_b = b in ("success", "terminal_failure")
    return dict(
        episode_id=tr["episode_id"], task_id="transport_recovery", root_group_id=group,
        evidence_tier=evidence_tier, cost_contract_id=contract, step=tr["step"],
        node_before=NODE_INDEX[a], node_after=NODE_INDEX[b],
        edge_id=tr.get("edge_index", NONE_EDGE), edge_type=tr["edge_type"],
        cost_before=float(tr["cost_before"]), cost_after=float(tr["cost_after"]),
        phi_before=float(phi_before), phi_after=float(phi_after),
        linear_before=float(tr["linear_before"]), linear_after=float(tr["linear_after"]),
        unordered_before=float(tr["unordered_before"]), unordered_after=float(tr["unordered_after"]),
        t0_ns=int(tr["t0_ns"]), t1_ns=int(tr["t1_ns"]), available_at_ns=int(tr["available_at_ns"]),
        success_before=success_a, success_after=success_b,
        terminal_before=term_a, terminal_after=term_b,
        raw_state_reference=tr.get("source_reference"),
        reference_known_at_ns=int(tr["available_at_ns"]),
        loss_episode_id=tr.get("loss_episode_id"),
        recovery_completed=bool(tr.get("recovery_completed")),
        recovered_loss_episode_id=tr.get("recovered_loss_episode_id"),
        classification=tr.get("classification"),
        node_before_label=a, node_after_label=b,
    )


def attach_phi(transitions: list[dict[str, Any]], AnchorBank, episode: str, object_id: str, goal_id: str):
    bank = AnchorBank()
    rows = []
    prev_phi = 0.0
    for i, tr in enumerate(transitions):
        tr = dict(tr, step=i)
        a, b = tr["node_before"], tr["node_after"]
        d0, d1 = tr.get("distance_before"), tr.get("distance_after")
        entered = (a != "in_transit" and b == "in_transit") or (a == "in_transit" and "in_transit" not in {k[3] for k in bank.anchors})
        # first time we are in in_transit after-state
        if b == "in_transit":
            obs = bank.observe((episode, object_id, goal_id, "in_transit"), d1, tr["available_at_ns"],
                               entered=(a != "in_transit"))
            phi_after = 0.0 if obs["phi"] is None else obs["phi"]
            phi_status = obs["status"]
        else:
            phi_after = 0.0
            phi_status = "STRUCTURAL_PHI_ZERO"
        if a == "in_transit":
            obs0 = bank.observe((episode, object_id, goal_id, "in_transit"), d0, tr["t0_ns"], entered=False)
            phi_before = 0.0 if obs0["phi"] is None else obs0["phi"]
        else:
            phi_before = 0.0
        # continuity: if previous row exists, phi_before should match previous phi_after
        if rows:
            phi_before = rows[-1]["phi_after"] if a == transitions[i-1]["node_after"] else phi_before
        tr["phi_status"] = phi_status
        rows.append(to_score_row(tr, phi_before=phi_before, phi_after=phi_after,
                                 evidence_tier="CAUSAL_STATE_REPLAY",
                                 contract="P1_NORMALIZED_RUNTIME_GRAPH_V1",
                                 group=episode.rsplit("_r", 1)[0] if "_r" in episode else episode))
        prev_phi = phi_after
    return rows, bank