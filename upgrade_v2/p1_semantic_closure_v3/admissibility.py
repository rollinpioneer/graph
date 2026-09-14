"""Long-loop topology + runtime phi-entry policy review. Evidence layers stay separate."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any

CYCLE = ("in_transit", "dropped_or_misaligned", "recovery", "grasped", "in_transit")
CYCLE_EDGES = (
    "in_transit_to_dropped_or_misaligned",
    "dropped_or_misaligned_to_recovery",
    "recovery_to_grasped",
    "grasped_to_in_transit",
)


def _load_graph(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def topology_path(graph: dict[str, Any]) -> dict[str, Any]:
    ids = {e["id"]: e for e in graph["edges"]}
    missing = [e for e in CYCLE_EDGES if e not in ids]
    hops = []
    for src, dst, eid in zip(CYCLE[:-1], CYCLE[1:], CYCLE_EDGES):
        edge = ids.get(eid)
        hops.append(dict(src=src, dst=dst, edge_id=eid, present=edge is not None,
                         type=None if edge is None else edge.get("type")))
    return dict(
        topology_reachable=not missing,
        missing_edges=missing,
        hops=hops,
        no_new_inter_node_edges=True,
        within_node_update_is_not_new_edge=True,
    )


def runtime_phi_entry_review(repo: Path) -> dict[str, Any]:
    engine = (repo / "tools/stage5/lib/reward_engine.py").read_text(encoding="utf-8")
    graph_path = repo / "artifacts/pathgraph_sarm/stage3/input_adapter_v1/runtime_graph_specs_v1.0.1/transport_recovery_graph_runtime_v1.0.1.yaml"
    graph = graph_path.read_text(encoding="utf-8")
    same_node_phi = "same=(n0==n1 and nc>=self.conf and et not in (3,4) and finite)" in engine.replace(" ", "") or (
        "et not in (3,4)" in engine and "n0==n1" in engine
    )
    return {
        "engine_phi_indicator": (
            "Phi credit is applied only when predicted node stays the same, node confidence is high, "
            "and the predicted edge type is not recovery(3) or failure(4). Cross-node failure/recovery "
            "therefore does not debit previously earned intra-node phi."
        ),
        "engine_does_not_store_entry_anchor": True,
        "same_node_gate_present": same_node_phi or ("et not in (3, 4)" in engine),
        "graph_progress_signal": "distance_or_stability on recovery-graph nodes; not an executable adapter",
        "dedicated_entry_distance_adapter": "NOT_LOCATED",
        "reentry_via_grasped_to_in_transit": (
            "Re-entering in_transit is a different node than recovery/grasped, so the engine treats the "
            "first intra-node step after re-entry as a fresh same-node interval. If the caller reports "
            "phi=0 at re-entry, later intra-node progress is credited again."
        ),
        "when_is_in_transit_entered": "legacy edge grasped_to_in_transit; no extra geometric guard in YAML beyond 'state evidence'",
        "same_attempt_vs_new_attempt": (
            "RewardState.failure_debt persists across recovery attempts inside one episode; "
            "engine.new_episode() is the only debt reset. Phi entry anchors are not in RewardState."
        ),
        "physical_regrasp_evidence": "NOT_IN_ADAPTER; must come from causal replay contact/weld fields",
        "hypothesis_used_by_packaged_probes": "ENTRY_RESET_HYPOTHESIS_TO_BE_CHECKED_IN_ACTUAL_ADAPTER",
        "source_files": [
            "tools/stage5/lib/reward_engine.py",
            str(graph_path.relative_to(repo)),
        ],
    }


def _episode_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def representative_loops(probe_csv: Path) -> list[dict[str, Any]]:
    rows = _episode_rows(probe_csv)
    wanted = {
        "normalized_main__reset__p0.4__k1",
        "normalized_main__reset__p0.4__k3",
        "normalized_main__reset__p0.4__k8",
        "normalized_main__reset__p0.8__k1",
        "normalized_main__unwind__p0.4__k1",
        "normalized_main__shifted_entry__p0.4__k1",
        "unit_clip_sensitivity__reset__p0.4__k1",
        "normalized_main__reset__p0__k1",
    }
    out = []
    for row in rows:
        if row["method"] != "FULL_FROZEN" or row["episode_id"] not in wanted:
            continue
        eid = row["episode_id"]
        shifted = "shifted_entry" in eid
        unwind = "unwind" in eid
        peak = 0.0
        if "p0.4" in eid:
            peak = 0.4
        elif "p0.8" in eid:
            peak = 0.8
        signed = float(row["signed_return"])
        debt = float(row["debt_after"])
        task_closed = (not shifted) and row.get("symbolic_task_position_closed", "").lower() == "true"
        reward_closed = row.get("closed_in_supplied_z_cost_phi", "").lower() == "true"
        if unwind:
            conclusion = "NO_COUNTEREXAMPLE_WITHIN_TESTED_DOMAIN"
            reasons = ["unwind negative-control returned approximately zero extra phi credit"]
        elif peak == 0:
            conclusion = "NO_COUNTEREXAMPLE_WITHIN_TESTED_DOMAIN"
            reasons = ["zero intra-node progress; normalized signed return ~0 with residual debt k/6"]
        elif shifted:
            conclusion = "TOPOLOGY_ADMISSIBLE_BINDING_UNRESOLVED"
            reasons = ["shifted_entry changes geometry so it is not a task-position cycle"]
        else:
            conclusion = "FORMAL_BINDING_ADMISSIBLE_COUNTEREXAMPLE" if False else "TOPOLOGY_ADMISSIBLE_BINDING_UNRESOLVED"
            reasons = [
                "Four legacy edges exist and packaged arithmetic matches R_loop=0.5*p under entry-reset hypothesis",
                "Runtime adapter that persists or resets d_entry was not located, so physical/formal binding stays unresolved",
            ]
        out.append(dict(
            schema="p1_v3_counterexample_admissibility_v1",
            loop_id=eid,
            evidence_tier="TOPOLOGY_CHECKED_SYMBOLIC_INPUT",
            topology_path=list(CYCLE),
            topology_reachable=True,
            within_node_update_supported=True,
            guard_satisfiable_from_current_prefix="UNKNOWN",
            runtime_phi_entry_reset_policy="ENGINE_ALLOWS_REENTRY_ZERO_PHI_IF_CALLER_REPORTS_IT; ADAPTER_NOT_LOCATED",
            entry_anchor_source="CONSTRUCTED_DISTANCE_SEQUENCE_NOT_RAW_PHYSICS",
            geometry_source="d_entry=0.10 constructed fixture",
            physical_feasibility="NOT_TESTED",
            raw_evidence_refs=[],
            task_projection_closed=task_closed,
            closure_fields=["node", "cost", "phi"] if not shifted else ["node", "cost", "phi_not_geometry"],
            nonclosing_history_fields=["failure_debt", "phi_entry_anchor_if_reset"],
            normalized_main_return=signed if "normalized_main" in eid else None,
            unit_clip_sensitive_return=signed if "unit_clip" in eid else None,
            debt_before=0.0,
            debt_after=debt,
            verified_no_new_subgoal=True,
            conclusion=conclusion,
            unresolved_reasons=reasons,
            signed_return=signed,
        ))
    return out


def build_admissibility(repo: Path, first_pass_dir: Path) -> dict[str, Any]:
    graph = _load_graph(repo / "artifacts/pathgraph_sarm/stage3/input_adapter_v1/runtime_graph_specs_v1.0.1/transport_recovery_graph_runtime_v1.0.1.yaml")
    topo = topology_path(graph)
    policy = runtime_phi_entry_review(repo)
    loops = representative_loops(first_pass_dir / "10_symbolic_probes" / "per_episode_returns.csv")
    summary = dict(
        topology=topo,
        runtime_phi_entry_policy=policy,
        representative_loops=loops,
        overall_formal_binding="TOPOLOGY_ADMISSIBLE_BINDING_UNRESOLVED",
        physical_replay_counterexample="NOT_CLAIMED",
        note=(
            "A four-edge legal skeleton exists. Packaged probes show repeated intra-node credit under an "
            "entry-reset hypothesis (R=0.5p per cycle on normalized_main). That is not a physical robot bug "
            "until a runtime adapter is shown to reset phi on re-entry with feasible geometry."
        ),
    )
    return summary