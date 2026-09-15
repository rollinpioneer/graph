"""Bridge existing V6-normalized states to a method ledger. Uses frozen V6 scorer."""
from __future__ import annotations
import csv, json, hashlib, math, sys
from collections import defaultdict
from pathlib import Path

V6_PKG = Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0")

def _load_v6():
    sys.path.insert(0, str(V6_PKG / "tools"))
    from reward_v6 import score_episode
    from engine_parity import load_engine
    from normalize_v5 import normalize, load_upstream
    return score_episode, load_engine, normalize, load_upstream

def extra_on_states(states):
    from .extra_methods import sparse_terminal, event_plus_count, MatchedEvents, unordered_valid_count
    bank = MatchedEvents()
    rows = []
    for i, (a, b) in enumerate(zip(states, states[1:])):
        r_s = sparse_terminal(a, b)
        r_e, dv, ev = event_plus_count(a, b, bank)
        r_u = unordered_valid_count(a, b)
        rows.append({"episode_id": a["episode_id"], "from_state_index": i, "to_state_index": i+1,
                     "available_at_ns": b["available_at_ns"], "SPARSE_TERMINAL": r_s,
                     "UNORDERED_VALID_COUNT": r_u,
                     "VALID_COUNT_PLUS_MATCHED_EVENTS_V1": r_e, "count_delta": dv, "event_delta": ev})
    return rows

def score_sequence(states, Engine):
    from .extra_methods import n_subgoals, valid_count
    score_episode, *_ = _load_v6()
    d, v, p = score_episode(states, Engine)
    extra = extra_on_states(states)
    by_step = {r["from_state_index"]: r for r in extra}
    ledger = []
    for row in d:
        step = row["step"]
        rec = {"episode_id": row["episode_id"], "method": row["method"],
               "from_state_index": step, "to_state_index": step+1,
               "available_at_ns": row.get("t1_ns") or states[step+1]["available_at_ns"],
               "reward": row["reward"]}
        ledger.append(rec)
    for r in extra:
        for m in ("SPARSE_TERMINAL", "UNORDERED_VALID_COUNT", "VALID_COUNT_PLUS_MATCHED_EVENTS_V1"):
            ledger.append({"episode_id": r["episode_id"], "method": m,
                           "from_state_index": r["from_state_index"], "to_state_index": r["to_state_index"],
                           "available_at_ns": r["available_at_ns"], "reward": r[m]})
    export_states = []
    for i, s in enumerate(states):
        export_states.append({
            "episode_id": s["episode_id"], "state_index": i, "available_at_ns": s["available_at_ns"],
            "evidence_tier": s.get("evidence_tier"), "psi": v[i]["psi"], "task": s["task"],
            "success": s.get("success"), "objects": {k: {"valid": o.get("valid"), "phase": o.get("phase"),
                                                         "held": o.get("held")} for k,o in s["objects"].items()},
            "events": s.get("events") or [],
            "provenance": s.get("provenance"),
        })
    return ledger, export_states, d, v
