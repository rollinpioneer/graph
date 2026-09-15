"""Macro windows from controller checkpoints, not reward-argmax boundaries."""
from __future__ import annotations
import json, math, sys
from pathlib import Path


def _pkg_loop():
    # local copy of closure comparison using packaged helper when available
    pass


def windows_for_episode(states: list[dict], checkpoints: list[dict], details: list[dict], values: list[dict]) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "unused"))
    from pathlib import Path as P
    # Import packaged loop_audit.closure by caller injecting it.
    raise RuntimeError("use build_windows()")


def build_windows(states, checkpoints, details, values, closure_fn, opportunities_fn) -> dict:
    loss_local = opportunities_fn(states, details, values)
    macros = []
    # Pair successive checkpoints named A* as start, next return attempt as end if recorded
    starts = [c for c in checkpoints if str(c.get("name","")).startswith("A")]
    for i, ck in enumerate(starts):
        sidx = int(ck["state_index"])
        # find first later state that matches closure vs checkpoint state
        end = None
        stat = None
        # search after this checkpoint, skipping the immediate neighborhood of 100ms (~5 frames)
        for j in range(sidx+6, len(states)):
            c = closure_fn(states[sidx], states[j])
            if c["status"] in ("EXACT_OBSERVED_TASK_RETURN", "BOUNDED_OBSERVED_TASK_RETURN"):
                end = j
                stat = c
                break
        row = {
            "episode_id": states[0]["episode_id"],
            "interval_kind": "MACRO_CYCLE",
            "checkpoint": ck.get("name"),
            "start": sidx,
            "end": end,
            "closure_status": "NO_OBSERVED_RETURN" if end is None else stat["status"],
            "geometry_closure": stat,
            "qpos_restored": False,
        }
        if end is None:
            macros.append(row)
            continue
        by = {}
        for r in details:
            val = r.get("reward")
            if val in (None, ""):
                continue
            by.setdefault(r["method"], []).append(float(val))
        dpsi = values[end]["psi"] - values[sidx]["psi"]
        for m, rs in by.items():
            R = math.fsum(rs[sidx:end])
            macros.append(dict(row, method=m, signed_return=R,
                               positive_weight_sum=math.fsum(max(0.0, r) for r in rs[sidx:end]),
                               candidate_endpoint_delta=dpsi,
                               interval_convention="states[s,e], transitions[s,e)",
                               reward_state_reset=False))
    return {"macro": macros, "loss_local": loss_local}
