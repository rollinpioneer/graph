"""Export collector raw to score-normalized states. New binding, old view as shadow."""
from __future__ import annotations
import json
from pathlib import Path
from .util import require_new, sha256_file, write_json, write_jsonl

OLD_COST = {"WAIT":4/6,"HELD":3/6,"TRANSPORT":2/6,"PLACED":1/6,"VALID":0.0,"LOST":5/6,"RECOVERING":4/6}
OLD_NODE = {"WAIT":"start","HELD":"grasped","TRANSPORT":"in_transit","PLACED":"placed","VALID":"success","LOST":"dropped_or_misaligned","RECOVERING":"recovery"}
EDGE = {"WAIT":"none","HELD":"forward","TRANSPORT":"forward","PLACED":"forward","VALID":"forward","LOST":"failure","RECOVERING":"recovery"}


def _legacy_phi(phase, dist, anchor):
    if phase != "TRANSPORT" or anchor is None or anchor <= 1e-9:
        return 0.0, anchor
    if anchor is None:
        return 0.0, dist if dist > 1e-9 else None
    return min(1.0, max(0.0, 1.0 - dist/anchor)), anchor


def episode_to_states(dest: Path) -> tuple[list[dict], dict]:
    ident = json.loads((dest/"identity.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (dest/"raw_state.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 2:
        raise ValueError(f"too few states {dest}")
    eid = ident["episode_id"]
    raw_sha = sha256_file(dest/"raw_state.jsonl")
    anchor = None
    out = []
    for i, r in enumerate(rows):
        o = r["objects"]["obj"]
        pos = o["pos"]; tgt = o["target_xy"]
        dist = ((pos[0]-tgt[0])**2 + (pos[1]-tgt[1])**2) ** 0.5
        phase = o["phase"]
        if phase == "TRANSPORT" and anchor is None and dist > 1e-9:
            anchor = dist
        lp, anchor = _legacy_phi(phase, dist, anchor)
        if i == 0:
            edge = "none"
        else:
            edge = EDGE.get(phase, "none")
            prev = rows[i-1]["objects"]["obj"]["phase"]
            if prev == phase:
                edge = "none"
        st = {
            "episode_id": eid,
            "state_index": i,
            "task": "recovery",
            "available_at_ns": int(r["physical_time_ns"]),
            "physical_time_ns": int(r["physical_time_ns"]),
            "capture_order": int(r["capture_order"]),
            "success": bool(o["valid"]),
            "terminal_failure": False,
            "gripper_closed": bool(r["gripper_closed"]),
            "eef": r["eef"],
            "events": r.get("events") or [],
            "objects": {
                "obj": {
                    "held": bool(o["held"]),
                    "valid": bool(o["valid"]),
                    "phase": phase,
                    "pos": list(pos),
                    "vel": list(o["vel"]),
                    "target_xy": list(tgt),
                    "quat": list(o.get("quat") or [1,0,0,0]),
                    "angular_vel": list(o.get("angular_vel") or [0,0,0]),
                }
            },
            "legacy": {"node": OLD_NODE[phase], "cost": OLD_COST[phase], "phi": lp},
            "legacy_edge_type": edge,
            "legacy_binding_status": "NEW_PHYSICS_SHADOW_LEGACY_VIEW_NOT_ORIGINAL_V5_GRAPH",
            "evidence_tier": ident.get("evidence_tier", "MUJOCO_CONSTRAINT_ASSISTED_STATE_VERIFICATION"),
            "provenance": {
                "source_file": str(dest/"raw_state.jsonl"),
                "source_row": i,
                "source_sha256": raw_sha,
                "family_id": ident.get("family_id"),
                "case_id": ident.get("case_id"),
                "orientation_available": True,
            },
        }
        out.append(st)
    meta = {"episode_id": eid, "n": len(out), "raw": str(dest), "identity": ident}
    return out, meta


def export_raw_tree(raw_root: Path, out: Path) -> dict:
    out = require_new(out)
    dests = sorted({p.parent for p in Path(raw_root).rglob("raw_state.jsonl")})
    all_states = []
    metas = []
    bad = []
    for d in dests:
        try:
            seq, meta = episode_to_states(d)
            all_states.extend(seq)
            metas.append(meta)
        except Exception as exc:
            bad.append({"path": str(d), "error": type(exc).__name__, "reason": str(exc)})
    write_jsonl(out/"states.jsonl", all_states)
    write_json(out/"manifest.json", {"episodes": len(metas), "states": len(all_states), "failures": bad, "meta": metas})
    write_json(out/"export_failures.json", bad)
    return {"episodes": len(metas), "states": len(all_states), "failures": len(bad), "out": str(out)}
