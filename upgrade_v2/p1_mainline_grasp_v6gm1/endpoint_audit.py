"""Opportunity compaction and independent raw endpoint audit for V6GM1."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

RETURN_STATUSES = ("EXACT_OBSERVED_TASK_RETURN", "BOUNDED_OBSERVED_TASK_RETURN")
POS_TOL = 0.001
VEL_TOL = 0.01


def _tf(v) -> str:
    if v is True or v == "true" or v == "True":
        return "true"
    if v is False or v == "false" or v == "False" or v in (None, ""):
        return "false"
    return "true" if v else "false"


def _present(v) -> bool:
    return v not in (None, "", "None", "null")


def compact_opportunities(raw_rows: list[dict]) -> list[dict]:
    uniq = {}
    for o in raw_rows:
        eid = str(o.get("episode_id") or "")
        oid = str(o.get("object_id") or "obj")
        oppid = str(o.get("loss_id") or o.get("opportunity_id") or "")
        if not eid or not oppid:
            continue
        key = (eid, oid, oppid)
        hold = _present(o.get("restoration"))
        closure = o.get("closure_status") or "NO_OBSERVED_RETURN"
        ret = closure in RETURN_STATUSES
        if ret and not hold:
            ret = False
        rec = {
            "episode_id": eid,
            "object_id": oid,
            "opportunity_id": oppid,
            "reference_valid": "true",
            "loss_observed": "true",
            "regrasp_confirmed": "true" if hold else "false",
            "return_verified": "true" if ret else "false",
            "attempts": "1",
            "closure_status": closure,
        }
        prev = uniq.get(key)
        if prev is None or (ret and prev.get("return_verified") != "true"):
            uniq[key] = rec
    return list(uniq.values())


def _dist(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _object_prefix(keys, oid: str):
    keys = set(keys)
    for pref in (oid, "obj"):
        if f"{pref}_x" in keys and f"{pref}_y" in keys and f"{pref}_z" in keys:
            return pref
    return None


def _load_timeseries(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _measure_raw(rows: list[dict], start, end, object_id: str) -> dict:
    out = {
        "independent_endpoint_verified": "false",
        "independent_endpoint_status": "NOT_EVALUATED_NO_RETURN_BOUNDARY",
        "independent_reason": "",
        "independent_eef_pos_delta": "",
        "independent_obj_pos_delta": "",
        "independent_obj_vel_delta": "",
        "independent_orientation_observed": "false",
    }
    if not _present(start) or not _present(end):
        return out
    try:
        s = int(start)
        e = int(end)
    except (TypeError, ValueError):
        out["independent_endpoint_status"] = "UNRESOLVED"
        out["independent_reason"] = "non_integer_boundary"
        return out
    if s < 0 or e < 0 or s >= len(rows) or e >= len(rows):
        out["independent_endpoint_status"] = "UNRESOLVED"
        out["independent_reason"] = "index_out_of_range"
        return out
    a, b = rows[s], rows[e]
    try:
        eef0 = (a["eef_x"], a["eef_y"], a["eef_z"])
        eef1 = (b["eef_x"], b["eef_y"], b["eef_z"])
        eef_d = _dist(eef0, eef1)
        pref = _object_prefix(a.keys(), object_id)
        if pref is None:
            out["independent_endpoint_status"] = "UNRESOLVED"
            out["independent_reason"] = "missing_object_columns"
            return out
        pos0 = (a[f"{pref}_x"], a[f"{pref}_y"], a[f"{pref}_z"])
        pos1 = (b[f"{pref}_x"], b[f"{pref}_y"], b[f"{pref}_z"])
        vel0 = (a[f"{pref}_vx"], a[f"{pref}_vy"], a[f"{pref}_vz"])
        vel1 = (b[f"{pref}_vx"], b[f"{pref}_vy"], b[f"{pref}_vz"])
        pos_d = _dist(pos0, pos1)
        vel_d = _dist(vel0, vel1)
    except KeyError as err:
        out["independent_endpoint_status"] = "UNRESOLVED"
        out["independent_reason"] = f"missing_column:{err}"
        return out
    out.update({
        "independent_endpoint_verified": "true",
        "independent_eef_pos_delta": f"{eef_d:.12g}",
        "independent_obj_pos_delta": f"{pos_d:.12g}",
        "independent_obj_vel_delta": f"{vel_d:.12g}",
        "independent_orientation_observed": "false",
    })
    if eef_d == 0.0 and pos_d == 0.0 and vel_d == 0.0:
        out["independent_endpoint_status"] = "EXACT_OBSERVED_TASK_RETURN"
    elif eef_d <= POS_TOL and pos_d <= POS_TOL and vel_d <= VEL_TOL:
        out["independent_endpoint_status"] = "BOUNDED_OBSERVED_TASK_RETURN"
    else:
        out["independent_endpoint_status"] = "NOT_CLOSED"
        out["independent_reason"] = "raw_pose_or_velocity_outside_declared_tolerance"
    return out


def _parse_geometry(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def build_endpoint_audit(raw_rows: list[dict], inventory: list[dict]) -> list[dict]:
    by_ep = {r["episode_id"]: r for r in inventory}
    grouped = {}
    for o in raw_rows:
        eid = str(o.get("episode_id") or "")
        oid = str(o.get("object_id") or "obj")
        oppid = str(o.get("loss_id") or o.get("opportunity_id") or "")
        if not eid or not oppid:
            continue
        key = (eid, oid, oppid)
        grouped.setdefault(key, []).append(o)

    ts_cache = {}
    out = []
    for key, rows in grouped.items():
        eid, oid, oppid = key
        base = rows[0]
        v6 = next((r for r in rows if r.get("method") == "V6_CAP_POTENTIAL"), None)
        src = v6 or base
        closure = src.get("closure_status") or "NO_OBSERVED_RETURN"
        geom = _parse_geometry(src.get("geometry_closure"))
        errors = geom.get("errors") or {}
        inv = by_ep.get(eid) or {}
        raw_path = inv.get("raw") or inv.get("raw_root") or ""
        measured = {
            "independent_endpoint_verified": "false",
            "independent_endpoint_status": "NOT_EVALUATED_NO_RETURN_BOUNDARY",
            "independent_reason": "no_raw_path" if not raw_path else "",
            "independent_eef_pos_delta": "",
            "independent_obj_pos_delta": "",
            "independent_obj_vel_delta": "",
            "independent_orientation_observed": "false",
        }
        if raw_path and _present(src.get("end")):
            p = Path(raw_path) / "timeseries.csv"
            if not p.is_file() and Path(raw_path).is_file():
                p = Path(raw_path)
            if p.is_file():
                if str(p) not in ts_cache:
                    ts_cache[str(p)] = _load_timeseries(p)
                measured = _measure_raw(ts_cache[str(p)], src.get("start"), src.get("end"), oid)
            else:
                measured["independent_endpoint_status"] = "UNRESOLVED"
                measured["independent_reason"] = "timeseries_missing"
        rec = {
            "episode_id": eid,
            "object_id": oid,
            "opportunity_id": oppid,
            "onset": src.get("onset") or "",
            "restoration": src.get("restoration") or "",
            "start": src.get("start") or "",
            "end": src.get("end") or "",
            "closure_status": closure,
            "legacy_return_reported": "true" if closure in RETURN_STATUSES else "false",
            "complete_physical_closure_certified": "false",
            "physical_cycle_claim": "UNRESOLVED",
            "loop_audit_eef_pos_delta": errors.get("eef_position", ""),
            "loop_audit_obj_pos_delta": errors.get(f"{oid}_pos") or errors.get("obj_pos") or "",
            "loop_audit_obj_vel_delta": errors.get(f"{oid}_vel") or errors.get("obj_vel") or "",
            "loop_audit_orientation_observed": "true" if geom.get("orientation_observed") else "false",
            "candidate_endpoint_delta": src.get("candidate_endpoint_delta") or "",
            "legacy_endpoint_delta": src.get("legacy_endpoint_delta") or "",
            "residual_against_reported_reference": src.get("residual_against_reported_reference") or "",
            "v6_signed_return": src.get("signed_return") if v6 else "",
            "candidate_lipschitz_bound": src.get("candidate_lipschitz_bound") or "",
            "evidence_tier": inv.get("evidence_tier") or "",
            "raw": raw_path,
            "note": "Independent measurement uses raw timeseries pose/velocity only; quaternion/attachment are absent so this is not complete physical closure.",
        }
        rec.update(measured)
        out.append(rec)
    out.sort(key=lambda r: (r["episode_id"], r["object_id"], r["opportunity_id"]))
    return out