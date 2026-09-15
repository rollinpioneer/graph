from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
from .util import require_new, write_csv, write_json
from .return_geometry import relative_position_in_eef, norm

CASES = {"C2_LOSS_RETURN_P40","C3_LOSS_RETURN_P80","C4_THREE_RETURNS_P40"}

def _load_jsonl(p):
    if not Path(p).is_file():
        return []
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def _n(a,b):
    return math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))

def audit_episode(dest: Path) -> dict:
    ident = json.loads((dest/"identity.json").read_text(encoding="utf-8"))
    raw = _load_jsonl(dest/"raw_state.jsonl")
    ckpts = _load_jsonl(dest/"checkpoint_events.jsonl")
    man = json.loads((dest/"manifest.json").read_text(encoding="utf-8")) if (dest/"manifest.json").is_file() else {}
    events = []
    for r in raw:
        events.extend(e.get("kind") for e in r.get("events") or [])
    loss = "LOSS" in events
    rehold = "HOLD_REESTABLISHED" in events
    ck = ckpts[0] if ckpts else None
    last = raw[-1] if raw else {}
    eef_err = obj_err = rel_shift = None
    taxonomy = "F9_UNRESOLVED"
    if not loss:
        taxonomy = "F0_NO_LOSS"
    elif not rehold:
        taxonomy = "F1_REGRASP_FAILED"
    elif ck and last:
        o = last["objects"]["obj"]
        # reconstruct checkpoint object/eef from saved fields
        p_obj_star = ck.get("object_pos") or ck.get("p_WO_star")
        p_eef_star = ck.get("eef") or ck.get("p_WE_star")
        if p_obj_star and p_eef_star:
            obj_err = _n(o["pos"], p_obj_star)
            eef_err = _n(last["eef"], p_eef_star)
            r0 = relative_position_in_eef(p_obj_star, p_eef_star, last.get("eef_quat") or [1,0,0,0])
            r1 = relative_position_in_eef(o["pos"], last["eef"], last.get("eef_quat") or [1,0,0,0])
            rel_shift = _n(r0, r1)
            if eef_err <= 0.012 and obj_err > 0.005:
                taxonomy = "F4_EEF_REACHED_OBJECT_NOT_RETURNED"
            elif eef_err > 0.012:
                taxonomy = "F3_EEF_TARGET_NOT_REACHED"
            elif obj_err <= 0.005 and float(norm(o["vel"])) > 0.01:
                taxonomy = "F5_OBJECT_RETURNED_NOT_SETTLED"
            else:
                taxonomy = "F8_CLOSURE_CONTRACT_MISMATCH"
    return {
        "episode_id": ident.get("episode_id"), "family_id": ident.get("family_id"),
        "case_id": ident.get("case_id"), "loss": loss, "rehold": rehold,
        "n_states": len(raw), "n_ckpt": len(ckpts),
        "eef_error_m": eef_err, "object_error_m": obj_err, "attachment_shift_m": rel_shift,
        "taxonomy": taxonomy, "pose_writes": man.get("direct_pose_overwrites_after_start"),
        "returns": man.get("returns"), "notes": ident.get("notes"), "raw": str(dest),
    }

def run_forensics(confirm_root: Path, eng_root: Path, out: Path) -> dict:
    out = require_new(out)
    rows = []
    for root in (Path(confirm_root), Path(eng_root)):
        for ident in Path(root).rglob("identity.json"):
            dest = ident.parent
            try:
                rec = json.loads(ident.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("case_id") not in CASES and not str(rec.get("case_id","")).startswith("C2") and not str(rec.get("case_id","")).startswith("C3") and not str(rec.get("case_id","")).startswith("C4"):
                continue
            if rec.get("case_id") not in CASES:
                continue
            rows.append(audit_episode(dest))
    write_csv(out/"episode_forensics.csv", rows)
    tax = defaultdict(int)
    for r in rows:
        tax[r["taxonomy"]] += 1
    write_csv(out/"failure_taxonomy.csv", [{"taxonomy": k, "n": v} for k,v in sorted(tax.items())])
    att = [r for r in rows if r.get("attachment_shift_m") is not None]
    write_csv(out/"attachment_transform_shift.csv", att)
    write_csv(out/"return_error_decomposition.csv", [{k: r[k] for k in ("episode_id","case_id","eef_error_m","object_error_m","attachment_shift_m","taxonomy")} for r in rows])
    summary = {
        "n": len(rows), "taxonomy": dict(tax),
        "eef_reached_object_not_returned": tax.get("F4_EEF_REACHED_OBJECT_NOT_RETURNED", 0),
        "median_eef_error": None, "median_object_error": None,
    }
    eefs = [r["eef_error_m"] for r in rows if r["eef_error_m"] is not None]
    objs = [r["object_error_m"] for r in rows if r["object_error_m"] is not None]
    if eefs:
        eefs.sort(); objs.sort()
        summary["median_eef_error"] = eefs[len(eefs)//2]
        summary["median_object_error"] = objs[len(objs)//2]
    write_json(out/"summary.json", summary)
    return summary
