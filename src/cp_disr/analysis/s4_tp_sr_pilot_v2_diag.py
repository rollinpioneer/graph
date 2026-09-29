"""Wave D classification and technical gate for the T_P_SR V2 pilot. Reads saved captures only; never constructs an environment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cp_disr.analysis.s1_integration import _atomic_json, sha256_file
from cp_disr.analysis.s4_tp_sr_pilot_v2 import (CODE_FILES, StopRun, INTERFERENCE_PHASES, RESOLVED_LABELS, VERIFIER_FACT_IDS, WAVES, load_config, mark_phase, rd, write_csv)

REQUIRED_ACTION_FILES = ("before_rgb.png", "before_depth.npy", "after_rgb.png", "after_depth.npy", "perception.json", "facts.json", "controller_trace.jsonl",
                         "evaluator.json", "qa_state.jsonl")
NORMAL = {"NORMAL_TERMINATION", "SUCCESS"}
AMENDMENT_ID = "WAVE_D_D7_CHECKER_AMENDMENT_1"
D7_TERMINAL = ("COMPLETED", "FAILED", "UNKNOWN")


def read_jsonl(p):
    p = Path(p)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def load_branch(out, wave, b):
    out = Path(out)
    phys = out / WAVES[wave]["dir"]
    bid = b["branch_id"]
    cap = out / "captures" / bid
    br = {"branch": b, "result": read_json(phys / "branch_results" / f"{bid}.json", {}), "journal": read_jsonl(phys / "witnesses" / f"branch_{bid}.jsonl"),
          "capture_dir": cap, "actions": [], "summary": read_json(cap / "branch_recorder_summary.json", {})}
    for d in sorted(cap.glob("action_*")):
        a = {"dir": d, "name": d.name,
             "files": {n: (d / n).is_file() and (d / n).stat().st_size > 0 for n in REQUIRED_ACTION_FILES},
             "facts": read_json(d / "facts.json"), "planner": read_json(d / "planner.json"), "evaluator": read_json(d / "evaluator.json"),
             "snapshot": read_json(d / "snapshot.json"), "trace": read_jsonl(d / "controller_trace.jsonl"), "contacts": read_jsonl(d / "contacts.jsonl"),
             "qa": read_jsonl(d / "qa_state.jsonl"), "perception": read_json(d / "perception.json", [])}
        begin = next((e for e in a["trace"] if e.get("event") == "skill_begin"), None)
        end = next((e for e in a["trace"] if e.get("event") == "skill_end"), None)
        a["candidate_id"] = (begin or {}).get("candidate_id")
        a["controller_exit"] = (end or {}).get("controller_exit")
        a["reported_steps"] = (end or {}).get("steps")
        a["step_records"] = sum(1 for e in a["trace"] if e.get("kind") == "step")
        br["actions"].append(a)
    return br


def fmap(facts):
    return {r["fact_id"]: r["value"] for r in (facts or [])}


def qa_tag(a, tag):
    return next((r for r in reversed(a["qa"]) if r.get("tag") == tag), None)


def hidden_held(qa, obj):
    if not qa:
        return None
    h = qa["hidden_truth"]
    p, eef, grip, table = h[obj], h["eef_pos"], h["gripper_qpos"], float(h["table_top_z"])
    lifted = p[2] > table + 0.02 + 0.04
    near = math.hypot(p[0] - eef[0], p[1] - eef[1]) < 0.05
    closed = abs(grip[0]) <= 0.026
    return bool(lifted and near and closed)


def hidden_inside(qa, obj="target"):
    if not qa:
        return None
    h = qa["hidden_truth"]
    p, c = h[obj], h["container"]
    return bool(abs(p[0] - c[0]) <= 0.030 and abs(p[1] - c[1]) <= 0.030 and p[2] <= float(h["table_top_z"]) + 0.12)


def other_contacts(a, other):
    """Finger/palm/target-vs-`other` contacts inside the sweep phases (in contact: distance <= 1 mm)."""
    ev = []
    for rec in a["contacts"]:
        if rec.get("phase") not in INTERFERENCE_PHASES:
            continue
        for p in rec.get("pairs", []):
            cats = {p["cat1"], p["cat2"]}
            partner = {"finger", "palm"} | ({"target"} if other == "interferer" else {"interferer"})
            if other in cats and cats & partner and p["distance"] <= 1e-3:
                ev.append({"step": rec["step"], "phase": rec["phase"], "sim_time": rec.get("sim_time"), "pair": [p["geom1"], p["geom2"]], "cats": sorted(cats),
                           "distance": p["distance"], "normal_force": p["normal_force"]})
    return ev


def precondition_true(contracts, cid, facts):
    from cp_disr.contracts import precondition_value
    from cp_disr.facts import Truth
    c = next((x for x in contracts if x.id == cid), None)
    if c is None:
        return None
    vals = {k: Truth(v) for k, v in facts.items()}
    return precondition_value(c, vals) == Truth.TRUE


def evidence(a, br):
    return {"source_frame": (a["perception"][-1]["frame_id"] if a["perception"] else None), "capture_dir": str(a["dir"]), "fact_records": "facts.json",
            "controller_event": "controller_trace.jsonl", "contact_event": "contacts.jsonl", "planner_input": "planner.json", "evaluator_record": "evaluator.json"}


def classify_branch(br, contracts):
    res = br["result"]
    if bool(res.get("task_success")):
        return {"label": "NOT_APPLICABLE_SUCCESS", "reason": "branch succeeded"}
    if not br["actions"]:
        return {"label": "UNRESOLVED", "reason": "no executed action captured"}
    a = br["actions"][-1]
    parts = (a["candidate_id"] or "").split(":")
    skill, obj = (parts[1] if len(parts) > 2 else ""), (parts[2] if len(parts) > 2 else "")
    facts = fmap(a["facts"])
    planner = a["planner"] or {}
    status = planner.get("status") or res.get("termination_reason")
    qa = qa_tag(a, "after")
    ev = evidence(a, br)
    if res.get("termination_reason") == "SYMBOLIC_EVALUATOR_MISMATCH" or planner.get("status") == "GOAL_ALREADY_SATISFIED":
        pub_inside, hid_inside = facts.get("p:Inside:target:container"), hidden_inside(qa)
        if pub_inside == "TRUE" and hid_inside is False:
            label = "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS"
        elif pub_inside == "TRUE" and hid_inside is True:
            label = "EVALUATOR_SYMBOLIC_DISAGREEMENT"
        elif pub_inside != "TRUE" and hid_inside is not None:
            label = "PLANNER_FACT_INPUT_ERROR"
        else:
            label = "UNRESOLVED"
        return {"label": label, "kind": "symbolic_goal", "checks": {"public_inside_target": pub_inside, "hidden_inside_target": hid_inside, "planner_status": planner.get("status"),
                                                                    "evaluator": a["evaluator"]}, "evidence": ev}
    nxt = {"target": "a:PLACE:target:container:v1", "interferer": "a:PLACE_BUFFER:interferer:buffer:v1"}.get(obj)
    other = "interferer" if obj == "target" else "target"
    held_h, held_p = hidden_held(qa, obj), facts.get(f"p:Held:{obj}")
    contacts = other_contacts(a, other)
    pre_true = precondition_true(contracts, nxt, {k: facts[k] for k in facts}) if nxt else None
    snap_mask = dict(zip((a["snapshot"] or {}).get("candidate_ids", []), (a["snapshot"] or {}).get("candidate_mask", [])))
    exit_ok = a["controller_exit"] in NORMAL
    chk = {"candidate": a["candidate_id"], "controller_exit": a["controller_exit"], "hidden_held": held_h, "public_held": held_p, "interferer_contact_events": len(contacts),
           "next_action": nxt, "next_action_precondition_true_on_public_facts": pre_true, "next_action_mask": snap_mask.get(nxt), "planner_status": planner.get("status"),
           "hidden_available": qa is not None}
    if qa is None or not a["facts"]:
        return {"label": "UNRESOLVED", "checks": chk, "reason": "missing hidden QA or facts", "evidence": ev}
    stopped = planner.get("status") == "NO_PLAN" or not exit_ok
    if not stopped:
        return {"label": "UNRESOLVED", "checks": chk, "reason": "branch failed without planner stop or abnormal controller exit", "evidence": ev}
    if (not held_h) and contacts and held_p != "TRUE" and (pre_true is False) and exit_ok:
        label = "PHYSICAL_GRASP_INTERFERENCE"
    elif (held_h and held_p != "TRUE") or ((not held_h) and held_p == "TRUE"):
        label = "PERCEPTION_VERIFIER_MISMATCH"
    elif pre_true and snap_mask.get(nxt) is True and planner.get("status") == "NO_PLAN":
        label = "PLANNER_FACT_INPUT_ERROR"
    elif (not held_h) and not contacts:
        label = "CONTROLLER_EXECUTION_FAILURE"
    else:
        label = "UNRESOLVED"
    return {"label": label, "checks": chk, "interferer_contacts": contacts[:10], "evidence": ev}


# ------------------------------------------------------------------ offline perception comparison (same saved frame, never online)
class _StubEnv:
    def __init__(self, layout, role):
        self._layout, self.second_role = layout, role

    def public_layout(self):
        return {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in self._layout.items()}


def compare_variants_on_frame(rgb, depth, calib, layout, eef, grip, role="interferer"):
    from cp_disr.platforms.libero import perception as P
    from cp_disr.platforms.libero.tp_sr_instrumentation import blobs_from_frame, nearest_palette_mask
    from cp_disr.platforms.libero.verifier import FactVerifier

    class M:
        pass

    outs = {}
    for name, fn in (("ORIGINAL_TOLERANCE_MASK", P._mask), ("NEAREST_PALETTE_MASK", nearest_palette_mask)):
        blobs, _t, reasons = blobs_from_frame(rgb, depth, calib, {k: np.array(v, dtype=float) if isinstance(v, list) else v for k, v in layout.items()}, fn)
        m = M()
        m.measurements = {"blobs": blobs, "eef_pos": eef, "gripper_qpos": grip, "unknown_reasons": reasons}
        recs = FactVerifier(_StubEnv(layout, role)).verify(m, None)
        outs[name] = {"blobs": {k: (None if v is None else {"xyz": v["xyz"], "pixels": v["pixels"]}) for k, v in blobs.items()},
                      "facts": {r.fact_id: str(getattr(r.value, "value", r.value)) for r in recs}}
    diffs = sorted(k for k in outs["ORIGINAL_TOLERANCE_MASK"]["facts"] if outs["ORIGINAL_TOLERANCE_MASK"]["facts"][k] != outs["NEAREST_PALETTE_MASK"]["facts"][k])
    return {**outs, "fact_differences": diffs}


def perception_comparison(out, branches):
    from PIL import Image
    rows = []
    for br in branches:
        for a in br["actions"]:
            for fr in a["perception"] or []:
                fid = fr["frame_id"]
                try:
                    rgb = np.array(Image.open(a["dir"] / f"{fid}_rgb.png"))
                    depth = np.load(a["dir"] / f"{fid}_depth_metric.npy")
                    calib = fr["camera_identity"]["calib"]
                    cmp_ = compare_variants_on_frame(rgb, depth, calib, fr["layout"], fr.get("eef_pos"), fr.get("gripper_qpos"))
                    online = fr["blob_xyz"]
                    rec_near = {k: (v or {}).get("xyz") for k, v in cmp_["NEAREST_PALETTE_MASK"]["blobs"].items()}
                    reproduces = all((online.get(k) is None and rec_near.get(k) is None) or (online.get(k) is not None and rec_near.get(k) is not None and
                                     np.allclose(online[k], rec_near[k], atol=1e-6)) for k in online)
                    rows.append({"branch_id": br["branch"]["branch_id"], "action": a["name"], "frame_id": fid, "offline_reproduces_online_local_variant": bool(reproduces),
                                 "fact_differences_original_vs_nearest": cmp_["fact_differences"], "original_facts": cmp_["ORIGINAL_TOLERANCE_MASK"]["facts"],
                                 "nearest_facts": cmp_["NEAREST_PALETTE_MASK"]["facts"]})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"branch_id": br["branch"]["branch_id"], "action": a["name"], "frame_id": fid, "error": f"{type(exc).__name__}: {exc}"})
    doc = {"note": "offline comparison on saved frames only; it never influenced the online branches", "frames": rows,
           "frames_with_fact_differences": sum(1 for r in rows if r.get("fact_differences_original_vs_nearest")),
           "frames_not_reproducing_online": sum(1 for r in rows if r.get("offline_reproduces_online_local_variant") is False)}
    _atomic_json(Path(out) / "diagnostics/perception_variant_comparison.json", doc)
    return doc


# ------------------------------------------------------------------ Wave D gate
def paired_initial_state(out, phys, branches, tol=1e-9):
    res = {}
    by_case = {}
    for b in branches:
        by_case.setdefault(b["case_id"], []).append(b)
    for case, bs in by_case.items():
        chk = [read_json(Path(phys) / "witnesses/initial_state_checks" / f"{b['branch_id']}.json") for b in bs]
        if len(chk) != 2 or any(c is None or c.get("status") != "OK" for c in chk):
            res[case] = {"ok": False, "reason": "missing initial_state_checks"}
            continue
        keys = chk[0]["compare_key"] == chk[1]["compare_key"]
        dq = float(np.max(np.abs(np.array(chk[0]["physical"]["qpos"]) - np.array(chk[1]["physical"]["qpos"]))))
        dv = float(np.max(np.abs(np.array(chk[0]["physical"]["qvel"]) - np.array(chk[1]["physical"]["qvel"]))))
        res[case] = {"ok": bool(keys and dq <= tol and dv <= tol), "compare_key_equal": keys, "qpos_max_abs_diff": dq, "qvel_max_abs_diff": dv,
                     "candidate_mask_equal": chk[0]["candidate_mask"] == chk[1]["candidate_mask"]}
    return res


def classify_diagnostics(root, config_path, out, amended=False, extra_gates=None):
    root, out = Path(root), Path(out)
    if not amended and (out / "diagnostics/wave_d_gate.json").exists():
        raise StopRun("STOPPED_ORIGINAL_GATE_IS_IMMUTABLE", "diagnostics/wave_d_gate.json exists; use the amended classification path")
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    import yaml
    cfg = load_config(config_path)
    phys = out / "diagnostics"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    manifest = yaml.safe_load(Path(reg["branches"][0]["manifest_path"]).read_text())
    rtm = manifest["runtime"]
    contracts = ground_task_contracts(Path(rtm["repository_path"]) / rtm["stage_2a_contract_path"], rtm["skill_timeouts"]["T_P_SR"])
    branches = [load_branch(out, "D", b) for b in reg["branches"]]
    attempts = rd(phys / "attempt_registry.json") if (phys / "attempt_registry.json").is_file() else {}
    rows, roots = [], {}
    for br in branches:
        b, r = br["branch"], br["result"]
        cls = classify_branch(br, contracts)
        roots[f"{b['case_id']}|{b['route']}"] = {"branch_id": b["branch_id"], **cls}
        rows.append({"branch_id": b["branch_id"], "case_id": b["case_id"], "route": b["route"], "attempt_state": (attempts.get(b["branch_id"]) or {}).get("state") if isinstance(attempts.get(b["branch_id"]), dict) else attempts.get(b["branch_id"]),
                     "execution_status": r.get("execution_status"), "termination_reason": r.get("termination_reason"), "task_success": r.get("task_success"),
                     "actions_captured": len(br["actions"]), "recorder_errors": r.get("recorder_errors"), "root_cause": cls["label"],
                     "env_reset_calls": (r.get("env_counts") or {}).get("reset_calls"), "worker_wall_seconds": r.get("worker_wall_seconds")})
    write_csv(phys / "branch_results.csv", rows)
    _atomic_json(phys / "root_cause_classification.json", {"labels_allowed": ["PHYSICAL_GRASP_INTERFERENCE", "CONTROLLER_EXECUTION_FAILURE", "PERCEPTION_VERIFIER_MISMATCH", "PLANNER_FACT_INPUT_ERROR",
                                                                                "UNRESOLVED", "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS", "EVALUATOR_SYMBOLIC_DISAGREEMENT"],
                                                            "rule": "NO_PLAN alone never determines a label; physical interference requires contact and state evidence", "branches": roots})
    comp = perception_comparison(out, branches)
    gate = wave_d_gate(out, cfg, reg, branches, roots, comp, amended=amended, extra_gates=extra_gates)
    mark_phase(out, "classify-diagnostics-amended" if amended else "classify-diagnostics", wave_d_status=gate["wave_d_status"])
    return gate


def wave_d_gate(out, cfg, reg, branches, roots, comp, amended=False, extra_gates=None):
    out = Path(out)
    phys = out / "diagnostics"
    led = rd(out / "budget_ledger.json")
    attempts = rd(phys / "attempt_registry.json") if (phys / "attempt_registry.json").is_file() else {}
    g = {}
    term = [br["result"].get("execution_status") for br in branches]
    g["D1"] = {"pass": len(branches) == 4 and all(t for t in term) and len(attempts) == 4, "detail": term}
    pair = paired_initial_state(out, phys, reg["branches"])
    g["D2"] = {"pass": bool(pair) and all(v["ok"] for v in pair.values()), "detail": pair}
    acts = [(br["branch"]["branch_id"], a) for br in branches for a in br["actions"]]
    g["D3"] = {"pass": bool(acts) and all(all(a["files"][n] for n in ("before_rgb.png", "before_depth.npy", "after_rgb.png", "after_depth.npy")) for _, a in acts),
               "actions": len(acts)}
    def facts_ok(a):
        f = a["facts"] or []
        return {r["fact_id"] for r in f} == set(VERIFIER_FACT_IDS) and all(k in r for r in f for k in ("value", "reason", "evidence_ids", "capture_time", "available_time",
                                                                                                        "last_confirmed_value", "last_confirmed_time"))
    g["D4"] = {"pass": bool(acts) and all(facts_ok(a) for _, a in acts)}
    g["D5"] = {"pass": bool(acts) and all(a["files"]["controller_trace.jsonl"] and a["step_records"] > 0 and a["step_records"] == a["reported_steps"] and
                                          any(e.get("event") == "phase_begin" for e in a["trace"]) for _, a in acts),
               "detail": [(a["name"], a["step_records"], a["reported_steps"]) for _, a in acts]}
    def planner_ok(br, a):
        ev = a["evaluator"] or {}
        if ev.get("terminated") or ev.get("truncated"):
            return True
        p = a["planner"]
        return bool(p) and {k for k in p["input_facts"]} == set(VERIFIER_FACT_IDS) and "status" in p and bool(p["candidate_ids"])
    g["D6"] = {"pass": bool(acts) and all(planner_ok(br, a) for br in branches for a in br["actions"])}
    chain = d7_identity_chain(out)
    used = {k: led[k]["used"] for k in ("wave_d_branch_attempts", "total_branch_attempts", "environment_resets", "standalone_capture_resets")}
    g["D7"] = {"pass": chain["pass"] and used == {"wave_d_branch_attempts": 4, "total_branch_attempts": 4, "environment_resets": 4, "standalone_capture_resets": 0},
               "check": "immutable identity and state chain (WAVE_D_D7_CHECKER_AMENDMENT_1)", "problems": chain["problems"], "per_branch": chain["per_branch"], "ledger": used,
               "pre_terminal_receipt_bytes": "PRE_TERMINAL_RECEIPT_BYTES_NOT_RETAINED"}
    r33, r57 = roots.get("T_P_SR_pool_33|direct", {}), roots.get("T_P_SR_pool_57|relocation", {})
    g["D8"] = {"pass": r33.get("label") in RESOLVED_LABELS, "label": r33.get("label")}
    g["D9"] = {"pass": r57.get("label") in RESOLVED_LABELS, "label": r57.get("label")}
    g["D10"] = {"pass": all(br["result"].get("global_perception_mask_unchanged") is True for br in branches), "detail": [br["result"].get("global_perception_mask_unchanged") for br in branches]}
    zero = ("provider_first_calls", "provider_retries", "representation_forwards", "rl_transitions", "optimizer_steps", "training_attempts", "elastic_attempts", "formal_test_episodes")
    g["D11"] = {"pass": all(led[k]["used"] == 0 for k in zero) and not any(br["result"].get("provider_module_loaded") for br in branches)}
    if extra_gates is not None:
        g.update(extra_gates())
    passed = all(v["pass"] for v in g.values())
    doc = {"wave_d_status": "PASS" if passed else "FAIL", "wave_p_released": bool(passed), "gates": g, "recorder_errors": [br["result"].get("recorder_errors") for br in branches],
           "perception_offline": {"frames_with_fact_differences": comp["frames_with_fact_differences"], "frames_not_reproducing_online": comp["frames_not_reproducing_online"]}}
    if not passed:
        doc.update({"status": "STOPPED", "next_action": "EXPLICIT_RESEARCH_DECISION"})
    if amended:
        doc.update({"amendment_id": AMENDMENT_ID, "original_gate_status": "FAIL", "original_failed_check": "D7",
                    "wave_d_status_detail": "PASS_AFTER_NONSCIENTIFIC_CHECKER_AMENDMENT" if passed else "FAIL_AFTER_AMENDED_CHECK"})
    _atomic_json(phys / ("wave_d_gate_amended.json" if amended else "wave_d_gate.json"), doc)
    return doc


# ------------------------------------------------------------------ D7 identity chain (WAVE_D_D7_CHECKER_AMENDMENT_1)
def d7_identity_chain(out):
    """Immutable identity and state-chain check for the wave-D attempts. Deliberately does NOT compare whole receipt files by hash: terminal completion
    legitimately rewrites the receipt (state, execution_status), so byte equality with the reservation-time receipt is not an identity check."""
    out = Path(out)
    phys = out / "diagnostics"
    reg_path = phys / "diagnostic_branch_registration.json"
    if not reg_path.is_file():
        reg_path = phys / "witnesses/e4_branch_registration.json"
    reg = read_json(reg_path) or {}
    reg_sha = sha256_file(reg_path) if reg_path.is_file() else ""
    rbs = reg.get("branches") or []
    events = [e for e in read_jsonl(phys / "budget_events.jsonl") if e.get("event_type") == "branch_attempt"]
    charges = [e for e in read_jsonl(out / "budget_events.jsonl") if e.get("event") == "charge"]
    registry = read_json(phys / "attempt_registry.json") or {}
    ids = [b.get("branch_id") for b in rbs]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("DUPLICATE_BRANCH_ID")
    if len(ids) != 4 or len(registry) != 4 or set(registry) != set(ids):
        problems.append("ATTEMPT_SET_NOT_EXACTLY_THE_FOUR_REGISTERED_BRANCHES")
    if any(e.get("branch_id") not in ids for e in events):
        problems.append("EVENT_FOR_UNREGISTERED_BRANCH")
    per, reset_by_case = {}, {}
    for b in rbs:
        bid = b.get("branch_id")
        pb = []
        receipt = read_json(phys / "witnesses/branch_receipts" / f"{bid}.json") or {}
        result = read_json(phys / "branch_results" / f"{bid}.json") or {}
        restore = read_json(phys / "witnesses/restore_receipts" / f"{bid}.json") or {}
        rr = restore.get("restore_receipt") or {}
        evs = [e for e in events if e.get("branch_id") == bid]
        seq = [e.get("status") for e in evs]
        # 1 branch identity
        if not receipt or receipt.get("branch_id") != bid:
            pb.append("RECEIPT_BRANCH_ID_MISMATCH")
        # 2 attempt / reservation identity across registration, receipt, events, registry, result
        if b.get("attempt_id") != bid or receipt.get("attempt_id") != bid:
            pb.append("ATTEMPT_ID_MISMATCH")
        if bid not in registry:
            pb.append("ATTEMPT_MISSING_FROM_REGISTRY")
        if result.get("branch_id") != bid:
            pb.append("RESULT_BRANCH_ID_MISMATCH")
        for k in ("attempt_id", "reservation_id"):
            if k in result and result[k] != (receipt.get(k)):
                pb.append(f"RESULT_{k.upper()}_MISMATCH")
        rids = {receipt.get("reservation_id")} | {e.get("reservation_id") for e in evs}
        if len(rids) != 1 or not next(iter(rids)):
            pb.append("RESERVATION_ID_MISMATCH")
        # 3 receipt binds the frozen registration file
        if not reg_sha or receipt.get("registration_sha256") != reg_sha:
            pb.append("REGISTRATION_SHA256_MISMATCH")
        # 4 exactly one RESERVED, one STARTED, one terminal event, in that order
        if len(seq) != 3 or seq[0] != "RESERVED" or seq[1] != "STARTED" or seq[2] not in D7_TERMINAL:
            pb.append("EVENT_SEQUENCE_NOT_RESERVED_STARTED_TERMINAL")
        term = evs[-1] if evs else {}
        # 5 terminal state agreement: receipt == registry == terminal event; execution status agrees with the branch result
        if not (receipt.get("state") == registry.get(bid) == term.get("status")) or receipt.get("state") not in D7_TERMINAL:
            pb.append("TERMINAL_STATE_DISAGREEMENT")
        if not (receipt.get("execution_status") == result.get("execution_status") == term.get("execution_status")):
            pb.append("EXECUTION_STATUS_DISAGREEMENT")
        if receipt.get("termination_reason") != term.get("termination_reason"):
            pb.append("TERMINATION_REASON_DISAGREEMENT")
        # 6 one attempt and one environment reset consumed
        mine = [c for c in charges if c.get("ref") == bid]
        for key in ("wave_d_branch_attempts", "total_branch_attempts", "environment_resets"):
            if sum(1 for c in mine if c.get("key") == key) != 1:
                pb.append(f"CHARGE_COUNT_NOT_1:{key}")
        if seq.count("RESERVED") != 1:
            pb.append("ATTEMPT_CONSUMED_NOT_EXACTLY_ONCE")
        if (result.get("env_counts") or {}).get("reset_calls") != 1:
            pb.append("RESET_CALLS_NOT_1")
        # 7 restore receipt identity against the registration
        if restore.get("branch_id") != bid or restore.get("case_id") != b.get("case_id"):
            pb.append("RESTORE_BRANCH_OR_CASE_MISMATCH")
        if not (rr.get("requested_case_id") == rr.get("applied_case_id") == b.get("case_id")):
            pb.append("RESTORE_CASE_ID_MISMATCH")
        if not (rr.get("requested_restore_seed") == rr.get("applied_restore_seed") == b.get("restore_seed")) or b.get("restore_seed") is None:
            pb.append("RESTORE_SEED_MISMATCH")
        cfg_id = rr.get("normalized_reset_config_sha256")
        if not cfg_id or not (rr.get("restore_checks") and all(rr["restore_checks"].values())):
            pb.append("RESTORE_RESET_CONFIG_IDENTITY_MISSING_OR_UNCHECKED")
        if "normalized_reset_config_sha256" in b and b["normalized_reset_config_sha256"] != cfg_id:
            pb.append("RESTORE_RESET_CONFIG_MISMATCH_WITH_REGISTRATION")
        reset_by_case.setdefault(b.get("case_id"), set()).add(cfg_id)
        rp = phys / "witnesses/branch_receipts" / f"{bid}.json"
        per[bid] = {"problems": pb, "reservation_id": receipt.get("reservation_id"), "event_sequence": seq, "terminal_state": receipt.get("state"),
                    "receipt_hash_pinned_at_restore_time": restore.get("budget_receipt_sha256"), "current_receipt_sha256": sha256_file(rp) if rp.is_file() else None,
                    "pre_terminal_receipt_bytes": "PRE_TERMINAL_RECEIPT_BYTES_NOT_RETAINED",
                    "registration_reset_config_identity": "PRESENT" if "normalized_reset_config_sha256" in b else "NOT_IN_REGISTRATION (restore receipt identity checked for self-consistency and pairing)"}
        problems += [f"{bid}:{x}" for x in pb]
    for case, s in reset_by_case.items():
        if len(s) != 1:
            problems.append(f"{case}:PAIRED_ROUTES_HAVE_DIFFERENT_RESET_CONFIG_IDENTITY")
    return {"pass": not problems, "problems": problems, "per_branch": per}


# ------------------------------------------------------------------ evidence manifest, amendment, summary, verify
def evidence_manifest(out):
    out = Path(out)
    files = {}

    def add(p):
        p = Path(p)
        if p.is_file() and not p.name.endswith(".lock"):
            files[p.relative_to(out).as_posix()] = sha256_file(p)

    for d in ("captures", "spec", "diagnostics/witnesses", "diagnostics/branch_results", "diagnostics/worker_logs"):
        for p in sorted((out / d).rglob("*")):
            add(p)
    for f in ("diagnostics/attempt_registry.json", "diagnostics/budget_events.jsonl", "diagnostics/budget_ledger.json", "diagnostics/diagnostic_branch_registration.json",
              "diagnostics/worker_events.jsonl", "diagnostics/throughput_report.json", "diagnostics/wave_d_gate.json", "budget_ledger.json", "budget_events.jsonl",
              "authorization.json", "source_identity.json", "scheduling_events.jsonl", "inventory/protected_before.json", "inventory/original_checker_source_identity.json"):
        add(out / f)
    return files


DERIVED = ("branch_results.csv", "root_cause_classification.json", "perception_variant_comparison.json")
ZERO_KEYS = ("provider_first_calls", "provider_retries", "representation_forwards", "rl_transitions", "optimizer_steps", "training_attempts", "elastic_attempts", "formal_test_episodes")


def _launches(out):
    return sum(1 for e in read_jsonl(Path(out) / "scheduling_events.jsonl") if e.get("event") == "branch_launch")


def amend_gate(root, config_path, out):
    root, out = Path(root), Path(out)
    phys = out / "diagnostics"
    am_path = phys / "gate_checker_amendment.json"
    if am_path.exists():
        raise StopRun("STOPPED_AMENDMENT_ALREADY_APPLIED", str(am_path))
    orig = phys / "wave_d_gate.json"
    og = read_json(orig) or {}
    failed = {k for k, v in (og.get("gates") or {}).items() if not v.get("pass")}
    if og.get("wave_d_status") != "FAIL" or failed != {"D7"}:
        raise StopRun("STOPPED_ORIGINAL_FAILURE_NOT_LIMITED_TO_D7", str(sorted(failed)))
    orig_sha = sha256_file(orig)
    ident = read_json(out / "inventory/original_checker_source_identity.json") or {}
    before = evidence_manifest(out)
    _atomic_json(out / "inventory/evidence_hashes_before_amendment.json", {"files": before, "count": len(before)})
    derived_before = {n: (sha256_file(phys / n) if (phys / n).is_file() else None) for n in DERIVED}
    ledger_before = rd(out / "budget_ledger.json")
    launches_before = _launches(out)
    tests = read_json(out / "inventory/amendment_test_receipt.json") or {}

    def extras():
        after = evidence_manifest(out)
        led = rd(out / "budget_ledger.json")
        diff = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        states = read_json(phys / "attempt_registry.json") or {}
        return {
            "X1_original_evidence_unchanged": {"pass": not diff, "files_hashed": len(before), "diff": diff[:20]},
            "X2_original_gate_file_unchanged": {"pass": sha256_file(orig) == orig_sha, "sha256": orig_sha},
            "X3_no_new_execution": {"pass": led == ledger_before and launches_before == _launches(out) == 4 and len(states) == 4 and all(v in D7_TERMINAL for v in states.values())
                                    and led["total_branch_attempts"]["used"] == 4 and all(led[k]["used"] == 0 for k in ZERO_KEYS + ("standalone_capture_resets",)),
                                    "launch_events": _launches(out)},
            "X4_tests": {"pass": bool(tests.get("targeted_all_passed")) and tests.get("new_full_suite_failures") == 0
                         and tests.get("full_suite_failure_classification") in ("NONE", "KNOWN_PREEXISTING_FAILURE"),
                         "receipt_present": bool(tests), "targeted_all_passed": tests.get("targeted_all_passed"), "new_full_suite_failures": tests.get("new_full_suite_failures"),
                         "full_suite_failure_classification": tests.get("full_suite_failure_classification")}}

    gate = classify_diagnostics(root, config_path, out, amended=True, extra_gates=extras)
    after = evidence_manifest(out)
    derived_after = {n: (sha256_file(phys / n) if (phys / n).is_file() else None) for n in DERIVED}
    led_after = rd(out / "budget_ledger.json")
    frozen = (read_json(out / "source_identity.json") or {}).get("files_sha256", {})
    code_now = {f: (sha256_file(root / f) if (root / f).is_file() else None) for f in frozen}
    checker_path = root / "src/cp_disr/analysis/s4_tp_sr_pilot_v2_diag.py"
    amended_sha = sha256_file(checker_path)
    chg = lambda prefix: any(before.get(k) != after.get(k) for k in set(before) | set(after) if k.startswith(prefix))
    manifest = {"amendment_id": AMENDMENT_ID, "original_checker_source_sha256": ident.get("original_checker_source_sha256"),
                "original_checker_derivation": ident.get("derivation"), "amended_checker_source_sha256": amended_sha, "original_gate_sha256": orig_sha,
                "original_gate_status": "FAIL", "original_failed_check": "D7", "original_failure_reason": "mutable budget receipt was compared by whole-file hash",
                "scientific_data_invalidated": False, "execution_completed_before_amendment": True, "branch_execution_code_changed": code_now != frozen,
                "runtime_data_changed": before != after, "branch_results_changed": chg("diagnostics/branch_results/") or chg("captures/"),
                "receipts_changed": chg("diagnostics/witnesses/"), "budgets_changed": led_after != ledger_before, "new_attempts": led_after["total_branch_attempts"]["used"] - ledger_before["total_branch_attempts"]["used"],
                "new_resets": led_after["environment_resets"]["used"] - ledger_before["environment_resets"]["used"],
                "derived_files_sha256_before": derived_before, "derived_files_sha256_after": derived_after, "derived_files_identical": derived_before == derived_after,
                "evidence_files_hashed": len(before), "amended_gate": "diagnostics/wave_d_gate_amended.json", "amended_gate_status": gate["wave_d_status"],
                "amended_gate_detail": gate.get("wave_d_status_detail"),
                "reason": "terminal completion legitimately mutates the receipt state, so whole-file hash equality is not a valid identity check"}
    _atomic_json(am_path, manifest)
    _atomic_json(out / "inventory/post_execution_checker_source_identity.json", {
        "branch_execution_source_identity": {"recorded_at": "freeze-spec (source_identity.json, never rewritten)", "files_sha256_frozen": frozen, "files_sha256_now": code_now,
                                             "all_unchanged": code_now == frozen},
        "post_execution_gate_checker": {"module": "src/cp_disr/analysis/s4_tp_sr_pilot_v2_diag.py", "original_sha256": ident.get("original_checker_source_sha256"),
                                        "amended_sha256": amended_sha, "amend_cli": "scripts/s4_tp_sr_pilot_v2_amend.py",
                                        "amend_cli_sha256": sha256_file(root / "scripts/s4_tp_sr_pilot_v2_amend.py"),
                                        "d7_test_sha256": sha256_file(root / "tests/test_tp_sr_v2_d7_amendment.py")},
        "note": "the post-execution checker identity is kept separate from branch execution identity; the spec-freeze identity was not rewritten"})
    rows = []
    for k in sorted(set(og["gates"]) | set(gate["gates"]), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 99, x)):
        o, a = og["gates"].get(k), gate["gates"].get(k)
        rows.append(f"| {k} | {o['pass'] if o else '-'} | {a['pass'] if a else '-'} |")
    md = ["# Wave D gate comparison (original vs amended)", "", f"Amendment: {AMENDMENT_ID} (non-scientific checker correction; no new attempt, reset, provider, RL or optimizer).", "",
          f"- original gate file: `diagnostics/wave_d_gate.json` sha256 `{orig_sha}` (byte-identical, read-only), status FAIL, failed check D7 only",
          "- original D7 failure: a mutable budget receipt was compared by whole-file hash; terminal completion legitimately rewrites the receipt",
          f"- amended gate: `diagnostics/wave_d_gate_amended.json`, status {gate['wave_d_status']} ({gate.get('wave_d_status_detail')}), wave_p_released {gate['wave_p_released']}",
          "- pre-terminal receipt bytes: PRE_TERMINAL_RECEIPT_BYTES_NOT_RETAINED (not a failure; no reconstruction)", "", "| gate | original pass | amended pass |", "|---|---|---|", *rows, "",
          "The checker amendment changed only the D7 criterion; D1-D6 and D8-D11 use unchanged code and inputs. X1-X4 are additional amended-gate conditions (evidence unchanged, original gate unchanged, no new execution, tests)."]
    (phys / "wave_d_gate_comparison.md").write_text("\n".join(md) + "\n")
    return gate


def summarize_wave_d(root, config_path, out):
    out = Path(out)
    phys = out / "diagnostics"
    g = read_json(phys / "wave_d_gate_amended.json") or {}
    rc = (read_json(phys / "root_cause_classification.json") or {}).get("branches", {})
    lines = ["# Wave D summary", "", f"- amended gate: {g.get('wave_d_status')} ({g.get('wave_d_status_detail')}), wave_p_released {g.get('wave_p_released')}", "- original gate: FAIL (D7 only), file preserved unchanged"]
    for k, v in rc.items():
        lines.append(f"- {k}: {v.get('label')} (branch {v.get('branch_id')})")
    (phys / "wave_d_summary.md").write_text("\n".join(lines) + "\n")
    return {"summary": str(phys / "wave_d_summary.md")}


def verify_wave_d(root, config_path, out):
    from cp_disr.analysis.s4_tp_sr_pilot_v2 import protected_inventory, protected_unchanged
    root, out = Path(root), Path(out)
    phys = out / "diagnostics"
    cfg = load_config(config_path)
    am = read_json(phys / "gate_checker_amendment.json") or {}
    g = read_json(phys / "wave_d_gate_amended.json") or {}
    before = (read_json(out / "inventory/evidence_hashes_before_amendment.json") or {}).get("files", {})
    now = evidence_manifest(out)
    inv = protected_inventory(root, cfg, out, "after")
    same, diff = protected_unchanged(out)
    led = rd(out / "budget_ledger.json")
    checks = {"original_gate_sha_matches_manifest": bool(am) and sha256_file(phys / "wave_d_gate.json") == am.get("original_gate_sha256"),
              "original_gate_still_fail_d7_only": (read_json(phys / "wave_d_gate.json") or {}).get("wave_d_status") == "FAIL",
              "evidence_unchanged_since_pre_amendment": bool(before) and before == now, "protected_evidence_unchanged": same,
              "ledger_4_4_4": led["wave_d_branch_attempts"]["used"] == led["total_branch_attempts"]["used"] == led["environment_resets"]["used"] == 4,
              "no_provider_rl_optimizer_use": all(led[k]["used"] == 0 for k in ZERO_KEYS + ("standalone_capture_resets",)),
              "amendment_flags_consistent": bool(am) and am.get("new_attempts") == 0 and am.get("branch_execution_code_changed") is False and am.get("runtime_data_changed") is False,
              "amended_gate_matches_manifest": g.get("wave_d_status") == am.get("amended_gate_status"),
              "wave_p_release_consistent": g.get("wave_p_released") is (g.get("wave_d_status") == "PASS"),
              "d1_d6_d8_d11_pass": all(g.get("gates", {}).get(k, {}).get("pass") for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D8", "D9", "D10", "D11"))}
    doc = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protected_files": inv["file_count"], "protected_diff": diff}
    _atomic_json(out / "verify_wave_d.json", doc)
    return doc
