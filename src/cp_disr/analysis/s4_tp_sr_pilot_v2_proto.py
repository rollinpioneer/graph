"""Wave P (minimal gripper-interference prototype): freeze, register, classify, decide, summarise, verify. Reads saved evidence only; never constructs an
environment or calls a provider."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from cp_disr.analysis import s4_tp_sr_pilot_v2_geom as G
from cp_disr.analysis.s1_integration import _atomic_json, sha256_file
from cp_disr.analysis.s4_tp_sr_pilot_v2 import (CARD_ID, CODE_FILES, INTERFERENCE_PHASES, StopRun, WAVES, build_manifest, init_phys_dir, jsonable, load_config, make_branch,
                                                mark_phase, perception_identity, protected_inventory, protected_unchanged, rd, sha_json, worker_dry_run, write_csv)
from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as D

CONFIG_IDS = ("V2_A_INTERFERENCE", "V2_A_OFFSET_CONTROL", "V2_B_INTERFERENCE", "V2_B_OFFSET_CONTROL")
LAYOUT_CASE = {"A": "T_P_SR_pool_33", "B": "T_P_SR_pool_57"}
SIDES = {"A": 1, "B": -1}             # fixed a priori: the two scenes use opposite sides
FIRST = {"direct": "a:PICK:target:v1", "relocation": "a:PICK:interferer:v1"}
ROUTE_ACTIONS = {"direct": ("a:PICK:target:v1", "a:PLACE:target:container:v1"),
                 "relocation": ("a:PICK:interferer:v1", "a:PLACE_BUFFER:interferer:buffer:v1", "a:PICK:target:v1", "a:PLACE:target:container:v1")}
HARD_BLOCK_TOKENS = ("Blocked", "ClearPath", "Occluded", "EasyGrasp", "SafeToPick")
PROTO_FILES = ("src/cp_disr/analysis/s4_tp_sr_pilot_v2_geom.py", "src/cp_disr/analysis/s4_tp_sr_pilot_v2_proto.py", "src/cp_disr/analysis/s4_tp_sr_pilot_v2_diag.py",
               "scripts/s4_tp_sr_pilot_v2_amend.py")


def _ro(p):
    Path(p).chmod(0o444)


def wave_d_passed(out):
    g = rd(Path(out) / "diagnostics/wave_d_gate_amended.json") if (Path(out) / "diagnostics/wave_d_gate_amended.json").is_file() else rd(Path(out) / "diagnostics/wave_d_gate.json")
    return g.get("wave_d_status") == "PASS" and g.get("wave_p_released") is True


def _seed(cfg_id):
    return int(hashlib.sha256(f"{cfg_id}|seed".encode()).hexdigest()[:8], 16) % 2147483648


def _decision(out, **f):
    p = Path(out) / "decision/decision_state.json"
    d = rd(p) if p.is_file() else {}
    d.update(f)
    _atomic_json(p, d)


def _route_check(by_id, facts0, mask, precondition_value, Truth):
    routes = {}
    for route, acts in ROUTE_ACTIONS.items():
        vals = {k: Truth(v) for k, v in facts0.items()}
        steps, ok = [], True
        for a in acts:
            c = by_id.get(a)
            pre = c is not None and precondition_value(c, vals) == Truth.TRUE
            steps.append({"action": a, "precondition_true": bool(pre)})
            if not pre:
                ok = False
                break
            for fid, t in c.effects.assignments().items():
                vals[fid] = t
        routes[route] = {"steps": steps, "goal_reached_by_nominal_effects": bool(ok and vals.get("p:Inside:target:container") == Truth.TRUE),
                         "first_action_legal_in_mask": bool(mask.get(acts[0]))}
    return routes


def contract_reachability(root, out, cfg):
    """Symbolic reachability of both nominal routes on the initial public facts. The semantic initial state (pool_33's saved facts: all objects on the table,
    nothing inside or at the buffer) is the template; the pool_57 saved facts (which carry a spurious Inside:interferer:container = TRUE from the verifier) are
    checked as a sensitivity variant, and both must pass."""
    from cp_disr.contracts import precondition_value
    from cp_disr.facts import Truth
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    root, out = Path(root), Path(out)
    saved = {"semantic_template(pool_33 saved facts)": rd(out / "captures/094286b08fcf5904/initial_public_facts.json"),
             "sensitivity(pool_57 saved facts)": rd(out / "captures/3c4f588df6e91b04/initial_public_facts.json")}
    tmpl = saved["semantic_template(pool_33 saved facts)"]
    if any(v["candidate_ids"] != tmpl["candidate_ids"] or v["candidate_mask"] != tmpl["candidate_mask"] for v in saved.values()):
        raise StopRun("STOPPED_CONTRACT_TEMPLATE", "candidate ids or mask differ between the saved wave D scenes")
    contracts = ground_task_contracts(root / "configs/runtime/tp_sr_v2_contract_registry.yaml", {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0})
    by_id = {c.id: c for c in contracts}
    mask = dict(zip(tmpl["candidate_ids"], tmpl["candidate_mask"]))
    variants, hard = {}, []
    for name, doc in saved.items():
        facts0 = {k: v["value"] for k, v in doc["facts"].items()}
        hard += [k for k in facts0 if any(t in k for t in HARD_BLOCK_TOKENS)]
        variants[name] = _route_check(by_id, facts0, mask, precondition_value, Truth)
        variants[name]["fact_values_differing_from_template"] = sorted(k for k in facts0 if facts0[k] != tmpl["facts"][k]["value"])
    ok = all(r["goal_reached_by_nominal_effects"] and r["first_action_legal_in_mask"] for v in variants.values() for r in (v["direct"], v["relocation"]))
    return {"candidate_mask": mask, "hard_blockage_facts": sorted(set(hard)), "variants": variants, "both_first_actions_legal": all(v[r]["first_action_legal_in_mask"] for v in variants.values() for r in ("direct", "relocation")),
            "both_routes_reachable": all(v[r]["goal_reached_by_nominal_effects"] for v in variants.values() for r in ("direct", "relocation")), "no_hard_blockage_fact": not hard,
            "pass": bool(ok and not hard), "routes": variants["semantic_template(pool_33 saved facts)"],
            "note": "reachability is symbolic on the initial public facts; Blocked/ClearPath facts are not registered so the geometry cannot leak into planning"}


def freeze_prototypes(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    proto = out / "prototype"
    if not wave_d_passed(out):
        raise StopRun("STOPPED_WAVE_P_NOT_RELEASED", "Wave D gate has not passed")
    if (proto / "prototype_configs.json").is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "prototypes are already frozen")
    proto.mkdir(parents=True, exist_ok=True)
    geo = G.measure_gripper(out)
    verts = geo.pop("_verts")
    _atomic_json(proto / "gripper_geometry.json", geo)
    np.savez(proto / "gripper_mesh_offsets.npz", **{f"{k}_{s}": v[i] for k, v in verts.items() for i, s in enumerate(("open", "closed"))})
    geo["_verts"] = verts
    diag_split = rd(out / "spec/T_P_SR_V2_diag_split.json")
    layouts = {r["case_id"]: r for r in diag_split["dev"]}
    n = int(cfg["wave_p"]["sweep_samples_per_segment"])
    rows, designs, sweep = [], {}, []
    for ci, (fam, case) in enumerate(LAYOUT_CASE.items()):
        lay = layouts[case]
        d = G.design_scene(geo, lay, SIDES[fam], n)
        designs[fam] = d
        if d["status"] != "OK":
            _atomic_json(proto / "design_stop.json", {"family": fam, "layout_case": case, **d})
            _decision(out, status="STOPPED", mechanism_feasibility="NOT_ESTABLISHED", next_action="EXPLICIT_RESEARCH_DECISION", stop_code=d["status"])
            raise StopRun(d["status"], d["reason"])
        for kind, key in (("INTERFERENCE", "interference"), ("OFFSET_CONTROL", "control")):
            cid = f"V2_{fam}_{kind}"
            r = d[key]
            rows.append({"case_id": cid, "split": "dev", "seed": _seed(cid), "target_xy": lay["target_xy"], "second_xy": [round(x, 6) for x in r["bar_xy"]], "container_xy": lay["container_xy"],
                         "buffer_xy": lay["buffer_xy"], "lid_closed": False, "second_role": "interferer", "second_half": d["bar_half"], "second_density": G.BAR["density"],
                         "second_friction": G.BAR["friction"], "design": {"family": fam, "kind": kind, "layout_case": case, "side": d["side"], "x_offset_from_target_m": d["x_offset_from_target_m"],
                                                                            **{k: v for k, v in r.items()}}})
            o_bar = d["own_grasp_opening_fraction"]
            for si, (xy, oe) in enumerate(((lay["target_xy"], 1.0), (r["bar_xy"], o_bar))):
                pts = G.sweep_points(geo["eef_start"], xy, n, oe)
                sweep.append(np.hstack([np.full((len(pts), 1), len(rows) - 1), np.full((len(pts), 1), si), pts]))
    np.save(proto / "gripper_sweep_points.npy", np.vstack(sweep))
    fam_dist = float(np.linalg.norm(np.array(layouts[LAYOUT_CASE["A"]]["target_xy"]) - np.array(layouts[LAYOUT_CASE["B"]]["target_xy"])))
    if fam_dist < 0.10:
        raise StopRun("STOPPED_CONFIGS_NOT_SPATIALLY_DISTINCT", f"A/B target distance {fam_dist:.3f} m")
    _atomic_json(proto / "gripper_sweep_model.json", {
        "columns_of_gripper_sweep_points_npy": ["config_index", "sweep_kind(0=target pick,1=bar own grasp)", "phase(0 approach,1 descend,2 press,3 close,4 lift_show)", "sample", "eef_x", "eef_y", "eef_z", "opening_fraction"],
        "phases": list(G.PHASES), "samples_per_linear_segment": n, "hover_dz_above_table": G.HOVER_DZ, "grasp_z": G.GRASP_Z, "press_dz": G.PRESS_DZ, "show_pose": list(G.SHOW_POSE),
        "object_slab_z": list(G.OBJ_ZRANGE), "graze_margin_m": G.Z_GRAZE_MARGIN, "bar": G.BAR, "bar_short_axis_limit_m": 0.8 * geo["usable_opening_open_m"],
        "bar_short_axis_m": 2 * G.BAR["hy"], "own_grasp_opening_fraction": designs["A"]["own_grasp_opening_fraction"], "control_min_clearance_m": G.CONTROL_MIN_CLEARANCE,
        "own_grasp_min_clearance_m": G.OWN_GRASP_MIN_CLEARANCE, "min_initial_gap_m": G.MIN_INITIAL_GAP,
        "geometry_basis": "real panda_gripper collision meshes and pad boxes placed with the saved wave D QA geom poses; vertices inside the object slab deeper than the graze margin",
        "revision_note": "a first AABB-based model over-estimated the palm height by 12 mm and made every scene infeasible; it was replaced by the mesh model BEFORE any wave P attempt (no wave P data existed)",
        "design_rule": "deterministic search, no outcome data; see design_scene docstring", "sides": SIDES, "layout_source": {k: v for k, v in LAYOUT_CASE.items()}, "family_target_distance_m": fam_dist})
    with (proto / "prototype_geometry.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "family", "kind", "target_xy", "bar_xy", "bar_half", "x_offset_from_target_m", "near_face_offset_m", "initial_gap_to_target_m", "target_sweep_clearance_m",
                    "own_grasp_sweep_clearance_m", "target_sweep_clearance_after_relocation_m", "closest_geom", "closest_phase"])
        for r in rows:
            d = r["design"]
            w.writerow([r["case_id"], d["family"], d["kind"], r["target_xy"], r["second_xy"], r["second_half"], d["x_offset_from_target_m"], d["near_face_offset_m"], d["initial_gap_to_target_m"],
                        d["target_sweep_clearance_m"], d.get("own_grasp_sweep_clearance_m", "NOT_APPLICABLE"), d.get("target_sweep_clearance_after_relocation_m", "NOT_APPLICABLE"),
                        (d["closest"] or {}).get("geom"), (d["closest"] or {}).get("phase")])
    _atomic_json(proto / "prototype_configs.json", {"card_id": CARD_ID, "configs": rows, "designs": designs, "rationale": "fixed a priori from measured gripper geometry; not tuned on outcomes"})
    reach = contract_reachability(root, out, cfg)
    _atomic_json(proto / "prototype_contract_reachability.json", reach)
    if not reach["pass"]:
        _decision(out, status="STOPPED", mechanism_feasibility="NOT_ESTABLISHED", next_action="EXPLICIT_RESEARCH_DECISION", stop_code="STOPPED_CONTRACT_GATE")
        raise StopRun("STOPPED_CONTRACT_GATE", json.dumps(reach["routes"]))
    split_path = out / "spec/T_P_SR_V2_proto_split.json"
    _atomic_json(split_path, {"task_id": "T_P_SR", "task_family_id": cfg["task_family_id"], "train": [], "dev": [{k: v for k, v in r.items()} for r in rows], "test": []})
    rel = os.path.relpath(split_path, root) if not str(split_path).startswith(str(root)) else str(split_path.relative_to(root))
    manifest_path = build_manifest(root, out, cfg, rel, "runtime_manifest_P.yaml")
    src_ident = {f: sha256_file(root / f) for f in CODE_FILES + PROTO_FILES if (root / f).is_file()}
    branches = []
    for r in rows:
        for rep in range(int(cfg["wave_p"]["repeats"])):
            for route, cand in FIRST.items():
                branches.append(make_branch("P", r["case_id"], cand, route, rep, manifest_path,
                                            {"source_config_sha256": sha_json(r), "config_kind": r["design"]["kind"], "runtime_source_sha256": src_ident["src/cp_disr/platforms/libero/tp_sr_v2_runtime.py"],
                                             "instrumentation_source_sha256": src_ident["src/cp_disr/platforms/libero/tp_sr_instrumentation.py"], "perception_identity": perception_identity()}))
    seeds = {}
    for b in branches:
        seeds.setdefault((b["case_id"], b["repeat"]), set()).add(b["restore_seed"])
    if len(branches) != 16 or len({b["branch_id"] for b in branches}) != 16 or any(len(v) != 1 for v in seeds.values()) or len(seeds) != 8:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "wave P must register 16 unique branches with paired restore seeds")
    _atomic_json(proto / "prototype_branch_registration.json", {"status": "REGISTERED", "card_id": CARD_ID, "physical_witness_episodes_cap": WAVES["P"]["cap"], "scenes": list(CONFIG_IDS),
                                                              "branches": branches})
    _atomic_json(proto / "source_identity_wave_p.json", {"files_sha256": src_ident, "note": "code identity at prototype freeze; the wave D branch-execution identity is in ../source_identity.json"})
    for p in proto.iterdir():
        _ro(p)
    mark_phase(out, "freeze-prototypes", prototype_configs=len(rows), prototype_branches=len(branches))
    return {"status": "FROZEN", "configs": [r["case_id"] for r in rows], "branches": len(branches), "family_target_distance_m": fam_dist}


def prepare_prototype_branches(root, config_path, out):
    root, out = Path(root), Path(out)
    if not wave_d_passed(out):
        raise StopRun("STOPPED_WAVE_P_NOT_RELEASED", "Wave D gate has not passed")
    src = out / "prototype/prototype_branch_registration.json"
    phys = out / "physical"
    dst = phys / "witnesses/e4_branch_registration.json"
    if dst.is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "wave P registration already exists")
    reg = rd(src)
    init_phys_dir(phys, WAVES["P"]["cap"])
    _atomic_json(dst, reg)
    _ro(dst)
    worker_dry_run(root, out, "P", reg["branches"][0]["branch_id"])
    mark_phase(out, "prepare-prototype-branches", wave_p_branches=len(reg["branches"]))
    return {"status": "PREPARED", "branches": len(reg["branches"])}


# ------------------------------------------------------------------ classification
def classify_prototype_branch(br, contracts):
    """Wave P classification. Same evidence rules as wave D except that physical interference is not required to end in a normal controller exit: a controller that
    could not complete because the gripper met the interferer ends abnormally, and that abnormal end is then the observed consequence (explanation path recorded)."""
    res = br["result"]
    if bool(res.get("task_success")):
        return {"label": "NOT_APPLICABLE_SUCCESS", "reason": "branch succeeded"}
    if not br["actions"]:
        return {"label": "UNRESOLVED", "reason": "no executed action captured"}
    a = br["actions"][-1]
    parts = (a["candidate_id"] or "").split(":")
    skill, obj = (parts[1] if len(parts) > 2 else ""), (parts[2] if len(parts) > 2 else "")
    facts, planner = D.fmap(a["facts"]), a["planner"] or {}
    qa = D.qa_tag(a, "after")
    ev = D.evidence(a, br)
    if res.get("termination_reason") == "SYMBOLIC_EVALUATOR_MISMATCH" or planner.get("status") == "GOAL_ALREADY_SATISFIED":
        pub_inside, hid_inside = facts.get("p:Inside:target:container"), D.hidden_inside(qa)
        if pub_inside == "TRUE" and hid_inside is False:
            label = "FALSE_SYMBOLIC_GOAL_FROM_PUBLIC_FACTS"
        elif pub_inside == "TRUE" and hid_inside is True:
            label = "EVALUATOR_SYMBOLIC_DISAGREEMENT"
        elif pub_inside != "TRUE" and hid_inside is not None:
            label = "PLANNER_FACT_INPUT_ERROR"
        else:
            label = "UNRESOLVED"
        return {"label": label, "kind": "symbolic_goal", "evidence": ev}
    nxt = {"target": "a:PLACE:target:container:v1", "interferer": "a:PLACE_BUFFER:interferer:buffer:v1"}.get(obj)
    other = "interferer" if obj == "target" else "target"
    held_h, held_p = D.hidden_held(qa, obj), facts.get(f"p:Held:{obj}")
    contacts = D.other_contacts(a, other)
    pre_true = D.precondition_true(contracts, nxt, {k: facts[k] for k in facts}) if nxt else None
    snap_mask = dict(zip((a["snapshot"] or {}).get("candidate_ids", []), (a["snapshot"] or {}).get("candidate_mask", [])))
    exit_ok = a["controller_exit"] in D.NORMAL
    status = planner.get("status")
    chk = {"candidate": a["candidate_id"], "controller_exit": a["controller_exit"], "hidden_held": held_h, "public_held": held_p, "interferer_contact_events": len(contacts),
           "next_action": nxt, "next_action_precondition_true_on_public_facts": pre_true, "next_action_mask": snap_mask.get(nxt), "planner_status": status, "hidden_available": qa is not None}
    if qa is None or not a["facts"]:
        return {"label": "UNRESOLVED", "checks": chk, "reason": "missing hidden QA or facts", "evidence": ev}
    if status == "NO_PLAN" and pre_true is False:
        path = "NO_PLAN_EXPLAINED_BY_POST_ACTION_PUBLIC_FACTS"
    elif status != "NO_PLAN" and not exit_ok:
        path = "ABNORMAL_CONTROLLER_EXIT_WITH_CONTACT_EVIDENCE"
    else:
        path = None
    if (not held_h) and contacts and held_p != "TRUE" and path:
        label = "PHYSICAL_GRASP_INTERFERENCE"
        chk["explanation_path"] = path
    elif (held_h and held_p != "TRUE") or ((not held_h) and held_p == "TRUE"):
        label = "PERCEPTION_VERIFIER_MISMATCH"
    elif pre_true and snap_mask.get(nxt) is True and status == "NO_PLAN":
        label = "PLANNER_FACT_INPUT_ERROR"
    elif (not held_h) and not contacts:
        label = "CONTROLLER_EXECUTION_FAILURE"
    else:
        label = "UNRESOLVED"
    return {"label": label, "checks": chk, "interferer_contacts": contacts[:10], "evidence": ev}


def engineering_failure(res, state):
    return bool(state == "UNKNOWN" or res.get("execution_status") in ("EXCEPTION", "ERROR") or not res or res.get("protocol_complete") is False and res.get("execution_status") == "EXCEPTION")


def contact_counts(br):
    n = {"finger_palm_vs_interferer": 0, "finger_palm_vs_target": 0, "target_vs_interferer": 0, "max_normal_force": 0.0, "measured": False}
    for a in br["actions"]:
        for rec in a["contacts"]:
            if rec.get("phase") not in INTERFERENCE_PHASES:
                continue
            for p in rec.get("pairs", []):
                cats = {p["cat1"], p["cat2"]}
                if p["distance"] > 1e-3:
                    continue
                if "interferer" in cats and cats & {"finger", "palm"}:
                    n["finger_palm_vs_interferer"] += 1
                elif "target" in cats and cats & {"finger", "palm"}:
                    n["finger_palm_vs_target"] += 1
                elif cats == {"target", "interferer"}:
                    n["target_vs_interferer"] += 1
                f = p.get("normal_force")
                if isinstance(f, (int, float)):
                    n["max_normal_force"] = max(n["max_normal_force"], float(f))
                    n["measured"] = True
    return n


def paired_restore(out, branches, tol=1e-9):
    phys = Path(out) / "physical"
    by = {}
    for b in branches:
        by.setdefault((b["case_id"], b["repeat"]), []).append(b)
    res = {}
    for (case, rep), bs in sorted(by.items()):
        k = f"{case}|{rep}"
        chk = [D.read_json(phys / "witnesses/initial_state_checks" / f"{b['branch_id']}.json") for b in bs]
        if len(bs) != 2 or any(c is None or c.get("status") != "OK" for c in chk):
            res[k] = {"ok": False, "reason": "missing initial_state_checks or pair incomplete"}
            continue
        same_seed = len({b["restore_seed"] for b in bs}) == 1
        dq = float(np.max(np.abs(np.array(chk[0]["physical"]["qpos"]) - np.array(chk[1]["physical"]["qpos"]))))
        dv = float(np.max(np.abs(np.array(chk[0]["physical"]["qvel"]) - np.array(chk[1]["physical"]["qvel"]))))
        keys = chk[0]["compare_key"] == chk[1]["compare_key"]
        mask = chk[0]["candidate_mask"] == chk[1]["candidate_mask"]
        res[k] = {"ok": bool(same_seed and keys and mask and dq <= tol and dv <= tol), "same_restore_seed": same_seed, "compare_key_equal": keys, "candidate_mask_equal": mask,
                  "qpos_max_abs_diff": dq, "qvel_max_abs_diff": dv, "branches": [b["branch_id"] for b in bs]}
    return res


def evaluate_mechanism(rows, pairing_ok, contract_ok, max_eng=0.125):
    """Pure decision rule of the runbook (section 15). rows: config_id, kind ('INTERFERENCE'/'OFFSET_CONTROL'), route, repeat, success, label, engineering_failure, contact_events."""
    reasons = []
    if len(rows) != 16:
        reasons.append(f"INCOMPLETE_WAVE_P ({len(rows)} of 16 branches)")
    eng = sum(1 for r in rows if r["engineering_failure"])
    if rows and eng / len(rows) > max_eng:
        reasons.append(f"ENGINEERING_FAILURE_RATE {eng}/{len(rows)} > {max_eng}")
    if not pairing_ok:
        reasons.append("PAIRED_RESTORE_INVALID")
    if not contract_ok:
        reasons.append("ROUTES_NOT_CONTRACT_REACHABLE")
    masq = [r for r in rows if not r["success"] and r["label"] in ("PERCEPTION_VERIFIER_MISMATCH", "PLANNER_FACT_INPUT_ERROR")]
    if masq:
        reasons.append(f"PERCEPTION_OR_PLANNER_CAUSE_PRESENT ({len(masq)} branches)")
    per = {}
    for cid in sorted({r["config_id"] for r in rows}):
        rs = [r for r in rows if r["config_id"] == cid]
        d, l = [r for r in rs if r["route"] == "direct"], [r for r in rs if r["route"] == "relocation"]
        per[cid] = {"kind": rs[0]["kind"], "direct_success": f"{sum(r['success'] for r in d)}/{len(d)}", "relocation_success": f"{sum(r['success'] for r in l)}/{len(l)}",
                    "direct_labels": [r["label"] for r in d], "relocation_labels": [r["label"] for r in l]}
        if rs[0]["kind"] == "INTERFERENCE":
            if not (len(l) == 2 and all(r["success"] for r in l)):
                reasons.append(f"{cid}: relocation not 2/2")
            if not (len(d) == 2 and sum(r["success"] for r in d) <= 1):
                reasons.append(f"{cid}: direct success > 1/2")
            bad = [r for r in d if not r["success"] and r["label"] != "PHYSICAL_GRASP_INTERFERENCE"]
            if bad or not any(not r["success"] for r in d):
                reasons.append(f"{cid}: direct non-success not attributed to PHYSICAL_GRASP_INTERFERENCE")
            if not any((not r["success"]) and r["label"] == "PHYSICAL_GRASP_INTERFERENCE" and r["contact_events"] > 0 for r in d):
                reasons.append(f"{cid}: no saved contact evidence for physical interference")
        else:
            if not (len(d) == 2 and all(r["success"] for r in d)):
                reasons.append(f"{cid}: control direct not 2/2")
            if any(r["label"] == "PHYSICAL_GRASP_INTERFERENCE" for r in rs):
                reasons.append(f"{cid}: physical interference labelled in a control")
    return {"verdict": "ESTABLISHED" if not reasons else "NOT_ESTABLISHED", "reasons": reasons, "per_config": per, "engineering_failures": eng, "branches": len(rows)}


def classify_prototypes(root, config_path, out):
    root, out = Path(root), Path(out)
    import yaml as _y
    from cp_disr.platforms.libero.tp_sr_runtime import ground_task_contracts
    phys = out / "physical"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    manifest = _y.safe_load(Path(reg["branches"][0]["manifest_path"]).read_text())
    rtm = manifest["runtime"]
    contracts = ground_task_contracts(Path(rtm["repository_path"]) / rtm["stage_2a_contract_path"], rtm["skill_timeouts"]["T_P_SR"])
    attempts = rd(phys / "attempt_registry.json") if (phys / "attempt_registry.json").is_file() else {}
    branches = [D.load_branch(out, "P", b) for b in reg["branches"]]
    rows, csv_rows, roots, contact_rows = [], [], {}, []
    for br in branches:
        b, r = br["branch"], br["result"]
        state = attempts.get(b["branch_id"])
        cls = classify_prototype_branch(br, contracts) if r else {"label": "UNRESOLVED", "reason": "no result"}
        cc = contact_counts(br)
        eng = engineering_failure(r, state)
        kind = b["config_kind"]
        rows.append({"config_id": b["case_id"], "kind": kind, "route": b["route"], "repeat": b["repeat"], "success": bool(r.get("task_success")), "label": cls["label"],
                     "engineering_failure": eng, "contact_events": cc["finger_palm_vs_interferer"]})
        csv_rows.append({"branch_id": b["branch_id"], "config_id": b["case_id"], "kind": kind, "route": b["route"], "repeat": b["repeat"], "attempt_state": state,
                         "execution_status": r.get("execution_status"), "termination_reason": r.get("termination_reason"), "task_success": r.get("task_success"),
                         "actions_captured": len(br["actions"]), "recorder_errors": r.get("recorder_errors"), "root_cause": cls["label"], "engineering_failure": eng,
                         "env_reset_calls": (r.get("env_counts") or {}).get("reset_calls"), "worker_wall_seconds": r.get("worker_wall_seconds")})
        contact_rows.append({"branch_id": b["branch_id"], "config_id": b["case_id"], "route": b["route"], "repeat": b["repeat"], **{k: cc[k] for k in ("finger_palm_vs_interferer", "finger_palm_vs_target", "target_vs_interferer", "max_normal_force", "measured")}})
        roots[b["branch_id"]] = {"config_id": b["case_id"], "route": b["route"], "repeat": b["repeat"], **cls}
    write_csv(phys / "branch_results.csv", csv_rows)
    write_csv(phys / "contact_summary.csv", contact_rows)
    pair = paired_restore(out, reg["branches"])
    _atomic_json(phys / "paired_restore.json", {"pairs": pair, "all_ok": bool(pair) and all(v["ok"] for v in pair.values())})
    _atomic_json(phys / "root_cause_classification.json", {"rule": "wave P classification (physical interference needs contact + state evidence; abnormal controller end allowed with contact evidence)", "branches": roots})
    reach = rd(out / "prototype/prototype_contract_reachability.json")
    gate = evaluate_mechanism(rows, bool(pair) and all(v["ok"] for v in pair.values()), bool(reach["pass"]))
    complete = len(attempts) == 16 and all(v in D.D7_TERMINAL for v in attempts.values())
    verdict = gate["verdict"]
    if not complete:
        state = {"status": "STOPPED", "mechanism_feasibility": "NOT_ESTABLISHED", "next_action": "EXPLICIT_RESEARCH_DECISION"}
    elif verdict == "ESTABLISHED":
        state = {"status": "COMPLETE", "mechanism_feasibility": "ESTABLISHED_FOR_CURATED_DEMONSTRATION", "next_action": "REQUEST_PROVIDER_AND_REPRESENTATION_QUALIFICATION"}
    else:
        state = {"status": "COMPLETE", "mechanism_feasibility": "NOT_ESTABLISHED", "next_action": "ABANDON_OR_REDESIGN_RELOCATION_MECHANISM"}
    (out / "decision").mkdir(exist_ok=True)
    _atomic_json(out / "decision/mechanism_gate.json", {**gate, "pairing_ok": bool(pair) and all(v["ok"] for v in pair.values()), "contract_reachable": bool(reach["pass"]), "attempts_terminal": complete, **state})
    _atomic_json(out / "decision/next_action.json", {"next_action": state["next_action"], "provider_authorized": False, "s2_authorized": False,
                                                      "note": "provider, representation and S2 are never auto-authorized"})
    _decision(out, wave_p_status=verdict, provider_authorized=False, s2_authorized=False, **state)
    mark_phase(out, "classify-prototypes", mechanism=verdict)
    return {**gate, **state}


def summarize(root, config_path, out):
    out = Path(out)
    g = rd(out / "decision/mechanism_gate.json")
    led = rd(out / "budget_ledger.json")
    lines = ["# CP-DISR-S4-TP-SR-PILOT-V2-1 final summary", "", f"- status: {g.get('status')}; mechanism_feasibility: {g.get('mechanism_feasibility')}; next_action: {g.get('next_action')}",
             f"- Wave D: amended gate PASS (original FAIL on D7 checker defect preserved); Wave P: {g.get('verdict')}",
             f"- budget used: total {led['total_branch_attempts']['used']}/{led['total_branch_attempts']['cap']}, wave D {led['wave_d_branch_attempts']['used']}, wave P {led['wave_p_branch_attempts']['used']}, resets {led['environment_resets']['used']}",
             "- provider / representation / RL / optimizer / elastic / formal test: 0", "", "## Per config", ""]
    for cid, v in (g.get("per_config") or {}).items():
        lines.append(f"- {cid} ({v['kind']}): direct {v['direct_success']}, relocation {v['relocation_success']}; direct labels {v['direct_labels']}")
    if g.get("reasons"):
        lines += ["", "## Reasons for NOT_ESTABLISHED", ""] + [f"- {r}" for r in g["reasons"]]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n")
    return {"summary": str(out / "final_summary.md")}


def verify(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    led = rd(out / "budget_ledger.json")
    inv = protected_inventory(root, cfg, out, "after")
    same, diff = protected_unchanged(out)
    g = rd(out / "decision/mechanism_gate.json") if (out / "decision/mechanism_gate.json").is_file() else {}
    zero = ("provider_first_calls", "provider_retries", "representation_forwards", "rl_transitions", "optimizer_steps", "training_attempts", "elastic_attempts", "formal_test_episodes", "standalone_capture_resets")
    checks = {"protected_evidence_unchanged": same, "budget_caps_respected": all(v["used"] <= v["cap"] for v in led.values()),
              "wave_p_le_16": led["wave_p_branch_attempts"]["used"] <= 16, "total_le_20": led["total_branch_attempts"]["used"] <= 20,
              "one_reset_per_attempt": led["environment_resets"]["used"] == led["total_branch_attempts"]["used"],
              "no_provider_rl_optimizer": all(led[k]["used"] == 0 for k in zero), "wave_d_amended_pass": wave_d_passed(out),
              "decision_files_present": bool(g), "provider_and_s2_not_authorized": rd(out / "decision/next_action.json").get("provider_authorized") is False if (out / "decision/next_action.json").is_file() else False}
    doc = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protected_files": inv["file_count"], "protected_diff": diff}
    _atomic_json(out / "verify.json", doc)
    return doc
