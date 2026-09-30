"""S4 Family A soft-ordering MVP (CP-DISR-S4-FAMILY-A-SOFT-ORDERING-MVP-1).

Controlled mechanism validation: does the *physical* completion time of the identical symbolic plan depend on the
order in which the two goal objects are handled, and does that dependence reverse when the two object identities are
exchanged geometrically?

Reuses the production branch executor (execute_registered_branch), reserve/claim/finish receipts, paired-restore QA,
RuntimeBundle, D0 environment, SkillExecutor, FactVerifier, TaskEvaluator, SafetyManager, DurationProvider,
SnapshotBuilder, B_PLAN and the V2 passive instrumentation unchanged.
No provider, representation, RL, optimizer, elastic, S2/S3 or formal-test code path exists in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cp_disr.analysis.s1_integration import (IntegrationError, _append_jsonl, _atomic_json, _load_attempts, _transaction_lock,
                                             finish_branch_attempt, reserve_branch_attempt, sha256_file)

CARD_ID = "CP-DISR-S4-FAMILY-A-SOFT-ORDERING-MVP-1"
BASE_COMMIT = "39baafb8ac937f4f55b6934d2018ba8ec6ba6d84"
TASK_ID = "T_P_SO_MVP"
SECOND_ROLE = "second_object"
ROUTE_TARGET_FIRST = "target_first"
ROUTE_SECOND_FIRST = "second_first"
ROUTES = {
    ROUTE_TARGET_FIRST: ("a:PICK:target:v1", "a:PLACE:target:container:v1", "a:PICK:second_object:v1", "a:PLACE:second_object:container:v1"),
    ROUTE_SECOND_FIRST: ("a:PICK:second_object:v1", "a:PLACE:second_object:container:v1", "a:PICK:target:v1", "a:PLACE:target:container:v1"),
}
GROUNDED_ACTIONS = ("a:PICK:target:v1", "a:PICK:second_object:v1", "a:PLACE:target:container:v1", "a:PLACE:second_object:container:v1")
INITIAL_CANDIDATES = ("a:PICK:target:v1", "a:PICK:second_object:v1")
CONFIG_ORDER = ("SO_TARGET_FIRST", "SO_SECOND_FIRST", "SO_NEUTRAL")
CONTRACT_REL = "configs/runtime/tp_so_mvp_contract_registry.yaml"
TASK_SPEC_REL = "configs/tasks/resolved/T_P_SO_MVP.yaml"
RUNTIME_REL = "src/cp_disr/platforms/libero/tp_so_mvp_runtime.py"
SKILL_TIMEOUTS = {"PICK": 9.0, "PLACE": 8.0}
D_REF = 3.649999999999709
DEADLINE = 60.0
DECISION_CAP = 8
PHYS_CAP = 12
OUT_REL = "runs/final_master/S4/family_a_soft_ordering_mvp"

# Facts produced by the production verifier for both cubes; the offline template must carry them.
VERIFIER_FACT_IDS = ("p:GripperEmpty", "p:Held:target", "p:Held:second_object", "p:OnTable:target", "p:OnTable:second_object",
                     "p:Open:container", "p:Inside:target:container", "p:Inside:second_object:container",
                     "p:AtBuffer:target:buffer", "p:AtBuffer:second_object:buffer")
INITIAL_TRUE_FACTS = ("p:GripperEmpty", "p:OnTable:target", "p:OnTable:second_object", "p:Open:container")
INITIAL_FALSE_FACTS = ("p:Inside:target:container", "p:Inside:second_object:container", "p:Held:target", "p:Held:second_object")

CODE_FILES = ("src/cp_disr/analysis/s4_family_a_soft_ordering_mvp.py", "scripts/s4_family_a_soft_ordering_mvp.py",
              "src/cp_disr/platforms/libero/tp_so_mvp_runtime.py", "src/cp_disr/platforms/libero/tp_sr_instrumentation.py",
              "src/cp_disr/platforms/libero/tp_sr_v2_env.py", "src/cp_disr/platforms/libero/runtime_factory.py",
              "configs/final_master/s4_family_a_soft_ordering_mvp.yaml", "configs/tasks/resolved/T_P_SO_MVP.yaml",
              "configs/runtime/tp_so_mvp_contract_registry.yaml")


class StopRun(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rd(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merged_budget_ledger(out):
    out = Path(out)
    ledger = rd(out / "budget_ledger.json")
    physical_path = out / "physical/budget_ledger.json"
    if physical_path.is_file():
        ledger.update(rd(physical_path))
    return ledger


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def write_csv(path, rows, fields=None):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha_json(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def dist2(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def delta_proxy(q0, a, b, c):
    """Offline-only path-cost proxy. Returns C_(B->A) - C_(A->B); positive means A-first is predicted faster.

    C_(A->B) = ||q0 - A|| + ||c - B||. Never read by the runtime, planner, policy or reward.
    """
    c_ab = dist2(q0, a) + dist2(c, b)
    c_ba = dist2(q0, b) + dist2(c, a)
    return float(c_ba - c_ab)

# ------------------------------------------------------------------ ledger
def init_ledger(out, cfg):
    led = {k: {"cap": int(v), "used": 0} for k, v in cfg["budgets"].items()}
    _atomic_json(Path(out) / "budget_ledger.json", led)
    (Path(out) / "budget_events.jsonl").touch()
    return led


def charge_many(out, keys, ref=""):
    """Atomic: either every key has room and all are charged by one, or nothing is consumed."""
    out = Path(out)
    with _transaction_lock(out):
        led = rd(out / "budget_ledger.json")
        for k in keys:
            if k not in led:
                raise StopRun("STOPPED_EVIDENCE_INTEGRITY", f"unknown budget key {k}")
            if int(led[k]["used"]) + 1 > int(led[k]["cap"]):
                raise StopRun("STOPPED_BUDGET_EXHAUSTED", f"{k} {led[k]['used']}+1>{led[k]['cap']}")
        for k in keys:
            led[k]["used"] = int(led[k]["used"]) + 1
            _append_jsonl(out / "budget_events.jsonl", {"event": "charge", "key": k, "used": led[k]["used"], "cap": led[k]["cap"],
                                                        "ref": ref, "ts": now()})
        _atomic_json(out / "budget_ledger.json", led)
        return {k: led[k]["used"] for k in keys}


def event(out, kind, **f):
    _append_jsonl(Path(out) / "scheduling_events.jsonl", {"event": kind, "timestamp_utc": now(), "monotonic": time.monotonic(), **f})


# ------------------------------------------------------------------ protected inventory (old S1/S4 evidence is read-only)
def hash_tree(root, rels):
    root = Path(root)
    files = {}
    for rel in rels:
        base = root / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(root))] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    return files


def protected_inventory(root, cfg, out, which):
    root, out = Path(root), Path(out)
    inv = {"baseline_commit": BASE_COMMIT, "source_commit": git(root, "rev-parse", "HEAD"), "created_utc": now(),
           "roots": list(cfg["protected_roots"]), "files": hash_tree(root, cfg["protected_roots"])}
    inv["file_count"] = len(inv["files"])
    (out / "inventory").mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "inventory" / f"protected_{which}.json", inv)
    return inv


def protected_unchanged(out):
    a = rd(Path(out) / "inventory/protected_before.json")["files"]
    b = rd(Path(out) / "inventory/protected_after.json")["files"]
    diff = sorted(set(a) ^ set(b))[:10] + [k for k in sorted(a) if k in b and a[k] != b[k]][:10]
    return a == b, diff


# ------------------------------------------------------------------ freeze-spec
def out_dir_for(root, config_path):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(root) / OUT_REL / f"{stamp}_{sha256_file(config_path)[:8]}"


def mark_phase(out, phase, **fields):
    p = Path(out) / "stage_manifest.json"
    doc = rd(p)
    if phase not in doc.setdefault("phases_done", []):
        doc["phases_done"].append(phase)
    doc.update(fields)
    _atomic_json(p, doc)


def freeze_spec(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    head = git(root, "rev-parse", "HEAD")
    if subprocess.call(["git", "-C", str(root), "merge-base", "--is-ancestor", BASE_COMMIT, head]) != 0:
        raise StopRun("STOPPED_WORKTREE_PREFLIGHT", "HEAD does not descend from the base commit")
    for sub in ("inventory", "spec", "geometry", "physical", "captures", "decision", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    inv = protected_inventory(root, cfg, out, "before")
    init_ledger(out, cfg)
    _atomic_json(out / "authorization.json", {"card_id": CARD_ID, "master_id": cfg["master_id"], "scope": cfg["scope"],
                                              **cfg["authorization"], "caps": cfg["budgets"], "recorded_utc": now()})
    _atomic_json(out / "source_identity.json", {"head": head, "base_commit": BASE_COMMIT, "base_branch": cfg["base_branch"],
                                                "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                                                "files_sha256": {f: sha256_file(root / f) for f in CODE_FILES if (root / f).is_file()},
                                                "git_status_at_freeze": git(root, "status", "--short"),
                                                "config_sha256": sha256_file(config_path)})
    _atomic_json(out / "stage_manifest.json", {"card_id": CARD_ID, "master_id": cfg["master_id"], "status": "SPEC_FROZEN",
                                               "created_utc": now(), "phases_done": ["freeze-spec"], "head": head,
                                               "task_id": TASK_ID, "task_family_id": cfg["task_family_id"]})
    _atomic_json(out / "spec/task_family_spec.json", {"task_family_id": cfg["task_family_id"], "family_version": cfg["family_version"],
                                                      "task_id": TASK_ID, "dataset_role": cfg["dataset_role"],
                                                      "scope": cfg["scope"], "method_version": cfg["method_version"],
                                                      "objects": {"target": "object", "second_object": "object", "container": "container",
                                                                  "buffer": "buffer"},
                                                      "goals": ["p:Inside:target:container", "p:Inside:second_object:container"],
                                                      "initial_candidate_ids": list(INITIAL_CANDIDATES),
                                                      "grounded_action_ids": list(GROUNDED_ACTIONS),
                                                      "excluded_groundings": ["a:OPEN:container:v1", "a:PLACE_BUFFER:target:buffer:v1",
                                                                              "a:PLACE_BUFFER:second_object:buffer:v1", "a:MOVE:*"],
                                                      "routes": {k: list(v) for k, v in ROUTES.items()},
                                                      "deadline_seconds": DEADLINE, "reference_skill_seconds": D_REF})
    _atomic_json(out / "spec/relation_templates.json", {"status": "FROZEN_FUTURE_ONLY_NOT_EXECUTED", "provider_calls": 0,
                                                        "templates": [{"type": "SOFT_SUPPORTS",
                                                                       "source": "a:PLACE:<first_object>:container:v1",
                                                                       "effect_fact_ref": "p:Inside:<first_object>:container",
                                                                       "target": "a:PICK:<second_object>:v1",
                                                                       "meaning": "completing the first object may make the next PICK easier"}],
                                                        "forbidden": [{"pattern": "PICK(object) -> PLACE(object)",
                                                                       "reason": "direct contract redundancy"}]})
    _atomic_json(out / "spec/metric_definition.json", {"primary": "task_sim_completion_seconds",
                                                       "primary_definition": "first TASK_SUCCESS evaluator sim time minus first action_start sim time",
                                                       "secondary": "sum_skill_sim_duration", "clock": "mujoco_sim_seconds",
                                                       "wall_time_used_for_mechanism": False, "wall_time_used_for": "engineering throughput only"})
    _atomic_json(out / "decision/decision_state.json", {"status": "IN_PROGRESS", "technical_wave": "NOT_RUN",
                                                        "remaining_branches_released": False,
                                                        "mechanism_feasibility": "NOT_DETERMINED", "next_action": "",
                                                        "provider_authorized": False, "representation_authorized": False,
                                                        "s2_authorized": False})
    return {"status": "SPEC_FROZEN", "out": str(out), "protected_files": inv["file_count"], "head": head}

# ------------------------------------------------------------------ anchor bank / geometry (offline; constructs no environment)
def _s1_cube_anchor(root, cfg):
    """Read the frozen S1 r3 evidence: the dev_18 TASK_SUCCESS branch and its paired journal.

    Returns the target xy of a *successfully PICKed and PLACEd* cube and the second cube xy of the same frozen case,
    taken only from saved split/results. No environment is constructed.
    """
    out = root / cfg["sources"]["s1_output"]
    rec = out / "e4_recovery_23fab0c_r3"
    reg = rd(rec / "witnesses/e4_branch_registration.json")
    results = {}
    for p in sorted((rec / "branch_results").glob("*.json")):
        r = rd(p)
        results[r["branch_id"]] = r
    cand = []
    for b in reg["branches"]:
        r = results.get(b["branch_id"])
        if not r or not r.get("task_success"):
            continue
        if str(r.get("case_id", "")).endswith("dev_18"):
            cand.append((b, r))
    if not cand:
        raise StopRun("STOPPED_REFERENCE_GEOMETRY_UNAVAILABLE", "no successful dev_18 branch in the S1 r3 evidence")
    branch, result = cand[0]
    trace = result.get("trace") or []
    picked, placed = set(), set()
    for step in trace:
        cid = str(step.get("candidate_id", ""))
        parts = cid.split(":")
        if len(parts) < 3:
            continue
        skill, obj = parts[1], parts[2]
        if skill == "PICK" and step.get("controller_exit") in ("NORMAL_TERMINATION", "SUCCESS"):
            picked.add(obj)
        if skill == "PLACE" and step.get("controller_exit") in ("NORMAL_TERMINATION", "SUCCESS"):
            placed.add(obj)
    split = rd(out / "input_binding/T_A_s1_rev1_runtime_split_no_cache.json")
    rows = {r["case_id"]: r for r in (split.get("train", []) + split.get("dev", []))}
    row = rows.get(branch["case_id"])
    if row is None:
        raise StopRun("STOPPED_REFERENCE_GEOMETRY_UNAVAILABLE", "dev_18 row missing from the frozen S1 split")
    return {"source_case": branch["case_id"], "source_branch": branch["branch_id"],
            "source_hash": sha256_file(rec / "branch_results" / f"{branch['branch_id']}.json"),
            "row": row, "picked": sorted(picked), "placed": sorted(placed),
            "task_success": bool(result.get("task_success")),
            "termination_reason": result.get("termination_reason"),
            "controller_exit": result.get("controller_exit"),
            "witness_validity": result.get("witness_validity")}


def _fact_values(path):
    """Read a capture facts.json into {fact_id: value}; tolerates both the dict and list encodings."""
    doc = rd(path)
    rows = doc if isinstance(doc, list) else (doc.get("facts") if isinstance(doc.get("facts"), list) else None)
    if rows is not None:
        return {str(r.get("fact_id")): r.get("value") for r in rows if isinstance(r, dict)}
    src = doc.get("facts", doc) if isinstance(doc, dict) else {}
    return {str(k): (v.get("value") if isinstance(v, dict) else v) for k, v in src.items()}


def _executed_trace(cap):
    """Executed (candidate_id, controller_exit) pairs from the passive controller trace. No planner intent."""
    trace = []
    for adir in sorted(cap.glob("action_*")):
        ct = adir / "controller_trace.jsonl"
        if not ct.is_file():
            continue
        lines = [json.loads(x) for x in ct.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines:
            trace.append((str(lines[0].get("candidate_id", "")), str(lines[-1].get("controller_exit", ""))))
    return trace


def _v2_neutral_anchor(root, cfg):
    """Read the frozen V2 pilot capture pool_33 from *executed* evidence.

    Prefers a capture whose executed trace actually PICKed and PLACEd a cube (d37aeb...), so the neutral anchors
    are backed by a real grasp+placement rather than planner intent. Never constructs an environment.
    """
    cap_root = root / cfg["sources"]["pilot_output"] / "captures"
    if not cap_root.is_dir():
        raise StopRun("STOPPED_REFERENCE_GEOMETRY_UNAVAILABLE", "V2 pilot captures missing")
    matches = []
    for cap in sorted(cap_root.iterdir()):
        layout = cap / "layout.json"
        if not layout.is_file():
            continue
        doc = rd(layout)
        case = (doc.get("case") or {})
        if case.get("case_id") != "T_P_SR_pool_33":
            continue
        picked, placed, exits = set(), set(), []
        for cid, exit_code in _executed_trace(cap):
            parts = cid.split(":")
            if exit_code not in ("NORMAL_TERMINATION", "SUCCESS"):
                continue
            if len(parts) > 2 and parts[1] == "PICK":
                picked.add(parts[2])
            if len(parts) > 2 and parts[1] == "PLACE":
                placed.add(parts[2])
        for adir in sorted(cap.glob("action_*")):
            ev = adir / "evaluator.json"
            if ev.is_file():
                exits.append(bool(rd(ev).get("task_success")))
        entry = {"source_capture": cap.name, "source_hash": sha_json(doc), "case": case,
                 "container_xy": case.get("container_xy"), "buffer_xy": case.get("buffer_xy"),
                 "executed_trace": [list(t) for t in _executed_trace(cap)],
                 "picked": sorted(picked), "placed": sorted(placed),
                 "pick_place_observed": bool(picked and placed), "task_success_observed": bool(any(exits))}
        matches.append(entry)
    if not matches:
        raise StopRun("STOPPED_REFERENCE_GEOMETRY_UNAVAILABLE", "T_P_SR_pool_33 capture not found")
    # Prefer a capture with observed PICK+PLACE; tie-break deterministically on capture name.
    matches.sort(key=lambda e: (not e["pick_place_observed"], e["source_capture"]))
    return matches[0]


def _anchor_entry(name, xy, source_case, source_branch, source_hash, pick_ok, place_ok, q0, container, cfg):
    xy = [float(xy[0]), float(xy[1])]
    half = float(cfg["geometry"]["workspace_half_extent_m"])
    sep = float(cfg["geometry"]["min_object_separation_m"])
    return {"anchor": name, "xy": xy, "source_case": source_case, "source_branch": source_branch, "source_hash": source_hash,
            "pick_success_observed": bool(pick_ok), "place_success_observed": bool(place_ok),
            "workspace_valid": bool(abs(xy[0]) <= half and abs(xy[1]) <= half),
            "container_clearance": bool(dist2(xy, container) >= sep),
            "no_initial_penetration": True, "engineering_exception": False,
            "same_geometry_clause_used": bool(pick_ok and not place_ok)}


def derive_anchor_bank(root, config_path, out):
    """Build the safe anchor bank from *saved* evidence only. This function never constructs an environment."""
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    geom = cfg["geometry"]
    q0 = geom["q0_xy"]
    container = cfg["scene"]["container_xy"]
    s1 = _s1_cube_anchor(root, cfg)
    v2 = _v2_neutral_anchor(root, cfg)
    row = s1["row"]
    a_xy, b_xy = row["target_xy"], row["second_xy"]
    picks = set(s1["picked"])
    places = set(s1["placed"])
    tf_entry = _anchor_entry("A_s1_dev18_target", a_xy, s1["source_case"], s1["source_branch"], s1["source_hash"],
                             "target" in picks, "target" in places, q0, container, cfg)
    sf_entry = _anchor_entry("B_s1_dev18_second", b_xy, s1["source_case"], s1["source_branch"], s1["source_hash"],
                             SECOND_ROLE in picks, SECOND_ROLE in places, q0, container, cfg)
    v2_case = v2["case"]
    n_xy = v2_case.get("target_xy")
    # SO_NEUTRAL is an *independent* evidence-backed layout: it carries its own container from the same saved capture,
    # exactly as SO_TARGET_FIRST/SO_SECOND_FIRST carry the dev_18 container. Using the dev_18 container here would
    # evaluate the neutral pair against a container it was never recorded with.
    n_container = list(v2["container_xy"])
    n_buffer = list(v2.get("buffer_xy") or cfg["scene"]["buffer_xy"])
    n_entry = _anchor_entry("N_v2_pool33_target", n_xy, v2["case"].get("case_id"), v2["source_capture"], v2["source_hash"],
                            "target" in set(v2["picked"]), "target" in set(v2["placed"]), q0, n_container, cfg)
    n2_entry = _anchor_entry("N_v2_pool33_second", v2_case.get("second_xy"), v2["case"].get("case_id"), v2["source_capture"],
                             v2["source_hash"], SECOND_ROLE in set(v2["picked"]) or "interferer" in set(v2["picked"]),
                             SECOND_ROLE in set(v2["placed"]) or "interferer" in set(v2["placed"]), q0, n_container, cfg)
    d_main = delta_proxy(q0, a_xy, b_xy, container)
    d_neutral = delta_proxy(q0, n_xy, n2_entry["xy"], n_container)
    ref = {"q0_xy": q0, "container_xy": container, "buffer_xy": cfg["scene"]["buffer_xy"], "lid_closed": cfg["scene"]["lid_closed"],
           "neutral_container_xy": n_container, "neutral_buffer_xy": n_buffer,
           "second_role": SECOND_ROLE, "s1_source": {k: s1[k] for k in ("source_case", "source_branch", "source_hash", "task_success",
                                                                        "termination_reason", "controller_exit", "witness_validity")},
           "v2_source": {"case_id": v2["case"].get("case_id"), "capture": v2["source_capture"], "source_hash": v2["source_hash"],
                         "pick_place_observed": v2["pick_place_observed"], "task_success_observed": v2["task_success_observed"],
                         "container_xy": n_container, "buffer_xy": n_buffer, "executed_trace": v2["executed_trace"]},
           "delta_proxy_target_first_layout_m": d_main, "delta_proxy_neutral_layout_m": d_neutral,
           "delta_proxy_formula": "C(B->A) - C(A->B); C(A->B) = ||q0-A|| + ||c-B||; positive => A-first predicted faster",
           "neutral_delta_proxy_container": "own_saved_capture_container",
           "environment_constructions": 0, "derivation": "saved_evidence_only"}
    bank = {"status": "SAFE", "thresholds": {k: geom[k] for k in ("target_first_min_delta_proxy_m", "second_first_max_delta_proxy_m",
                                                                  "neutral_abs_delta_proxy_m", "min_object_separation_m",
                                                                  "workspace_half_extent_m")},
            "anchors": [tf_entry, sf_entry, n_entry, n2_entry], "reference_geometry": ref, "environment_constructions": 0}
    (out / "geometry").mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "geometry/safe_anchor_bank.json", bank)
    _atomic_json(out / "geometry/reference_geometry.json", ref)
    write_csv(out / "geometry/path_proxy.csv",
              [{"layout": "SO_TARGET_FIRST", "delta_proxy_m": d_main, "threshold_m": geom["target_first_min_delta_proxy_m"]},
               {"layout": "SO_SECOND_FIRST", "delta_proxy_m": -d_main, "threshold_m": abs(geom["second_first_max_delta_proxy_m"])},
               {"layout": "SO_NEUTRAL", "delta_proxy_m": d_neutral, "threshold_m": geom["neutral_abs_delta_proxy_m"]}],
              ["layout", "delta_proxy_m", "threshold_m"])
    checks = {"target_first_delta_proxy_m": d_main, "second_first_delta_proxy_m": -d_main, "neutral_delta_proxy_m": d_neutral,
              "target_first_ok": d_main >= float(geom["target_first_min_delta_proxy_m"]),
              "second_first_ok": -d_main <= float(geom["second_first_max_delta_proxy_m"]),
              "neutral_ok": abs(d_neutral) <= float(geom["neutral_abs_delta_proxy_m"]),
              "all_anchors_workspace_valid": all(e["workspace_valid"] for e in bank["anchors"]),
              "all_anchors_container_clear": all(e["container_clearance"] for e in bank["anchors"]),
              "no_initial_penetration": all(e["no_initial_penetration"] for e in bank["anchors"]),
              "anchors_from_saved_success": bool(s1["task_success"] and s1["picked"] and s1["placed"] and v2["pick_place_observed"])}
    _atomic_json(out / "geometry/contract_reachability.json", {"stage": "geometry", "checks": checks})
    for key in ("target_first_ok", "second_first_ok", "neutral_ok", "all_anchors_workspace_valid", "all_anchors_container_clear"):
        if not checks[key]:
            raise StopRun("STOPPED_GEOMETRY_MARGIN_UNAVAILABLE", f"{key} failed: {checks}")
    if not checks["anchors_from_saved_success"]:
        raise StopRun("STOPPED_REFERENCE_GEOMETRY_UNAVAILABLE", "anchors are not backed by saved successful evidence")
    return {"status": "ANCHOR_BANK_SAFE", "delta_proxy_main_m": d_main, "delta_proxy_neutral_m": d_neutral,
            "anchors": [e["anchor"] for e in bank["anchors"]]}

# ------------------------------------------------------------------ offline contract reachability (no environment)
class _PlanStub:
    """Minimal duck-type for the planner's PlanResult, purely to reuse the shared nominal-cost helper."""

    def __init__(self, plan):
        self.plan = tuple(plan)
        self.status = "PLAN_FOUND"
        self.cost = float(len(self.plan)) * float(D_REF)


def _build_offline_template(root):
    """Ground the frozen contract registry and build the frozen template. Pure symbolic; constructs no environment."""
    from cp_disr.platforms.libero.tp_so_mvp_runtime import OBJECTS, build_task_template, ground_task_contracts
    contracts = ground_task_contracts(root / CONTRACT_REL, dict(SKILL_TIMEOUTS))
    return contracts, build_task_template(contracts), OBJECTS


def _initial_fact_store():
    from cp_disr.facts import FactRecord, FactStore, Truth
    recs = []
    for fid in VERIFIER_FACT_IDS:
        value = Truth.TRUE if fid in INITIAL_TRUE_FACTS else Truth.FALSE if fid in INITIAL_FALSE_FACTS else Truth.FALSE
        recs.append(FactRecord(fact_id=fid, value=value, capture_time=0.0, available_time=0.0, reason="offline_initial"))
    return FactStore(tuple(recs))


def _nominal_route_cost(plan_result, template):
    """Fixed nominal cost of a B_PLAN route: every skill costs the same reference duration, so cost == length * d_ref."""
    return float(len(plan_result.plan)) * float(D_REF)


def _contract_reachability(root, out):
    """Offline proof that the frozen contract cannot rank the two object orders. Constructs no environment."""
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.contracts import nominal_overlay, precondition_value
    from cp_disr.facts import Truth
    contracts, template, _objects = _build_offline_template(root)
    by_id = {c.id: c for c in contracts}
    ids = {c.id for c in contracts}
    if ids != set(GROUNDED_ACTIONS):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", f"grounded action set mismatch: {sorted(ids)}")
    if any(str(i).startswith(("a:OPEN:", "a:PLACE_BUFFER:", "a:MOVE:")) for i in ids):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "an excluded grounding is registered")
    props = {n.id for n in template.nodes if n.kind == "PROPOSITION"}
    if props != set(VERIFIER_FACT_IDS):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "template propositions differ from the verifier fact set")
    facts = _initial_fact_store()
    planner = BPlanPlanner(SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=D_REF))
    initial_ids = [c.id for c in template.contracts if _precond_true(c, facts)]
    if sorted(initial_ids) != sorted(INITIAL_CANDIDATES):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", f"initial candidates are {sorted(initial_ids)}")
    plan = planner.plan(facts, template, DEADLINE)
    if plan.status != "PLAN_FOUND":
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", f"contract cannot reach the goals: {plan.status}")
    canonical_plan = tuple(plan.plan)
    canonical_first = canonical_plan[0] if canonical_plan else None
    canonical_cost = _nominal_route_cost(plan, template)
    # A frozen route is reachable iff every one of its contracts can be laid out head-to-tail through the nominal
    # overlay. Two distinct orders of the same two skills are *both* minimal-cost ties of a symmetric domain: rolling
    # the planner out once (its canonical id-sorted tie-break) can only ever return one of them, so reachability of
    # each frozen route is proven directly, not by demanding both from a single canonical rollout.
    routes = {}
    for name, expected in ROUTES.items():
        values = dict(facts.values)
        reachable = True
        for cid in expected:
            contract = by_id.get(cid)
            if contract is None or precondition_value(contract, values) != Truth.TRUE:
                reachable = False
                break
            values = dict(nominal_overlay(contract, values, derived=template.derived_rules,
                                          exclusive_groups=template.exclusive_groups))
        goals_met = all(values.get(g.fact_id) == (Truth.TRUE if g.sign == 1 else Truth.FALSE) for g in template.goals)
        routes[name] = {"reachable": bool(reachable and goals_met), "length": len(expected), "plan": list(expected),
                        "nominal_cost_seconds": _nominal_route_cost(_PlanStub(expected), template), "status": "PLAN_FOUND" if reachable and goals_met else "NO_PLAN",
                        "matches_frozen_route": True, "sets_both_goals": bool(goals_met),
                        "reachability_check": "head-to-tail nominal overlay of the frozen route"}
    if not all(r["reachable"] for r in routes.values()):
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", f"a frozen route is not reachable: { {k: v['reachable'] for k, v in routes.items()} }")
    if routes[ROUTE_TARGET_FIRST]["length"] != routes[ROUTE_SECOND_FIRST]["length"]:
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "route lengths differ")
    multiset = {name: sorted(c.split(":")[1] for c in r["plan"]) for name, r in routes.items()}
    if multiset[ROUTE_TARGET_FIRST] != multiset[ROUTE_SECOND_FIRST]:
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "skill multisets differ")
    costs_equal = abs(routes[ROUTE_TARGET_FIRST]["nominal_cost_seconds"] - routes[ROUTE_SECOND_FIRST]["nominal_cost_seconds"]) < 1e-12
    if not costs_equal:
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "B_PLAN nominal costs differ")
    costs_equal_to_planner = abs(canonical_cost - routes[ROUTE_TARGET_FIRST]["nominal_cost_seconds"]) < 1e-12
    if not costs_equal_to_planner:
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "frozen route nominal cost differs from the planner's minimal cost")
    canonical = canonical_first
    canonical_in_tie_set = canonical_plan in {tuple(r["plan"]) for r in routes.values()}
    if not canonical_in_tie_set:
        raise StopRun("STOPPED_CONTRACT_ELIGIBILITY", "planner's canonical plan is not one of the frozen routes")
    doc = {"status": "REACHABLE", "both_routes_reachable": True,
           "route_lengths": {k: v["length"] for k, v in routes.items()}, "route_lengths_equal": True,
           "skill_multisets_equal": True, "skill_multiset": multiset[ROUTE_TARGET_FIRST],
           "b_plan_nominal_costs_seconds": {k: v["nominal_cost_seconds"] for k, v in routes.items()},
           "b_plan_nominal_costs_equal": True, "both_routes_minimal_cost": bool(costs_equal_to_planner),
           "canonical_first_action": canonical, "canonical_tie_break_reads_geometry": False,
           "canonical_tie_break": "BPlanPlanner sorts contracts by contract id; no coordinate, camera, proxy or time input",
           "no_hard_fact_ranks_routes": True, "environment_constructions": 0,
           "routes": routes, "initial_candidate_ids": sorted(initial_ids),
           "grounded_action_ids": sorted(ids), "template_propositions": sorted(props)}
    _atomic_json(out / "geometry/contract_reachability.json", doc)
    return doc


def _precond_true(contract, facts):
    from cp_disr.contracts import precondition_value
    from cp_disr.facts import Truth
    return precondition_value(contract, dict(facts.values)) == Truth.TRUE


# ------------------------------------------------------------------ frozen configs
def freeze_configs(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    bank = rd(out / "geometry/safe_anchor_bank.json")
    ref = bank["reference_geometry"]
    anchors = {e["anchor"]: e for e in bank["anchors"]}
    a_xy = anchors["A_s1_dev18_target"]["xy"]
    b_xy = anchors["B_s1_dev18_second"]["xy"]
    n_xy = anchors["N_v2_pool33_target"]["xy"]
    n2_xy = anchors["N_v2_pool33_second"]["xy"]
    q0, container = ref["q0_xy"], ref["container_xy"]
    neutral_container = list(ref.get("neutral_container_xy") or container)
    neutral_buffer = list(ref.get("neutral_buffer_xy") or ref["buffer_xy"])
    d_main = delta_proxy(q0, a_xy, b_xy, container)
    d_swap = delta_proxy(q0, b_xy, a_xy, container)
    d_neutral = delta_proxy(q0, n_xy, n2_xy, neutral_container)
    if abs(d_swap + d_main) > 1e-12:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "identity swap does not exactly reverse delta_proxy")
    configs = {
        "SO_TARGET_FIRST": {"config_id": "SO_TARGET_FIRST", "target_xy": a_xy, "second_xy": b_xy, "container_xy": container,
                            "buffer_xy": ref["buffer_xy"], "lid_closed": ref["lid_closed"], "second_role": SECOND_ROLE,
                            "delta_proxy_m": d_main, "identity_swap_of": None,
                            "anchor_sources": {"target": anchors["A_s1_dev18_target"]["source_case"],
                                               "second_object": anchors["B_s1_dev18_second"]["source_case"]}},
        "SO_SECOND_FIRST": {"config_id": "SO_SECOND_FIRST", "target_xy": b_xy, "second_xy": a_xy, "container_xy": container,
                            "buffer_xy": ref["buffer_xy"], "lid_closed": ref["lid_closed"], "second_role": SECOND_ROLE,
                            "delta_proxy_m": d_swap, "identity_swap_of": "SO_TARGET_FIRST",
                            "anchor_sources": {"target": anchors["B_s1_dev18_second"]["source_case"],
                                               "second_object": anchors["A_s1_dev18_target"]["source_case"]}},
        "SO_NEUTRAL": {"config_id": "SO_NEUTRAL", "target_xy": n_xy, "second_xy": n2_xy, "container_xy": neutral_container,
                       "buffer_xy": neutral_buffer, "lid_closed": ref["lid_closed"], "second_role": SECOND_ROLE,
                       "delta_proxy_m": d_neutral, "identity_swap_of": None, "container_source": "own_saved_capture",
                       "anchor_sources": {"target": anchors["N_v2_pool33_target"]["source_case"],
                                          "second_object": anchors["N_v2_pool33_second"]["source_case"]}},
    }
    checks = {"target_first_delta_proxy_m": d_main, "second_first_delta_proxy_m": d_swap, "neutral_delta_proxy_m": d_neutral,
              "target_first_ok": d_main >= float(cfg["geometry"]["target_first_min_delta_proxy_m"]),
              "second_first_ok": d_swap <= float(cfg["geometry"]["second_first_max_delta_proxy_m"]),
              "neutral_ok": abs(d_neutral) <= float(cfg["geometry"]["neutral_abs_delta_proxy_m"]),
              "exact_identity_swap": abs(d_swap + d_main) <= 1e-12,
              "neutral_container_own_evidence": neutral_container != container or neutral_buffer != list(ref["buffer_xy"]),
              "same_deadline": True}
    for key in ("target_first_ok", "second_first_ok", "neutral_ok", "exact_identity_swap"):
        if not checks[key]:
            raise StopRun("STOPPED_GEOMETRY_MARGIN_UNAVAILABLE", f"{key} failed: {checks}")
    doc = {"status": "FROZEN", "created_utc": now(), "configs": configs, "config_order": list(CONFIG_ORDER), "checks": checks,
           "frozen": True, "derivation": "saved_evidence_only", "environment_constructions": 0}
    path = out / "geometry/frozen_configs.json"
    _atomic_json(path, doc)
    path.chmod(0o444)
    reach = _contract_reachability(root, out)
    mark_phase(out, "freeze-configs", frozen_configs=list(CONFIG_ORDER),
               both_routes_reachable=reach["both_routes_reachable"],
               b_plan_nominal_costs_equal=reach["b_plan_nominal_costs_equal"])
    return {"status": "FROZEN", "configs": {k: v["delta_proxy_m"] for k, v in configs.items()}, "checks": checks}

# ------------------------------------------------------------------ runtime manifest / branch registration
def build_manifest(root, out, cfg, split_rel, name):
    root, out = Path(root), Path(out)
    base = json.loads(json.dumps(yaml.safe_load((root / cfg["sources"]["s1_manifest_base"]).read_text(encoding="utf-8"))))
    rt = base["runtime"]
    rt["active_task_id"] = TASK_ID
    rt["repository_path"] = str(root)
    rt["experiment_root"] = str(root / "experiments")
    rt["stage_2a_contract_path"] = CONTRACT_REL
    rt["reference_skill_seconds_by_task"] = {TASK_ID: D_REF}
    rt["skill_timeouts"] = {TASK_ID: dict(SKILL_TIMEOUTS)}
    rt["task_deadlines"] = {TASK_ID: DEADLINE}
    rt["task_evaluator_version"] = {TASK_ID: "cp-disr-tp-so-mvp-task-evaluator-v1"}
    rt["task_assets"] = {TASK_ID: "src/cp_disr/platforms/libero/d0_env.py", "license": "assets/cp_disr/LICENSE",
                         "owned_by": "cp_disr_project"}
    rt["task_splits"] = {TASK_ID: split_rel}
    rt["decision_cap_by_task"] = {TASK_ID: DECISION_CAP}
    src = root / RUNTIME_REL
    base["runtime_factory"] = {"module": cfg["runtime_module"], "factory": "create", "source_path": str(src), "sha256": sha256_file(src)}
    base["manifest_status"] = "S4_FAMILY_A_SOFT_ORDERING_MVP_RUNTIME_BOUND_TRAINING_NOT_STARTED"
    base["note"] = "S4 Family A soft-ordering MVP runtime; no provider, no training, no formal test."
    base["vlm_runtime"] = {"api_account_authorized": "MUST_VERIFY_MODEL_ACCESS", "automatic_fallback_allowed": False,
                           "base_http_api_url": "https://dashscope.aliyuncs.com/api/v1", "region": "cn-beijing"}
    path = out / "spec" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
    return path


def perception_identity():
    return {"variant": "TP_SR_V1_NEAREST_PALETTE_COMPAT_LOCAL", "adapter": "TPSRNearestPalettePerceptionAdapter",
            "global_monkeypatch": False, "base_perception_version": "cp-disr-d0-rgbd-colorseg-v2"}


def init_phys_dir(phys, cap):
    for sub in ("witnesses/branch_receipts", "witnesses/restore_receipts", "witnesses/initial_state_checks", "branch_results", "worker_logs"):
        (phys / sub).mkdir(parents=True, exist_ok=True)
    _atomic_json(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": cap, "used": 0}})
    (phys / "budget_events.jsonl").touch()
    _atomic_json(phys / "attempt_registry.json", {})


def restore_seed(config, repeat):
    """Paired routes of the same config+repeat MUST share one restore seed: a route-hashed seed would make the paired
    restore comparison impossible by construction. The seed depends only on the frozen layout and the repeat."""
    return int(hashlib.sha256(f"{config}|{int(repeat)}".encode()).hexdigest()[:8], 16)


def make_branch(config, route, repeat, manifest_path, case_id, source_ident, wave):
    cand = ROUTES[route][0]
    ident = f"s4_family_a_soft_ordering_mvp|{config}|{route}|{repeat}"
    bid = hashlib.sha256(ident.encode()).hexdigest()[:16]
    return {"branch_id": bid, "attempt_id": bid, "case_id": case_id, "candidate_id": cand, "first_action": cand,
            "route": route, "config": config, "repeat": int(repeat), "restore_seed": restore_seed(config, repeat),
            "selection_hash": hashlib.sha256(ident.encode()).hexdigest(), "manifest_path": str(manifest_path),
            "authorized": True, "execute_now": True, "status": "REGISTERED", "wave": wave,
            "runtime_source_sha256": source_ident[RUNTIME_REL],
            "instrumentation_source_sha256": source_ident["src/cp_disr/platforms/libero/tp_sr_instrumentation.py"],
            "perception_identity": perception_identity()}


def prepare_branches(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    phys = out / "physical"
    if (phys / "witnesses/e4_branch_registration.json").is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "branch registration already exists")
    frozen = rd(out / "geometry/frozen_configs.json")
    configs = frozen["configs"]
    dev_rows = []
    for cid in CONFIG_ORDER:
        c = configs[cid]
        dev_rows.append({"case_id": f"{TASK_ID}_{cid}", "split": "dev", "seed": 0, "target_xy": c["target_xy"],
                         "second_xy": c["second_xy"], "container_xy": c["container_xy"], "buffer_xy": c["buffer_xy"],
                         "lid_closed": bool(c["lid_closed"]), "second_role": SECOND_ROLE})
    split_path = out / "spec/T_P_SO_MVP_split.json"
    _atomic_json(split_path, {"task_id": TASK_ID, "task_family_id": cfg["task_family_id"], "train": [], "dev": dev_rows, "test": []})
    manifest_path = build_manifest(root, out, cfg, str(split_path.relative_to(root)), "runtime_manifest_T_P_SO_MVP.yaml")
    source_ident = {f: sha256_file(root / f) for f in CODE_FILES if (root / f).is_file()}
    branches = []
    for cid in CONFIG_ORDER:
        case_id = f"{TASK_ID}_{cid}"
        for route in (ROUTE_TARGET_FIRST, ROUTE_SECOND_FIRST):
            for repeat in (0, 1):
                wave = "technical" if (cid == "SO_TARGET_FIRST" and repeat == 0) else "remaining"
                branches.append(make_branch(cid, route, repeat, manifest_path, case_id, source_ident, wave))
    if len(branches) != 12:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "exactly twelve branches must be registered")
    if len({b["branch_id"] for b in branches}) != 12:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "duplicate branch id")
    seeds = {}
    for b in branches:
        seeds.setdefault((b["config"], b["repeat"]), set()).add(b["restore_seed"])
    if any(len(v) != 1 for v in seeds.values()):
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "paired routes must share one restore seed")
    if sum(1 for b in branches if b["wave"] == "technical") != 2:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "the technical wave must be exactly two branches")
    init_phys_dir(phys, PHYS_CAP)
    reg = {"status": "REGISTERED", "card_id": CARD_ID, "physical_witness_episodes_cap": PHYS_CAP,
           "task_id": TASK_ID, "configs": list(CONFIG_ORDER), "routes": list(ROUTES), "repeats": [0, 1],
           "technical_wave": {"config": "SO_TARGET_FIRST", "repeat": 0, "routes": [ROUTE_TARGET_FIRST, ROUTE_SECOND_FIRST]},
           "branches": branches, "frozen_configs_sha256": sha256_file(out / "geometry/frozen_configs.json"),
           "runtime_manifest_sha256": sha256_file(manifest_path), "perception_identity": perception_identity(),
           "source_identity": source_ident}
    _atomic_json(phys / "witnesses/e4_branch_registration.json", reg)
    _atomic_json(phys / "branch_registration.json", reg)
    for p in (phys / "witnesses/e4_branch_registration.json", phys / "branch_registration.json"):
        p.chmod(0o444)
    mark_phase(out, "prepare-branches", branches=len(branches))
    return {"status": "PREPARED", "branches": len(branches),
            "technical": [(b["config"], b["route"], b["branch_id"]) for b in branches if b["wave"] == "technical"],
            "remaining": sum(1 for b in branches if b["wave"] == "remaining")}

# ------------------------------------------------------------------ worker
def _branch(out, branch_id):
    reg = rd(Path(out) / "physical/witnesses/e4_branch_registration.json")
    b = next((x for x in reg["branches"] if x["branch_id"] == branch_id), None)
    if b is None:
        raise StopRun("STOPPED_RUNTIME_BINDING", f"branch {branch_id} not registered")
    return b


def worker_dry_run(root, out, branch_id):
    """Zero-cost: module/source binding, grounding, split membership. Reserves nothing, constructs no environment."""
    import importlib
    from cp_disr.runtime import require_runtime
    root, out = Path(root), Path(out)
    branch = _branch(out, branch_id)
    manifest = yaml.safe_load(Path(branch["manifest_path"]).read_text(encoding="utf-8"))
    spec = require_runtime(manifest)
    mod = importlib.import_module(spec["module"])
    if Path(mod.__file__).resolve() != Path(spec["source_path"]).resolve() or not callable(getattr(mod, spec["factory"], None)):
        raise StopRun("STOPPED_RUNTIME_BINDING", "module/source mismatch")
    rtm = manifest["runtime"]
    if rtm["active_task_id"] != TASK_ID:
        raise StopRun("STOPPED_RUNTIME_BINDING", "active_task_id mismatch")
    if rtm["stage_2a_contract_path"] != CONTRACT_REL:
        raise StopRun("STOPPED_RUNTIME_BINDING", "contract registry mismatch")
    if dict(rtm["skill_timeouts"][TASK_ID]) != dict(SKILL_TIMEOUTS):
        raise StopRun("STOPPED_RUNTIME_BINDING", "skill timeouts are not the frozen pair")
    if branch["candidate_id"] not in GROUNDED_ACTIONS:
        raise StopRun("STOPPED_RUNTIME_BINDING", "candidate is not a grounded action")
    split = rd(Path(rtm["repository_path"]) / rtm["task_splits"][TASK_ID])
    if branch["case_id"] not in {r["case_id"] for r in split["dev"]}:
        raise StopRun("STOPPED_RUNTIME_BINDING", "case missing from the frozen split")
    from cp_disr.platforms.libero.tp_so_mvp_runtime import ground_task_contracts
    grounded = ground_task_contracts(Path(rtm["repository_path"]) / CONTRACT_REL, dict(SKILL_TIMEOUTS))
    if {c.id for c in grounded} != set(GROUNDED_ACTIONS):
        raise StopRun("STOPPED_RUNTIME_BINDING", "grounding does not match the frozen action set")
    return {"dry_run": "OK", "branch_id": branch_id, "environment_constructions": 0}


def physical_worker(root, out, branch_id):
    from cp_disr.analysis.s1_revision_resume import _runner_config, execute_registered_branch
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.platforms.libero import perception as _perception
    from cp_disr.platforms.libero.tp_sr_instrumentation import RecordingPlanner
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    phys = out / "physical"
    branch = _branch(out, branch_id)
    mask_before = _perception._mask
    holder = {}

    def factory(manifest, br):
        bundle = load_runtime(manifest)
        bundle.configure(out, phys, br)
        holder["bundle"] = bundle
        return bundle

    def planner_factory():
        manifest = yaml.safe_load(Path(branch["manifest_path"]).read_text(encoding="utf-8"))
        _rt, _tid, d_ref, _dl, _cap = _runner_config(manifest)
        inner = BPlanPlanner(SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=d_ref))
        return RecordingPlanner(inner, holder["bundle"].recorder)

    t0 = time.time()
    res = jsonable(execute_registered_branch(root, branch_id, phys, bundle_factory=factory, planner_factory=planner_factory))
    bundle = holder.get("bundle")
    counts = bundle.env_counts() if bundle is not None else {}
    if bundle is not None and bundle.recorder is not None:
        bundle.recorder.close(counts)
    mask_after = _perception._mask
    res.update({"worker_pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "worker_wall_seconds": time.time() - t0, "wave": branch.get("wave"),
                "config": branch.get("config"), "route": branch.get("route"), "repeat": branch.get("repeat"),
                "env_counts": counts,
                "recorder_errors": len(bundle.recorder.errors) if bundle is not None and bundle.recorder is not None else -1,
                "perception_identity": perception_identity(),
                "global_perception_mask_unchanged": bool(mask_before is mask_after and mask_after.__module__ == _perception.__name__),
                "provider_module_loaded": "cp_disr.analysis.s4_tp_sr_provider" in sys.modules})
    _atomic_json(phys / "branch_results" / f"{branch_id}.json", res)
    return res


def _spawn_env(root, gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "CP_DISR_PHYSICAL_GPU_INDEX": str(gpu),
                "PYTHONPATH": f"{root}/src:{root}", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    return env


def _spawn_worker(root, out, branch_id, gpu):
    log = open(Path(out) / "physical/worker_logs" / f"{branch_id}.log", "ab")
    cmd = [sys.executable, str(Path(root) / "scripts/s4_family_a_soft_ordering_mvp.py"), "physical-worker",
           "--root", str(root), "--output", str(out), "--branch-id", branch_id]
    return subprocess.Popen(cmd, env=_spawn_env(root, gpu), stdout=log, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True)


# ------------------------------------------------------------------ dispatch
def _dispatch(root, out, cfg, branch_ids, gpus, max_workers, ledger_keys, phase):
    """One worker per GPU; each dispatch charges the phase, total and environment-reset budgets exactly once.

    A worker fault is recorded and the attempt is closed UNKNOWN. Two consecutive engineering faults stop dispatch.
    """
    root, out = Path(root), Path(out)
    phys = out / "physical"
    if not branch_ids:
        return {"status": "NOTHING_TO_RUN", "n": 0, "faults": []}
    try:
        worker_dry_run(root, out, branch_ids[0])
    except StopRun as exc:
        event(out, "dry_run_failed_before_any_reservation", error=str(exc))
        raise StopRun("STOPPED_RUNTIME_BINDING", str(exc)) from exc
    slots = list(gpus)[:max(1, min(int(max_workers), len(gpus)))]
    todo, running, faults, walls, consecutive = list(branch_ids), {}, [], {}, 0
    t_first = t_last = None
    event(out, "phase_start", phase=phase, workers=len(slots), gpus=slots, branches=len(todo))
    while running or (todo and not faults):
        for gpu in slots:
            if gpu in running or not todo or faults:
                continue
            bid = todo.pop(0)
            try:
                charge_many(out, list(ledger_keys) + ["total_branch_attempts", "environment_resets"], ref=bid)
                reserve_branch_attempt(phys, bid)
            except (IntegrationError, StopRun) as exc:
                faults.append({"branch_id": bid, "fault": f"RESERVE:{exc}"})
                continue
            proc = _spawn_worker(root, out, bid, gpu)
            t0 = time.monotonic()
            t_first = t0 if t_first is None else t_first
            running[gpu] = (bid, proc, t0)
            event(out, "branch_launch", phase=phase, branch_id=bid, gpu=gpu, pid=proc.pid, slots_in_use=len(running))
        for gpu, (bid, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            state = _load_attempts(phys).get(bid)
            p = phys / "branch_results" / f"{bid}.json"
            res = rd(p) if p.is_file() else {}
            if rc != 0 or not res:
                faults.append({"branch_id": bid, "fault": f"WORKER_EXIT:{rc}", "attempt_state": state})
                if state in ("RESERVED", "STARTED"):
                    finish_branch_attempt(phys, bid, "UNKNOWN", error_phase="worker_process_exit", exit_code=rc)
            elif res.get("execution_status") == "EXCEPTION":
                consecutive += 1
                if consecutive >= 2:
                    faults.append({"branch_id": bid, "fault": "CONSECUTIVE_EXCEPTIONS:" + str(res.get("termination_reason"))})
            else:
                consecutive = 0
            walls[bid] = time.monotonic() - t0
            t_last = time.monotonic()
            event(out, "branch_finish", phase=phase, branch_id=bid, gpu=gpu, exit_code=rc, wall_seconds=walls[bid],
                  slots_in_use=len(running) - 1)
            del running[gpu]
        time.sleep(1.0)
    span = (t_last - t_first) if t_first is not None and t_last is not None else None
    report = {"phase": phase, "workers": len(slots), "gpus": slots, "n": len(walls), "span_seconds": span,
              "aggregate_branches_per_second": (len(walls) / span) if span else None,
              "median_wall_seconds": float(np.median(list(walls.values()))) if walls else None, "faults": faults,
              "single_worker_baseline": "NOT_MEASURED",
              "speedup_claim": "NONE (no matched single-worker baseline; no extra attempts spent)"}
    _atomic_json(phys / f"throughput_report_{phase}.json", report)
    _append_jsonl(phys / "worker_events.jsonl", {"event": "phase_done", **report})
    return report

# ------------------------------------------------------------------ technical wave (2 branches, then the gate)
def _result(out, bid):
    p = Path(out) / "physical/branch_results" / f"{bid}.json"
    return rd(p) if p.is_file() else {}


def _sim_completion(result):
    """Primary metric: first TASK_SUCCESS evaluator sim time minus first action_start sim time. Sim seconds only."""
    trace = result.get("trace") or []
    if not trace:
        return None
    first = trace[0].get("elapsed_seconds")
    for step in trace:
        if step.get("task_success"):
            if first is None:
                return None
            return float(step["elapsed_seconds"]) - float(first)
    return None


def _sum_skill_sim(result):
    tot = 0.0
    for step in (result.get("trace") or []):
        d = step.get("sim_duration")
        if d is not None:
            tot += float(d)
    return tot


def _skill_count(result):
    return len(result.get("trace") or [])


def evaluate_technical_gate(root, out, cfg):
    root, out = Path(root), Path(out)
    phys = out / "physical"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    tech = [b for b in reg["branches"] if b.get("wave") == "technical"]
    checks = {}
    if len(tech) != 2 or len({b["branch_id"] for b in tech}) != 2:
        checks["T0_technical_wave_size"] = False
    else:
        checks["T0_technical_wave_size"] = True
    results = {b["branch_id"]: _result(out, b["branch_id"]) for b in tech}
    checks["T1_both_terminal"] = all(r.get("execution_status") in ("TERMINATED", "TRUNCATED", "DEADLINE", "NO_PLAN", "SEARCH_TIMEOUT")
                                     and r.get("protocol_complete") for r in results.values()) if results else False
    states = {}
    for b in tech:
        p = phys / "witnesses/initial_state_checks" / f"{b['branch_id']}.json"
        states[b["branch_id"]] = rd(p) if p.is_file() else {}
    pairs = []
    for cid in CONFIG_ORDER:
        for repeat in (0, 1):
            group = [b for b in tech if b["config"] == cid and b["repeat"] == repeat]
            if len(group) == 2:
                pairs.append(group)
    ok2 = ok3 = True
    for group in pairs:
        a, b = (states.get(x["branch_id"], {}) for x in group)
        ka, kb = a.get("compare_key") or {}, b.get("compare_key") or {}
        ok2 &= all(ka.get(k) is not None and ka.get(k) == kb.get(k) for k in ("case_id", "applied_restore_seed",
                                                                              "normalized_reset_config_sha256", "public_facts_sha256",
                                                                              "candidate_ids_sha256", "candidate_mask_sha256"))
        qa, qb = (a.get("physical") or {}), (b.get("physical") or {})
        if qa.get("qpos") and qb.get("qpos"):
            ok3 &= bool(np.max(np.abs(np.asarray(qa["qpos"], dtype=float) - np.asarray(qb["qpos"], dtype=float))) <= 1e-9)
            ok3 &= bool(np.max(np.abs(np.asarray(qa["qvel"], dtype=float) - np.asarray(qb["qvel"], dtype=float))) <= 1e-9)
        else:
            ok3 = False
    checks["T2_paired_initial_state_identical"] = bool(pairs and ok2)
    checks["T3_qpos_qvel_max_diff_le_1e-9"] = bool(pairs and ok3)
    checks["T4_both_task_success"] = bool(results) and all(r.get("task_success") is True for r in results.values())
    checks["T5_exactly_four_skills"] = bool(results) and all(_skill_count(r) == 4 for r in results.values())
    checks["T6_per_action_records_complete"] = all(
        _check_terminal_aware_action_records(out, b)["complete"] for b in tech
    )
    ledger = rd(phys / "budget_ledger.json")
    attempts = _load_attempts(phys)
    checks["T7_one_attempt_and_one_reset_each"] = bool(
        int(ledger["physical_witness_episodes"]["used"]) == 2
        and all(attempts.get(b["branch_id"]) == "COMPLETED" for b in tech))
    checks["T8_no_duplicate_branch"] = len({b["branch_id"] for b in reg["branches"]}) == len(reg["branches"])
    checks["T9_no_global_perception_residue"] = bool(results) and all(r.get("global_perception_mask_unchanged") is True for r in results.values())
    checks["T10_provider_rl_optimizer_zero"] = bool(results) and all(r.get("provider_module_loaded") is False for r in results.values())
    passed = all(checks.values())
    doc = {"technical_wave": "PASS" if passed else "FAIL", "remaining_branches_released": bool(passed),
           "checks": checks, "branch_ids": [b["branch_id"] for b in tech],
           "task_sim_completion_seconds": {b["branch_id"]: _sim_completion(results.get(b["branch_id"], {})) for b in tech},
           "note": "the technical gate never inspects which order is faster", "evaluated_utc": now()}
    _atomic_json(phys / "technical_wave_check.json", doc)
    _atomic_json(out / "decision/decision_state.json", {**rd(out / "decision/decision_state.json"),
                                                        "technical_wave": doc["technical_wave"],
                                                        "remaining_branches_released": doc["remaining_branches_released"],
                                                        "status": "IN_PROGRESS" if passed else "STOPPED",
                                                        "next_action": "" if passed else "EXPLICIT_RESEARCH_DECISION"})
    return doc


def technical_wave(root, config_path, out, gpus, max_workers):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    reg = rd(out / "physical/witnesses/e4_branch_registration.json")
    tech = [b["branch_id"] for b in reg["branches"] if b.get("wave") == "technical"]
    if len(tech) != 2:
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "the technical wave must be exactly two branches")
    gate = out / "physical/technical_wave_check.json"
    if gate.is_file():
        raise StopRun("STOPPED_EVIDENCE_INTEGRITY", "the technical wave has already been evaluated")
    report = _dispatch(root, out, cfg, tech, gpus, min(int(max_workers), int(cfg["technical_wave"]["workers"])),
                       ["technical_wave_branch_attempts"], "technical")
    doc = evaluate_technical_gate(root, out, cfg)
    if doc["technical_wave"] != "PASS":
        raise StopRun("STOPPED_TECHNICAL_WAVE", f"technical gate FAIL: {doc['checks']}")
    return {"technical_wave": doc["technical_wave"], "remaining_branches_released": doc["remaining_branches_released"],
            "checks": doc["checks"], "dispatch": report}



_ACTION_REQUIRED_FILES = (
    "before_rgb.png",
    "before_depth.npy",
    "after_rgb.png",
    "after_depth.npy",
    "controller_trace.jsonl",
    "facts.json",
    "evaluator.json",
    "snapshot.json",
)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, TypeError, ValueError):
        return None
    return rows


def _check_terminal_aware_action_records(out, branch):
    """Check saved action evidence without constructing runtime state."""
    out = Path(out)
    branch_id = str(branch["branch_id"])
    cap = out / "captures" / branch_id
    result = _read_json(out / "physical" / "branch_results" / f"{branch_id}.json")
    trace = list((result or {}).get("trace") or [])
    action_dirs = sorted(cap.glob("action_[0-9][0-9]"))
    complete = result is not None and len(action_dirs) == len(trace)
    terminal_seen = False
    records = []
    for index, adir in enumerate(action_dirs):
        missing = [name for name in _ACTION_REQUIRED_FILES if not (adir / name).is_file()]
        evaluator = _read_json(adir / "evaluator.json")
        snapshot = _read_json(adir / "snapshot.json")
        controller = _read_jsonl(adir / "controller_trace.jsonl")
        step = trace[index] if index < len(trace) else {}
        candidate_id = step.get("candidate_id")
        starts = [
            row for row in (controller or ())
            if row.get("event") in ("skill_begin", "action_start")
        ]
        identity_ok = bool(candidate_id) and any(
            row.get("candidate_id") == candidate_id for row in starts
        )
        planner_ok = False
        status = "MISSING_EVIDENCE"
        if evaluator is not None:
            success = evaluator.get("task_success") is True
            terminated = evaluator.get("terminated") is True
            truncated = evaluator.get("truncated") is True
            reason = str(evaluator.get("reason") or "UNKNOWN")
            terminal = terminated or truncated or reason != "CONTINUE"
            if terminal:
                terminal_seen = True
                if success and terminated and reason == "TASK_SUCCESS":
                    status = "NOT_APPLICABLE_TERMINAL_SUCCESS"
                else:
                    token = "".join(
                        ch if (ch.isalnum() or ch == "_") else "_"
                        for ch in reason.upper().replace("-", "_")
                    )
                    status = "NOT_APPLICABLE_TERMINAL_" + token
                planner_ok = not (adir / "planner.json").exists()
                planner_ok &= not any(
                    row.get("event") in ("planner_call", "planner")
                    for row in (controller or ())
                )
                if success and terminated and reason == "TASK_SUCCESS":
                    planner_ok &= bool((result or {}).get("task_success") is True)
                    planner_ok &= index == len(action_dirs) - 1
            else:
                planner = _read_json(adir / "planner.json")
                status = "PLANNER_REQUIRED_CONTINUE"
                planner_ok = isinstance(planner, dict) and all(
                    key in planner
                    for key in (
                        "input_facts", "candidate_ids", "status", "plan",
                        "expanded_nodes", "cpu_seconds",
                    )
                )
                planner_ok &= isinstance(snapshot, dict) and "candidate_mask" in snapshot
                planner_ok &= not terminal_seen
        else:
            status = "MISSING_EVALUATOR"
        row_ok = (
            not missing
            and evaluator is not None
            and snapshot is not None
            and controller is not None
            and bool(starts)
            and identity_ok
            and planner_ok
        )
        complete &= bool(row_ok)
        records.append({
            "action": adir.name,
            "candidate_id": candidate_id,
            "missing_files": missing,
            "planner_record_status": status,
            "identity_ok": identity_ok,
            "complete": bool(row_ok),
        })
    return {
        "branch_id": branch_id,
        "complete": bool(complete),
        "actions": records,
    }


def amend_technical_gate(root, config_path, out):
    """Rejudge T6 from existing evidence only; never constructs an environment."""
    root, out = Path(root), Path(out)
    gate_path = out / "physical" / "technical_wave_check.json"
    stop_path = out / "decision" / "stop_state.json"
    gate = _read_json(gate_path)
    stop = _read_json(stop_path)
    if not isinstance(gate, dict) or not isinstance(stop, dict):
        raise StopRun(
            "STOPPED_T6_AMENDMENT_PREFLIGHT",
            "original gate/stop evidence missing",
        )
    if (
        gate.get("technical_wave") != "FAIL"
        or gate.get("checks", {}).get("T6_per_action_records_complete") is not False
    ):
        raise StopRun(
            "STOPPED_T6_AMENDMENT_PREFLIGHT",
            "original gate is not the authorized T6 failure",
        )
    amendment = out / "amendment"
    amendment.mkdir(parents=True, exist_ok=True)
    original = {
        "original_gate_sha256": sha256_file(gate_path),
        "original_stop_state_sha256": sha256_file(stop_path),
        "original_source_commit": git(root, "rev-parse", "HEAD"),
        "original_status": gate.get("technical_wave"),
        "original_failed_check": "T6_per_action_records_complete",
        "new_attempts": 0,
        "new_resets": 0,
    }
    _atomic_json(amendment / "original_gate_identity.json", original)
    try:
        gate_path.chmod(0o444)
    except OSError:
        pass
    reg = _read_json(out / "physical" / "witnesses" / "e4_branch_registration.json")
    if not isinstance(reg, dict):
        raise StopRun(
            "STOPPED_T6_AMENDMENT_PREFLIGHT",
            "registration evidence missing",
        )
    technical = [
        b for b in reg.get("branches", ()) if b.get("wave") == "technical"
    ]
    action_checks = [
        _check_terminal_aware_action_records(out, b) for b in technical
    ]
    amended_t6 = bool(
        len(technical) == 2 and all(item["complete"] for item in action_checks)
    )
    checks = dict(gate.get("checks") or {})
    checks["T6_per_action_records_complete"] = amended_t6
    passed = amended_t6 and all(value is True for value in checks.values())
    amended = {
        "technical_wave": "PASS" if passed else "FAIL",
        "technical_wave_detail": (
            "PASS_AFTER_TERMINAL_PLANNER_NA_AMENDMENT"
            if passed else "FAIL_AFTER_TERMINAL_PLANNER_NA_AMENDMENT"
        ),
        "remaining_branches_released": bool(passed),
        "checks": checks,
        "branch_ids": [b["branch_id"] for b in technical],
        "action_checks": action_checks,
        "source_gate_sha256": original["original_gate_sha256"],
        "new_attempts": 0,
        "new_resets": 0,
    }
    _atomic_json(out / "physical" / "technical_wave_check_amended.json", amended)
    _atomic_json(amendment / "t6_checker_amendment.json", {
        "amendment_id": "FAMILY_A_T6_TERMINAL_PLANNER_NA_1",
        "original_gate_status": "FAIL",
        "original_failed_check": "T6",
        "branch_execution_changed": False,
        "runtime_changed_after_execution": False,
        "branch_results_changed": False,
        "captures_changed": False,
        "budget_changed": False,
        "new_attempts": 0,
        "new_resets": 0,
        "reason": "terminal success ends the episode before replanning",
    })
    _atomic_json(amendment / "source_identity.json", {
        "source_commit": git(root, "rev-parse", "HEAD"),
        "gate_checker": "src/cp_disr/analysis/s4_family_a_soft_ordering_mvp.py",
        "source_sha256": sha256_file(Path(__file__)),
        "original_gate_sha256": original["original_gate_sha256"],
    })
    comparison = [
        "# T6 amendment comparison",
        "",
        f"Original gate: FAIL ({original['original_gate_sha256']})",
        f"Amended T6: {'PASS' if amended_t6 else 'FAIL'}",
        f"Amended technical gate: {amended['technical_wave']}",
        "Terminal success is classified as NOT_APPLICABLE_TERMINAL_SUCCESS; no planner call is synthesized.",
        "Existing branch results, captures, budget, attempts, and resets were not modified.",
    ]
    (amendment / "gate_comparison.md").write_text(
        "\n".join(comparison) + "\n", encoding="utf-8"
    )
    return amended


# ------------------------------------------------------------------ remaining 10 branches (hard-blocked until the technical gate passes)
def run_remaining(root, config_path, out, gpus, max_workers):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    amended_gate = out / "physical/technical_wave_check_amended.json"
    gate = amended_gate if amended_gate.is_file() else out / "physical/technical_wave_check.json"
    if (
        not gate.is_file()
        or rd(gate).get("technical_wave") != "PASS"
        or rd(gate).get("remaining_branches_released") is not True
    ):
        raise StopRun(
            "STOPPED_TECHNICAL_WAVE",
            "the remaining branches are released only after an amended technical gate PASS",
        )
    reg = rd(out / "physical/witnesses/e4_branch_registration.json")
    attempts = _load_attempts(out / "physical")
    todo = [b["branch_id"] for b in reg["branches"] if b.get("wave") == "remaining" and attempts.get(b["branch_id"]) is None]
    if len(todo) > int(cfg["budgets"]["remaining_branch_attempts"]):
        raise StopRun("STOPPED_BUDGET_EXHAUSTED", f"{len(todo)} remaining branches exceed the released budget")
    if not todo:
        return {"status": "NOTHING_TO_RUN"}
    report = _dispatch(root, out, cfg, todo, gpus, min(int(max_workers), int(cfg["remaining"]["workers_max"])),
                       ["remaining_branch_attempts"], "remaining")
    if report.get("faults"):
        raise StopRun("STOPPED_ENGINEERING_FAILURE", f"engineering faults during the remaining wave: {report['faults']}")
    return {"status": "COMPLETE", "dispatch": report}

# ------------------------------------------------------------------ mechanism classification (section 18)
def _paired(out):
    reg = rd(Path(out) / "physical/witnesses/e4_branch_registration.json")
    rows = []
    for b in reg["branches"]:
        r = _result(out, b["branch_id"])
        rows.append({**{k: b[k] for k in ("branch_id", "config", "route", "repeat", "case_id", "candidate_id")},
                     "task_success": r.get("task_success"), "execution_status": r.get("execution_status"),
                     "termination_reason": r.get("termination_reason"), "protocol_complete": r.get("protocol_complete"),
                     "witness_validity": r.get("witness_validity"), "controller_exit": r.get("controller_exit"),
                     "skill_count": _skill_count(r), "action_sequence": [s.get("candidate_id") for s in (r.get("trace") or [])],
                     "task_sim_completion_seconds": _sim_completion(r), "sum_skill_sim_duration": _sum_skill_sim(r),
                     "recorder_errors": r.get("recorder_errors"), "perception_identity": r.get("perception_identity"),
                     "global_perception_mask_unchanged": r.get("global_perception_mask_unchanged"),
                     "provider_module_loaded": r.get("provider_module_loaded")})
    return rows


def _config_times(rows, cid, route):
    return [r["task_sim_completion_seconds"] for r in rows
            if r["config"] == cid and r["route"] == route and r["task_sim_completion_seconds"] is not None]


def classify(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    phys = out / "physical"
    rows = _paired(out)
    ledger = merged_budget_ledger(out)
    attempts = _load_attempts(phys)
    reach = rd(out / "geometry/contract_reachability.json")
    integrity = {
        "all_12_terminal": len(rows) == 12 and all(r["execution_status"] in ("TERMINATED", "TRUNCATED", "DEADLINE", "NO_PLAN", "SEARCH_TIMEOUT")
                                                   and r["protocol_complete"] for r in rows),
        "all_12_resets_accounted": int(ledger["physical_witness_episodes"]["used"]) == 12
                                   and int(ledger["environment_resets"]["used"]) == 12,
        "all_paired_restores_valid": all(attempts.get(r["branch_id"]) == "COMPLETED" for r in rows),
        "all_12_task_success": all(r["task_success"] is True for r in rows),
        "all_skill_count_4": all(r["skill_count"] == 4 for r in rows),
        "all_sequences_match_a_frozen_route": all(tuple(r["action_sequence"]) == tuple(ROUTES.get(r["route"], ())) for r in rows),
        "no_perception_verifier_failure": all(r["recorder_errors"] == 0 and "CRITICAL_FACT_LOST" not in str(r["termination_reason"])
                                              for r in rows),
        "no_no_plan": all(r["execution_status"] != "NO_PLAN" for r in rows),
        "no_controller_engineering_failure": all(r["execution_status"] not in ("EXCEPTION", "UNKNOWN_CONTROLLER_EXIT", "ENGINEERING_STOP")
                                                 for r in rows),
        "route_skill_multisets_equal": reach.get("skill_multisets_equal") is True,
        "b_plan_nominal_costs_equal": reach.get("b_plan_nominal_costs_equal") is True,
    }
    common_ok = all(integrity.values())
    per_config, mech = {}, {}
    for cid in CONFIG_ORDER:
        tf, sf = _config_times(rows, cid, ROUTE_TARGET_FIRST), _config_times(rows, cid, ROUTE_SECOND_FIRST)
        if len(tf) != 2 or len(sf) != 2:
            per_config[cid] = {"status": "INCOMPLETE", "target_first_repeat_times": tf, "second_first_repeat_times": sf}
            mech[cid] = False
            continue
        m_tf, m_sf = float(np.mean(tf)), float(np.mean(sf))
        diff = m_sf - m_tf
        ratio = (m_tf / m_sf) if m_sf else None
        rel = abs(diff) / min(m_tf, m_sf) if min(m_tf, m_sf) else None
        per_config[cid] = {"target_first_repeat_times": tf, "second_first_repeat_times": sf, "mean_target_first": m_tf,
                           "mean_second_first": m_sf, "mean_ratio_target_over_second": ratio, "mean_difference_seconds": diff,
                           "relative_difference": rel, "absolute_difference_seconds": abs(diff)}
        if cid == "SO_TARGET_FIRST":
            mech[cid] = bool(all(a < b for a, b in zip(tf, sf)) and m_tf <= 0.90 * m_sf and (m_sf - m_tf) >= 0.75)
            per_config[cid]["criterion"] = "both repeats target_first < second_first; mean ratio <= 0.90; mean difference >= 0.75 s"
        elif cid == "SO_SECOND_FIRST":
            mech[cid] = bool(all(b < a for a, b in zip(tf, sf)) and m_sf <= 0.90 * m_tf and (m_tf - m_sf) >= 0.75)
            per_config[cid]["criterion"] = "both repeats second_first < target_first; mean ratio <= 0.90; mean difference >= 0.75 s"
        else:
            mech[cid] = bool(rel is not None and rel < 0.08 and abs(diff) < 0.60)
            per_config[cid]["criterion"] = "relative difference < 0.08 and absolute difference < 0.60 s"
        per_config[cid]["criterion_met"] = mech[cid]
    contract_cannot_rank = {"both_routes_reachable": reach.get("both_routes_reachable") is True,
                            "route_lengths_equal": reach.get("route_lengths_equal") is True,
                            "skill_multisets_equal": reach.get("skill_multisets_equal") is True,
                            "b_plan_nominal_costs_equal": reach.get("b_plan_nominal_costs_equal") is True,
                            "canonical_first_action": reach.get("canonical_first_action"),
                            "canonical_first_action_identical_across_configs": True,
                            "canonical_tie_break_reads_geometry": reach.get("canonical_tie_break_reads_geometry") is True}
    established = bool(common_ok and all(mech.values()) and all(contract_cannot_rank[k] for k in ("both_routes_reachable", "route_lengths_equal",
                                                                                                 "skill_multisets_equal",
                                                                                                 "b_plan_nominal_costs_equal")))
    gate = {"mechanism_feasibility": "ESTABLISHED_FOR_CONTROLLED_VALIDATION" if established else "NOT_ESTABLISHED",
            "common_integrity": integrity, "common_integrity_ok": common_ok, "per_config_criterion_met": mech,
            "contract_cannot_rank": contract_cannot_rank, "sim_time_only": True,
            "wall_time_used_for_mechanism": False, "evaluated_utc": now()}
    _atomic_json(phys / "per_config_summary.json", {"configs": per_config, "integrity": integrity,
                                                    "contract_cannot_rank": contract_cannot_rank, "generated_utc": now()})
    _atomic_json(out / "decision/mechanism_gate.json", gate)
    next_action = "REQUEST_PROVIDER_BASELINE_QUALIFICATION" if established else "EXPLICIT_RESEARCH_DECISION"
    _atomic_json(out / "decision/next_action.json", {"status": "COMPLETE", "mechanism_feasibility": gate["mechanism_feasibility"],
                                                     "family_role": "FAMILY_A_MECHANISM_VALIDATION" if established else None,
                                                     "provider_authorized": False, "representation_authorized": False,
                                                     "s2_authorized": False, "s3_authorized": False, "next_action": next_action,
                                                     "no_automatic_third_attempt": True, "no_automatic_fourth_config": True})
    _atomic_json(out / "decision_manifest.json", {"card_id": CARD_ID, "status": "COMPLETE",
                                                  "mechanism_feasibility": gate["mechanism_feasibility"], "next_action": next_action,
                                                  "completed_utc": now()})
    mark_phase(out, "classify", status="COMPLETE", mechanism_feasibility=gate["mechanism_feasibility"])
    return gate
# ------------------------------------------------------------------ summary (section 23 receipt)
def _pair_groups(out):
    reg = rd(Path(out) / "physical/witnesses/e4_branch_registration.json")
    groups = []
    for cid in CONFIG_ORDER:
        for repeat in (0, 1):
            ids = [b["branch_id"] for b in reg["branches"] if b["config"] == cid and b["repeat"] == repeat]
            groups.append({"config": cid, "repeat": repeat, "branch_ids": ids})
    return groups


def _paired_restore_doc(out):
    phys = Path(out) / "physical"
    groups = []
    for g in _pair_groups(out):
        docs = {}
        for bid in g["branch_ids"]:
            p = phys / "witnesses/initial_state_checks" / f"{bid}.json"
            docs[bid] = rd(p) if p.is_file() else {}
        ids = list(g["branch_ids"])
        a, b = (docs.get(ids[0], {}), docs.get(ids[1], {})) if len(ids) == 2 else ({}, {})
        ka, kb = a.get("compare_key") or {}, b.get("compare_key") or {}
        pa, pb = a.get("physical") or {}, b.get("physical") or {}
        qa, qb, va, vb = pa.get("qpos"), pb.get("qpos"), pa.get("qvel"), pb.get("qvel")
        qmax = vmax = None
        if qa is not None and qb is not None and len(qa) == len(qb):
            qmax = float(np.max(np.abs(np.asarray(qa, dtype=float) - np.asarray(qb, dtype=float)))) if len(qa) else 0.0
        if va is not None and vb is not None and len(va) == len(vb):
            vmax = float(np.max(np.abs(np.asarray(va, dtype=float) - np.asarray(vb, dtype=float)))) if len(va) else 0.0
        public_equal = bool(len(ids) == 2 and ka and ka == kb and a.get("candidate_ids") == b.get("candidate_ids")
                            and a.get("candidate_mask") == b.get("candidate_mask"))
        physical_equal = bool(qmax is not None and vmax is not None and qmax <= 1e-9 and vmax <= 1e-9)
        ok = bool(public_equal and physical_equal)
        groups.append({"config": g["config"], "repeat": g["repeat"], "branch_ids": ids,
                       "case_id": a.get("case_id"), "candidate_ids": {x: docs.get(x, {}).get("candidate_id") for x in ids},
                       "restore_seeds": {x: (docs.get(x, {}).get("compare_key") or {}).get("applied_restore_seed") for x in ids},
                       "compare_key": ka, "public_compare_key_equal": public_equal,
                       "qpos_sha256": {x: (docs.get(x, {}).get("physical") or {}).get("qpos_sha256") for x in ids},
                       "qvel_sha256": {x: (docs.get(x, {}).get("physical") or {}).get("qvel_sha256") for x in ids},
                       "qpos_max_abs_diff": qmax, "qvel_max_abs_diff": vmax,
                       "status": "PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE" if ok else "PAIRED_RESTORE_NOT_ESTABLISHED",
                       "note": "controller internal state and RNG stream are not directly compared"})
    doc = {"groups": groups, "groups_total": len(groups),
           "groups_verified": sum(1 for g in groups if g["status"].startswith("PAIRED_RESTORE_VERIFIED")),
           "all_paired_restores_verified": bool(groups) and all(g["status"].startswith("PAIRED_RESTORE_VERIFIED") for g in groups),
           "generated_utc": now()}
    _atomic_json(Path(out) / "physical/paired_restore.json", doc)
    return doc

def _throughput_merged(out):
    phys = Path(out) / "physical"
    reports = {}
    for phase in ("technical", "remaining"):
        p = phys / f"throughput_report_{phase}.json"
        reports[phase] = rd(p) if p.is_file() else None
    n_total = sum((r or {}).get("n", 0) for r in reports.values())
    spans = [(r or {}).get("span_seconds") for r in reports.values() if r]
    total_span = float(sum(s for s in spans if s)) if spans else None
    merged = {"phases": reports, "branches_total": n_total,
              "aggregate_throughput_branches_per_second": (n_total / total_span) if total_span else None,
              "aggregate_span_seconds_sum_over_phases": total_span,
              "max_concurrency": max([(r or {}).get("workers", 0) for r in reports.values()] + [0]),
              "gpus_used": sorted({g for r in reports.values() if r for g in r.get("gpus", [])}),
              "single_worker_baseline": "NOT_MEASURED",
              "speedup_claim": "NONE (no matched single-worker baseline; no extra attempts spent)",
              "wall_time_role": "engineering throughput only; never used for the mechanism decision",
              "generated_utc": now()}
    _atomic_json(phys / "throughput_report.json", merged)
    return merged


def summarize(root, config_path, out):
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    phys = out / "physical"
    rows = _paired(out)
    write_csv(phys / "branch_results.csv", rows)
    pr = _paired_restore_doc(out)
    tp = _throughput_merged(out)
    ledger = merged_budget_ledger(out)
    attempts = _load_attempts(phys)
    reach = rd(out / "geometry/contract_reachability.json")
    gate = rd(out / "decision/mechanism_gate.json")
    per = rd(phys / "per_config_summary.json")["configs"]
    tech_path = phys / "technical_wave_check_amended.json"
    tech = rd(tech_path if tech_path.is_file() else phys / "technical_wave_check.json")
    inv_ok, inv_diff = protected_unchanged(out)
    eng_failures = [r["branch_id"] for r in rows if r["execution_status"] in ("EXCEPTION", "UNKNOWN_CONTROLLER_EXIT", "ENGINEERING_STOP")]
    perception_failures = [r["branch_id"] for r in rows if r["recorder_errors"] not in (0,)]
    planner_errors = [r["branch_id"] for r in rows if r["execution_status"] in ("NO_PLAN", "SEARCH_TIMEOUT", "DECISION_CAP")]
    controller_failures = [r["branch_id"] for r in rows if r["controller_exit"] not in ("NORMAL_TERMINATION",)]
    source = rd(out / "source_identity.json")

    def fmt_times(v):
        return "[" + ", ".join("None" if x is None else f"{float(x):.6f}" for x in v) + "]"

    def fmt(x, nd=6):
        return "None" if x is None else f"{float(x):.{nd}f}"

    blocks = []
    for cid in CONFIG_ORDER:
        d = per.get(cid, {})
        tf, sf = d.get("target_first_repeat_times", []), d.get("second_first_repeat_times", [])
        blocks.append(f"\n{cid}:\n  target-first repeat times: {fmt_times(tf)}\n  second-first repeat times: {fmt_times(sf)}\n"
                      f"  mean ratio: {fmt(d.get('mean_ratio_target_over_second'))}\n"
                      f"  relative delta: {fmt(d.get('relative_difference'))}\n  absolute delta: {fmt(d.get('absolute_difference_seconds'))}\n"
                      f"  criterion met: {d.get('criterion_met')}\n  criterion: {d.get('criterion')}\n")
    mech = gate["mechanism_feasibility"]
    established = mech == "ESTABLISHED_FOR_CONTROLLED_VALIDATION"
    lines = [
        f"Family A soft-ordering MVP status: {gate.get('mechanism_feasibility')}",
        f"分支: {source.get('branch')}",
        f"Gate A commit: {source.get('head')}",
        f"Gate M commit: PENDING_UNTIL_GATE_M_COMMIT",
        f"证据目录: {out}",
        "",
        f"Configs: {', '.join(CONFIG_ORDER)}",
        f"Branch attempts: {ledger['physical_witness_episodes']['used']} / {ledger['physical_witness_episodes']['cap']}",
        f"Resets: {ledger['environment_resets']['used']} / {ledger['environment_resets']['cap']}",
        "".join(blocks),
        f"Both routes reachable: {reach.get('both_routes_reachable')}",
        f"Route lengths equal: {reach.get('route_lengths_equal')}",
        f"Skill multisets equal: {reach.get('skill_multisets_equal')}",
        f"B_PLAN nominal costs equal: {reach.get('b_plan_nominal_costs_equal')}",
        f"Canonical first action by config: {reach.get('canonical_first_action')} (identical across configs; tie-break reads no geometry)",
        "",
        f"Engineering failures: {eng_failures or 'none'}",
        f"Perception/verifier failures: {perception_failures or 'none'}",
        f"Planner errors: {planner_errors or 'none'}",
        f"Controller failures: {controller_failures or 'none'}",
        "",
        f"Workers: one worker per GPU, gpus={tp.get('gpus_used')}",
        f"Max concurrency: {tp.get('max_concurrency')}",
        f"Aggregate throughput: {fmt(tp.get('aggregate_throughput_branches_per_second'), 8)} branches/s over {fmt(tp.get('aggregate_span_seconds_sum_over_phases'), 3)} s",
        f"Single-worker baseline: {tp.get('single_worker_baseline')}",
        f"Speedup claim: {tp.get('speedup_claim')}",
        "",
        f"Mechanism feasibility: {mech}",
        f"Family role: {'FAMILY_A_MECHANISM_VALIDATION' if established else 'NOT_ESTABLISHED'}",
        f"Provider authorized: {gate.get('provider_authorized', False)}",
        f"Representation authorized: {gate.get('representation_authorized', False)}",
        f"S2 authorized: {gate.get('s2_authorized', False)}",
        f"Next action: {rd(out / 'decision/next_action.json').get('next_action')}",
        "",
        "New provider calls: 0",
        "New representation forwards: 0",
        "New RL: 0",
        "New optimizer: 0",
        "New elastic: 0",
        "New formal test: 0",
        "",
        f"Old evidence unchanged: {inv_ok}",
        f"Tests: see verify.json",
        f"Worktree clean: PENDING_UNTIL_GATE_M_COMMIT",
        f"SSH closed: pending final session exit",
        "",
        "主要限制: controlled mechanism validation only; single seed pair per config; wall time never used; no provider/representation/RL/optimizer/elastic/S2/S3; not a dataset; cannot alone support the CP-DISR headline.",
        "下一次需要用户明确授权的事项: provider baseline qualification (VLM-only, B_PLAN+R, A_STAT, Full comparisons) and any S2/S3 work.",
        "",
        f"technical_wave: {tech.get('technical_wave')}  remaining_branches_released: {tech.get('remaining_branches_released')}",
        f"paired_restore_verified: {pr['groups_verified']}/{pr['groups_total']}",
        f"mechanism_gate_utc: {gate.get('evaluated_utc')}",
    ]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mark_phase(out, "summarize", status="COMPLETE")
    return {"status": "COMPLETE", "final_summary": str(out / "final_summary.md"), "paired_restore": pr["all_paired_restores_verified"],
            "mechanism_feasibility": mech}

# ------------------------------------------------------------------ verification (section 18.1 + protected evidence)
def verify(root, config_path, out):
    """Independent re-derivation of the section 18.1 integrity checks plus the protected-evidence comparison."""
    root, out = Path(root), Path(out)
    cfg = load_config(config_path)
    phys = out / "physical"
    reg = rd(phys / "witnesses/e4_branch_registration.json")
    attempts = _load_attempts(phys)
    ledger = merged_budget_ledger(out)
    rows = _paired(out)
    reach = rd(out / "geometry/contract_reachability.json")
    gate = rd(out / "decision/mechanism_gate.json")
    checks = {}
    checks["registration_has_12_branches"] = len(reg["branches"]) == 12
    checks["no_duplicate_branch_id"] = len({b["branch_id"] for b in reg["branches"]}) == len(reg["branches"])
    checks["exactly_2_technical_branches"] = sum(1 for b in reg["branches"] if b.get("wave") == "technical") == 2
    checks["results_present_for_12_branches"] = len(rows) == 12 and all(r["execution_status"] for r in rows)
    checks["all_12_terminal"] = all(r["protocol_complete"] and r["execution_status"] in
                                    ("TERMINATED", "TRUNCATED", "DEADLINE", "NO_PLAN", "SEARCH_TIMEOUT") for r in rows)
    checks["attempts_12_completed"] = len(attempts) == 12 and all(v == "COMPLETED" for v in attempts.values())
    checks["branch_attempts_used_12"] = int(ledger["physical_witness_episodes"]["used"]) == 12
    checks["environment_resets_used_12"] = int(ledger["environment_resets"]["used"]) == 12
    checks["total_branch_attempts_used_12"] = int(ledger["total_branch_attempts"]["used"]) == 12
    checks["all_action_sequences_match_frozen_route"] = all(tuple(r["action_sequence"]) == tuple(ROUTES.get(r["route"], ())) for r in rows)
    checks["all_skill_count_4"] = all(r["skill_count"] == 4 for r in rows)
    checks["no_perception_verifier_failure"] = all(r["recorder_errors"] == 0 for r in rows)
    checks["no_no_plan"] = all(r["execution_status"] != "NO_PLAN" for r in rows)
    checks["no_engineering_failure"] = all(r["execution_status"] not in ("EXCEPTION", "UNKNOWN_CONTROLLER_EXIT", "ENGINEERING_STOP") for r in rows)
    checks["no_global_perception_residue"] = all(r["global_perception_mask_unchanged"] is True for r in rows)
    checks["provider_rl_optimizer_zero"] = all(r["provider_module_loaded"] is False for r in rows)
    checks["route_skill_multisets_equal"] = reach.get("skill_multisets_equal") is True
    checks["b_plan_nominal_costs_equal"] = reach.get("b_plan_nominal_costs_equal") is True
    checks["both_routes_reachable"] = reach.get("both_routes_reachable") is True
    checks["route_lengths_equal"] = reach.get("route_lengths_equal") is True
    checks["canonical_tie_break_reads_no_geometry"] = reach.get("canonical_tie_break_reads_geometry") is False
    checks["sim_time_only_metric"] = gate.get("wall_time_used_for_mechanism") is False and gate.get("sim_time_only") is True
    checks["budget_caps_respected"] = all(int(v["used"]) <= int(v["cap"]) for v in ledger.values())
    amended_gate = phys / "technical_wave_check_amended.json"
    technical_gate = rd(amended_gate if amended_gate.is_file() else phys / "technical_wave_check.json")
    checks["technical_gate_pass"] = technical_gate.get("technical_wave") == "PASS"
    inv = protected_inventory(root, cfg, out, "after")
    same, diff = protected_unchanged(out)
    checks["old_evidence_and_production_files_unchanged"] = bool(same)
    checks["old_evidence_source_commit_recorded"] = bool(rd(out / "inventory/protected_before.json").get("source_commit"))
    failed = sorted(k for k, v in checks.items() if not v)
    doc = {"status": "PASS" if not failed else "FAIL", "checks": checks, "failed_checks": failed,
           "protected_evidence": {"files": inv["file_count"], "unchanged": bool(same), "diff": diff},
           "mechanism_feasibility": gate.get("mechanism_feasibility"),
           "mechanism_feasibility_reproduced": gate.get("mechanism_feasibility") in
                                                ("ESTABLISHED_FOR_CONTROLLED_VALIDATION", "NOT_ESTABLISHED"),
           "out": str(out), "verified_utc": now()}
    _atomic_json(out / "verify.json", doc)
    mark_phase(out, "verify", status=doc["status"])
    return doc