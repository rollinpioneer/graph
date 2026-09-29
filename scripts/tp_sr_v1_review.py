#!/usr/bin/env python3
"""Zero-new-sample review of the T_P_SR V1 saved evidence (416c621e).

Reads JSON/CSV/JSONL/source text only. No environment, runtime, provider, policy or budget imports.
Subcommands: geometry, branches, assemble, verify.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

NR = "NOT_RECORDED"
LABELS = ("OBSERVED_SUCCESS_WITH_EXECUTION_TIME", "OBSERVED_NON_SUCCESS_AT_PLANNER_STOP", "SYMBOLIC_EVALUATOR_MISMATCH",
          "CONTROLLER_STATUS_MAPPING_REVIEW_REQUIRED", "ROOT_CAUSE_NOT_RECORDED")
EXECUTOR_REL = "src/cp_disr/platforms/libero/skill_executor.py"
PERCEPTION_REL = "src/cp_disr/platforms/libero/perception.py"
RUNTIME_REL = "src/cp_disr/platforms/libero/tp_sr_runtime.py"
CASE_SCENES_SPECIAL = {"T_P_SR_pool_33": "pool_33: only STRONG_HELPFUL scene; failure evidence for the direct route reviewed separately",
                       "T_P_SR_pool_57": "pool_57: symbolic/evaluator mismatch review"}
BRANCH_COLS = ["case_id", "route", "repeat", "branch_id", "attempt_state", "raw_controller_exit", "last_evaluator_success",
               "last_evaluator_terminated", "last_evaluator_reason", "planner_status", "planner_expanded_nodes",
               "executed_action_count", "last_measured_sim_time", "post_action_facts_ref", "post_action_rgb_ref",
               "post_action_depth_ref", "controller_subphase_trace_ref", "physical_contact_evidence_ref",
               "causal_interpretation_status"]


def rcsv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def wcsv(p, cols, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def rjson(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def wjson(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def rjsonl(p):
    out = []
    try:
        for ln in Path(p).read_text().splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    except Exception:
        pass
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def minmax(v):
    v = [x for x in v if x is not None]
    return {"min": min(v), "max": max(v), "n": len(v)} if v else {"min": None, "max": None, "n": 0}


def truthy(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def find_root(source: Path, root_arg: str | None):
    if root_arg:
        return Path(root_arg)
    for p in [source] + list(source.parents):
        if (p / "src").is_dir():
            return p
    return Path.cwd()


def ensure(out: Path, *subs):
    for s in subs:
        (out / s).mkdir(parents=True, exist_ok=True)


def pool_rows(src):
    return rcsv(src / "pool/pool_index.csv")


def opp_rows(src):
    return {r["case_id"]: r for r in rcsv(src / "opportunity/scene_opportunity.csv")}


def branch_rows(src):
    return rcsv(src / "physical/branch_results.csv")


def is_helpful(label):
    return "HELPFUL" in str(label).upper()


# ------------------------------------------------------------------ geometry
def geometry(source: Path, out: Path, root: Path):
    ensure(out, "geometry")
    pool = pool_rows(source)
    opp = opp_rows(source)
    br = branch_rows(source)
    rows = []
    for p in pool:
        cid = p["case_id"]
        o = opp.get(cid)
        b = [x for x in br if x["case_id"] == cid]
        srow = rjson(source / f"static/rows/{cid}.json", {}) or {}
        rows.append({
            "case_id": cid, "occlusion_bin": p["occlusion_bin"],
            "distance_bin_originally_named_occlusion": p["occlusion_bin"],
            "axis": p["axis"], "seed": p["seed"],
            "target_interferer_xy_distance": p["target_interferer_xy_distance"],
            "approach_clearance_proxy": p["approach_clearance_proxy"],
            "camera_projected_overlap": p["camera_projected_overlap"],
            "interferer_buffer_distance": p["interferer_buffer_distance"],
            "target_container_distance": p["target_container_distance"],
            "target_grasp_margin_proxy": p["target_grasp_margin_proxy"],
            "physical_scene": "yes" if o else "no",
            "n_branches": len(b),
            "opportunity": o["opportunity"] if o else "NOT_PHYSICALLY_TESTED",
            "direct_success": o["direct_success"] if o else "",
            "reloc_success": o["reloc_success"] if o else "",
            "static_row_present": "yes" if srow else "no",
            "static_qa_blob_recorded": "yes" if "qa_blob_xyz" in srow else NR,
        })
    cols = list(rows[0].keys()) if rows else ["case_id"]
    wcsv(out / "geometry/geometry_outcome_join.csv", cols, rows)

    bins = sorted({r["occlusion_bin"] for r in rows})
    per_bin = {}
    for b in bins:
        rr = [r for r in rows if r["occlusion_bin"] == b]
        phys = [r for r in rr if r["physical_scene"] == "yes"]
        bb = [x for x in br if x["occlusion_bin"] == b]

        def cnt(route):
            xs = [x for x in bb if x["route"] == route]
            return {"successes": sum(truthy(x["success"]) for x in xs), "denominator": len(xs)}
        per_bin[b] = {
            "pool_count": len(rr), "physical_scene_count": len(phys),
            "target_interferer_xy_distance": minmax([fnum(r["target_interferer_xy_distance"]) for r in rr]),
            "approach_clearance_proxy": minmax([fnum(r["approach_clearance_proxy"]) for r in rr]),
            "camera_projected_overlap": minmax([fnum(r["camera_projected_overlap"]) for r in rr]),
            "direct": cnt("direct"), "relocation": cnt("relocation"),
        }
    nz_pool = [r["case_id"] for r in rows if (fnum(r["camera_projected_overlap"]) or 0) > 0]
    nz_phys = [r["case_id"] for r in rows if (fnum(r["camera_projected_overlap"]) or 0) > 0 and r["physical_scene"] == "yes"]
    helpful = [r["case_id"] for r in rows if is_helpful(r["opportunity"])]
    sem = {
        "pool_count": len(rows), "physical_scene_count": sum(r["physical_scene"] == "yes" for r in rows),
        "nonzero_projected_overlap_pool_count": len(nz_pool), "nonzero_projected_overlap_physical_count": len(nz_phys),
        "per_bin": per_bin, "helpful_case_ids": helpful,
        "helpful_case_geometry": [{k: r[k] for k in ("case_id", "occlusion_bin", "axis", "target_interferer_xy_distance",
                                                     "approach_clearance_proxy", "camera_projected_overlap",
                                                     "interferer_buffer_distance", "target_grasp_margin_proxy")}
                                  for r in rows if r["case_id"] in helpful],
        "field_renaming": {"old_name": "occlusion_bin", "derived_name": "distance_bin_originally_named_occlusion",
                           "old_csv_modified": False},
        "bin_meaning": "bins order the target-interferer centre distance, not a verified camera occlusion or gripper interference level",
        "distinguished_notions": {
            "centre_distance_near": "target_interferer_xy_distance (pool_index)",
            "camera_occlusion_of_target": "camera_projected_overlap (pool_index); a projection proxy, not an observed occlusion",
            "real_gripper_interference": "approach/descend/close/lift contact of the finger/palm/held object with the interferer; NOT measured by any recorded field",
        },
        "approach_clearance_proxy_definition": "centre_distance - 0.06; NOT a real gripper collision distance",
        "overlap_zero_implies_no_mechanical_interference": False,
        "overlap_zero_note": "zero camera projected overlap says nothing about finger/palm sweep interference; cannot be inferred",
        "outcome_labels_are": "route-level heuristic labels, not verified physical harm or occlusion removal",
    }
    wjson(out / "geometry/bin_semantics.json", sem)

    ex = root / EXECUTOR_REL
    excerpt, ex_sha = NR, NR
    if ex.exists():
        txt = ex.read_text()
        m = re.search(r"    def _pick\(self.*?(?=\n    def _place\()", txt, re.S)
        excerpt = m.group(0) if m else NR
        ex_sha = hashlib.sha256(txt.encode()).hexdigest()
    md = ["# T_P_SR V1 geometry / mechanism review (saved data only)", "",
          "No new environment, provider, reset, skill, physical attempt, representation forward, RL or optimizer was used.", "",
          "## What the bins are", "",
          f"Pool configs: {sem['pool_count']}; physically tested scenes: {sem['physical_scene_count']}. "
          f"Configs with nonzero camera projected overlap: pool {len(nz_pool)}, physical {len(nz_phys)}.", "",
          "| bin | pool | physical | xy distance min..max | clearance proxy min..max | overlap min..max | direct | relocation |",
          "|---|---|---|---|---|---|---|---|"]
    for b, v in per_bin.items():
        d, c, o = v["target_interferer_xy_distance"], v["approach_clearance_proxy"], v["camera_projected_overlap"]
        md.append(f"| {b} | {v['pool_count']} | {v['physical_scene_count']} | {d['min']:.4f}..{d['max']:.4f} | "
                  f"{c['min']:.4f}..{c['max']:.4f} | {o['min']}..{o['max']} | "
                  f"{v['direct']['successes']}/{v['direct']['denominator']} | {v['relocation']['successes']}/{v['relocation']['denominator']} |"
                  if d["n"] and c["n"] else f"| {b} | {v['pool_count']} | - | - | - | - | - | - |")
    md += ["", "The field `occlusion_bin` is a distance bin. It is kept unchanged in old CSVs; the derived column "
           "`distance_bin_originally_named_occlusion` is used here.", "",
           "## Three notions that must not be merged", "",
           "1. Centre distance near (recorded). 2. Camera occlusion of the target (only a projection proxy recorded). "
           "3. Real gripper approach/descend/close/lift interference (not recorded and not measurable from saved fields). "
           "`approach_clearance_proxy = centre_distance - 0.06` is arithmetic on the centre distance and is not a gripper collision distance. "
           "Zero projected overlap therefore does not imply that mechanical interference is absent.", "",
           f"Helpful scenes: {helpful}.", "",
           "## Read-only look at SkillExecutor._pick (hypotheses, not findings)", "",
           f"Source sha256 {ex_sha}. Sequence: hover above object xy at hover z, descend to grasp z, press 8 mm lower, close hold, lift to the fixed show pose "
           "(0.02, -0.08, table_top+0.16), refresh perception.", "",
           "- Hover (APPROACH): the vertical hover above the target xy may be affected by an interferer only if it is tall enough to reach hover z; hypothesis.",
           "- Descend/press (INTERACT): finger opening axes near the target xy may collide with a nearby interferer; depends on the finger axis versus the target-interferer axis.",
           "- Close hold: closing fingers may push an interferer within finger travel; hypothesis.",
           "- Lift to show pose: a held target could drag or hit a neighbour; hypothesis.",
           "- Which phase a given `axis` (x or y) affects is unverified: no sub-phase trace or contact evidence was saved.", "",
           "```python", excerpt, "```"]
    (out / "geometry/mechanism_review.md").write_text("\n".join(md) + "\n")
    return {"worker": "geometry", "rows": len(rows)}


# ------------------------------------------------------------------ branches
def label_for(row, journal_last_exit):
    reason = row.get("termination_reason", "")
    if truthy(row.get("success")):
        return "OBSERVED_SUCCESS_WITH_EXECUTION_TIME"
    if reason == "SYMBOLIC_EVALUATOR_MISMATCH":
        return "SYMBOLIC_EVALUATOR_MISMATCH"
    if reason == "UNKNOWN_CONTROLLER_EXIT":
        return "CONTROLLER_STATUS_MAPPING_REVIEW_REQUIRED"
    if reason == "NO_PLAN":
        return "OBSERVED_NON_SUCCESS_AT_PLANNER_STOP"
    return "ROOT_CAUSE_NOT_RECORDED"


def branches(source: Path, out: Path, root: Path):
    ensure(out, "branches")
    br = branch_rows(source)
    opp = opp_rows(source)
    rows = []
    for x in br:
        bid = x["branch_id"]
        j = rjsonl(source / f"physical/witnesses/branch_{bid}.jsonl")
        res = rjson(source / f"physical/branch_results/{bid}.json", {}) or {}
        acts = [e for e in j if e.get("phase") == "action_complete"]
        evs = [e for e in j if e.get("phase") == "evaluator_complete"]
        pls = [e for e in j if e.get("phase") == "planner_return"]
        times = [e["relative_time"] for e in j if e.get("relative_time") is not None]
        last_exit = acts[-1]["controller_exit"] if acts else (res.get("controller_exit") or NR)
        rows.append({
            "case_id": x["case_id"], "route": x["route"], "repeat": x["repeat"], "branch_id": bid,
            "attempt_state": x["attempt_state"], "raw_controller_exit": last_exit,
            "last_evaluator_success": evs[-1].get("task_success") if evs else NR,
            "last_evaluator_terminated": evs[-1].get("terminated") if evs else NR,
            "last_evaluator_reason": evs[-1].get("reason") if evs else NR,
            "planner_status": pls[-1].get("status") if pls else NR,
            "planner_expanded_nodes": pls[-1].get("expanded_nodes") if pls else NR,
            "executed_action_count": len(acts),
            "last_measured_sim_time": times[-1] if times else NR,
            "post_action_facts_ref": NR, "post_action_rgb_ref": NR, "post_action_depth_ref": NR,
            "controller_subphase_trace_ref": NR, "physical_contact_evidence_ref": NR,
            "causal_interpretation_status": label_for(x, last_exit),
            "_trace": res.get("trace", []),
        })
    rows.sort(key=lambda r: (r["case_id"], r["route"], str(r["repeat"])))
    wcsv(out / "branches/termination_review.csv", BRANCH_COLS, rows)

    scenes = []
    for cid in sorted({r["case_id"] for r in rows}):
        rr = [r for r in rows if r["case_id"] == cid]
        srow = rjson(source / f"static/rows/{cid}.json", {}) or {}
        labels = sorted({r["causal_interpretation_status"] for r in rr if r["route"] == "relocation"})
        raw = sorted({str(r["raw_controller_exit"]) for r in rr})
        notes = []
        if cid in CASE_SCENES_SPECIAL:
            notes.append(CASE_SCENES_SPECIAL[cid])
        if any(r["causal_interpretation_status"] == "CONTROLLER_STATUS_MAPPING_REVIEW_REQUIRED" for r in rr):
            notes.append("raw exit code(s) " + ",".join(sorted({t["controller_exit"] for r in rr for t in r["_trace"]})) +
                         " are legal executor codes not mapped by the T_P_SR classifier; not an illegal unknown code")
        if any(r["causal_interpretation_status"] == "OBSERVED_NON_SUCCESS_AT_PLANNER_STOP" for r in rr):
            notes.append("planner stopped with NO_PLAN after a normal controller exit; no recorded facts explain why -> unknown")
        scenes.append({
            "case_id": cid, "opportunity": (opp.get(cid) or {}).get("opportunity", NR),
            "direct_outcomes": "|".join(f"{r['causal_interpretation_status']}@{r['last_measured_sim_time']}" for r in rr if r["route"] == "direct"),
            "relocation_outcomes": "|".join(f"{r['causal_interpretation_status']}@{r['last_measured_sim_time']}" for r in rr if r["route"] == "relocation"),
            "raw_controller_exits": ";".join(raw),
            "static_qa_blob_xyz": "recorded" if "qa_blob_xyz" in srow else NR,
            "rgb_depth_post_action": NR, "post_action_fact_values": NR,
            "relocation_failure_explained": "no" if any(l != "OBSERVED_SUCCESS_WITH_EXECUTION_TIME" for l in labels) else "n/a",
            "notes": " ; ".join(notes),
        })
    scols = ["case_id", "opportunity", "direct_outcomes", "relocation_outcomes", "raw_controller_exits", "static_qa_blob_xyz",
             "rgb_depth_post_action", "post_action_fact_values", "relocation_failure_explained", "notes"]
    wcsv(out / "branches/scene_evidence_review.csv", scols, scenes)

    n = len(rows)
    gaps = {
        "branches_reviewed": n,
        "items": {k: {"status": NR, "branches_lacking": n} for k in (
            "post_action_fact_values", "post_action_rgb", "post_action_depth", "controller_subphase_trace",
            "physical_contact_evidence", "held_object_state", "interferer_pose_after_action")},
        "journal_records_verified_fact_count_only": True,
        "never_filled_from": ["contract effects", "nominal overlay", "seed", "pool geometry"],
        "labels_allowed": list(LABELS),
        "normal_termination_is_not_grasp_success": True,
        "harmful_label_is_not_verified_physical_harm": True,
        "pool_33_is_not_verified_occlusion_removal": True,
        "provider_representation_status": "NOT_RUN",
    }
    wjson(out / "branches/recording_gaps.json", gaps)

    rt = root / RUNTIME_REL
    pc = root / PERCEPTION_REL
    lines = [l.strip() for l in rt.read_text().splitlines()
             if re.search(r"_nearest_palette_mask|install_nearest_palette_mask|perception\._mask", l)] if rt.exists() else []
    scope = ["# Runtime perception patch scope (read from source text)", "",
             f"- tp_sr_runtime.py sha256: {sha(rt) if rt.exists() else NR}",
             f"- perception.py sha256 (file bytes): {sha(pc) if pc.exists() else NR}", "",
             "Relevant source lines:", "", "```"] + lines + ["```", "",
             "- Dynamic replacement target: `perception._mask` (module-level attribute assigned at runtime).",
             "- Timing: installed when a T_P_SR runtime bundle is created, before observations are taken.",
             "- Process scope: process-wide for the lifetime of that worker process; not limited to one branch.",
             "- Unchanged file bytes of perception.py do not mean unchanged behaviour: the nearest-palette mask changes mask assignment for "
             "pixels claimed by several per-colour tolerance masks, and may change held/shaded interferer pixel counts (hypothesis, unverified).",
             "- Effect on the observed relocation outcomes (e.g. NO_PLAN after PICK interferer) has not been separated from other causes."]
    (out / "branches/runtime_patch_scope.md").write_text("\n".join(scope) + "\n")
    return {"worker": "branches", "rows": n}


# ------------------------------------------------------------------ assemble
def budget_request():
    return {
        "status": "PROPOSED_NOT_AUTHORIZED",
        "purpose": "TP_SR_SENSOR_DIAGNOSIS_AND_MINIMAL_GEOMETRY_PROTOTYPE",
        "requested_branch_attempts_total": 20, "diagnostic_attempts": 4, "prototype_attempts": 16,
        "approved_branch_attempts": 0, "part_b_authorized": False, "execute_now": False,
        "provider_calls_requested": 0, "rl_attempts_requested": 0, "optimizer_steps_requested": 0,
        "elastic_attempts_requested": 0, "standalone_capture_resets_requested": 0,
        "s2_s3_formal_test_authorized": False,
        "note": "Before any execution: instrument reset counts, declare the bootstrap/restore call chain, and save observations "
                "(facts, RGB, depth, sub-phase trace) during the branch itself without a separate capture replay. Old budgets "
                "(S1 20 reserved; S4 64 static + 96 physical) stay separate.",
    }


PROPOSAL = """# Minimal geometry prototype proposal (NOT AUTHORIZED; Part B is a proposal only)

Status: PROPOSED_NOT_AUTHORIZED. approved_branch_attempts = 0. execute_now = false.

## V1 close-out (unchanged)

T_P_SOFT_RELOCATION_V1: qualification not established; provider/representation NOT_RUN (not observed zero).
Direct route succeeded in 46/48 physical branches; relocation in 26/48. The bins are centre-distance bins, camera
projected overlap was zero in nearly all configs, and no recorded field observes real gripper interference. V1 evidence is retained as-is.

## Single mechanism for V2 candidate

A movable interferer intrudes into the target's real gripper sweep region (approach, descend, close, lift). The interferer's own
grasp point remains reachable. Moving it to the buffer frees the sweep, so relocation can reduce direct-route interference.
The mechanism is defined on real gripper/object collision geometry and the control path, not on camera overlap or centre distance.
pool_33 is not a V2 positive example (its failure evidence was never recorded).

## Proposed new configs (4 only)

- 2 configs with expected path interference (interferer placed inside the finger/palm sweep of the target grasp).
- 2 matched offset controls (same geometry, interferer shifted outside the sweep).
- Asymmetric interferer shape/orientation may be used only if it matches gripper opening and grasp height;
  where the current controller cannot support it, mark "not implemented" instead of forcing it.

## Constraints

- Target publicly localizable at start; both first actions pass safety checks; both nominal routes completable.
- No new contract dead ends; no initial penetration, no disabled collisions, no teleport.
- No per-route timeout tweaks, random failures or safety removal; controller fixes are shared by both routes and all future methods.
- State differences from V1 are listed explicitly; V1 assets, runtime and observations are not overwritten.
- V1 scenes where direct succeeds and relocation costs may serve as low-opportunity control candidates; they are not verified harmful-prior data.

## Staged request (proposal)

20 branch attempts: 4 diagnostic (sensor/recording check) + 16 prototype. Requires separate approval. Provider, RL, optimizer,
elastic and standalone capture resets: 0. S2/S3/formal test/method upgrade not requested.
"""


def assemble(source: Path, out: Path, root: Path):
    ensure(out, "raw", "geometry", "branches", "proposal", "tests")
    need = ["geometry/geometry_outcome_join.csv", "geometry/bin_semantics.json", "geometry/mechanism_review.md",
            "branches/termination_review.csv", "branches/scene_evidence_review.csv", "branches/recording_gaps.json",
            "branches/runtime_patch_scope.md"]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        raise SystemExit("assemble: missing worker outputs: " + ",".join(missing))
    sem = rjson(out / "geometry/bin_semantics.json")
    (out / "proposal/minimal_geometry_prototype.md").write_text(PROPOSAL)
    wjson(out / "proposal/budget_request.json", budget_request())
    rows = rcsv(out / "branches/termination_review.csv")
    lab = {}
    for r in rows:
        lab[r["causal_interpretation_status"]] = lab.get(r["causal_interpretation_status"], 0) + 1
    asm = {
        "source": str(source), "pool_count": sem["pool_count"], "physical_scene_count": sem["physical_scene_count"],
        "nonzero_projected_overlap_pool_count": sem["nonzero_projected_overlap_pool_count"],
        "nonzero_projected_overlap_physical_count": sem["nonzero_projected_overlap_physical_count"],
        "helpful_case_ids": sem["helpful_case_ids"], "branch_label_counts": lab,
        "provider_status": "NOT_RUN", "provider_calls_observed": "NOT_RUN",
        "representation_status": "NOT_RUN",
        "new_provider_calls": 0, "new_environment_constructions": 0, "new_resets": 0, "new_skill_executions": 0,
        "new_physical_attempts": 0, "new_representation_forwards": 0, "new_rl_or_optimizer": 0,
        "part_b_authorized": False, "cpu_workers": 2, "gpu_used": False, "speedup": "NOT_MEASURED",
        "output_sha256": {n: sha(out / n) for n in need + ["proposal/minimal_geometry_prototype.md", "proposal/budget_request.json"]},
    }
    wjson(out / "assembly.json", asm)
    return {"assembled": True}


# ------------------------------------------------------------------ verify
FORBIDDEN_IMPORT_ROOTS = {"cp_disr", "torch", "robosuite", "libero", "mujoco", "openai", "requests", "httpx", "numpy",
                          "gymnasium", "gym", "transformers", "subprocess"}


def scan_imports(path: Path):
    bad = []
    for n in ast.walk(ast.parse(path.read_text())):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [n.module or ""]
        for m in mods:
            if m.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                bad.append(m)
    return bad


def verify(source: Path, out: Path, root: Path):
    checks = {}
    need = ["geometry/geometry_outcome_join.csv", "geometry/bin_semantics.json", "geometry/mechanism_review.md",
            "branches/termination_review.csv", "branches/scene_evidence_review.csv", "branches/recording_gaps.json",
            "branches/runtime_patch_scope.md", "proposal/minimal_geometry_prototype.md", "proposal/budget_request.json", "assembly.json"]
    checks["required_outputs_present"] = all((out / n).exists() for n in need)
    b = rjson(out / "proposal/budget_request.json", {}) or {}
    checks["budget_proposed_only"] = (b.get("status") == "PROPOSED_NOT_AUTHORIZED" and b.get("approved_branch_attempts") == 0
                                      and b.get("execute_now") is False and b.get("part_b_authorized") is False
                                      and b.get("requested_branch_attempts_total") == 20
                                      and b.get("diagnostic_attempts") == 4 and b.get("prototype_attempts") == 16
                                      and all(b.get(k) == 0 for k in ("provider_calls_requested", "rl_attempts_requested",
                                                                       "optimizer_steps_requested", "elastic_attempts_requested",
                                                                       "standalone_capture_resets_requested")))
    a = rjson(out / "assembly.json", {}) or {}
    checks["provider_not_run_not_zero"] = a.get("provider_calls_observed") == "NOT_RUN" and a.get("provider_status") == "NOT_RUN"
    checks["no_new_samples"] = all(a.get(k) == 0 for k in ("new_provider_calls", "new_environment_constructions", "new_resets",
                                                            "new_skill_executions", "new_physical_attempts",
                                                            "new_representation_forwards", "new_rl_or_optimizer"))
    rows = rcsv(out / "branches/termination_review.csv") if (out / "branches/termination_review.csv").exists() else []
    checks["labels_allowed"] = all(r["causal_interpretation_status"] in LABELS for r in rows)
    checks["unrecorded_not_filled"] = all(r["post_action_rgb_ref"] == NR and r["post_action_facts_ref"] == NR for r in rows)
    checks["script_no_forbidden_imports"] = not scan_imports(Path(__file__))
    pb = out / "protected_before.json"
    changed, missing = [], []
    if pb.exists():
        before = rjson(pb)["sha256"]
        for p, h in before.items():
            fp = root / p
            if not fp.exists():
                missing.append(p)
            elif sha(fp) != h:
                changed.append(p)
    checks["protected_unchanged"] = pb.exists() and not changed and not missing
    res = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "changed": changed[:20], "missing": missing[:20]}
    wjson(out / "verify.json", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("geometry", "branches", "assemble", "verify"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--root", default="")
    a = ap.parse_args(argv)
    src, out = Path(a.source).resolve(), Path(a.output).resolve()
    root = find_root(src, a.root or None)
    res = {"geometry": geometry, "branches": branches, "assemble": assemble, "verify": verify}[a.command](src, out, root)
    print(json.dumps(res, ensure_ascii=False)[:2000])
    return 1 if a.command == "verify" and res.get("status") != "PASS" else 0


if __name__ == "__main__":
    sys.exit(main())
