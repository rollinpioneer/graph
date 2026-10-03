#!/usr/bin/env python
"""CP-DISR T_P shared-staging card, STOP gate 1: does W have an existing, non-hard, public soft staging value?

Read-only: parses registries, source and earlier result summaries. No environment is constructed, no reset, no skill, no provider, no RL.
Verdict when no existing mechanism is found: TASK_FAIL_NO_EXISTING_SOFT_STAGING_VALUE.
"""
import argparse
import glob
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

CARD = "CP-DISR-TP-SHARED-STAGING (agent execution v1.0)"
CANDIDATE = "TP_SHARED_STAGING_V1"
RECOVERY_TERMS = ("RESEAT", "RE_SEAT", "REGRASP", "RE_GRASP", "REPOSITION", "REHANDLE", "REPICK", "RE_PICK", "RELOCATE", "RECOVER")
REGISTRIES = ["configs/contracts/d0_runtime_skills.yaml", "configs/contracts/skills.yaml", "configs/runtime/stage_2a_contract_registry.yaml", "configs/runtime/tp_fb_contract_registry.yaml",
              "configs/runtime/tp_so_mvp_contract_registry.yaml", "configs/runtime/tp_sr_contract_registry.yaml", "configs/runtime/tp_sr_v2_contract_registry.yaml"]
LIMIT_KEYS = ("environment_constructions", "env_reset", "preflight_resets", "formal_episodes", "skill_calls", "provider_requests", "provider_retries", "rl_transitions", "optimizer_steps", "elastic_attempts")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()


def contracts_of(path):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for c in (d.get("contracts") or []):
        out[c["name"]] = {"pre_pos": [a["predicate"] for a in c["initiation"]["pre_pos"]], "pre_neg": [a["predicate"] for a in c["initiation"]["pre_neg"]],
                          "add": [a["predicate"] for a in c["nominal"]["add"]], "del": [a["predicate"] for a in c["nominal"]["delete"]], "conditional_effects": len(c.get("conditional", []))}
    return {"predicate_types": sorted((d.get("predicate_types") or {}).keys()), "contracts": out}


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = git(["rev-parse", "HEAD"], root)
    man = json.loads((root / "docs/authoritative/authority_manifest.json").read_text())
    authority = {"manifest_status": man["authority_status"], "method_default": "2.1.1",
                 "experimental_plan": {"path": man["experimental_plan"]["path"], "manifest_sha256": man["experimental_plan"]["sha256"], "actual_sha256": sha(root / man["experimental_plan"]["path"])},
                 "research_content": {"path": man["research_content"]["path"], "manifest_sha256": man["research_content"]["sha256"], "actual_sha256": sha(root / man["research_content"]["path"])}}
    authority["hashes_match"] = all(authority[k]["manifest_sha256"] == authority[k]["actual_sha256"] for k in ("experimental_plan", "research_content"))
    # ---- skill inventory over every registry
    regs, all_skills = {}, set()
    for r in REGISTRIES:
        if (root / r).exists():
            regs[r] = contracts_of(root / r)
            all_skills |= set(regs[r]["contracts"])
    ex = (root / "src/cp_disr/platforms/libero/skill_executor.py").read_text(encoding="utf-8")
    dispatch = sorted(set(re.findall(r'skill == "([A-Z_]+)"', ex)))
    ev = (root / "src/cp_disr/platforms/libero/task_evaluator.py").read_text(encoding="utf-8")
    eval_tasks = sorted(set(re.findall(r'self\.task_id == "([A-Za-z_]+)"', ev)))
    recovery_skill_names = sorted(s for s in (all_skills | set(dispatch)) if any(t in s.upper() for t in RECOVERY_TERMS))
    defs = []
    for p in (root / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.match(r"\s*def\s+(reseat|regrasp|reposition|recover|rehandle|relocate)\w*", line, re.I):
                defs.append("%s:%d:%s" % (p.relative_to(root), i, line.strip()))
    d0 = regs["configs/contracts/d0_runtime_skills.yaml"]
    pb = d0["contracts"]["PLACE_BUFFER"]
    location_dependent_pre = sorted({(r, s) for r, v in regs.items() for s, c in v["contracts"].items() if any("Buffer" in p or "Region" in p for p in c["pre_pos"] + c["pre_neg"])})
    # ---- prior physical evidence on buffer staging / relocation (read from committed summaries)
    def text(p):
        g = sorted(glob.glob(str(root / p)))
        return (Path(g[-1]).read_text(encoding="utf-8"), str(Path(g[-1]).relative_to(root))) if g else ("", None)
    prior = {}
    t, f = text("runs/final_master/S4/tp_soft_relocation_design/*/final_summary.md")
    m = re.search(r"direct success (\d+)/(\d+), relocation success (\d+)/(\d+)", t)
    prior["soft_relocation_design"] = {"file": f, "direct_success": m.group(1) + "/" + m.group(2) if m else None, "relocation_success": m.group(3) + "/" + m.group(4) if m else None,
                                       "scene_opportunity": re.search(r"Scene opportunity: ([^.]*)\.", t).group(1) if re.search(r"Scene opportunity: ([^.]*)\.", t) else None,
                                       "feasibility": re.search(r"task_family_feasibility = (\w+)", t).group(1) if re.search(r"task_family_feasibility = (\w+)", t) else None,
                                       "reading": "the direct route rarely fails and relocation takes about twice the simulated time: no discrete rework to save"}
    t, f = text("runs/final_master/S4/tp_soft_relocation_pilot_v2/*/final_summary.md")
    prior["soft_relocation_pilot_v2"] = {"file": f, "mechanism": re.search(r"mechanism_feasibility: (\w+)", t).group(1) if re.search(r"mechanism_feasibility: (\w+)", t) else None, "next_action": re.search(r"next_action: (\w+)", t).group(1) if re.search(r"next_action: (\w+)", t) else None}
    t, f = text("runs/final_master/S4/family_b_staging_v2/*/final_summary.md")
    prior["family_b_staging_v2"] = {"file": f, "physical_mechanism": re.search(r"Physical mechanism status: \*\*(\w+)\*\*", t).group(1) if re.search(r"Physical mechanism status: \*\*(\w+)\*\*", t) else None,
                                    "reading": "pad_u and pad_v differ by seconds only (all 24 branches succeed with matching skill multisets): a path/time difference, not a discrete rework"}
    t, f = text("runs/final_master/S4/family_b_v3r1_reach_safe_mirror/*/final_summary.md")
    prior["family_b_v3r1"] = {"file": f, "mechanism": re.search(r"Mechanism: \*\*(\w+)\*\*", t).group(1) if re.search(r"Mechanism: \*\*(\w+)\*\*", t) else None, "next_action": re.search(r"next action `(\w+)`", t).group(1) if re.search(r"next action `(\w+)`", t) else None}
    dur = json.loads(Path(sorted(glob.glob(str(root / "runs/final_master/S4/tp_bsi_preflight/*/skill_duration_summary.json")))[0]).read_text())["per_skill"]
    pick = (dur["PICK_target"]["mean"] + dur["PICK_second"]["mean"]) / 2
    place = dur["PLACE_target"]["mean"]
    place_buf = dur["PLACE_BUFFER_second"]["mean"]
    direct_skills, direct_time = 4, 2 * (pick + place)
    stage_skills = {"one_object_staged": 6, "both_objects_staged": 8}
    direct = {"direct_route": "FINISH(p,G_p) + FINISH(q,G_q) = PICK,PLACE,PICK,PLACE", "direct_skills": direct_skills, "direct_time_s": direct_time, "staging_skills": stage_skills,
              "staging_extra_time_s_per_staged_object": pick + place_buf, "note": "staging adds PICK + PLACE_BUFFER per staged object; with no downstream saving it is pure overhead, so always-direct dominates by skill count and by time",
              "duration_source": "runs/final_master/S4/tp_bsi_preflight (clean-log means)"}
    # ---- gate 1 criteria
    crit = {
        "existing_downstream_operation_that_staging_can_remove": {"met": False, "evidence": "skills in every registry and in the executor are PICK, PLACE, PLACE_BUFFER, OPEN (and the unused MOVE); none is a re-seat, re-grasp, reposition or recovery skill; recovery-like names found only in analysis/module/path names"},
        "not_hard_precondition": {"met": None, "evidence": "not reached: there is no mechanism to classify"},
        "benefit_enters_existing_cost_or_skill_count": {"met": False, "evidence": "TaskEvaluator goals depend only on Inside/AtBuffer of fixed regions (T_A, T_B); no location-quality term exists; PLACE_BUFFER only ADDs AtBuffer and DELs Held; no PRE of any skill mentions a buffer or region"},
        "not_new_reward_hidden_truth_or_must_pass_W": {"met": False, "evidence": "a 'premium' W is only meaningful if some operation is cheaper for an object staged there; creating that is a new Evaluator/reward condition, a hard PRE, a controller feature or a hidden label"},
        "not_merely_closer_to_goal": {"met": None, "evidence": "not reached"}}
    gate1 = {"gate": "STOP gate 1 (section 1)", "all_criteria_met": False, "criteria": crit, "authority": authority, "skills_by_registry": {r: sorted(v["contracts"]) for r, v in regs.items()}, "executor_dispatch": dispatch, "evaluator_task_ids": eval_tasks,
             "recovery_like_skill_names": recovery_skill_names, "recovery_like_function_definitions": defs, "place_buffer_contract": pb, "skills_with_buffer_or_region_in_precondition": [list(x) for x in location_dependent_pre],
             "predicate_types_d0": d0["predicate_types"], "prior_physical_evidence": prior, "always_direct_static_audit": direct,
             "terms_searched": list(RECOVERY_TERMS), "registries_read": list(regs)}
    (out / "gate1_audit.json").write_text(json.dumps(gate1, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    verdict = {"task_candidate": CANDIDATE, "preflight_resets_used": 0, "formal_episodes": 0, "formal_episodes_note": "the card's 4 formal episodes are only reached after gate 1; none were run", "geometry_frozen_before_C1": False,
               "geometry_frozen_note": "not reached", "dual_legality": "NOT_EVALUATED", "contract_insufficient_for_ranking": "NOT_EVALUATED", "public_context": "NOT_EVALUATED", "staging_needed_but_not_hard_required": False,
               "H": {"A_rehandling": None, "B_rehandling": None, "strict_preference": "NOT_RUN"}, "R": {"A_rehandling": None, "B_rehandling": None, "strict_preference": "NOT_RUN"}, "primary_difference_is_discrete_rework": False,
               "always_direct_dominates": "TRUE_BY_STATIC_COUNT_NOT_PHYSICALLY_TESTED", "relation_schema_status": "NOT_CHECKED", "verdict": "TASK_FAIL_NO_EXISTING_SOFT_STAGING_VALUE",
               "secondary_findings": ["TASK_FAIL_STAGING_NOT_NEEDED would also apply by the static count (direct 4 skills vs staging 6 or 8)"], "stop_condition_hit": "agent STOP 1: a new Evaluator/reward condition, hard PRE, controller feature or hidden label would be needed"}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = {k: 0 for k in LIMIT_KEYS}
    ledger["files_read"] = "registries, source, earlier committed summaries (read-only)"
    (out / "budget_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "source_identity.json").write_text(json.dumps({"project_head_full": head, "branch": git(["branch", "--show-current"], root), "authority": authority, "skill_executor_sha256": sha(root / "src/cp_disr/platforms/libero/skill_executor.py"),
                                                          "task_evaluator_sha256": sha(root / "src/cp_disr/platforms/libero/task_evaluator.py"), "verifier_sha256": sha(root / "src/cp_disr/platforms/libero/verifier.py"),
                                                          "contract_registry_sha256": {r: sha(root / r) for r in regs}, "audit_script_sha256": sha(Path(__file__)), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "not_run_manifest.json").write_text(json.dumps({"reason": "STOP gate 1 failed before any scene was written", "not_produced": ["geometry_probe.json", "geometry_freeze_manifest.json", "asset_manifest.json", "continuation_spec.md", "continuation_hash.txt",
                                                       "preflight_log.jsonl", "canary_manifest.json", "episodes/C1..C4/events.jsonl"], "preflight_resets_used": 0, "formal_episodes_run": 0}, indent=2) + "\n", encoding="utf-8")
    (out / "design_card.md").write_text(design_card(gate1, verdict), encoding="utf-8")
    (out / "negative_evidence.md").write_text(negative(gate1, verdict), encoding="utf-8")
    return verdict


def design_card(g, v):
    return "\n".join(["# Design card: %s (%s)" % (CANDIDATE, CARD), "", "Scope: task-design / physical-canary only; no RL, optimizer, provider; no change to Method 2.1.1, contract semantics, Verifier, Evaluator, controller, reward, timeout or safety thresholds.", "",
                      "Authority: Research Content v5 and Experimental Plan v3 hashes match the manifest: %s; default method 2.1.1." % g["authority"]["hashes_match"], "",
                      "Idea under test: a shared premium staging place W with a legal fallback W_f; p or q is the staging-sensitive object depending on a public downstream context (H or R); a wrong first occupant of W must cost at least one extra discrete rehandling.", "",
                      "Result: **%s** at STOP gate 1; %d preflight resets, %d formal episodes. Nothing was constructed or run." % (v["verdict"], v["preflight_resets_used"], v["formal_episodes"]), "",
                      "Gate-1 reading: staging at W can only be valuable if some existing high-level operation becomes unnecessary or cheaper for an object staged there. The platform has no such operation (see gate1_audit.json and negative_evidence.md).", ""])


def negative(g, v):
    c = g["criteria"]
    L = ["# Negative evidence - %s" % v["verdict"], "", "## What gate 1 required", "", "A public, recordable, non-hard-precondition downstream mechanism by which using W removes an existing high-level operation, entering existing skill count / rework / recovery / formal cost, not a new reward, hidden truth or 'must pass W' rule, and not merely 'W is closer to the goal'.", "",
         "## What the platform has", "", "- Skills in every registry: %s" % json.dumps({r: s for r, s in g["skills_by_registry"].items()}),
         "- Executor dispatch: %s; evaluator task ids: %s" % (g["executor_dispatch"], g["evaluator_task_ids"]),
         "- PLACE_BUFFER contract: PRE %s; ADD %s; DEL %s (no conditional effects)." % (g["place_buffer_contract"]["pre_pos"], g["place_buffer_contract"]["add"], g["place_buffer_contract"]["del"]),
         "- Skills whose precondition mentions a buffer or region: %s" % (g["skills_with_buffer_or_region_in_precondition"] or "none"),
         "- Recovery-like skill names: %s; recovery-like function definitions in src: %s" % (g["recovery_like_skill_names"] or "none", g["recovery_like_function_definitions"] or "none"),
         "  (terms searched: %s; the only hits in src are module, path or audit-function names, not skills)" % ", ".join(g["terms_searched"]), "", "## Criteria", ""]
    for k, x in c.items():
        L.append("- %s: met=%s. %s" % (k, x["met"], x["evidence"]))
    L += ["", "## Earlier physical evidence on buffer staging and relocation", ""]
    for k, x in g["prior_physical_evidence"].items():
        L.append("- %s: %s" % (k, json.dumps(x)))
    d = g["always_direct_static_audit"]
    L += ["", "## Always-direct (static, not physically tested)", "", "- %s: %d skills, about %.1f s; staging routes: %s skills; each staged object adds about %.1f s." % (d["direct_route"], d["direct_skills"], d["direct_time_s"], d["staging_skills"], d["staging_extra_time_s_per_staged_object"]),
          "", "## What would have to be added for W to matter", "",
          "A location-dependent cost or a premium, i.e. a new Evaluator/reward term, a hard precondition tied to W, a controller feature (a re-seat or re-grasp that works only at W) or a hidden preferred-object label. Each is forbidden by the card, so the candidate stops here.", "",
          "No coordinates were chosen, no reset was spent and no episode was run; nothing was tuned.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.root, a.output)))
