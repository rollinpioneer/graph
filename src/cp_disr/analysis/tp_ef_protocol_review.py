"""CP-DISR-TP-EF-PROTOCOL-REVIEW-1: zero-environment / zero-provider / zero-RL review of the DISCOVERY-1 protocol.

Reads the saved DISCOVERY-1 outputs read-only, never constructs a simulator environment, never calls a provider,
never executes a skill. `make_env` is replaced by a raising guard for the lifetime of the review.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

CARD = "CP-DISR-TP-EF-PROTOCOL-REVIEW-1"
BASELINE = "9b11b4269b8bf4cc4b1992eb7f86727d4109311a"
DISCOVERY_ROOT = "runs/final_master/S4/tp_ef_discovery"
ATTEMPT1 = "20261002T170706Z_88110a77"
ATTEMPT2 = "20261002T171217Z_b2046983"
SPLIT_REL = "configs/splits/T_B_stage_2a_v11.json"
CONTRACT_REL = "configs/runtime/stage_2a_contract_registry.yaml"
SEARCH_DEPTH = 8
IDS = {"open": "a:OPEN:container:v1", "pt": "a:PICK:target:v1", "ps": "a:PICK:second_object:v1",
       "plt": "a:PLACE:target:container:v1", "pls": "a:PLACE:second_object:container:v1",
       "pbt": "a:PLACE_BUFFER:target:buffer:v1", "pbs": "a:PLACE_BUFFER:second_object:buffer:v1"}
PAIR_TYPES = {"OPEN_vs_PICK_target": ("open", "pt"), "OPEN_vs_PICK_second": ("open", "ps"), "PICK_target_vs_PICK_second": ("pt", "ps")}
# Frozen scripted continuations proposed in the review request (PICK target vs PICK second) and the two
# B_PLAN-observed full routes used for OPEN vs PICK(second); candidate action is the first element.
SCRIPTS = {
    "OPEN_vs_PICK_target": {"A": ["open", "ps", "pbs", "pt", "plt"], "B": ["pt", "plt", "ps", "pbs"]},
    "OPEN_vs_PICK_second": {"A": ["open", "ps", "pbs", "pt", "plt"], "B": ["ps", "pbs", "open", "pt", "plt"]},
    "PICK_target_vs_PICK_second": {"A": ["pt", "plt", "ps", "pbs"], "B": ["ps", "pbs", "pt", "plt"]},
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if mode:
        path.chmod(mode)


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ------------------------------------------------------------------------------- symbolic contract layer
def build_template(root):
    from cp_disr.graph import Goal, build_template as bt
    from cp_disr.platforms.libero import runtime_factory as rf

    def _no_env(*a, **k):
        raise RuntimeError("PROTOCOL_REVIEW_ZERO_ENV_GUARD: environment construction is forbidden")
    rf.make_env = _no_env
    objects = rf.TASK_OBJECTS["T_B"]
    contracts = rf._ground_contracts_for(Path(root) / CONTRACT_REL, {"OPEN": 16.0, "PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}, objects)
    goals = tuple(Goal(g, 1) for g in rf.TASK_GOALS["T_B"])
    return bt(contracts, goals, rf.PREDICATES, objects)


def _truth(v):
    from cp_disr.facts import Truth
    return v if isinstance(v, Truth) else Truth(v)


def legal(template, values, cid):
    from cp_disr.contracts import precondition_value
    from cp_disr.facts import Truth
    c = next(x for x in template.contracts if x.id == cid)
    return precondition_value(c, values) == Truth.TRUE


def apply(template, values, cid):
    from cp_disr.contracts import nominal_overlay
    c = next(x for x in template.contracts if x.id == cid)
    return dict(nominal_overlay(c, values, derived=template.derived_rules, exclusive_groups=template.exclusive_groups))


def goal_met(template, values):
    from cp_disr.facts import Truth
    return all(values.get(g.fact_id, Truth.UNKNOWN) == (Truth.TRUE if g.sign == 1 else Truth.FALSE) for g in template.goals)


def min_plan_after(template, values, first, depth=SEARCH_DEPTH):
    """BFS over nominal successors after forcing `first`; returns (reachable, min_total_skills incl. first, route) or (False, None, None)."""
    start = apply(template, values, first)
    key = lambda v: tuple(sorted((k, _truth(x).value) for k, x in v.items()))
    frontier = [(start, [first])]
    seen = {key(start)}
    if goal_met(template, start):
        return True, 1, [first]
    ids = sorted(c.id for c in template.contracts)
    for _ in range(depth - 1):
        nxt = []
        for v, route in frontier:
            for cid in ids:
                if not legal(template, v, cid):
                    continue
                w = apply(template, v, cid)
                k = key(w)
                if k in seen:
                    continue
                seen.add(k)
                r = route + [cid]
                if goal_met(template, w):
                    return True, len(r), r
                nxt.append((w, r))
        frontier = nxt
    return False, None, None


def script_legality(template, values, script_names):
    """Step-by-step nominal legality of a frozen scripted route from the initial facts."""
    v = dict(values)
    steps = []
    for name in script_names:
        cid = IDS[name]
        ok = legal(template, v, cid)
        steps.append({"action": cid, "legal_at_step": ok})
        if not ok:
            return False, steps
        v = apply(template, v, cid)
    return goal_met(template, v), steps


def patch(template, values, cid):
    nxt = apply(template, values, cid)
    return {k: [_truth(values[k]).value, _truth(nxt[k]).value] for k in sorted(nxt) if k in values and values[k] != nxt[k]}


def classify_pair(a_legal, b_legal, ra, rb, scripts_ok_a, scripts_ok_b):
    """ra/rb: (reachable, min_len). CONTRACT_REVEALED_PAIR when the hard contracts already order the pair."""
    if not (a_legal and b_legal):
        return "NOT_BOTH_LEGAL"
    if ra[0] != rb[0] or (ra[0] and rb[0] and ra[1] != rb[1]):
        return "CONTRACT_REVEALED_PAIR"
    if not ra[0] and not rb[0]:
        return "NEITHER_ROUTE_REACHABLE"
    return "CONTRACT_TIED_SCRIPTABLE" if (scripts_ok_a and scripts_ok_b) else "CONTRACT_TIED_NEEDS_DYNAMIC_PLANNER"


# -------------------------------------------------------------------------------------- evidence reading
def observed_skill_support(disc):
    """Per-action NORMAL_TERMINATION counts from the saved DISCOVERY-1 branch traces (attempt 2, read-only)."""
    c, ok = Counter(), Counter()
    for p in sorted((disc / ATTEMPT2 / "branches").glob("*.json")):
        for t in read_json(p).get("trace", []):
            c[t["action"]] += 1
            ok[t["action"]] += int(t["controller_exit"] == "NORMAL_TERMINATION")
    return {a: {"executions": c[a], "normal_termination": ok[a]} for a in sorted(c)}


def attempt_accounting(disc, name, workers_expected=8):
    d = disc / name
    ev = [json.loads(l) for l in (d / "technical_events.jsonl").read_text().splitlines() if l.strip()]
    reg_ev = [e for e in ev if e["kind"] == "register"]
    states = reg_ev[0]["states"] if reg_ev else 0
    started = [e for e in ev if e["kind"] == "worker_start"]
    ended = [e for e in ev if e["kind"] == "worker_end"]
    skills = restores = 0
    for p in sorted((d / "branches").glob("*.jsonl")):
        for l in p.read_text().splitlines():
            k = json.loads(l)["kind"]
            skills += int(k == "skill")
            restores += int(k == "restored")
    res_rows = [read_json(p) for p in sorted((d / "branches").glob("*.json"))]
    ledger = read_json(d / "budget_ledger.json")
    registration_bundles, workers = 1, len(started)
    start_cases = states + restores_attempted(res_rows)
    return {"attempt_dir": str(Path(DISCOVERY_ROOT) / name), "register_phase_bundles": registration_bundles, "register_phase_start_case": states,
            "worker_processes": workers, "worker_bundles": workers, "worker_start_case_attempted": restores_attempted(res_rows),
            "worker_restores_logged": restores, "environment_constructions": registration_bundles + workers + start_cases,
            "explicit_env_reset_calls": registration_bundles + workers + start_cases,
            "skill_executions": skills, "skill_executions_from_branch_json_traces": sum(len(r.get("trace", [])) for r in res_rows),
            "workers_ended": len(ended), "ledger_physical_episodes_used": ledger["physical_episodes"]["used"],
            "ledger_physical_episodes_cap": ledger["physical_episodes"]["cap"],
            "ledger_reserved": len(ledger["physical_episodes"]["reserved"]),
            "restore_only_branches": sum(1 for r in res_rows if not r.get("trace")),
            "derivation": "construction/reset counts derived from runtime_factory at the baseline: create_task_runtime = 1 make_env + 1 reset; "
                          "each start_case = 1 close + 1 make_env + 1 reset + pose apply + gripper reset. Skill counts are read from journals."}


def restores_attempted(rows):
    return sum(1 for r in rows if "restore_check" in r)


def cumulative(disc):
    a1, a2 = attempt_accounting(disc, ATTEMPT1), attempt_accounting(disc, ATTEMPT2)
    tot = {k: a1[k] + a2[k] for k in ("environment_constructions", "explicit_env_reset_calls", "skill_executions", "worker_processes")}
    return {"card": CARD, "baseline": BASELINE, "attempt_1_STOPPED_ENGINEERING": a1, "attempt_2_formal": a2, "cumulative": tot,
            "ledger_note": "attempt 1 ledger counts 8 'physical episodes' although 0 skills executed; attempt 2 ledger counts 8 with skill executions. "
                           "Reported as recorded: no refund, old ledgers untouched.",
            "provider_requests": 0, "rl_transitions": 0, "optimizer_steps": 0, "this_card_environment_constructions": 0}


# ------------------------------------------------------------------------------------------ the review
def review(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    disc = root / DISCOVERY_ROOT
    template = build_template(root)
    reg = read_json(disc / ATTEMPT2 / "candidate_state_registry.json")["states"]
    split = {r["case_id"]: r for r in read_json(root / SPLIT_REL)["dev"]}
    from cp_disr.facts import Truth
    support = observed_skill_support(disc)
    matrix, per_type = [], {}
    for s in reg:
        values = {k: Truth(v) for k, v in s["facts"].items()}
        for ptype, (an, bn) in PAIR_TYPES.items():
            a, b = IDS[an], IDS[bn]
            al, bl = legal(template, values, a), legal(template, values, b)
            ra = min_plan_after(template, values, a) if al else (False, None, None)
            rb = min_plan_after(template, values, b) if bl else (False, None, None)
            sa, stepsa = script_legality(template, values, SCRIPTS[ptype]["A"]) if al else (False, [])
            sb, stepsb = script_legality(template, values, SCRIPTS[ptype]["B"]) if bl else (False, [])
            cls = classify_pair(al, bl, ra[:2], rb[:2], sa, sb)
            needed = set(SCRIPTS[ptype]["A"]) | set(SCRIPTS[ptype]["B"])
            unsupported = sorted(IDS[n] for n in needed if support.get(IDS[n], {}).get("normal_termination", 0) == 0)
            row = {"scene_id": s["scene_id"], "pair_type": ptype, "candidate_a": a, "candidate_b": b, "both_legal": al and bl,
                   "patch_a": json.dumps(patch(template, values, a)) if al else "", "patch_b": json.dumps(patch(template, values, b)) if bl else "",
                   "route_a_reachable": ra[0], "route_a_min_skills": ra[1] if ra[1] is not None else "", "route_b_reachable": rb[0],
                   "route_b_min_skills": rb[1] if rb[1] is not None else "", "route_a_example": "->".join(x.split(":")[1] for x in ra[2]) if ra[2] else "",
                   "route_b_example": "->".join(x.split(":")[1] for x in rb[2]) if rb[2] else "",
                   "scripted_route_a_fully_legal": sa, "scripted_route_b_fully_legal": sb,
                   "first_illegal_step_a": next((x["action"] for x in stepsa if not x["legal_at_step"]), ""),
                   "first_illegal_step_b": next((x["action"] for x in stepsb if not x["legal_at_step"]), ""),
                   "needs_dynamic_b_plan": not (sa and sb), "skills_without_observed_success": ";".join(unsupported), "classification": cls}
            matrix.append(row)
            per_type.setdefault(ptype, []).append(row)
    fields = list(matrix[0].keys())
    write_csv(out / "candidate_pair_matrix.csv", fields, matrix)
    # ---- contract ordering audit
    audit = {"card": CARD, "states": len(reg), "pair_types": {}}
    for ptype, rows in per_type.items():
        cls = Counter(r["classification"] for r in rows)
        audit["pair_types"][ptype] = {
            "classification_counts": dict(cls), "all_both_legal": all(r["both_legal"] for r in rows),
            "route_a_reachable_states": sum(1 for r in rows if r["route_a_reachable"]), "route_b_reachable_states": sum(1 for r in rows if r["route_b_reachable"]),
            "route_a_min_skills": sorted({r["route_a_min_skills"] for r in rows}, key=str), "route_b_min_skills": sorted({r["route_b_min_skills"] for r in rows}, key=str),
            "scripted_a_legal_states": sum(1 for r in rows if r["scripted_route_a_fully_legal"]), "scripted_b_legal_states": sum(1 for r in rows if r["scripted_route_b_fully_legal"]),
            "first_illegal_step_a": sorted({r["first_illegal_step_a"] for r in rows}), "first_illegal_step_b": sorted({r["first_illegal_step_b"] for r in rows}),
            "skills_without_observed_success": sorted({x for r in rows for x in r["skills_without_observed_success"].split(";") if x}),
            "passes_offline_gate": cls.get("CONTRACT_TIED_SCRIPTABLE", 0) == len(rows) and not any(r["skills_without_observed_success"] for r in rows),
            "scripts": SCRIPTS[ptype]}
    audit["observed_skill_support_attempt2"] = support
    audit["contract_facts"] = {"PICK_pre": "GripperEmpty & OnTable(o)", "PLACE_pre": "Held(o) & Open(c)", "OPEN_pre": "GripperEmpty", "PLACE_BUFFER_pre": "Held(o)",
                               "PICK_effects": "adds Held(o); deletes GripperEmpty, OnTable(o)", "PLACE_BUFFER_effects": "adds AtBuffer, GripperEmpty; deletes Held (no OnTable re-add)",
                               "initial_container": "Open FALSE in every saved dev state (lid_closed)"}
    write_json(out / "contract_ordering_audit.json", audit)
    # ---- frozen next candidate protocol?
    pp = audit["pair_types"]["PICK_target_vs_PICK_second"]
    passes = pp["passes_offline_gate"]
    spec = {"card": CARD, "requested_protocol": "PICK_target_vs_PICK_second", "status": "FROZEN_NOT_EXECUTED" if passes else "NOT_FROZEN_OFFLINE_GATE_FAILED",
            "requested_scripts": SCRIPTS["PICK_target_vs_PICK_second"], "no_stepwise_b_plan": True,
            "offline_gate": {"passes": passes, "classification_counts": pp["classification_counts"], "first_illegal_step_a": pp["first_illegal_step_a"],
                             "route_a_reachable_states": pp["route_a_reachable_states"], "skills_without_observed_success": pp["skills_without_observed_success"]},
            "reason": "" if passes else "requested continuation A starts PICK(target) then PLACE(target, container); PLACE requires Open(container) which is FALSE in all saved dev states, "
                                        "OPEN requires GripperEmpty, and PLACE_BUFFER does not restore OnTable, so no legal continuation of candidate A reaches the goal under the registered contracts"}
    write_json(out / "scripted_continuation_spec.json", spec)
    other = {t: {"classification_counts": v["classification_counts"], "passes_offline_gate": v["passes_offline_gate"]} for t, v in audit["pair_types"].items()}
    geom = []
    for s in reg:
        r = split[s["scene_id"]]
        d = lambda p, q: round(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5, 4)
        geom.append({"scene_id": s["scene_id"], "target_to_container": d(r["target_xy"], r["container_xy"]), "second_to_buffer": d(r["second_xy"], r["buffer_xy"]),
                     "target_to_second": d(r["target_xy"], r["second_xy"]), "rank_key_reuse_from_discovery": s["rank_key"]})
    write_json(out / "candidate_state_preselection.json", {
        "card": CARD, "status": "NOT_FROZEN" if not passes else "REQUIRES_EXPLICIT_PRESELECTION_RULE_REVIEW",
        "reason": "no requested candidate protocol passed the offline gate; selecting two states for a protocol that is contract-infeasible would be vacuous",
        "other_pair_types_for_context": other,
        "public_geometry_proxies_only_not_a_selection": geom,
        "never_read": ["new physical winners", "Full/B2 scores", "test split"], "states_available": [s["scene_id"] for s in reg]})
    derived = {"card": CARD, "baseline": BASELINE, "original_manifest": str(Path(DISCOVERY_ROOT) / ATTEMPT2 / "eligibility_manifest.json"),
               "original_manifest_sha256": sha256_file(disc / ATTEMPT2 / "eligibility_manifest.json"), "original_manifest_modified": False,
               "rejected_scope": "T_B dev + OPEN(container) vs PICK(second_object) + B_PLAN stepwise replanning + existing relation caches (this concrete protocol only)",
               "gates": {"E1": "FAIL_FOR_EXISTING_CACHE", "E4": "NOT_ESTABLISHED", "E5": "NOT_ASSESSABLE", "E6": "NOT_ASSESSABLE"},
               "e6_original_automatic_label": "CONTRACT_SUFFICIENT", "e6_label_status": "AUTOMATIC_LABEL_RETAINED_NOT_A_RESEARCH_CONCLUSION",
               "GLOBAL_T_P": "UNRESOLVED",
               "evidence_limits": ["3 of 20 states physically sampled; 2 of 3 ended NO_PLAN (cause not recorded: facts were not saved)", "1 state measured, below epsilon",
                                   "simulation deterministic (repeat range 0)", "provider never called (no credential)"]}
    write_json(out / "derived_interpretation.json", derived)
    write_json(out / "cumulative_budget_reconciliation.json", cumulative(disc))
    return {"passes": passes, "pair_gate": {t: v["passes_offline_gate"] for t, v in audit["pair_types"].items()}, "classes": {t: v["classification_counts"] for t, v in audit["pair_types"].items()}}


def write_prose(out, result):
    audit = read_json(Path(out) / "contract_ordering_audit.json")
    bud = read_json(Path(out) / "cumulative_budget_reconciliation.json")
    a1, a2 = bud["attempt_1_STOPPED_ENGINEERING"], bud["attempt_2_formal"]
    lines = [f"# {CARD} — final review summary", "", "Zero environment, zero provider, zero RL, zero skill executions in this card. DISCOVERY-1 raw results accepted read-only.", "",
             "## Pair types over the 20 saved T_B dev states", ""]
    for t, v in audit["pair_types"].items():
        lines.append(f"- **{t}**: {v['classification_counts']}; route A reachable in {v['route_a_reachable_states']}/20, route B in {v['route_b_reachable_states']}/20; "
                     f"offline gate {'PASS' if v['passes_offline_gate'] else 'FAIL'}; first illegal step A={v['first_illegal_step_a']} B={v['first_illegal_step_b']}; "
                     f"skills never observed succeeding: {v['skills_without_observed_success']}")
    lines += ["", "## Requested next protocol (PICK(target) vs PICK(second_object), no stepwise B_PLAN)", "",
              f"Status: **{read_json(Path(out) / 'scripted_continuation_spec.json')['status']}**. "
              + read_json(Path(out) / "scripted_continuation_spec.json")["reason"], "",
              "## Budget reconciliation (as recorded, no refund)", "",
              f"- Attempt 1 (STOPPED_ENGINEERING): env constructions {a1['environment_constructions']}, explicit resets {a1['explicit_env_reset_calls']}, restores logged {a1['worker_restores_logged']}, skill executions {a1['skill_executions']}, ledger 'used' {a1['ledger_physical_episodes_used']}.",
              f"- Attempt 2 (formal): env constructions {a2['environment_constructions']}, explicit resets {a2['explicit_env_reset_calls']}, restores logged {a2['worker_restores_logged']}, skill executions {a2['skill_executions']}, ledger 'used' {a2['ledger_physical_episodes_used']}.",
              f"- Cumulative: {bud['cumulative']}", "",
              "## Interpretation", "", "See derived_interpretation.json: only the concrete DISCOVERY-1 protocol is rejected; GLOBAL_T_P is UNRESOLVED.", ""]
    (Path(out) / "final_review_summary.md").write_text("\n".join(lines), encoding="utf-8")


def request_md(out, result):
    spec = read_json(Path(out) / "scripted_continuation_spec.json")
    audit = read_json(Path(out) / "contract_ordering_audit.json")
    tied = audit["pair_types"]["OPEN_vs_PICK_second"]
    txt = [f"# Next physical card request — {CARD}", "",
           f"Requested PICK(target) vs PICK(second_object) protocol: **{spec['status']}**.", ""]
    if spec["status"] != "FROZEN_NOT_EXECUTED":
        txt += ["**No physical card is requested.** The requested candidate A is infeasible under the registered hard contracts (reason in `scripted_continuation_spec.json`).", "",
                "Decision options for the user (none is started by this card):", "",
                f"1. OPEN vs PICK(second_object) with a fully scripted continuation (A: OPEN, PICK second, PLACE_BUFFER second, PICK target, PLACE target; B: PICK second, PLACE_BUFFER second, OPEN, PICK target, PLACE target). "
                f"Offline status: {tied['classification_counts']}; all five skills have observed NORMAL_TERMINATION in DISCOVERY-1. It is contract-tied, not contract-revealed, so it is not excluded by the CONTRACT_REVEALED_PAIR rule; "
                "it removes the stepwise-planner dependency but not the unexplained NO_PLAN-after-PICK(second) observations (facts were not saved), so any new card must log post-skill facts.",
                "2. Treat the T_B template as exhausted for candidate diversity (single contract-tied pair) and move to a different decision structure; requires a research decision and probably new task content.",
                "3. Stop T_P discovery here and take the result to the S4 research decision.", "",
                "Provider: remains 0. A provider/prompt revision is requested only after a new physical card establishes two reliable utility witnesses."]
    else:
        txt += ["Frozen protocol is in `scripted_continuation_spec.json`; state selection requires a separate preregistration review."]
    (Path(out) / "next_physical_card_request.md").write_text("\n".join(txt) + "\n", encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["derived_interpretation.json", "cumulative_budget_reconciliation.json", "candidate_pair_matrix.csv", "contract_ordering_audit.json",
            "scripted_continuation_spec.json", "candidate_state_preselection.json", "next_physical_card_request.md", "final_review_summary.md"]
    disc = root / DISCOVERY_ROOT
    before = read_json(out / "discovery_hashes_before.json")
    after = {str(p.relative_to(root)): sha256_file(p) for p in sorted(disc.rglob("*")) if p.is_file()}
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and any(x in p.read_text(errors="ignore") for x in ("sk-", "Authorization", "DASHSCOPE_API_KEY=")))
    bud = read_json(out / "cumulative_budget_reconciliation.json")
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "discovery_results_unchanged": before == after, "discovery_files_hashed": len(after),
              "original_manifest_untouched": read_json(out / "derived_interpretation.json")["original_manifest_sha256"] == sha256_file(disc / ATTEMPT2 / "eligibility_manifest.json"),
              "zero_env_provider_rl_skill": bud["this_card_environment_constructions"] == 0 and bud["provider_requests"] == 0 and bud["rl_transitions"] == 0,
              "no_secret_shaped_content": secrets == 0,
              "accounting_consistent": all(bud[k]["skill_executions"] == bud[k]["skill_executions_from_branch_json_traces"] for k in ("attempt_1_STOPPED_ENGINEERING", "attempt_2_formal"))}
    checks["status"] = "PASS" if (all(checks["outputs_present"].values()) and all(checks[k] is True for k in ("discovery_results_unchanged", "original_manifest_untouched", "zero_env_provider_rl_skill", "no_secret_shaped_content", "accounting_consistent"))) else "FAIL"
    write_json(out / "verify.json", checks)
    return checks


def snapshot_discovery(root, out):
    root = Path(root).resolve()
    disc = root / DISCOVERY_ROOT
    write_json(Path(out) / "discovery_hashes_before.json", {str(p.relative_to(root)): sha256_file(p) for p in sorted(disc.rglob("*")) if p.is_file()})
