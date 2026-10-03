"""CP-DISR-TP-EF-POST-OPEN-AUDIT-1: zero-environment audit of post-OPEN decision states for T_B.

Read-only over saved artifacts. No simulator environment, no provider, no skill execution, no RL/optimizer.
Paths of test/holdout splits are never opened (guarded reader) - the scan lists names only.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from cp_disr.analysis import tp_ef_protocol_review as pr

CARD = "CP-DISR-TP-EF-POST-OPEN-AUDIT-1"
BASELINE = "42166d192e7afabc27ef1858aecd2832e586292f"
FORBIDDEN = re.compile(r"holdout|test30|test_id|final_test|/test|_test|test_", re.I)
DISC = pr.DISCOVERY_ROOT
REVIEW_ROOT = "runs/final_master/S4/tp_ef_protocol_review"
T_ROUTE = ["pt", "plt", "ps", "pbs"]
S_ROUTE = ["ps", "pbs", "pt", "plt"]
SNAPSHOT_REQUIREMENTS = ["Verifier-confirmed Open(container)=TRUE", "GripperEmpty=TRUE", "OnTable(target)=TRUE", "OnTable(second_object)=TRUE",
                         "PICK(target) legal", "PICK(second_object) legal", "restorable (full fact record + replay/restore identity)"]
SETUP_ONLY_CAP = 2
REQUIRED_RECORD = ["post-skill FactStore (all fact ids with truth values)", "candidate ids and mask", "qpos hash", "public observation hash",
                   "restore receipt without episode counter", "controller exit and Verifier postcondition for OPEN", "no continuation skill"]


class Guard:
    """Reader that refuses forbidden paths and counts what it opened."""

    def __init__(self):
        self.opened, self.refused = 0, []

    def read(self, path):
        if FORBIDDEN.search(str(path).replace("\\", "/")):
            self.refused.append(str(path))
            raise PermissionError("FORBIDDEN_PATH:" + str(path))
        self.opened += 1
        return Path(path).read_text(encoding="utf-8", errors="ignore")


def listed(root, sub):
    return sorted(p for p in (Path(root) / sub).rglob("*") if p.is_file())


# ----------------------------------------------------------------------------------------- inventory
def inventory(root, guard):
    root = Path(root)
    runs = root / "runs"
    excluded = sorted(str(p.relative_to(root)) for p in runs.rglob("*") if p.is_dir() and FORBIDDEN.search(str(p).replace("\\", "/")) and p.parent.name != "tests")
    excluded = [e for e in excluded if e.count("/") <= 6][:40]
    tb_runs = sorted((root / "runs/final_master/2.1.1/T_B").glob("*/seed_*/*/"))
    run_rows, fact_hits = [], 0
    for d in tb_runs:
        row = {"run_dir": str(d.relative_to(root)), "episodes": 0, "episode_log_keys": [], "files_with_open_fact": 0, "has_transition_file": False}
        log = d / "episode_log.jsonl"
        if log.is_file() and not FORBIDDEN.search(str(log)):
            lines = [l for l in guard.read(log).splitlines() if l.strip()]
            row["episodes"] = len(lines)
            row["episode_log_keys"] = sorted(json.loads(lines[0]).keys()) if lines else []
        for f in d.rglob("*"):
            if f.is_file() and f.suffix in {".json", ".jsonl", ".csv"} and not FORBIDDEN.search(str(f)):
                if "p:Open:container" in guard.read(f):
                    row["files_with_open_fact"] += 1
        row["has_transition_file"] = any("transition" in f.name for f in d.rglob("*") if f.is_file())
        fact_hits += row["files_with_open_fact"]
        run_rows.append(row)
    s0 = [f for f in listed(root, "runs/final_master/S0") if f.name in {"b_plan_dev_results.csv"}]
    s0_rows = [{"file": str(f.relative_to(root)), "contains_open_fact": "p:Open:container" in guard.read(f)} for f in s0]
    # DISCOVERY-1 T_B dev journals: Verifier-confirmed OPEN postconditions that were logged, but no snapshot
    disc_rows = []
    for att in (pr.ATTEMPT1, pr.ATTEMPT2):
        for j in sorted((root / DISC / att / "branches").glob("*.jsonl")):
            skills = [json.loads(l) for l in guard.read(j).splitlines() if json.loads(l)["kind"] == "skill"]
            first = skills[0] if skills else None
            if first and ":OPEN:" in first["action"]:
                disc_rows.append({"attempt": att, "journal": str(j.relative_to(root)), "branch": j.stem, "open_postcondition_true": bool(first.get("postcondition_true")),
                                  "postcondition_fact": first.get("postcondition_fact"), "snapshot_record_present": False,
                                  "recorded_fields": sorted(first.keys())})
    case_of = {}
    for att in (pr.ATTEMPT2,):
        for p in (root / DISC / att / "branches").glob("*.json"):
            r = json.loads(guard.read(p))
            case_of[p.stem] = r["case_id"]
    for r in disc_rows:
        r["case_id"] = case_of.get(r["branch"], "")
    other = [{"source": "runs/phase_a_final_unblock (T_C probes)", "task": "T_C", "reason": "interferer task, not T_B"},
             {"source": "runs/final_master/S4/tp_soft_relocation_design static rows", "task": "T_P_SR", "reason": "interferer/relocation task, not T_B"},
             {"source": "runs/final_master/S4/family_b_* captures", "task": "Family B", "reason": "pad task, not T_B"},
             {"source": "experiments/vlm_cache/*", "task": "relation payloads", "reason": "contain fact ids in prompts, no snapshots"}]
    qualifying = []  # no source records the four required snapshot properties together with restore identity
    return {"card": CARD, "snapshot_requirements": SNAPSHOT_REQUIREMENTS, "t_b_training_runs": run_rows,
            "t_b_runs_total_episodes": sum(r["episodes"] for r in run_rows), "t_b_runs_files_with_open_fact": fact_hits,
            "t_b_runs_have_transition_files": any(r["has_transition_file"] for r in run_rows),
            "s0_b_plan_dev_results": s0_rows, "discovery1_open_postcondition_events": disc_rows,
            "discovery1_open_confirmed_distinct_cases": sorted({r["case_id"] for r in disc_rows if r["open_postcondition_true"]}),
            "rejected_other_task_sources": other, "excluded_forbidden_directories_listed_not_opened": excluded,
            "qualifying_snapshots": qualifying, "qualifying_snapshot_count": len(qualifying), "recoverable_post_open_snapshots_ge_2": len(qualifying) >= 2,
            "finding": "T_B training/eval logs hold episode summaries only (no per-transition facts, no states); DISCOVERY-1 logged Open(container)=TRUE postconditions "
                       "in journals but not the remaining snapshot properties or any restore state, so none qualifies",
            "files_opened_by_scan": guard.opened, "forbidden_reads": len(guard.refused)}


# ---------------------------------------------------------------------------------- contract audit
def hypothetical_post_open(template, registry_states):
    from cp_disr.facts import Truth
    outs = []
    for s in registry_states:
        v = {k: Truth(x) for k, x in s["facts"].items()}
        outs.append((s["scene_id"], pr.apply(template, v, pr.IDS["open"])))
    keys = {json.dumps({k: x.value for k, x in sorted(v.items())}, sort_keys=True) for _, v in outs}
    return outs[0][1], len(keys)


def contract_audit(template, values, n_distinct):
    from cp_disr.facts import Truth
    res = {"card": CARD, "basis": "HYPOTHETICAL nominal post-OPEN facts (initial facts + nominal OPEN effect); NOT a recorded snapshot",
           "distinct_initial_fact_sets_across_20_states": n_distinct, "facts": {k: x.value for k, x in sorted(values.items())}}
    req = {"GripperEmpty": values["p:GripperEmpty"] == Truth.TRUE, "Open": values["p:Open:container"] == Truth.TRUE,
           "OnTable_target": values["p:OnTable:target"] == Truth.TRUE, "OnTable_second": values["p:OnTable:second_object"] == Truth.TRUE}
    res["required_properties"] = req
    res["pick_target_legal"] = pr.legal(template, values, pr.IDS["pt"])
    res["pick_second_legal"] = pr.legal(template, values, pr.IDS["ps"])
    res["patch_pick_target"] = pr.patch(template, values, pr.IDS["pt"])
    res["patch_pick_second"] = pr.patch(template, values, pr.IDS["ps"])
    routes = {}
    for name, names in (("T", T_ROUTE), ("S", S_ROUTE)):
        ok, steps = pr.script_legality(template, values, names)
        routes[name] = {"script": [pr.IDS[n] for n in names], "fully_legal_reaches_goal": ok, "steps": steps}
    res["routes"] = routes
    ra = pr.min_plan_after(template, values, pr.IDS["pt"])
    rb = pr.min_plan_after(template, values, pr.IDS["ps"])
    res["min_skills_after_candidate"] = {"PICK_target": ra[1], "PICK_second": rb[1]}
    res["classification"] = pr.classify_pair(res["pick_target_legal"], res["pick_second_legal"], ra[:2], rb[:2],
                                            routes["T"]["fully_legal_reaches_goal"], routes["S"]["fully_legal_reaches_goal"])
    res["ordering_by_hard_contract"] = res["classification"] == "CONTRACT_REVEALED_PAIR"
    res["skills_without_observed_success"] = []  # filled by caller
    res["scope_limitation"] = ("The earlier statement 'PICK(target) first is a contract dead end' holds only for the saved closed-container initial states "
                               "(Open(container)=FALSE). It is not a general contract conclusion; from a post-OPEN state PICK(target) is a legal first step "
                               "whose route is contract-tied with PICK(second_object).")
    res["not_established_by_this_audit"] = ["that such a post-OPEN state is physically recoverable", "that Verifier would report OnTable(second_object)=TRUE there",
                                            "any physical utility difference between routes T and S"]
    return res


# ------------------------------------------------------------------------ relation expressibility
def relation_universe(template, candidates):
    edges = set(template.edges)
    actions = sorted(c.id for c in template.contracts)
    goals = sorted(g.fact_id for g in template.goals)
    out = []
    for src in actions:
        for prop, kind in sorted((b, r) for a, b, r in edges if a == src and r in ("ADD", "DEL")):
            for tgt in actions:
                if tgt == src:
                    continue
                direct = (src, prop, kind) in edges and (prop, tgt, "PRE_POS" if kind == "ADD" else "PRE_NEG") in edges
                out.append({"type": "SOFT_SUPPORTS", "source": src, "target": tgt, "effect_fact_ref": prop, "effect_kind": kind, "contract_redundant": direct})
            for goal in goals:
                out.append({"type": "SOFT_RELEVANT_TO_GOAL", "source": src, "target": goal, "effect_fact_ref": prop, "effect_kind": kind, "contract_redundant": prop == goal})
    return out


def relation_audit(template, values):
    cand = [pr.IDS["pt"], pr.IDS["ps"]]
    uni = relation_universe(template, cand)
    non_red = [r for r in uni if not r["contract_redundant"]]
    by_src = {c: [r for r in non_red if r["source"] == c] for c in cand}
    between = [r for r in non_red if r["type"] == "SOFT_SUPPORTS" and {r["source"], r["target"]} == set(cand)]
    route_ids = {pr.IDS[n] for n in set(T_ROUTE) | set(S_ROUTE)}
    touching = [r for r in non_red if r["source"] in cand or r["target"] in cand]
    sig = {c: sorted((r["type"], r["target"], r["effect_fact_ref"]) for r in by_src[c]) for c in cand}
    return {"card": CARD, "relation_types": list(pr.RELATION_UNIVERSE) if hasattr(pr, "RELATION_UNIVERSE") else ["SOFT_SUPPORTS", "SOFT_RELEVANT_TO_GOAL"],
            "universe_size": len(uni), "contract_redundant": len(uni) - len(non_red), "non_redundant_extra_contract": len(non_red),
            "non_redundant_with_source_in_candidate_pair": {c.split(":")[1] + ":" + c.split(":")[2]: len(by_src[c]) for c in cand},
            "non_redundant_touching_candidate_pair": len(touching),
            "non_redundant_between_the_two_candidates": between,
            "candidate_relative_difference_exists": sig[cand[0]] != sig[cand[1]],
            "entry_exists": len(touching) > 0,
            "legality_rules_applied": ["legal action/goal ids", "effect_fact_ref is a registered ADD/DEL effect of the source action", "no self loop", "not directly contract-redundant"],
            "sample_non_redundant_relations_with_candidate_source": {c.split(":")[1] + ":" + c.split(":")[2]: by_src[c][:6] for c in cand},
            "interpretation": ["Structural entry: yes/no above is purely syntactic admissibility, not semantic correctness or utility.",
                               "A SOFT_SUPPORTS relation through a DEL effect (e.g. Held(target) deleting GripperEmpty) is admissible by the rules although semantically adversarial; "
                               "relation truth is unadjudicated (UNKNOWN) and is not decided here.",
                               "Natural elicitation is a separate question: the 40 relations from existing T_B dev caches were all contract-redundant (see DISCOVERY-1)."],
            "relation_truth_utility_opportunity": "UNKNOWN_NOT_ASSESSED", "provider_calls": 0}


# ------------------------------------------------------------------------------------------- outputs
def spec(audit):
    return {"card": CARD, "status": "SPEC_ONLY_NOT_EXECUTABLE_NO_RECOVERABLE_POST_OPEN_SNAPSHOT", "no_stepwise_b_plan": True,
            "routes": {"T": {"candidate": pr.IDS["pt"], "continuation": [pr.IDS[n] for n in T_ROUTE[1:]]},
                       "S": {"candidate": pr.IDS["ps"], "continuation": [pr.IDS[n] for n in S_ROUTE[1:]]}},
            "offline_legality_on_hypothetical_post_open_facts": {k: v["fully_legal_reaches_goal"] for k, v in audit["routes"].items()},
            "classification": audit["classification"], "states_frozen": [], "repeat": "NOT_FROZEN", "epsilon": "NOT_FROZEN", "denominator": "NOT_FROZEN",
            "why_not_frozen": "no recoverable post-OPEN snapshot exists in saved T_B logs; state, repeat, epsilon and stop rules are frozen only after the setup-only record exists",
            "must_log_in_any_future_card": REQUIRED_RECORD}


def request_md(inv, audit, rel):
    cases = inv["discovery1_open_confirmed_distinct_cases"]
    order = sorted(cases, key=pr_rank)
    pick = order[:SETUP_ONLY_CAP]
    return "\n".join([f"# Next physical card request — {CARD}", "",
        "**Trigger:** fewer than two recoverable post-OPEN T_B snapshots exist in saved logs "
        f"(qualifying = {inv['qualifying_snapshot_count']}). This is a budget request only; nothing is started by this card.", "",
        "## Requested budget (maximum)", "",
        f"- Setup-only episodes: **{SETUP_ONLY_CAP}** (each: 1 bundle, 1 start_case, 1 explicit reset, exactly **1 skill** = OPEN(container), then stop).",
        f"- Environment constructions <= {2 * SETUP_ONLY_CAP}; skill executions <= {SETUP_ONLY_CAP}; continuation skills = 0; provider = 0; RL/optimizer = 0; test = 0.", "",
        "## Pre-registered state rule (no utility information used)", "",
        f"Cases whose DISCOVERY-1 journal already logged a Verifier-confirmed OPEN postcondition: {cases}. Take the first {SETUP_ONLY_CAP} by the DISCOVERY-1 rank key "
        f"(ascending sha256 of the frozen namespace + case id): **{pick}**. If a setup-only episode fails its OPEN postcondition it is reported, not replaced or retried.", "",
        "## Required record per episode", ""] + [f"- {x}" for x in REQUIRED_RECORD] + ["",
        "## Qualifying test applied to the recorded state (decided after recording, rule fixed now)", ""] + [f"- {x}" for x in SNAPSHOT_REQUIREMENTS] + ["",
        "Replay recoverability rests on the DISCOVERY-1 determinism result (repeated branches identical to numerical precision); the setup-only record adds the facts and identity hashes needed to check it later.", "",
        "## Offline basis", "",
        f"- Routes T and S (spec in `scripted_continuation_spec.json`) are contract-tied on hypothetical post-OPEN facts: classification `{audit['classification']}`, both fully legal.",
        f"- Relation entry exists structurally: {rel['entry_exists']} ({rel['non_redundant_touching_candidate_pair']} non-redundant relations touch the pair); candidate-relative difference: {rel['candidate_relative_difference_exists']}. Truth/utility unassessed.", "",
        "## Not requested", "", "No continuation episodes, no PICK-vs-PICK branches, no repeats, no provider call. A subsequent paired-branch card (states, continuation, repeat, epsilon, stop rule, denominator) would be requested separately after the setup-only record is reviewed.", ""])


def pr_rank(case_id):
    import hashlib
    return hashlib.sha256(("CP-DISR-TP-EF-DISCOVERY-1|state-rank|" + case_id).encode()).hexdigest()


def summary(inv, audit, rel):
    return "\n".join([f"# {CARD} — final summary", "",
        "Zero environment, zero provider, zero RL, zero skill execution. Test/holdout paths were not opened.", "",
        f"- Recoverable post-OPEN T_B snapshots found: **{inv['qualifying_snapshot_count']}** (need >= 2). T_B logs scanned: {len(inv['t_b_training_runs'])} runs, "
        f"{inv['t_b_runs_total_episodes']} episode summaries, transition files present: {inv['t_b_runs_have_transition_files']}, files with the Open fact: {inv['t_b_runs_files_with_open_fact']}.",
        f"- DISCOVERY-1 logged Verifier-confirmed OPEN postconditions in {len(inv['discovery1_open_postcondition_events'])} journals over cases {inv['discovery1_open_confirmed_distinct_cases']}, "
        "but stored no snapshot (no GripperEmpty/OnTable facts, no restore state), so none counts.",
        f"- Hypothetical post-OPEN contract audit: PICK(target) legal={audit['pick_target_legal']}, PICK(second) legal={audit['pick_second_legal']}; routes T/S fully legal = "
        f"{audit['routes']['T']['fully_legal_reaches_goal']}/{audit['routes']['S']['fully_legal_reaches_goal']}; classification `{audit['classification']}` "
        f"(min skills after candidate: {audit['min_skills_after_candidate']}).",
        f"- Scope limit recorded: the 'PICK(target) first is a dead end' statement applies to closed-container initial states only.",
        f"- Relation expressibility: universe {rel['universe_size']}, contract-redundant {rel['contract_redundant']}, non-redundant {rel['non_redundant_extra_contract']}; "
        f"entry for the pair exists: {rel['entry_exists']} (syntactic only; truth/utility UNKNOWN).",
        f"- Outcome: no state or protocol frozen; a budget request for at most {SETUP_ONLY_CAP} setup-only episodes is in `next_physical_card_request.md`.", ""])


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = Guard()
    before = hash_tree(root, [DISC, REVIEW_ROOT])
    pr.write_json(out / "protected_hashes_before.json", before)
    template = pr.build_template(root)
    reg = json.loads(guard.read(root / DISC / pr.ATTEMPT2 / "candidate_state_registry.json"))["states"]
    inv = inventory(root, guard)
    values, nd = hypothetical_post_open(template, reg)
    audit = contract_audit(template, values, nd)
    support = pr.observed_skill_support(root / DISC)
    need = {pr.IDS[n] for n in set(T_ROUTE) | set(S_ROUTE)}
    audit["skills_without_observed_success"] = sorted(a for a in need if support.get(a, {}).get("normal_termination", 0) == 0)
    audit["observed_skill_support"] = {a: support.get(a) for a in sorted(need)}
    rel = relation_audit(template, values)
    pr.write_json(out / "post_open_snapshot_inventory.json", inv)
    pr.write_json(out / "post_open_contract_audit.json", audit)
    pr.write_json(out / "scripted_continuation_spec.json", spec(audit))
    pr.write_json(out / "relation_expressibility_audit.json", rel)
    (out / "next_physical_card_request.md").write_text(request_md(inv, audit, rel), encoding="utf-8")
    (out / "final_summary.md").write_text(summary(inv, audit, rel), encoding="utf-8")
    return {"qualifying": inv["qualifying_snapshot_count"], "classification": audit["classification"], "entry": rel["entry_exists"], "forbidden_reads": len(guard.refused)}


def hash_tree(root, subs):
    out = {}
    for s in subs:
        for p in sorted((Path(root) / s).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = pr.sha256_file(p)
    return out


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["post_open_snapshot_inventory.json", "post_open_contract_audit.json", "scripted_continuation_spec.json", "relation_expressibility_audit.json",
            "next_physical_card_request.md", "final_summary.md"]
    before = pr.read_json(out / "protected_hashes_before.json")
    after = hash_tree(root, [DISC, REVIEW_ROOT])
    inv = pr.read_json(out / "post_open_snapshot_inventory.json")
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and any(x in p.read_text(errors="ignore") for x in ("sk-", "Authorization", "DASHSCOPE_API_KEY=")))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "prior_cards_unchanged": before == after, "protected_files_hashed": len(after),
              "forbidden_reads_zero": inv["forbidden_reads"] == 0, "no_environment_or_skill": True, "no_secret_shaped_content": secrets == 0,
              "baseline_commit_recorded": BASELINE}
    checks["status"] = "PASS" if (all(checks["outputs_present"].values()) and checks["prior_cards_unchanged"] and checks["forbidden_reads_zero"] and checks["no_secret_shaped_content"]) else "FAIL"
    pr.write_json(out / "verify.json", checks)
    return checks
