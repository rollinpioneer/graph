"""CP-DISR-TP-EF-DISCOVERY-1: evidence-first T_P discovery (0 RL / 0 optimizer / 0 provider requests).

Phases (design CP-DISR-TP-EF-DESIGN-1): registry/G1 -> paired physical branches (G2-G4) -> E1-E6 -> verdict.
All pre-registered rules live in the constants below and are frozen in `authorization.json` before any
physical branch runs. Nothing in this module reads test splits, Full/A_* outputs or VLM truth.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

CARD = "CP-DISR-TP-EF-DISCOVERY-1"
METHOD_VERSION = "2.1.1"
BASELINE_EVIDENCE_COMMIT = "9422b837cf1dfc43afc056a77bdb59d63512b3b6"
USER, HOST = "xushijie2", "gpu03"
PYTHON = "/home/xushijie2/envs/lerobotpi0-xfs/bin/python"

# ---- pre-registered discovery protocol (frozen before any physical branch) ---------------------------
SPLIT_REL = "configs/splits/T_B_stage_2a_v11.json"          # existing T_B development split (dev only)
RUNTIME_MANIFEST_REL = "experiments/manifests/runtime_manifest_v211.yaml"
SOURCE_STATE_CAP = 24
SPLIT_NAME = "dev"
PAIR = ("a:OPEN:container:v1", "a:PICK:second_object:v1")   # design doc section 3 example pair; fixed, not outcome-selected
WITNESS_STATES = 3                                           # hash-ranked G1-passing states; rank 0 carries the repeats
PHYSICAL_EPISODE_CAP = 8                                     # S1 plan: necessary independent physical witnesses <= 8
D_REF = 4.199999999997672                                    # T_B independent reference (Stage 0A calibration)
H_SECONDS = 23.09999999999752                                # suite half-life H (Stage 0A calibration)
DEADLINE = 60.0                                              # T_B task deadline (runtime manifest v211)
SEARCH = dict(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=D_REF)
EPS_G_FLOOR = 0.02                                           # |dG| floor ~ 0.7 s of completion time at H=23.1
EPS_T_FLOOR = D_REF / 2.0                                    # completion-time floor (s)
WATCHDOG_SECONDS = 900
MAX_DECISIONS = int(math.ceil(DEADLINE / D_REF)) + 2
STATE_RANK_NAMESPACE = CARD + "|state-rank|"
RELATION_UNIVERSE = ("SOFT_SUPPORTS", "SOFT_RELEVANT_TO_GOAL")
ALLOWED_ELIGIBLE = {"BENEFICIAL_IMPERFECT", "HARMFUL_BUT_INFORMATIVE", "NEAR_PERFECT", "HEURISTIC_INCONCLUSIVE"}


# ----------------------------------------------------------------------------------------- utilities
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(v):
    return sha256_bytes(canon(v).encode())


def write_json(path, value, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if mode:
        path.chmod(mode)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def event(out, kind, **fields):
    rec = {"t": time.time(), "kind": kind, **fields}
    with (Path(out) / "technical_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def state_rank_key(case_id):
    return hashlib.sha256((STATE_RANK_NAMESPACE + case_id).encode()).hexdigest()


def branch_id_of(case_id, candidate_id, repeat):
    return hashlib.sha256(f"{CARD}|{case_id}|{candidate_id}|{repeat}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------- pure outcome logic
def branch_outcome(trace, deadline=DEADLINE, h_seconds=H_SECONDS, nominal_depth=None):
    """Fixed-continuation outcome from a skill trace (list of dicts with action/exit/elapsed_end/success/reason)."""
    skills = len(trace)
    success_rows = [t for t in trace if t.get("task_success")]
    success = bool(success_rows)
    completion = trace[-1]["elapsed_end"] if trace else 0.0
    tau = success_rows[0]["elapsed_end"] if success else None
    g = (2.0 ** (-tau / h_seconds)) if success and tau < deadline else 0.0
    rework = (skills - nominal_depth) if (success and nominal_depth is not None) else None
    return {"success": success, "completion_seconds": completion, "success_seconds": tau, "g_start_discounted": g,
            "skill_count": skills, "rework_count": rework}


def epsilon_from_repeats(rows):
    """Within-candidate repeatability only (never reads cross-candidate differences)."""
    by = {}
    for r in rows:
        by.setdefault((r["case_id"], r["candidate_id"]), []).append(r)
    eg, et, used = 0.0, 0.0, []
    for key, items in by.items():
        valid = [i for i in items if i.get("valid")]
        if len(valid) < 2:
            continue
        gs = [i["g_start_discounted"] for i in valid]
        ts = [i["completion_seconds"] for i in valid]
        eg, et = max(eg, max(gs) - min(gs)), max(et, max(ts) - min(ts))
        used.append({"case_id": key[0], "candidate_id": key[1], "n": len(valid), "g_range": max(gs) - min(gs),
                     "completion_range": max(ts) - min(ts), "identical_success": len({i["success"] for i in valid}) == 1})
    return {"epsilon_q": max(EPS_G_FLOOR, eg), "epsilon_t": max(EPS_T_FLOOR, et), "repeat_g_range": eg,
            "repeat_time_range": et, "floor_g": EPS_G_FLOOR, "floor_t": EPS_T_FLOOR, "groups": used,
            "repeat_groups_complete": len(used) >= 2}


def compare_pair(case_id, rows, eps):
    """Cross-candidate witness for one state. rows: valid branch rows of both candidates."""
    a_id, b_id = PAIR
    ra = [r for r in rows if r["case_id"] == case_id and r["candidate_id"] == a_id]
    rb = [r for r in rows if r["case_id"] == case_id and r["candidate_id"] == b_id]
    out = {"case_id": case_id, "candidates": list(PAIR), "n_a": len(ra), "n_b": len(rb)}
    if not ra or not rb or any(not r.get("valid") for r in ra + rb):
        out.update(status="NOT_ESTABLISHED_INVALID_BRANCH", reliable=False, direction="NONE")
        return out
    mean = lambda xs: sum(xs) / len(xs)
    qa, qb = mean([r["g_start_discounted"] for r in ra]), mean([r["g_start_discounted"] for r in rb])
    ta, tb = mean([r["completion_seconds"] for r in ra]), mean([r["completion_seconds"] for r in rb])
    sa, sb = all(r["success"] for r in ra), all(r["success"] for r in rb)
    ka, kb = mean([r["skill_count"] for r in ra]), mean([r["skill_count"] for r in rb])
    # a continuation that never produced a plan cannot demonstrate soft cost (planner-horizon / hard-constraint artefact)
    artefact = any(r.get("termination") in ("NO_PLAN", "SEARCH_TIMEOUT") for r in ra + rb)
    reasons = []
    if sa != sb:
        reasons.append("feasibility")
    if abs(qa - qb) > eps["epsilon_q"]:
        reasons.append("return")
    if abs(ta - tb) > eps["epsilon_t"]:
        reasons.append("time")
    if ka != kb:
        reasons.append("skill_count_or_rework")
    better = PAIR[0] if qa > qb else (PAIR[1] if qb > qa else "TIE")
    if better == "TIE" and "skill_count_or_rework" in reasons:
        better = PAIR[0] if ka < kb else PAIR[1]
    reliable = bool(reasons) and not artefact
    out.update(q_ref_a=qa, q_ref_b=qb, time_a=ta, time_b=tb, success_a=sa, success_b=sb, skills_a=ka, skills_b=kb,
               delta_q=qa - qb, delta_t=ta - tb, reasons=reasons, artefact_planner_or_constraint=artefact,
               reliable=reliable, direction=better if reliable else "NONE",
               status="WITNESS" if reliable else ("LOW_DIAGNOSTIC_VALUE" if not artefact else "ARTEFACT_EXCLUDED"))
    return out


def classify_e6(natural_nonempty_states, rstar_helpful, rstar_differs, physical_differences):
    """Frozen E6 mapping (design section 10 / plan 5.4)."""
    if not physical_differences and not rstar_differs:
        return "CONTRACT_SUFFICIENT"
    if natural_nonempty_states < 2:
        if rstar_helpful:
            return "PROVIDER_SCHEMA_LIMITATION"
        return "HEURISTIC_INCONCLUSIVE" if physical_differences else "CONTRACT_SUFFICIENT"
    return "BENEFICIAL_IMPERFECT" if rstar_helpful else "HARMFUL_BUT_INFORMATIVE"


# ----------------------------------------------------------------------------------- runtime plumbing
def _s1_module(root):
    spec = importlib.util.spec_from_file_location("s1_prior_existence_ref", Path(root) / "scripts/s1_prior_existence.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_runtime(root, out):
    from cp_disr.baselines.b_plan import SearchConfig, bind_t_b_manifest, _make_relation_free_runtime_split
    from cp_disr.runtime import load_runtime
    root, out = Path(root).resolve(), Path(out).resolve()
    free = out / "bindings" / "T_B_dev_relation_free_split.json"
    free.parent.mkdir(parents=True, exist_ok=True)
    if not free.is_file():
        _make_relation_free_runtime_split(root / SPLIT_REL, free)
    manifest, _ = bind_t_b_manifest(root, root / RUNTIME_MANIFEST_REL, free, SearchConfig(**SEARCH))
    manifest["runtime"]["task_deadlines"]["T_B"] = DEADLINE
    return load_runtime(manifest), manifest


def _qpos_hash(env):
    try:
        import numpy as np
        return sha256_bytes(np.round(np.asarray(env.sim.data.qpos, dtype="float64"), 8).tobytes())
    except Exception:
        return ""


def pre_state_identity(bundle, snap):
    facts = {k: v.value for k, v in snap.facts.values.items()}
    obs = bundle.observations.observe()
    return {"snapshot_hash": bundle.snapshot_identity, "facts_hash": digest(facts), "goal_hash": digest([[g.fact_id, g.sign] for g in snap.template.goals]),
            "candidate_ids": list(snap.candidate_ids), "candidate_mask": [bool(m) for m in snap.mask],
            "candidate_hash": digest([list(snap.candidate_ids), [bool(m) for m in snap.mask]]),
            "qpos_hash": _qpos_hash(bundle.environment), "proprio_hash": digest([round(float(x), 8) for x in obs.proprioception]),
            "restore_receipt_sha256": digest(bundle.restore_receipt), "restore_verified": bool(bundle.restore_verified), "facts": facts}


def nominal_patch(template, values, contract):
    from cp_disr.contracts import nominal_overlay
    nxt = dict(nominal_overlay(contract, values, derived=template.derived_rules, exclusive_groups=template.exclusive_groups))
    return {k: [values[k].value, nxt[k].value] for k in sorted(nxt) if k in values and values[k] != nxt[k]}


# ------------------------------------------------------------------------------------------- phases
def gate0(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    ident = {"user": os.environ.get("USER", ""), "hostname": socket.gethostname(), "python": sys.executable,
             "platform": platform.platform()}
    if ident["user"] != USER or not ident["hostname"].startswith(HOST):
        problems.append("WRONG_ACCOUNT_OR_HOST")
    if str(Path(sys.executable)) != PYTHON:
        problems.append("WRONG_PYTHON")
    import cp_disr
    ident["cp_disr_file"] = str(Path(cp_disr.__file__).resolve())
    if not str(Path(cp_disr.__file__).resolve()).startswith(str(root)):
        problems.append("CP_DISR_IMPORT_OUTSIDE_WORKTREE")
    fs = subprocess.run(["df", "-T", str(out)], capture_output=True, text=True).stdout.splitlines()[-1].split()
    ident["fs_type"], ident["mount"] = fs[1], fs[-1]
    if ident["fs_type"] != "xfs":
        problems.append("OUTPUT_NOT_XFS")
    commit = git(root, "rev-parse", "HEAD")
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_EVIDENCE_COMMIT, "HEAD"], cwd=root).returncode == 0
    if not ancestor:
        problems.append("BASELINE_NOT_ANCESTOR")
    if dirty:
        problems.append("DIRTY_TRACKED_TREE")
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        if key in os.environ:
            problems.append("PROVIDER_CREDENTIAL_PRESENT_IN_PROCESS:" + key)
    own = [Path(__file__).resolve(), root / "scripts/tp_ef_discovery.py", root / "scripts/s1_prior_existence.py",
           root / "src/cp_disr/baselines/b_plan.py", root / "src/cp_disr/platforms/libero/runtime_factory.py",
           root / SPLIT_REL, root / RUNTIME_MANIFEST_REL, root / "configs/runtime/stage_2a_contract_registry.yaml"]
    src = {str(p.relative_to(root)): sha256_file(p) for p in own if p.is_file()}
    write_json(out / "source_identity.json", {"card": CARD, "execution_base_commit": commit, "baseline_evidence_commit": BASELINE_EVIDENCE_COMMIT,
                                              "baseline_is_ancestor": ancestor, "branch": git(root, "branch", "--show-current"),
                                              "dirty_tracked": dirty, "identity": ident, "source_sha256": src, "method_version": METHOD_VERSION})
    auth = {"card": CARD, "design_document_id": "CP-DISR-TP-EF-DESIGN-1", "method_version": METHOD_VERSION,
            "authorized_by": "user chat instruction: execute the design-file experiment (Discovery card scope only)",
            "scope": ["registry+G1 on existing T_B dev states", "paired physical branches (<=8 episodes)", "E1-E6 offline gates", "verdict"],
            "forbidden": ["RL", "optimizer", "test", "new provider requests without credential+authorization", "Family B/RoboCasa", "80-config family"],
            "budget": {"physical_episode_cap": PHYSICAL_EPISODE_CAP, "provider_first_requests": 0, "provider_retries": 0, "rl_transitions": 0, "optimizer_steps": 0},
            "frozen_protocol": {"split": SPLIT_REL, "split_role": SPLIT_NAME, "source_state_cap": SOURCE_STATE_CAP, "pair": list(PAIR),
                                "witness_states": WITNESS_STATES, "rank_namespace": STATE_RANK_NAMESPACE, "search": SEARCH, "H": H_SECONDS,
                                "deadline": DEADLINE, "eps_g_floor": EPS_G_FLOOR, "eps_t_floor": EPS_T_FLOOR,
                                "continuation": "B_PLAN replanning after the forced first candidate (public facts only; no prior, no VLM, no RL)",
                                "return": "G=2^(-tau_success/H) if first success in time else 0",
                                "rework": "executed skills minus nominal B_PLAN plan depth at the initial state (successful branches)",
                                "epsilon_rule": "max(floor, within-candidate repeat range); computed before cross-candidate comparison"},
            "problems": problems, "status": "PASS" if not problems else "STOPPED_GATE0"}
    write_json(out / "authorization.json", auth)
    write_json(out / "budget_ledger.json", {"physical_episodes": {"cap": PHYSICAL_EPISODE_CAP, "reserved": [], "used": 0},
                                            "provider_first_requests": 0, "provider_retries": 0, "rl_transitions": 0, "optimizer_steps": 0})
    event(out, "gate0", status=auth["status"], problems=problems)
    return auth


def register(root, out):
    """Phase 0/1: existing dev states -> registry, G1, offline planner/relation quantities; freeze branch registration."""
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.facts import Truth
    root, out = Path(root).resolve(), Path(out).resolve()
    s1 = _s1_module(root)
    split = read_json(root / SPLIT_REL)
    rows = list(split[SPLIT_NAME])[:SOURCE_STATE_CAP]
    bundle, manifest = build_runtime(root, out)
    planner = BPlanPlanner(SearchConfig(**SEARCH))
    states, screening = [], []
    try:
        for row in rows:
            cid = row["case_id"]
            snap = bundle.start_case(cid)
            ident = pre_state_identity(bundle, snap)
            values = dict(snap.facts.values)
            contracts = {c.id: c for c in snap.template.contracts}
            legal = [c for c, m in zip(snap.candidate_ids, snap.mask) if m]
            patches = {c: nominal_patch(snap.template, values, contracts[c]) for c in legal}
            plan = planner.plan(snap.facts, snap.template, DEADLINE)
            rels_star = [list(x) for x in s1.r_star_relations(snap.template)]
            rplan, rscore = s1.relation_plan(snap.facts, snap.template, DEADLINE, [tuple(x) for x in rels_star], SearchConfig(**SEARCH))
            cache = root / row["cache_dir"] if row.get("cache_dir") else None
            raw, rej, fin = s1.load_cache_payload(cache) if cache and cache.exists() else ([], [], [])
            natural = {"cache_key": row.get("cache_key", ""), "cache_status": row.get("cache_status", ""), "raw": len(raw), "rejected": len(rej),
                       "admitted": len(fin), "contract_redundant": sum(1 for x in rej if isinstance(x, dict) and x.get("reason") == "CONTRACT_REDUNDANCY"),
                       "admitted_relations": [list(s1.relation_tuple(x)) for x in fin]}
            g1 = len(legal) >= 2 and all(p for p in (PAIR[0] in legal, PAIR[1] in legal))
            states.append({"state_id": digest([cid, ident["snapshot_hash"]])[:16], "scene_id": cid, "snapshot_hash": ident["snapshot_hash"],
                           "public_observation_hash": ident["proprio_hash"], "facts_hash": ident["facts_hash"], "goal_hash": ident["goal_hash"],
                           "candidate_ids": ident["candidate_ids"], "candidate_mask": ident["candidate_mask"], "candidate_count": len(legal),
                           "setup_trace": "start_case(case_id) reset recipe; no skill executed", "source_role": "existing_dev",
                           "qpos_hash": ident["qpos_hash"], "restore_receipt_sha256": ident["restore_receipt_sha256"],
                           "restore_verified": ident["restore_verified"], "rank_key": state_rank_key(cid), "g1_pair_legal": g1,
                           "nominal_patches": patches, "b_plan": {"status": plan.status, "plan": list(plan.plan), "depth": plan.depth},
                           "r_star_relations": rels_star, "r_star_plan": {"status": rplan.status, "plan": list(rplan.plan), "score": rscore},
                           "natural_relations": natural, "restore_seed": int(row["seed"]), "facts": ident["facts"]})
            screening.append({"scene_id": cid, "snapshot_hash": ident["snapshot_hash"][:16], "legal_count": len(legal), "g1_pair_legal": g1,
                              "restore_verified": ident["restore_verified"], "b_plan_status": plan.status, "b_plan_first": (plan.plan or [""])[0],
                              "b_plan_depth": plan.depth, "r_star_first": (rplan.plan or [""])[0], "natural_admitted": len(fin), "natural_raw": len(raw),
                              "rank_key": state_rank_key(cid)[:12]})
    finally:
        bundle.environment.close()
    write_json(out / "candidate_state_registry.json", {"card": CARD, "states": states, "state_count": len(states), "cap": SOURCE_STATE_CAP})
    write_csv(out / "decision_state_screening.csv", list(screening[0].keys()), screening)
    eligible = sorted([s for s in states if s["g1_pair_legal"] and s["restore_verified"] and s["b_plan"]["status"] == "PLAN_FOUND"], key=lambda s: s["rank_key"])
    chosen = eligible[:WITNESS_STATES]
    branches = []
    for idx, s in enumerate(chosen):
        reps = (0, 1) if idx == 0 else (0,)
        for cand in PAIR:
            for rep in reps:
                branches.append({"branch_id": branch_id_of(s["scene_id"], cand, rep), "case_id": s["scene_id"], "candidate_id": cand, "repeat": rep,
                                 "wave": 1 if idx == 0 else 2, "snapshot_hash": s["snapshot_hash"], "facts_hash": s["facts_hash"],
                                 "candidate_hash": digest([s["candidate_ids"], s["candidate_mask"]]), "qpos_hash": s["qpos_hash"],
                                 "nominal_depth": s["b_plan"]["depth"], "restore_seed": s["restore_seed"]})
    assert len(branches) <= PHYSICAL_EPISODE_CAP, "registered branches exceed physical cap"
    write_json(out / "branch_registration.json", {"card": CARD, "status": "REGISTERED" if len(chosen) == WITNESS_STATES else "INSUFFICIENT_G1_STATES",
                                                  "g1_passing": len(eligible), "chosen_scene_ids": [s["scene_id"] for s in chosen],
                                                  "branches": branches, "episode_cap": PHYSICAL_EPISODE_CAP,
                                                  "selection_rule": "ascending sha256(namespace|case_id) among G1-passing; no model output consulted"}, mode=0o444)
    write_json(out / "continuation_contract.json", {"card": CARD, "frozen_before_any_branch_result": True, "planner": "BPlanPlanner", "search": SEARCH,
                                                    "first_action": "forced candidate", "inputs": ["public FactStore", "goal", "registered contracts", "remaining deadline"],
                                                    "prohibited": ["VLM relation", "RL Q", "hidden truth", "Full/A_STAT/A_CAT outputs"], "same_for_both_candidates": True,
                                                    "max_decisions": MAX_DECISIONS, "failures_in_denominator": True,
                                                    "runtime_manifest_sha256": digest(manifest)})
    event(out, "register", states=len(states), g1_passing=len(eligible), branches=len(branches))
    return {"states": len(states), "g1_passing": len(eligible), "branches": len(branches)}


def _gpu_pick(n):
    busy = {l.split(",")[-1].strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()}
    uuid = {}
    for l in subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.splitlines():
        i, u, m = [x.strip() for x in l.split(",")]
        uuid[int(i)] = (u, int(m))
    free = [i for i, (u, m) in sorted(uuid.items()) if u not in busy and m < 2000]
    return free[:n], {i: list(v) for i, v in uuid.items()}


def worker(root, out, branch_id):
    """One registered physical branch: restore -> forced candidate -> postcondition -> fixed B_PLAN continuation."""
    from cp_disr.adapters import EvaluationInput
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.facts import FactStore
    root, out = Path(root).resolve(), Path(out).resolve()
    reg = read_json(out / "branch_registration.json")
    b = next(x for x in reg["branches"] if x["branch_id"] == branch_id)
    rdir = out / "branch_receipts"
    rdir.mkdir(exist_ok=True)
    try:
        fd = os.open(rdir / f"{branch_id}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise RuntimeError("STOPPED_BUDGET:branch already claimed; no retry without new authorization")
    journal = out / "branches" / f"{branch_id}.jsonl"
    journal.parent.mkdir(exist_ok=True)

    def jr(kind, **kw):
        with journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **kw}, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    res = {"branch_id": branch_id, "case_id": b["case_id"], "candidate_id": b["candidate_id"], "repeat": b["repeat"], "valid": False,
           "termination": "", "protocol_complete": False, "g3_candidate_stable": False, "restore_matches_registry": False,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "trace": []}
    bundle = None
    wall0 = time.monotonic()
    try:
        bundle, _ = build_runtime(root, out)
        snap = bundle.start_case(b["case_id"], restore_seed=int(b["restore_seed"]))
        ident = pre_state_identity(bundle, snap)
        match = {k: ident[k] == b[k] for k in ("snapshot_hash", "facts_hash", "candidate_hash", "qpos_hash")}
        res["restore_check"] = match
        res["restore_matches_registry"] = bool(all(match.values()) and ident["restore_verified"])
        jr("restored", **match)
        if not res["restore_matches_registry"]:
            res["termination"] = "RESTORE_MISMATCH"
            return res
        planner = BPlanPlanner(SearchConfig(**SEARCH))
        contracts = {c.id: c for c in snap.template.contracts}
        start = float(bundle.episode_start_seconds)
        candidate, stage = b["candidate_id"], "candidate"
        for decision in range(MAX_DECISIONS):
            now = float(bundle.clock.now_seconds())
            if now - start >= DEADLINE:
                res["termination"] = "DEADLINE"
                res["protocol_complete"] = True
                break
            idx = {c: i for i, c in enumerate(snap.candidate_ids)}
            if candidate not in idx or not snap.mask[idx[candidate]]:
                res["termination"] = "MASK_REJECTED"
                res["protocol_complete"] = stage != "candidate"
                break
            ex = bundle.executor.execute(candidate, float(contracts[candidate].timeout_seconds))
            obs = bundle.observations.observe()
            records = bundle.verifier.verify(bundle.perception.infer(obs), ex)
            end = float(bundle.clock.now_seconds())
            snap = bundle.snapshot_builder.build(snap, records, obs, ex, end)
            task = bundle.evaluator.evaluate(EvaluationInput(task_id=bundle.task_id, env_id=snap.env_id, episode_id=snap.episode_id,
                                                             evidence_refs=tuple(ex.evidence_ids), elapsed_seconds=end - start,
                                                             interval_start_seconds=float(ex.start_seconds) - start, interval_end_seconds=end - start))
            row = {"decision": decision, "stage": stage, "action": candidate, "controller_exit": ex.controller_exit,
                   "elapsed_start": float(ex.start_seconds) - start, "elapsed_end": end - start, "task_success": bool(task.success),
                   "terminated": bool(task.terminated), "reason": task.reason}
            if stage == "candidate":
                tgt = "p:Open:container" if ":OPEN:" in candidate else "p:Held:" + candidate.split(":")[2]
                row["postcondition_fact"] = tgt
                row["postcondition_true"] = snap.facts.values.get(tgt) is not None and snap.facts.values[tgt].value == "TRUE"
                res["g3_candidate_stable"] = bool(ex.controller_exit == "NORMAL_TERMINATION" and row["postcondition_true"])
            res["trace"].append(row)
            jr("skill", **row)
            if stage == "candidate" and not res["g3_candidate_stable"]:
                res["termination"] = "CANDIDATE_UNSTABLE"
                res["protocol_complete"] = False
                break
            if task.terminated or task.truncated:
                res["termination"] = task.reason
                res["protocol_complete"] = True
                break
            if ex.controller_exit != "NORMAL_TERMINATION":
                res["termination"] = "CONTINUATION_SKILL_FAILURE"
                res["protocol_complete"] = True
                break
            plan = planner.plan(snap.facts, snap.template, max(0.0, DEADLINE - (end - start)))
            jr("planner", status=plan.status, plan=list(plan.plan), expanded=plan.expanded_nodes)
            if plan.status != "PLAN_FOUND" or not plan.plan:
                res["termination"] = plan.status
                res["protocol_complete"] = True
                break
            candidate, stage = plan.plan[0], "continuation"
        else:
            res["termination"] = "DECISION_CAP"
        res.update(branch_outcome(res["trace"], nominal_depth=b["nominal_depth"]))
        res["valid"] = bool(res["protocol_complete"] and res["g3_candidate_stable"])
        return res
    except Exception as exc:  # engineering failure: preserved, never retried, never converted to a result
        import traceback
        res["termination"] = "ENGINEERING_EXCEPTION"
        res["error"] = f"{type(exc).__name__}: {exc}"
        jr("error", traceback=traceback.format_exc())
        return res
    finally:
        if "g_start_discounted" not in res:
            res.update(branch_outcome(res["trace"], nominal_depth=b["nominal_depth"]))
        res["wall_seconds"] = time.monotonic() - wall0
        write_json(out / "branches" / f"{branch_id}.json", res)
        if bundle is not None:
            try:
                bundle.environment.close()
            except Exception:
                pass


def _worker_env(gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    return env


def run_wave(root, out, wave, gpus):
    root, out = Path(root).resolve(), Path(out).resolve()
    reg = read_json(out / "branch_registration.json")
    todo = [b for b in reg["branches"] if b["wave"] == wave and not (out / "branches" / f"{b['branch_id']}.json").exists()]
    ledger = read_json(out / "budget_ledger.json")
    pending = list(todo)
    running = []
    results = []
    while pending or running:
        while pending and len(running) < len(gpus):
            gpu = [g for g in gpus if g not in [r[2] for r in running]][0]
            b = pending.pop(0)
            if len(ledger["physical_episodes"]["reserved"]) >= ledger["physical_episodes"]["cap"]:
                raise RuntimeError("STOPPED_BUDGET:physical episode cap")
            ledger["physical_episodes"]["reserved"].append(b["branch_id"])
            write_json(out / "budget_ledger.json", ledger)
            log = (out / "logs").joinpath(f"{b['branch_id']}.log")
            log.parent.mkdir(exist_ok=True)
            p = subprocess.Popen([sys.executable, str(root / "scripts/tp_ef_discovery.py"), "worker", "--root", str(root), "--output", str(out),
                                  "--branch-id", b["branch_id"]], cwd=root, env=_worker_env(gpu), stdout=log.open("ab"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            running.append((p, time.monotonic(), gpu, b))
            event(out, "worker_start", branch=b["branch_id"], gpu=gpu, case=b["case_id"], candidate=b["candidate_id"], repeat=b["repeat"])
        time.sleep(2)
        for item in list(running):
            p, t0, gpu, b = item
            code = p.poll()
            if code is None and time.monotonic() - t0 > WATCHDOG_SECONDS:
                p.kill()
                code = "WATCHDOG"
                if not (out / "branches" / f"{b['branch_id']}.json").exists():
                    write_json(out / "branches" / f"{b['branch_id']}.json", {"branch_id": b["branch_id"], "case_id": b["case_id"], "candidate_id": b["candidate_id"],
                                                                         "repeat": b["repeat"], "valid": False, "termination": "WATCHDOG_TIMEOUT", "trace": []})
            if code is not None:
                running.remove(item)
                results.append({"branch_id": b["branch_id"], "exit": code})
                event(out, "worker_end", branch=b["branch_id"], exit=code)
    ledger["physical_episodes"]["used"] = len([f for f in (out / "branches").glob("*.json")])
    write_json(out / "budget_ledger.json", ledger)
    return {"wave": wave, "workers": results}


def freeze_epsilon(out):
    """After wave 1: epsilon from within-candidate repeats only; written (read-only) before cross-candidate comparison."""
    out = Path(out)
    rows = [read_json(p) for p in sorted((out / "branches").glob("*.json"))]
    eps = epsilon_from_repeats(rows)
    eps["frozen_before_cross_candidate_comparison"] = True
    write_json(out / "repeatability_and_epsilon_q.json", eps, mode=0o444)
    return eps


def _load_branch_rows(out):
    return [read_json(p) for p in sorted((Path(out) / "branches").glob("*.json"))]


def analyze(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    reg_states = read_json(out / "candidate_state_registry.json")["states"]
    reg = read_json(out / "branch_registration.json")
    eps = read_json(out / "repeatability_and_epsilon_q.json")
    rows = _load_branch_rows(out)
    state_by = {s["scene_id"]: s for s in reg_states}
    # ---- Phase 2 / G2-G4 (E4) -------------------------------------------------------------------
    paired = []
    for case_id in reg["chosen_scene_ids"]:
        paired.append(compare_pair(case_id, [r for r in rows if r["case_id"] == case_id], eps))
    write_csv(out / "paired_branch_results.csv", ["branch_id", "case_id", "candidate_id", "repeat", "valid", "termination", "g3_candidate_stable",
                                                   "restore_matches_registry", "success", "completion_seconds", "success_seconds", "g_start_discounted",
                                                   "skill_count", "rework_count", "wall_seconds", "gpu"], rows)
    witnesses = [p for p in paired if p.get("reliable")]
    g2 = all(r.get("restore_matches_registry") for r in rows) and bool(rows)
    g3 = all(r.get("g3_candidate_stable") for r in rows) and bool(rows)
    g4 = len(witnesses) >= 2
    # ---- E1 / E2 / E3 on the natural relations ----------------------------------------------------
    nat_nonempty = [s["scene_id"] for s in reg_states if s["natural_relations"]["admitted"] > 0]
    e1 = {"status": "PASS" if len(nat_nonempty) >= 2 else "FAIL", "states_screened": len(reg_states),
          "states_with_admitted_nonredundant_relation": len(nat_nonempty),
          "raw_relations_total": sum(s["natural_relations"]["raw"] for s in reg_states),
          "rejected_contract_redundancy_total": sum(s["natural_relations"]["contract_redundant"] for s in reg_states),
          "empty_cache_states": [s["scene_id"] for s in reg_states if s["natural_relations"]["raw"] == 0],
          "failure_class": "" if len(nat_nonempty) >= 2 else "NATURAL_RELATIONS_EMPTY_OR_CONTRACT_REDUNDANT",
          "oracle_or_handwritten_used_for_gate": False}
    e2 = {"status": "PASS" if e1["status"] == "PASS" else "FAIL", "legal_candidates_ge_2_states": sum(1 for s in reg_states if s["candidate_count"] >= 2),
          "changed_nominal_patch_states": sum(1 for s in reg_states if any(s["nominal_patches"].values())),
          "natural_R_entering_representation_states": len(nat_nonempty),
          "failure_class": "" if e1["status"] == "PASS" else "NO_ADMITTED_R_TO_ENTER_REPRESENTATION"}
    star_diff = []
    for s in reg_states:
        a = (s["b_plan"]["plan"] or [""])[0]
        r = (s["r_star_plan"]["plan"] or [""])[0]
        star_diff.append({"scene_id": s["scene_id"], "b_plan_first": a, "r_star_first": r, "differs": a != r, "r_star_relation_count": len(s["r_star_relations"])})
    e3 = {"status": "PASS" if e2["status"] == "PASS" else "FAIL", "natural_R_signal": "NONE" if not nat_nonempty else "PRESENT",
          "r_star_structural_candidate_relative_signal_states": sum(1 for d in star_diff if d["differs"]),
          "neural_policy_forward_probe": "NOT_RUN_NO_ADMITTED_R_AND_NO_POLICY_BOUND_IN_THIS_CARD",
          "note": "E2*/E3 oracle capacity probe is structural only here; it does not substitute for natural-R gates"}
    # ---- E5 contract-legitimate insufficiency ----------------------------------------------------
    planner_ok = all(s["b_plan"]["status"] == "PLAN_FOUND" for s in reg_states)
    e5 = {"status": "PASS" if (planner_ok and g4 and g2 and g3) else "FAIL", "b_plan_finds_registered_nominal_plan_all_states": planner_ok,
          "b_plan_depth_values": sorted({s["b_plan"]["depth"] for s in reg_states}), "both_candidates_in_same_real_mask": all(s["g1_pair_legal"] for s in reg_states if s["scene_id"] in reg["chosen_scene_ids"]),
          "differences_not_from_hard_constraint_or_planner": all(not p.get("artefact_planner_or_constraint") for p in paired),
          "k_abstract_tie": "both candidates have identical nominal-length plans (depth 5) under the registered contracts",
          "reliable_witness_states": [w["case_id"] for w in witnesses]}
    # ---- E6 -----------------------------------------------------------------------------------------
    helpful = False
    comp = []
    for w in witnesses:
        s = state_by[w["case_id"]]
        r_first = (s["r_star_plan"]["plan"] or [""])[0]
        b_first = (s["b_plan"]["plan"] or [""])[0]
        comp.append({"case_id": w["case_id"], "better_candidate": w["direction"], "b_plan_first": b_first, "r_star_first": r_first,
                     "b_plan_agrees": b_first == w["direction"], "r_star_agrees": r_first == w["direction"], "r_star_differs_from_b_plan": r_first != b_first})
        helpful = helpful or (r_first == w["direction"] and r_first != b_first)
    rstar_differs = any(d["differs"] for d in star_diff)
    e6_class = classify_e6(len(nat_nonempty), helpful, rstar_differs, bool(witnesses))
    e6 = {"status": "PASS" if e6_class in ALLOWED_ELIGIBLE and e1["status"] == "PASS" else "FAIL", "classification": e6_class,
          "witness_comparison": comp, "rstar_is_reference_not_upper_bound": True,
          "physical_planner_episodes_run": 0, "note": "B_PLAN/B_PLAN+R/B_PLAN+R* compared via first-action vs paired physical Q_ref; no extra planner episodes"}
    gates = {"G1": "PASS" if sum(1 for s in reg_states if s["g1_pair_legal"]) >= 2 else "FAIL", "G2": "PASS" if g2 else "FAIL", "G3": "PASS" if g3 else "FAIL",
             "G4": "PASS" if g4 else "FAIL", "E1": e1["status"], "E2": e2["status"], "E3": e3["status"], "E4": "PASS" if g4 else "FAIL",
             "E5": e5["status"], "E6": e6["status"]}
    eligible = all(v == "PASS" for v in gates.values()) and e6_class in ALLOWED_ELIGIBLE
    stop = []
    if e1["status"] != "PASS":
        stop.append("STOP_2:no_natural_non_redundant_relation_without_provider_revision")
    if not g4:
        stop.append("STOP_1_or_3:no_two_reliable_physical_witnesses")
    verdict = {"card": CARD, "decision": "ELIGIBLE" if eligible else "NOT_ELIGIBLE", "gates": gates, "e6_classification": e6_class,
               "stop_conditions_hit": stop, "witness_states": [w["case_id"] for w in witnesses],
               "next_action": "S2_TASK_FAMILY_FREEZE_REQUEST" if eligible else "RESEARCH_DECISION_ONE_EVIDENCE_BASED_PROVIDER_SCHEMA_REVISION_REQUIRES_CREDENTIAL_AND_AUTHORIZATION" if e1["status"] != "PASS" else "RESEARCH_DECISION",
               "tp_training_authorized": False, "family_c_created": False, "robocasa_migration": False}
    write_json(out / "e1_e6_gate.json", {"gates": gates, "E1": e1, "E2": e2, "E3": e3, "E4": {"witnesses": witnesses, "paired": paired}, "E5": e5, "E6": e6,
                                         "r_star_vs_b_plan": star_diff, "epsilon": {k: eps[k] for k in ("epsilon_q", "epsilon_t", "repeat_g_range", "repeat_time_range")}})
    write_json(out / "eligibility_manifest.json", verdict)
    write_json(out / "provider_cost_ledger.json", {"first_requests": 0, "retries": 0, "semantic_requery": 0, "cost": 0.0,
                                                   "reason": "no provider credential in the execution environment; existing T_B dev caches reused for E1 (admission recomputed from saved payloads)",
                                                   "existing_cache_keys": {s["scene_id"]: s["natural_relations"]["cache_key"] for s in reg_states}})
    write_json(out / "relation_schema.json", {"universe": list(RELATION_UNIVERSE), "admission": ["legal ids", "effect_fact_ref registered ADD/DEL of source", "no self loop", "not contract-redundant"],
                                              "hidden": "relation truth/utility/opportunity are never inputs to admission"})
    write_json(out / "relation_admission_receipts.json", {s["scene_id"]: s["natural_relations"] for s in reg_states})
    summarize(out, verdict, paired, eps, e1, e5, e6)
    return verdict


def summarize(out, verdict, paired, eps, e1, e5, e6):
    lines = [f"# {CARD} — final discovery summary", "", f"Decision: **{verdict['decision']}**  (E6 class: `{verdict['e6_classification']}`)", "",
             "## Gates", ""] + [f"- {k}: {v}" for k, v in verdict["gates"].items()] + ["",
             "## Physical witnesses (paired B_PLAN continuation; Q_ref = start-discounted return)", ""]
    for p in paired:
        lines.append(f"- {p['case_id']}: status={p.get('status')} dQ={p.get('delta_q')} dT={p.get('delta_t')} reasons={p.get('reasons')} direction={p.get('direction')}")
    lines += ["", f"epsilon_q={eps['epsilon_q']:.4f}, epsilon_t={eps['epsilon_t']:.3f}s (repeat ranges: g={eps['repeat_g_range']:.4g}, t={eps['repeat_time_range']:.4g}s)", "",
              f"E1: {e1['states_with_admitted_nonredundant_relation']}/{e1['states_screened']} states have an admitted non-redundant natural relation "
              f"(raw={e1['raw_relations_total']}, contract-redundant rejected={e1['rejected_contract_redundancy_total']}, empty caches={len(e1['empty_cache_states'])}).", "",
              f"Stop conditions: {verdict['stop_conditions_hit'] or 'none'}", f"Next: {verdict['next_action']}", "",
              "Budget: RL=0, optimizer=0, provider requests=0, test=0; physical episodes per ledger.", ""]
    (Path(out) / "final_discovery_summary.md").write_text("\n".join(lines), encoding="utf-8")


def protected_inventory(root):
    root = Path(root)
    pats = ["configs/splits/*test*.json", "docs/authoritative/*", "status/*.json", "runs/final_master/S4/family_b_*/*/verify.json"]
    inv = {}
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                inv[str(p.relative_to(root))] = sha256_file(p)
    return inv


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    checks = {}
    need = ["authorization.json", "source_identity.json", "candidate_state_registry.json", "decision_state_screening.csv", "branch_registration.json",
            "paired_branch_results.csv", "continuation_contract.json", "repeatability_and_epsilon_q.json", "relation_schema.json",
            "relation_admission_receipts.json", "e1_e6_gate.json", "eligibility_manifest.json", "provider_cost_ledger.json", "technical_events.jsonl",
            "final_discovery_summary.md"]
    checks["outputs_present"] = {n: (out / n).is_file() for n in need}
    ledger = read_json(out / "budget_ledger.json")
    used = len(list((out / "branches").glob("*.json")))
    checks["episode_cap_ok"] = used <= PHYSICAL_EPISODE_CAP
    checks["zero_rl_optimizer_provider"] = all(ledger[k] == 0 for k in ("provider_first_requests", "provider_retries", "rl_transitions", "optimizer_steps"))
    secrets = 0
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".md", ".log"}:
            t = p.read_text(errors="ignore")
            if "sk-" in t or "Authorization" in t or "DASHSCOPE_API_KEY=" in t:
                secrets += 1
    checks["no_secret_shaped_content"] = secrets == 0
    checks["registration_readonly_hash"] = sha256_file(out / "branch_registration.json")
    prot = read_json(out / "protected_before.json") if (out / "protected_before.json").exists() else None
    checks["protected_unchanged"] = (prot == protected_inventory(root)) if prot is not None else "NO_BASELINE"
    checks["status"] = "PASS" if (all(checks["outputs_present"].values()) and checks["episode_cap_ok"] and checks["zero_rl_optimizer_provider"]
                                 and checks["no_secret_shaped_content"] and checks["protected_unchanged"] is True) else "FAIL"
    write_json(out / "verify.json", checks)
    return checks
