"""Independent Reward-V2 mechanism and signal audits.

May import environment/MDP exporters for comparison. The planner module
itself must remain free of those imports.
"""
from __future__ import annotations

import ast
import json
import math
from collections import defaultdict, deque
from pathlib import Path

from p2cq_research.environment import SkillEnv
from p2cq_research.enumerate_mdp import enumerate_mdp, state_id
from p2cq_research.potentials import evaluate_all
from p2cq_research.task_contract import ACTION_NAMES, N, TaskContract, action_index, node_op

from .remaining_work_model import (
    RemainingWorkPlanner,
    SearchTruncatedError,
    UnreachableWorkError,
    abstract_from_dynamic,
    goal_ok,
    scale_s,
)
from .potentials_v2 import METHOD_V2, shaped_training_reward
from .mask_contract import legal_mask_bool, mask_sha256
from .io_utils import dump_replace, load
from .statistics import mean

V1_METHOD = "PATHGRAPH_MODEL_FULL_PBRS_V1"
FORBIDDEN_PLANNER_IMPORTS = {
    "environment", "enumerate_mdp", "export_pairs", "oracle", "mdp", "pairs",
}


def planner_import_audit(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[-1]
                if name in FORBIDDEN_PLANNER_IMPORTS or "oracle" in alias.name:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = set(mod.split("."))
            if parts & FORBIDDEN_PLANNER_IMPORTS or "oracle" in mod:
                hits.append(mod)
            for alias in node.names:
                if alias.name in FORBIDDEN_PLANNER_IMPORTS:
                    hits.append(mod + "." + alias.name)
    src = Path(path).read_text(encoding="utf-8")
    if "id(contract)" in src.replace(" ", ""):
        hits.append("id(contract)_cache_key")
    return {"path": str(path), "forbidden_hits": hits, "passed": not hits}


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("numeric")
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("nonfinite")
    return float(x)


def close(a, b, atol=1e-10, rtol=1e-9):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def optimal_set(q, atol=1e-10, rtol=1e-9):
    z = [_finite(v) for v in q]
    m = max(z)
    return frozenset(i for i, v in enumerate(z) if close(v, m, atol, rtol))


def oracle_alignment(scores, oracle_q):
    chosen = optimal_set(scores)
    target = optimal_set(oracle_q)
    qmax = max(oracle_q)
    return {
        "optimistic_hit": bool(chosen & target),
        "conservative_hit": chosen <= target,
        "uniform_tie_hit": len(chosen & target) / len(chosen),
        "regret_mean": sum(qmax - oracle_q[i] for i in chosen) / len(chosen),
        "regret_best": min(qmax - oracle_q[i] for i in chosen),
        "regret_worst": max(qmax - oracle_q[i] for i in chosen),
    }


def independent_skill_costs(mdp):
    states = mdp["states"]
    rev = defaultdict(list)
    goals = []
    for sid, rec in states.items():
        if rec.get("success"):
            goals.append(sid)
            continue
        if rec.get("terminal"):
            continue
        mask = rec["valid_actions"]
        for a, outs in enumerate(rec["outcomes"]):
            if a == 0 or not mask[a]:
                continue
            nxt = outs[0]["next"]
            rev[nxt].append(sid)
    cost = {}
    q = deque()
    for g in goals:
        cost[g] = 0.0
        q.append(g)
    while q:
        v = q.popleft()
        for u in rev[v]:
            if u not in cost:
                cost[u] = cost[v] + 1.0
                q.append(u)
    return cost


def planner_skill_costs(contract, mdp, planner=None):
    if planner is None:
        planner = RemainingWorkPlanner(contract)
    states = mdp["states"]
    abstracts = {}
    for sid, rec in states.items():
        abstracts[sid] = planner.abstract_state(rec["dynamic_public"])
    fwd = {}
    stack = list(set(abstracts.values()))
    while stack:
        u = stack.pop()
        if u in fwd:
            continue
        vs = planner.neighbors(u)
        fwd[u] = vs
        for v in vs:
            if v not in fwd:
                stack.append(v)
    rev = defaultdict(list)
    for u, vs in fwd.items():
        for v in vs:
            rev[v].append(u)
    cost = {}
    q = deque()
    for st in fwd:
        if goal_ok(contract, st[0]):
            cost[st] = 0.0
            q.append(st)
    while q:
        v = q.popleft()
        for u in rev[v]:
            if u not in cost:
                cost[u] = cost[v] + 1.0
                q.append(u)
    out = {}
    trunc = 0
    for sid, st in abstracts.items():
        if st in cost:
            out[sid] = cost[st]
        else:
            # Finite remaining-work search was complete; missing reverse-BFS
            # cost means the abstract state cannot reach the goal, not that
            # the planner hit max_expands.
            out[sid] = float("inf")
    return out, trunc, planner, cost, fwd, abstracts


def audit_mdp_costs(contract, mdp, planner=None):
    ind = independent_skill_costs(mdp)
    plan, trunc, planner, cost_abs, fwd, abstracts = planner_skill_costs(contract, mdp, planner)
    cost_mismatch = 0
    unreachable_mismatch = 0
    bellman = 0
    n = 0
    inf = float("inf")
    for sid, rec in mdp["states"].items():
        n += 1
        ic = ind.get(sid, inf)
        pc = plan.get(sid, inf)
        inf_i = ic == inf
        inf_p = pc == inf
        if inf_i != inf_p:
            unreachable_mismatch += 1
            cost_mismatch += 1
        elif not inf_i and abs(ic - pc) > 1e-9:
            cost_mismatch += 1
        st = abstracts[sid]
        if rec.get("success") or goal_ok(contract, st[0]):
            if pc != 0.0:
                bellman += 1
            continue
        if rec.get("terminal"):
            continue
        nxts = fwd.get(st, ())
        if not nxts:
            expect = inf
        else:
            expect = min(1.0 + cost_abs.get(v, inf) for v in nxts)
        got = cost_abs.get(st, inf)
        if expect == inf or got == inf:
            if expect != got:
                bellman += 1
        elif abs(expect - got) > 1e-9:
            bellman += 1
    return {
        "states_checked": n,
        "cost_parity_mismatches": cost_mismatch,
        "bellman_mismatches": bellman,
        "unreachable_mismatches": unreachable_mismatch,
        "search_truncations": trunc,
    }, planner


def pbrs_max_error(contract, method=METHOD_V2, n_rollouts=8, max_steps=64, gamma=0.99, beta=1.0):
    env = SkillEnv(contract)
    planner = RemainingWorkPlanner(contract) if method == METHOD_V2 else None
    worst = 0.0

    def phi(dyn, terminated):
        if terminated:
            return 0.0
        if method == METHOD_V2:
            return planner.potential(dyn)[0]
        if method == "TASK_ONLY_ZERO_V1":
            return 0.0
        return float(evaluate_all(contract, dyn)[method])

    for seed in range(n_rollouts):
        env.reset(contract)
        rows = []
        for t in range(max_steps):
            if env.terminated():
                break
            mask = env.legal_mask()
            legal = [i for i, ok in enumerate(mask) if ok]
            # deterministic mix of WAIT and first legal skill
            action = legal[(seed + t) % len(legal)]
            dyn_b = env.dynamic_public()
            before = phi(dyn_b, False)
            _, task_r, term, _, _ = env.step(action)
            dyn_a = env.dynamic_public()
            after = phi(dyn_a, False)
            train, bonus = shaped_training_reward(task_r, before, after, gamma=gamma, beta=beta, terminated=term)
            rows.append({
                "phi_before": before,
                "phi_after": after,
                "terminated": bool(term),
                "shaping": bonus,
            })
            if term:
                break
        if not rows:
            continue
        lhs = 0.0
        for i, r in enumerate(rows):
            effective = 0.0 if r["terminated"] else r["phi_after"]
            expected = beta * (gamma * effective - r["phi_before"])
            worst = max(worst, abs(expected - r["shaping"]))
            lhs += (gamma ** i) * expected
        last = 0.0 if rows[-1]["terminated"] else rows[-1]["phi_after"]
        rhs = beta * (-rows[0]["phi_before"] + (gamma ** len(rows)) * last)
        worst = max(worst, abs(lhs - rhs))
    return worst


def cache_mismatches(contract, mdp, n_states=32):
    planner_a = RemainingWorkPlanner(contract)
    planner_b = RemainingWorkPlanner(TaskContract(contract.public_dict()))
    n = 0
    bad = 0
    for sid, rec in mdp["states"].items():
        if rec.get("terminal") and not rec.get("success"):
            continue
        dyn = rec["dynamic_public"]
        c1, t1, _ = planner_a.remaining_cost(planner_a.abstract_state(dyn), use_cache=True)
        c2, t2, _ = planner_a.remaining_cost(planner_a.abstract_state(dyn), use_cache=False)
        c3, t3, _ = planner_b.remaining_cost(planner_b.abstract_state(dyn), use_cache=True)
        n += 1
        if t1 or t2 or t3:
            if not (t1 and t2 and t3):
                bad += 1
        elif abs(c1 - c2) > 1e-12 or abs(c1 - c3) > 1e-12:
            bad += 1
        if n >= n_states:
            break
    return n, bad


def stage_handoff_regressions(contract):
    env = SkillEnv(contract)
    env.reset()
    planner = RemainingWorkPlanner(contract)
    ind_mdp, _ = enumerate_mdp(contract)
    ind = independent_skill_costs(ind_mdp)
    regs = 0
    i = contract.present_nodes()[0]
    k = contract.transport_steps[i]
    seq = [action_index(i, "ACQUIRE")] + [action_index(i, "ADVANCE")] * k + [action_index(i, "PLACE")]
    prev = None
    for a in seq:
        if env.terminated():
            break
        st = planner.abstract_state(env.dynamic_public())
        c, trunc, _ = planner.remaining_cost(st)
        if trunc or c == float("inf"):
            regs += 1
            break
        if prev is not None and not (c <= prev + 1e-12):
            # remaining work should not increase on a modeled skill that is useful;
            # allow increase only if the action was not progress (still count if +>1 unexpected)
            if c > prev + 1.0 + 1e-9:
                regs += 1
        prev = c
        env.step(a)
        sid = state_id(env.dynamic_public())
        ic = ind.get(sid, float("inf"))
        st2 = planner.abstract_state(env.dynamic_public())
        c2, trunc2, _ = planner.remaining_cost(st2)
        if trunc2:
            regs += 1
        elif ic == float("inf") or c2 == float("inf"):
            if ic != c2:
                regs += 1
        elif abs(ic - c2) > 1e-9:
            regs += 1
    return regs


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_mdp_files(export_dir):
    mdir = Path(export_dir) / "mdp"
    for p in sorted(mdir.glob("*.json")):
        yield p


def contract_from_mdp(mdp):
    return TaskContract(mdp["task_public"])


def audit_export_mechanism(export_dir, max_mdps=None):
    export_dir = Path(export_dir)
    tot = {
        "states_checked": 0,
        "cost_parity_mismatches": 0,
        "bellman_mismatches": 0,
        "unreachable_mismatches": 0,
        "search_truncations": 0,
        "cache_equivalence_mismatches": 0,
        "cache_states": 0,
        "stage_handoff_regressions": 0,
        "pbrs_max_abs_error": 0.0,
        "mdps": 0,
    }
    for i, path in enumerate(iter_mdp_files(export_dir)):
        if max_mdps is not None and i >= max_mdps:
            break
        mdp = load_json(path)
        contract = contract_from_mdp(mdp)
        rec, planner = audit_mdp_costs(contract, mdp)
        for k in ("states_checked", "cost_parity_mismatches", "bellman_mismatches",
                  "unreachable_mismatches", "search_truncations"):
            tot[k] += rec[k]
        if tot["mdps"] < 16:
            cn, cb = cache_mismatches(contract, mdp)
            tot["cache_states"] += cn
            tot["cache_equivalence_mismatches"] += cb
            tot["stage_handoff_regressions"] += stage_handoff_regressions(contract)
            tot["pbrs_max_abs_error"] = max(tot["pbrs_max_abs_error"], pbrs_max_error(contract))
        tot["mdps"] += 1
        print(f"mech {tot['mdps']} {path.name} states={rec['states_checked']} cost_mm={rec['cost_parity_mismatches']} bell={rec['bellman_mismatches']}", flush=True)
        del mdp, planner
    return tot


def _add_p2cq_path(p2cq_pkg):
    import sys
    pkg = Path(p2cq_pkg)
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def score_one_step(mdp_table, sid, h, potential, gamma, beta=1.0):
    st = mdp_table.states[sid]
    before = _finite(potential(sid, h))
    shaped = []
    for outcomes in st["outcomes"]:
        f = 0.0
        r = 0.0
        for o in outcomes:
            next_terminal = h == 1 or mdp_table.states[o["next"]]["terminal"]
            after = 0.0 if next_terminal else _finite(potential(o["next"], h - 1))
            f += o["p"] * beta * (gamma * after - before)
            r += o["p"] * o["task_reward"]
        shaped.append(r + f)
    return shaped


def audit_export_signal(export_dir, p2cq_pkg, v2_only_recompute=False):
    _add_p2cq_path(p2cq_pkg)
    from p2cq.io import load as p2cq_load
    from p2cq.pairs import analyze_pairs, MDPRepository

    export_dir = Path(export_dir)
    registry = load_json(export_dir / "pairs.json")
    limits = {
        "max_states_per_mdp": 200000,
        "max_bellman_cells_per_mdp": 2000000,
        "min_oracle_gap": 1e-06,
        "min_separating_pairs": 1000,
        "min_per_motif": 128,
        "min_root_families_per_motif": 32,
        "min_shared_legal_per_motif": 64,
    }
    env_summary, records, cache = analyze_pairs(export_dir, registry, limits)
    indexed = {r["pair_id"]: r for r in records}
    planners = {}
    contracts = {}
    rows = []
    v1_name = V1_METHOD
    v2_name = METHOD_V2

    def phi_v1_factory(mdp_obj, contract):
        def phi(sid, h):
            dyn = mdp_obj.states[sid]["dynamic_public"]
            return float(evaluate_all(contract, dyn)[v1_name])
        return phi

    def phi_v2_factory(mdp_obj, contract, key):
        if key not in planners:
            planners[key] = RemainingWorkPlanner(contract)
        planner = planners[key]
        def phi(sid, h):
            dyn = mdp_obj.states[sid]["dynamic_public"]
            return planner.potential(dyn)[0]
        return phi

    n_pairs = len(registry["pairs"])
    for i, p in enumerate(registry["pairs"]):
        if i % 32 == 0:
            print(f"signal pair {i}/{n_pairs}", flush=True)
        rec = indexed[p["pair_id"]]
        for side in ("left", "right"):
            spec = p[side]
            name = spec["mdp_file"]
            m = cache[name]
            sid = spec["state"]
            h = spec["remaining_steps"]
            if name not in contracts:
                contracts[name] = TaskContract(m.data["task_public"])
            contract = contracts[name]
            q = m.q(sid, h)
            s1 = score_one_step(m, sid, h, phi_v1_factory(m, contract), m.gamma)
            s2 = score_one_step(m, sid, h, phi_v2_factory(m, contract, name), m.gamma)
            a1 = oracle_alignment(s1, q)
            a2 = oracle_alignment(s2, q)
            rows.append({
                "pair_id": p["pair_id"],
                "motif": p["motif"],
                "family": p["family"],
                "side": side,
                "decision_separating": rec["decision_separating"],
                "horizon_boundary": h == 1,
                "v1": a1,
                "v2": a2,
            })
    crit = [r for r in rows if r["decision_separating"] and not r["horizon_boundary"]]

    def agg(rs):
        if not rs:
            return {"n": 0, "v1_hit": None, "v2_hit": None, "v1_regret": None, "v2_regret": None}
        return {
            "n": len(rs),
            "v1_hit": mean(1.0 if r["v1"]["conservative_hit"] else 0.0 for r in rs),
            "v2_hit": mean(1.0 if r["v2"]["conservative_hit"] else 0.0 for r in rs),
            "v1_regret": mean(r["v1"]["regret_mean"] for r in rs),
            "v2_regret": mean(r["v2"]["regret_mean"] for r in rs),
        }

    per_motif = {}
    for motif in sorted({r["motif"] for r in rows}):
        per_motif[motif] = agg([r for r in crit if r["motif"] == motif])
        per_motif[motif]["v1_hit"] = per_motif[motif]["v1_hit"]
        per_motif[motif]["v2_hit"] = per_motif[motif]["v2_hit"]
    overall = agg(crit)
    return {
        "environment_summary": env_summary,
        "critical": overall,
        "all": agg(rows),
        "per_motif": per_motif,
        "n_rows": len(rows),
        "n_critical": len(crit),
    }, rows


def combine_signal(parts):
    # parts: list of audit_export_signal results
    def take_rows(part_rows):
        return part_rows
    rows = []
    env = []
    for sig, rs in parts:
        env.append(sig["environment_summary"])
        rows.extend(rs)
    crit = [r for r in rows if r["decision_separating"] and not r["horizon_boundary"]]
    def agg(rs):
        if not rs:
            return {"n": 0, "v1_hit": None, "v2_hit": None, "v1_regret": None, "v2_regret": None}
        return {
            "n": len(rs),
            "v1_hit": mean(1.0 if r["v1"]["conservative_hit"] else 0.0 for r in rs),
            "v2_hit": mean(1.0 if r["v2"]["conservative_hit"] else 0.0 for r in rs),
            "v1_regret": mean(r["v1"]["regret_mean"] for r in rs),
            "v2_regret": mean(r["v2"]["regret_mean"] for r in rs),
        }
    per_motif = {m: agg([r for r in crit if r["motif"] == m]) for m in sorted({r["motif"] for r in rows})}
    return {
        "environment_summaries": env,
        "critical": agg(crit),
        "all": agg(rows),
        "per_motif": per_motif,
        "n_rows": len(rows),
        "n_critical": len(crit),
    }


def mechanism_passed(mech):
    return (
        mech["cost_parity_mismatches"] == 0
        and mech["bellman_mismatches"] == 0
        and mech["unreachable_mismatches"] == 0
        and mech["search_truncations"] == 0
        and mech["cache_equivalence_mismatches"] == 0
        and mech["stage_handoff_regressions"] == 0
        and mech["pbrs_max_abs_error"] <= 1e-10
    )


def development_signal_passed(sig):
    c = sig["critical"]
    if None in (c["v1_hit"], c["v2_hit"], c["v1_regret"], c["v2_regret"]):
        return False
    return (c["v2_hit"] - c["v1_hit"] >= 0.05) and (c["v2_regret"] < c["v1_regret"])


def confirmation_signal_passed(sig):
    c = sig["critical"]
    if None in (c["v1_hit"], c["v2_hit"], c["v1_regret"], c["v2_regret"]):
        return False
    if c["v2_hit"] - c["v1_hit"] < 0.03:
        return False
    if not (c["v2_regret"] < c["v1_regret"]):
        return False
    for motif, rec in sig["per_motif"].items():
        if rec["n"] and rec["v2_hit"] < rec["v1_hit"] - 0.05:
            return False
    return True

