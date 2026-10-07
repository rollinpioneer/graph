"""C1-BW-SCALE-ORDER-PROTOTYPE-V1 helpers: table-distractor padding of the single-tower training trajectories (MG-S8-PAD), exact relabelling and validation, the 48-problem factorial
development board, the common offline state bank, and the six-way error taxonomy. No planner answer ever reaches a model input; nothing here changes a frozen rule or scorer.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from . import a04p_controls as AC
from . import eval_a03 as E
from . import gp_attribution as A
from . import goal_progress as GP
from . import scorer_control as SC
from . import state as S

CARD = "C1-BW-SCALE-ORDER-PROTOTYPE-V1"
BASE_COMMIT = "e8ffa9a9b09a40dbf90025581861719a52a7a559"
BRANCH = "codex/cp-disr-c1-bw-scale-order-prototype-v1"
PLAN_REL = "docs/c1_blocksworld/CP_DISR_C1_Scale_Order_Interaction_Next_Experiment_v1.md"
SO_REL = E.BASE_REL + "/scale_order_prototype_v1"
SC_RUN_REL = SC.SC_REL + "/20261007T040106Z_0c391fe0"
BOARD_REL = "configs/splits/c1_bw_scale_order_board48_v1.json"
NAMESPACE = "C1-BW-SCALE-ORDER-BOARD-v1"
RUN_ID = "R-C1-SO-MG-S8-PAD-0"
PAD_VERSIONS = (6, 7, 8)
EPOCHS, CKPT_EPOCHS, SHUFFLE_SEED = 100, (20, 40, 60, 80, 100), 0
# (cell id, n, non-trivial goal towers k, max tower height h, full height partition)
BOARD_CELLS = (("n6k1h2", 6, 1, 2, (2, 1, 1, 1, 1)), ("n6k1h4", 6, 1, 4, (4, 1, 1)), ("n6k2h2", 6, 2, 2, (2, 2, 1, 1)), ("n6k2h4", 6, 2, 4, (4, 2)),
               ("n8k1h2", 8, 1, 2, (2, 1, 1, 1, 1, 1, 1)), ("n8k1h4", 8, 1, 4, (4, 1, 1, 1, 1)), ("n8k2h2", 8, 2, 2, (2, 2, 1, 1, 1, 1, 1, 1)), ("n8k2h4", 8, 2, 4, (4, 2, 1, 1)))
BOARD_PER_CELL, BOARD_MAX_ATTEMPTS, BOARD_P_STACK = 6, 2000, 0.88
OLD_ROOTS = SC.OLD_ROOTS + (SC_RUN_REL,)
FROZEN = tuple(dict.fromkeys(SC.FROZEN + ("src/cp_disr/blocksworld/scorer_control.py", "scripts/c1_bw_scorer_control.py")))
CLASSES = ("NEUTRAL_PREP_WRONG", "REFUSE_NECESSARY_DESTRUCTION", "WRONG_DESTRUCTION_CHOICE", "AVOIDABLE_DESTRUCTION", "NONOPT_DIRECT_PROGRESS", "MISSED_DIRECT_PROGRESS", "OTHER_AMBIGUOUS")


# ------------------------------------------------------------------------------------------------ padding
def pad_colors(parent_case_id, total, extra):
    h0 = hashlib.sha256(("%s|%d" % (parent_case_id, total)).encode()).digest()[0] & 1
    return tuple((h0 + j) & 1 for j in range(extra))                         # alternating from a hashed start: balanced to within one block


def pad_problem(problem, total, parent_case_id):
    extra = total - problem.n
    assert extra > 0 and problem.names == S.default_names(problem.n)
    return S.Problem(S.default_names(total), tuple(problem.colors) + pad_colors(parent_case_id, total, extra), tuple(problem.init) + (S.TABLE,) * extra, tuple(problem.goal) + (S.TABLE,) * extra)


def pad_state(state, extra):
    return tuple(state) + (S.TABLE,) * extra


def version_for(slot, epoch):
    """0 = original, 1..3 = pad to 6/7/8. Even epochs original; odd epochs cycle the three pads with a per-slot offset: every parent meets all four versions, 50% original."""
    return 0 if epoch % 2 == 0 else 1 + (slot + epoch // 2) % 3


def _pad_worker(args):
    from . import planner as P
    from .environment import Case
    parent, trajs = args
    out = {"cases": [], "trajs": [], "labels": {}, "stats": Counter(), "exceptions": []}
    for total in PAD_VERSIONS:
        extra = total - parent["n"]
        case_id = "%s__PAD%d" % (parent["case_id"], total)
        base_p = S.Problem(tuple(parent["names"]), tuple(parent["colors"]), tuple(parent["init"]), tuple(parent["goal"]))
        problem = pad_problem(base_p, total, parent["case_id"])
        case = Case(case_id, "train_pad", problem, parent["optimal_length"], parent["step_cap"])
        out["cases"].append({"case_id": case_id, "split": "train_pad", "names": list(problem.names), "colors": list(problem.colors), "init": list(problem.init), "goal": list(problem.goal),
                             "optimal_length": parent["optimal_length"], "step_cap": parent["step_cap"], "parent": parent["case_id"], "total": total})
        solver = P.Solver()
        lab = {}

        def label(state):
            if state not in lab:
                L, acts = solver.optimal_actions(state, problem.goal)
                lab[state] = (L, [S.action_id(problem.names, a) for a in acts], GP.candidate_distances(solver, case, state))
            return lab[state]
        for t in trajs:
            states = [pad_state(s, extra) for s in t["states"]]
            astar, bad = [], False
            for i in range(len(t["actions"])):
                L, ids, dist = label(states[i])
                out["stats"]["decisions"] += 1
                ok_len = L == t["remaining"][i]
                ok_sup = set(t["astar"][i]) <= set(ids)
                ok_act = t["actions"][i] in dist
                out["stats"]["remaining_mismatch"] += not ok_len
                out["stats"]["orig_astar_not_subset"] += not ok_sup
                out["stats"]["orig_action_illegal"] += not ok_act
                out["stats"]["astar_size_orig"] += len(t["astar"][i])
                out["stats"]["astar_size_pad"] += len(ids)
                out["stats"]["decisions_with_extra_optimal_actions"] += len(set(ids) - set(t["astar"][i])) > 0
                bad |= not (ok_len and ok_sup and ok_act)
                astar.append(ids)
                out["labels"]["%s|%s" % (case_id, ",".join(map(str, states[i])))] = dist
            if bad:
                out["exceptions"].append(t["tid"] + "|PAD%d" % total)
            nt = dict(t)
            nt.update({"tid": "%s|PAD%d" % (t["tid"], total), "case_id": case_id, "n_blocks": total, "states": [list(s) for s in states], "astar": astar})
            out["trajs"].append(nt)
        out["stats"]["planner_expanded_nodes"] += solver.stats.expanded
    out["stats"] = dict(out["stats"])
    return out


def build_padding(cases, d_train, procs=48):
    """Pad every parent trajectory to 6/7/8 blocks and relabel it exactly. Returns (padded cases, padded trajectories, rank labels, validation)."""
    import multiprocessing as mp
    by_case = {}
    for t in d_train:
        by_case.setdefault(t["case_id"], []).append(t)
    jobs = []
    for c in cases:
        if c.case_id in by_case:
            parent = {"case_id": c.case_id, "n": c.problem.n, "names": list(c.problem.names), "colors": list(c.problem.colors), "init": list(c.problem.init), "goal": list(c.problem.goal),
                      "optimal_length": c.optimal_length, "step_cap": c.step_cap}
            jobs.append((parent, by_case[c.case_id]))
    t0 = time.time()
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as pool:
        res = pool.map(_pad_worker, jobs, chunksize=1)
    pcases, ptrajs, labels, stats, exc = [], [], {}, Counter(), []
    for r in res:
        pcases += r["cases"]
        ptrajs += r["trajs"]
        labels.update(r["labels"])
        stats.update(r["stats"])
        exc += r["exceptions"]
    order = {t["tid"]: i for i, t in enumerate(d_train)}
    ptrajs.sort(key=lambda t: (order[t["tid"].rsplit("|PAD", 1)[0]], t["tid"]))
    n = stats["decisions"]
    val = {"parent_trajectories": len(d_train), "padded_trajectories": len(ptrajs), "padded_cases": len(pcases), "decisions_checked": n, "remaining_length_mismatch": stats["remaining_mismatch"], "original_astar_not_subset_of_padded_astar": stats["orig_astar_not_subset"],
           "original_action_illegal": stats["orig_action_illegal"], "trajectories_with_exception": len(exc), "exception_tids": sorted(exc)[:200], "decisions_with_extra_optimal_actions": stats["decisions_with_extra_optimal_actions"],
           "mean_astar_size_original": round(stats["astar_size_orig"] / n, 4), "mean_astar_size_padded": round(stats["astar_size_pad"] / n, 4), "planner_expanded_nodes": stats["planner_expanded_nodes"],
           "label_wall_seconds_total": round(time.time() - t0, 1), "label_processes": min(procs, len(jobs)), "unique_rank_label_states": len(labels)}
    return pcases, ptrajs, labels, val


def epoch_trajectories(d_train, padded_by_tid, epoch):
    out = []
    for i, t in enumerate(d_train):
        v = version_for(i, epoch)
        out.append(t if v == 0 else padded_by_tid["%s|PAD%d" % (t["tid"], PAD_VERSIONS[v - 1])])
    return out


def schedule_counts(n_slots, epochs=EPOCHS):
    c = Counter(version_for(i, e) for i in range(n_slots) for e in range(epochs))
    covered = all({version_for(i, e) for e in range(epochs)} == {0, 1, 2, 3} for i in range(n_slots))
    return {"slot_epochs": dict(sorted(c.items())), "every_parent_meets_all_four_versions": bool(covered), "original_share": c[0] / (n_slots * epochs)}


# ------------------------------------------------------------------------------------------------ 48-problem factorial board
def gen_board_cell(builder, forbidden, key, n, heights, quota, max_attempts, rejects):
    """First ``quota`` qualified, deduplicated problems of one (n, k, h) cell; planner / labels only, no model."""
    from . import canonical as K
    from . import generator as Gen
    out, attempt = [], 0
    while len(out) < quota and attempt < max_attempts:
        p = Gen.make_problem(key, n, len(out), heights, S.RED, attempt, p_stack=BOARD_P_STACK)
        attempt += 1
        if p is None:
            rejects["generator_none"] += 1
            continue
        h = K.problem_iso_hash(p)
        if h in forbidden:
            rejects["seen_problem_class"] += 1
            continue
        q = builder.qualify(p, need_destruction=True)
        if q is None:
            rejects["unqualified_or_unsolved_or_label_unavailable"] += 1
            continue
        if q["labels"]["requires_goal_destruction"] is None:
            rejects["destruction_label_unknown"] += 1
            continue
        forbidden.add(h)
        out.append((p, q, h))
    return out, attempt


def board_excluded(root, d_train, train_cases):
    forbidden, by_source = SC.excluded_hashes(root, d_train, train_cases)
    fresh = {c["problem_iso_hash"] for c in json.loads((Path(root) / SC.FRESH_REL).read_text())["cases"]}
    by_source["scorer_control_fresh112"] = len(fresh)
    return forbidden | fresh, by_source


# ------------------------------------------------------------------------------------------------ goal change and error taxonomy
def action_table(problem):
    from . import contracts as C
    return {S.action_id(problem.names, a): a for a in C.all_actions(problem.n)}


def goal_change(problem, state, action):
    """(atoms made true, atoms made false) among the final goal atoms by one action."""
    b, a = S.goal_atom_truth(problem, state), S.goal_atom_truth(problem, S.apply(state, action))
    return {k for k in b if a[k] and not b[k]}, {k for k in b if b[k] and not a[k]}


def classify(problem, state, sel_id, opt_ids, table=None):
    """Plan section 3.3 priority order; ``sel_id`` must not be in ``opt_ids``. All / exists are over the FULL optimal set."""
    table = table or action_table(problem)
    add, dele = goal_change(problem, state, table[sel_id])
    ch = {o: goal_change(problem, state, table[o]) for o in opt_ids}
    neutral = lambda g: not g[0] and not g[1]
    if not add and not dele and all(neutral(g) for g in ch.values()):
        return "NEUTRAL_PREP_WRONG"
    if not dele and all(g[1] for g in ch.values()):
        return "REFUSE_NECESSARY_DESTRUCTION"
    if dele and all(g[1] for g in ch.values()):
        return "WRONG_DESTRUCTION_CHOICE"
    if dele and any(not g[1] for g in ch.values()):
        return "AVOIDABLE_DESTRUCTION"
    if add and not dele:
        return "NONOPT_DIRECT_PROGRESS"
    if not add and any(g[0] and not g[1] for g in ch.values()):
        return "MISSED_DIRECT_PROGRESS"
    return "OTHER_AMBIGUOUS"


def episode_error_stats(problem, ep, with_c3, table=None):
    """Per-episode raw / executed errors, their classes and the C3 available-set denominator, rebuilt from the saved decision records (raw = saved original prediction)."""
    table = table or action_table(problem)
    ds = ep["decisions"]
    visited = {tuple(ds[0]["state"])} if ds else set()
    out = {"decisions": len(ds), "raw_err": 0, "exec_err": 0, "rewrite": 0, "no_opt_available": 0, "first_raw_err": None, "first_exec_err": None, "raw_classes": Counter(), "exec_classes": Counter(),
           "unique_raw_err_states": set(), "first_raw_class": None}
    for i, d in enumerate(ds):
        state = tuple(d["state"])
        visited.add(state)
        opt = set(d["optimal_actions"])
        raw, sel = d.get("raw", d["selected"]), d["selected"]
        if raw not in opt:
            out["raw_err"] += 1
            c = classify(problem, state, raw, d["optimal_actions"], table)
            out["raw_classes"][c] += 1
            out["unique_raw_err_states"].add(state)
            if out["first_raw_err"] is None:
                out["first_raw_err"], out["first_raw_class"] = i, c
        if sel not in opt:
            out["exec_err"] += 1
            out["exec_classes"][classify(problem, state, sel, d["optimal_actions"], table)] += 1
            if out["first_exec_err"] is None:
                out["first_exec_err"] = i
        if raw in opt and sel not in opt:
            out["rewrite"] += 1
        if with_c3:
            legal = [a for a in d["probs"]]
            avail = [a for a in legal if S.apply(state, table[a]) not in visited]
            out["no_opt_available"] += bool(avail) and not (set(avail) & opt)
    return out


# ------------------------------------------------------------------------------------------------ common offline state bank
def bank_states(case, solver):
    """<= 5 states per problem, chosen from the planner's first optimal plan and one fixed non-optimal deviation; no model score is consulted."""
    plan = solver.one_optimal_plan(case.problem.init, case.problem.goal)
    states = [case.problem.init]
    for a in plan:
        states.append(S.apply(states[-1], a))
    L = len(plan)
    picks = {"init": 0, "p33": round(L / 3), "p66": round(2 * L / 3), "pre_final": L - 1}
    out, seen = [], set()
    for kind, i in picks.items():
        if 0 <= i < L and states[i] not in seen:
            seen.add(states[i])
            out.append((kind, states[i], i))
    m = L // 2
    _, opt = solver.optimal_actions(states[m], case.problem.goal)
    optset = {S.action_id(case.problem.names, a) for a in opt}
    for a in sorted(S.legal_actions(states[m]), key=lambda x: S.action_id(case.problem.names, x)):
        if S.action_id(case.problem.names, a) not in optset:
            s2 = S.apply(states[m], a)
            if s2 not in seen and solver.cost_to_go(s2, case.problem.goal) > 0:
                out.append(("deviation", s2, m + 1))
            break
    return out


def state_record(case, state, kind, solver):
    from . import goal_progress as G
    L, acts = solver.optimal_actions(state, case.problem.goal)
    table = action_table(case.problem)
    opt_ids = [S.action_id(case.problem.names, a) for a in acts]
    ch = {o: goal_change(case.problem, state, table[o]) for o in opt_ids}
    dist = G.candidate_distances(solver, case, state)
    return {"case_id": case.case_id, "state": list(state), "kind": kind, "L": L, "optimal": opt_ids, "dist": dist, "all_optimal_neutral": all(not g[0] and not g[1] for g in ch.values()),
            "must_destroy": all(g[1] for g in ch.values()), "direct_progress_available": any(g[0] and not g[1] for g in ch.values())}


def score_state(policy, case, rec):
    from . import imitation as I
    snap = I.snapshot_at(case, tuple(rec["state"]), 0)
    with torch.no_grad():
        out = policy(snap, policy.initial_hidden())
    ids = snap.candidate_ids
    legal = [i for i, m in enumerate(snap.mask) if m]
    logits = out.logits.detach().cpu()
    probs = out.distribution.probs.detach().cpu()
    top = AC.argmax_tie(logits, ids, legal)
    opt = set(rec["optimal"])
    best_opt = max(float(logits[i]) for i in legal if ids[i] in opt)
    non = [float(logits[i]) for i in legal if ids[i] not in opt]
    return {"top1": ids[top], "top1_optimal": ids[top] in opt, "optimal_mass": float(sum(float(probs[i]) for i in legal if ids[i] in opt)), "margin": (best_opt - max(non)) if non else None,
            "local_excess": 1 + rec["dist"][ids[top]] - rec["L"], "n_legal": len(legal)}


# ------------------------------------------------------------------------------------------------ fixtures (train-split cases and artificial states only)
def check_fixtures(cases, d_train, device):
    import random
    from . import imitation as I
    from . import planner as P
    from .environment import BwEpisode
    out = {}
    p = cases[0].problem
    q = pad_problem(p, 8, cases[0].case_id)
    out["pad_keeps_indices_names_goal"] = bool(q.names == S.default_names(8) and q.init[:p.n] == p.init and q.goal[:p.n] == p.goal and all(x == S.TABLE for x in q.init[p.n:] + q.goal[p.n:]) and S.is_valid(q.init))
    cols = [pad_colors("x", 8, 5).count(0), pad_colors("x", 8, 5).count(1)]
    out["pad_colours_balanced"] = abs(cols[0] - cols[1]) <= 1
    out["schedule"] = schedule_counts(len(d_train))
    out["schedule_pass"] = bool(out["schedule"]["every_parent_meets_all_four_versions"] and abs(out["schedule"]["original_share"] - 0.5) < 1e-9)
    rng = random.Random(7)
    solver = P.Solver()
    classes, total = Counter(), 0
    for c in rng.sample(cases, 12):
        ep = BwEpisode(c)
        tab = action_table(c.problem)
        for _ in range(12):
            if ep.done:
                break
            _, opt = solver.optimal_actions(ep.state, c.problem.goal)
            opt_ids = [S.action_id(c.problem.names, a) for a in opt]
            non = [S.action_id(c.problem.names, a) for a in S.legal_actions(ep.state) if S.action_id(c.problem.names, a) not in opt_ids]
            if non:
                k = classify(c.problem, ep.state, rng.choice(non), opt_ids, tab)
                assert k in CLASSES
                classes[k] += 1
                total += 1
            ep.step(rng.choice(opt_ids))
    out["classify_total_function"] = total > 40
    out["classify_class_counts_on_random_states"] = dict(classes)
    # relabelling on two parents: original mapped trajectories keep their remaining length and their optimal actions
    by_case = {}
    for t in d_train:
        by_case.setdefault(t["case_id"], []).append(t)
    checks = []
    for c in (cases[0], cases[len(cases) // 2]):
        parent = {"case_id": c.case_id, "n": c.problem.n, "names": list(c.problem.names), "colors": list(c.problem.colors), "init": list(c.problem.init), "goal": list(c.problem.goal), "optimal_length": c.optimal_length, "step_cap": c.step_cap}
        r = _pad_worker((parent, by_case[c.case_id][:3]))
        checks.append({"decisions": r["stats"]["decisions"], "remaining_mismatch": r["stats"].get("remaining_mismatch", 0), "astar_not_subset": r["stats"].get("orig_astar_not_subset", 0), "illegal": r["stats"].get("orig_action_illegal", 0)})
    out["relabel_on_two_parents"] = checks
    out["relabel_pass"] = all(x["decisions"] > 0 and x["remaining_mismatch"] == 0 and x["astar_not_subset"] == 0 and x["illegal"] == 0 for x in checks)
    c = cases[5]
    bs = bank_states(c, solver)
    out["bank_at_most_five_no_goal"] = bool(len(bs) <= 5 and all(solver.cost_to_go(s, c.problem.goal) > 0 for _, s, _ in bs))
    # the frozen scorer reads a padded n=8 snapshot (finite masked logits)
    pol = SC.load_scorer(Path(__file__).resolve().parents[3], "MG_C3", device)
    pc = cases[0]
    from .environment import Case
    prob = pad_problem(pc.problem, 8, pc.case_id)
    snap = I.snapshot_at(Case("pad_fx", "t", prob, pc.optimal_length, pc.step_cap), prob.init, 0)
    with torch.no_grad():
        o = pol(snap, pol.initial_hidden())
    legal = [i for i, m in enumerate(snap.mask) if m]
    out["scorer_reads_padded_n8"] = bool(torch.isfinite(o.logits[legal]).all()) and len(snap.candidate_ids) > len(I.snapshot_at(pc, pc.problem.init, 0).candidate_ids)
    out["pass"] = bool(out["pad_keeps_indices_names_goal"] and out["pad_colours_balanced"] and out["schedule_pass"] and out["classify_total_function"] and out["relabel_pass"] and out["bank_at_most_five_no_goal"] and out["scorer_reads_padded_n8"])
    return out


# ------------------------------------------------------------------------------------------------ identity
def source_identity(root, new_sources):
    root = Path(root)
    frozen = {}
    for rel in FROZEN:
        import subprocess
        now = E.sha256_file(root / rel)
        base = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", "%s:%s" % (BASE_COMMIT, rel)])).hexdigest()
        frozen[rel] = {"sha256": now, "sha256_at_base_commit": base, "unchanged_since_base": now == base}
    return {"base_commit": BASE_COMMIT, "frozen_sources": frozen, "new_sources": {r: {"sha256": E.sha256_file(root / r)} for r in new_sources if (root / r).is_file()},
            "old_root_tree_ids_at_base": {rel: E.git(root, "rev-parse", "%s:%s" % (BASE_COMMIT, rel)) for rel in OLD_ROOTS}, "all_frozen_unchanged": all(v["unchanged_since_base"] for v in frozen.values())}


def check_identity(root, run_root):
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    own = str(run_root.relative_to(root))
    if E.git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % own):
        raise E.A03Error("tracked source tree is dirty")
    reg = json.loads((run_root / "prep" / "source_identity.json").read_text())
    for rel, v in reg["frozen_sources"].items():
        if E.sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("frozen source changed: %s" % rel)
    for rel, v in reg["new_sources"].items():
        if E.sha256_file(root / rel) != v["sha256"]:
            raise E.A03Error("registered source changed: %s" % rel)
    for rel in OLD_ROOTS:
        if E.git(root, "rev-parse", "HEAD:%s" % rel) != reg["old_root_tree_ids_at_base"][rel]:
            raise E.A03Error("an old result directory changed: %s" % rel)
    return E.git(root, "rev-parse", "HEAD")


# ------------------------------------------------------------------------------------------------ scorers
def load_pad_scorer(root, run_root, which="final", device=None):
    from ..torch_rl import load_checkpoint
    acct = json.loads((Path(run_root) / "runs" / RUN_ID / "training_accounting.json").read_text())
    ck = Path(root) / acct["checkpoints"][which]["path"]
    assert E.sha256_file(ck) == acct["checkpoints"][which]["sha256"]
    model = A.GoalProgressModelGoal(GP.load_base(root, device), "global", "production").to(device)
    load_checkpoint(ck, model)
    model.eval()
    return GP.ScorePolicy(model)
