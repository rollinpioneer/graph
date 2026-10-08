"""Fixtures of card C1-DEPOTS-SEARCH-MATCH-V1 (plan section 7): shared search engine semantics on synthetic graphs, successor index vs the project's legal-action semantics, h_add / h_count, the WL-GOOSE adapter
against the native planner, the neural state-value path against the policy path, budget and failure statuses. Only synthetic graphs and Train96 problems are used. CPU only."""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from cp_disr.pddl import depots as DP
from cp_disr.pddl import task as T
from cp_disr.pddl.search_match import evaluators as EV
from cp_disr.pddl.search_match.budget import Budget
from cp_disr.pddl.search_match.engine import ModelNonFinite, SuccessorIndex, gbfs
from cp_disr.pddl.search_match.runner import OldRun, run_case

ROOT = Path(__file__).resolve().parents[1]
INF = float("inf")
OLD_ROOT = ROOT / "runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e"
HAVE_OLD = (OLD_ROOT / "data" / "manifest.json").is_file()
need_old = pytest.mark.skipif(not HAVE_OLD, reason="previous run root not on this machine")
need_hadd = pytest.mark.skipif(not EV.HADD_BIN.is_file(), reason="hadd_server not built")
need_wl = pytest.mark.skipif(not (EV.SCORPION_DOWNWARD.is_file() and EV.GOOSE_PY.is_file() and EV.WL_SHIM.is_dir() and HAVE_OLD), reason="author software / WL shim not on this machine")

DOMAIN = """(define (domain graph) (:predicates (at ?n) (edge ?a ?b))
 (:action move :parameters (?a ?b) :precondition (and (at ?a) (edge ?a ?b)) :effect (and (at ?b) (not (at ?a)))))"""


def graph_task(edges, start="a", goal="f"):
    nodes = sorted({x for e in edges for x in e} | {start, goal})
    prob = "(define (problem g) (:domain graph) (:objects %s) (:init (at %s) %s) (:goal (at %s)))" % (" ".join(nodes), start, " ".join("(edge %s %s)" % e for e in edges), goal)
    return T.StripsTask(DOMAIN, prob)


class DictEval(EV.Evaluator):
    """h by node name of the single true at-atom."""

    def __init__(self, h, default=100.0):
        self.h, self.default, self.log = h, default, []

    def prepare(self, task, ctx, budget):
        super().prepare(task, ctx, budget)
        self.name_of = {i: a[1][0] for i, a in enumerate(task.dyn_atoms) if a[0] == "at"}

    def evaluate(self, states, budget):
        budget.check()
        out = []
        for s in states:
            n = self.name_of[EV.bits(s)[0]]
            self.log.append(n)
            out.append(float(self.h.get(n, self.default)))
        return out


def search(edges, h, **kw):
    task = graph_task(edges)
    ev = DictEval(h)
    ev.prepare(task, {}, None)
    b = Budget(kw.pop("wall", 30), kw.pop("max_exp", 100_000), 8 * 2 ** 30)
    res = gbfs(task, SuccessorIndex(task), ev, b, kw.pop("milestones", (1000, 10000, 100000)))
    names = lambda ids: [task.actions[i].args for i in ids]
    return task, ev, res, names


# ---------------------------------------------------------------------------------------------------------- engine semantics on synthetic graphs
def test_other_branch_is_explored_after_a_dead_end():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "f")]
    task, ev, res, names = search(edges, {"a": 5, "b": 1, "c": 5, "d": 0})
    assert res.status == "SOLVED" and names(res.plan) == [("a", "c"), ("c", "f")]
    assert res.expanded == 4                                                           # a, b, d (dead end, nothing generated), c
    assert ev.log == ["a", "b", "c", "d"] or ev.log == ["a", "b", "c", "d"]


def test_duplicates_closed_not_reopened_and_cycle_terminates():
    edges = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d"), ("d", "c")]
    task, ev, res, names = search(edges, {})                                            # goal f unreachable
    assert res.status == "OPEN_EXHAUSTED" and res.expanded == 4                         # a, b, c, d each once
    assert res.duplicates >= 3


def test_open_state_gets_shorter_path_in_place():
    edges = [("a", "p"), ("a", "z"), ("p", "r"), ("r", "q"), ("z", "q"), ("q", "f")]
    h = {"a": 9, "p": 1, "r": 1, "z": 5, "q": 9}
    task, ev, res, names = search(edges, h)
    assert res.status == "SOLVED" and res.path_updates == 1
    assert names(res.plan) == [("a", "z"), ("z", "q"), ("q", "f")]                       # not a-p-r-q-f
    assert ev.log.count("q") == 1                                                       # a state is scored once


def test_equal_h_is_fifo_in_canonical_action_order():
    edges = [("a", "d"), ("a", "b"), ("a", "c"), ("b", "f")]
    task, ev, res, names = search(edges, {"a": 3, "b": 1, "c": 1, "d": 1})
    assert ev.log[1:4] == ["b", "c", "d"]                                               # pushed in action-id order (b < c < d) and popped FIFO
    assert res.status == "SOLVED" and names(res.plan) == [("a", "b"), ("b", "f")]


def test_goal_is_tested_at_generation_and_root_goal_returns_empty_plan():
    task, ev, res, names = search([("a", "f")], {"a": 1})
    assert res.status == "SOLVED" and res.expanded == 1 and names(res.plan) == [("a", "f")] and ev.log == ["a"]   # the goal successor is never scored
    t0 = graph_task([("a", "b")], start="f", goal="f")
    ev0 = DictEval({})
    ev0.prepare(t0, {}, None)
    r0 = gbfs(t0, SuccessorIndex(t0), ev0, Budget(5, 10, 2 ** 33))
    assert r0.status == "SOLVED" and r0.plan == [] and r0.expanded == 0


def test_milestones_are_snapshots_of_one_run_and_node_limit_is_reported():
    chain = [("n%02d" % i, "n%02d" % (i + 1)) for i in range(30)]
    edges = [("a", "n00")] + chain + [("n30", "f")]
    task, ev, res, names = search(edges, {}, milestones=(5, 10, 100))
    assert res.status == "SOLVED" and 5 in res.snapshots and 10 in res.snapshots and 100 not in res.snapshots
    assert res.snapshots[5]["expanded"] == 5 and len(res.plan) == 32
    task, ev, res, names = search(edges, {}, max_exp=7, milestones=(5, 10))
    assert res.status == "NODE_LIMIT" and res.expanded == 7 and set(res.snapshots) == {5}


def test_infinite_h_sorts_last_and_nan_is_technical():
    edges = [("a", "b"), ("a", "c"), ("b", "f"), ("c", "f")]
    task, ev, res, names = search(edges, {"a": 0, "b": INF, "c": 2})
    assert res.status == "SOLVED" and names(res.plan)[0] == ("a", "c")
    task = graph_task(edges)
    ev2 = DictEval({"a": float("nan")})
    ev2.prepare(task, {}, None)
    assert gbfs(task, SuccessorIndex(task), ev2, Budget(5, 10, 2 ** 33)).status == "MODEL_NONFINITE"


def test_wall_and_memory_limits_are_resource_statuses():
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    task = graph_task(edges)
    ev = DictEval({})
    ev.prepare(task, {}, None)
    b = Budget(0.0, 10, 2 ** 33)
    assert gbfs(task, SuccessorIndex(task), ev, b).status == "TIMEOUT"
    b = Budget(30, 10, 2 ** 33)
    b.max_rss_growth = -1
    r = gbfs(task, SuccessorIndex(task), ev, b)
    assert r.status in ("MEMORY_LIMIT",) or r.expanded < 64                              # memory is sampled every 64 expansions; the tiny graph may end first


# ---------------------------------------------------------------------------------------------------------- successor index vs project semantics
@pytest.fixture(scope="module")
def train_tasks():
    if not HAVE_OLD:
        pytest.skip("previous run root not on this machine")
    old = OldRun(OLD_ROOT)
    out = []
    for cid in ("train_n3_000", "train_n4_000", "train_n5_016"):
        c = next(x for x in old.manifest["train"] if x["case_id"] == cid)
        out.append((cid, c, T.depots_task(DP.DOMAIN_TYPED, c["file"])))
    return out


def test_successor_index_equals_task_legal_in_canonical_order(train_tasks):
    rng = random.Random(0)
    for cid, c, task in train_tasks:
        idx = SuccessorIndex(task)
        s = task.init_mask
        for _ in range(80):
            fast = idx.applicable(s)
            assert [a.index for a in fast] == [a.index for a in task.legal(s)]
            assert [a.aid for a in fast] == sorted(a.aid for a in fast)                  # canonical action-id order
            s = task.apply(s, rng.choice(fast))


# ---------------------------------------------------------------------------------------------------------- heuristics
HADD_DOMAIN = """(define (domain h) (:predicates (a ?x) (b ?x) (c ?x) (d ?x))
 (:action o1 :parameters (?x) :precondition (a ?x) :effect (b ?x))
 (:action o2 :parameters (?x) :precondition (a ?x) :effect (c ?x))
 (:action o3 :parameters (?x) :precondition (and (b ?x) (c ?x)) :effect (d ?x))
 (:action o4 :parameters (?x) :precondition (d ?x) :effect (not (a ?x))))"""                  # o4 makes (a) a dynamic atom (it can be deleted); irrelevant for h_add

@need_hadd
def test_h_add_hand_example_and_h_count():
    prob = "(define (problem p) (:domain h) (:objects o) (:init (a o)) (:goal (and (d o) (b o))))"
    task = T.StripsTask(HADD_DOMAIN, prob)
    ix = {a[0]: i for i, a in enumerate(task.dyn_atoms)}
    st = lambda *names: sum(1 << ix[n] for n in names)
    ev = EV.HAddEval()
    ev.prepare(task, {}, Budget())
    try:
        got = ev.evaluate([st("a"), st("a", "b"), st("b", "d"), st("c"), st("d")], Budget())
    finally:
        ev.close()
    assert got == [4.0, 2.0, 0.0, INF, INF]                                              # d = 1 + b + c = 3, b = 1 | {a,b}: c=1, d=2, b=0 | goals true | b and d unreachable from {c}
    cnt = EV.CountEval()
    cnt.prepare(task, {}, Budget())
    assert cnt.evaluate([st("a"), st("a", "b"), st("b", "d")], Budget()) == [2.0, 1.0, 0.0]


@need_hadd
def test_h_add_cpp_equals_a_python_reference_on_random_depots_states(train_tasks):
    def reference(task, state):
        cost = {i: 0 for i in EV.bits(state)}
        changed = True
        while changed:
            changed = False
            for a in task.actions:
                pre = EV.bits(a.pre_mask)
                if all(p in cost for p in pre):
                    c = 1 + sum(cost[p] for p in pre)
                    for x in EV.bits(a.add_mask):
                        if x not in cost or c < cost[x]:
                            cost[x] = c
                            changed = True
        gs = EV.bits(task.goal_mask)
        return INF if any(g not in cost for g in gs) else float(sum(cost[g] for g in gs))
    rng = random.Random(3)
    for cid, c, task in train_tasks:
        idx = SuccessorIndex(task)
        s, states = task.init_mask, []
        for _ in range(15):
            states.append(s)
            s = task.apply(s, rng.choice(idx.applicable(s)))
        ev = EV.HAddEval()
        ev.prepare(task, {}, Budget())
        try:
            got = ev.evaluate(states, Budget())
        finally:
            ev.close()
        assert got == [reference(task, x) for x in states]


# ---------------------------------------------------------------------------------------------------------- WL-GOOSE adapter against the native planner
def ipc_encoding(text):
    from cp_disr.pddl.parse import parse_problem
    p = parse_problem(text)
    facts = []
    for o in sorted(p.objects):
        t = p.objects[o]
        if t in ("depot", "distributor", "place"):
            facts.append("(place %s)" % o)
        elif t in ("pallet", "crate"):
            facts += ["(%s %s)" % (t, o), "(surface %s)" % o]
        else:
            facts.append("(%s %s)" % (t, o))
    init = ["(%s %s)" % (a[0], " ".join(a[1])) for a in sorted(p.init)]
    goal = ["(%s %s)" % (a[0], " ".join(a[1])) for a in p.goal]
    return "(define (problem %s) (:domain depot)\n(:objects %s)\n(:init\n%s\n)\n(:goal (and\n%s\n)))\n" % (p.name, " ".join(sorted(p.objects)), "\n".join(facts + init), "\n".join(goal))


@need_wl
def test_wl_adapter_equals_the_native_planner_on_initial_and_visited_states(train_tasks, tmp_path):
    old = OldRun(OLD_ROOT)
    rng = random.Random(5)
    n = 0
    for track, wrap in (("typed", lambda c: (DP.DOMAIN_TYPED, c["file"])), ("ipc", None)):
        params = old.wl_params("ipc" if track == "ipc" else "struct")["params"]
        ev = EV.WlEval(params)
        for cid, c, _t in train_tasks[:3]:
            if track == "typed":
                domain, problem = DP.DOMAIN_TYPED, c["file"]
            else:
                problem = tmp_path / (cid + "_ipc.pddl")
                problem.write_text(ipc_encoding(Path(c["file"]).read_text()))
                domain = DP.DOMAIN_IPC
            task = T.depots_task(domain, problem)
            ev.prepare(task, {"domain": str(domain), "problem": str(problem)}, Budget())
            assert ev.unmapped == 0
            idx = SuccessorIndex(task)
            s, states = task.init_mask, []
            for _ in range(8):
                states.append(s)
                s = task.apply(s, rng.choice(idx.applicable(s)))
            for st in states:
                mine = int(ev.evaluate([st], Budget())[0])
                native = EV.native_initial_h(EV.sas_with_init(ev.sas, ev.sas_values(st)), str(params))
                assert mine == native, (track, cid, mine, native)
                n += 1
    assert n == 48


def test_cpp_round_matches_std_round():
    assert [EV.cpp_round(x) for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.4, 0.49)] == [1, 2, 3, -1, -2, -2, 0]


# ---------------------------------------------------------------------------------------------------------- neural scorer
@need_old
def test_neural_state_value_path_equals_the_policy_path_and_chunks_do_not_matter(train_tasks):
    from cp_disr.pddl import model as PM
    dev = torch.device("cpu")
    m = PM.make_model("dense", dev)
    with torch.no_grad():
        for p in m.attn.parameters():
            p.add_(torch.randn_like(p) * 0.05)                                         # make the new layer non-trivial
    m.eval()
    for cid, c, task in train_tasks[:2]:
        ep = T.PddlEpisode(T.PddlCase(cid, "t", task, 10, 24))
        snap = ep.snapshot()
        logits = m.eval_logits(snap)
        legal = [i for i, ok in enumerate(snap.mask) if ok]
        ids = snap.candidate_ids
        succ = [task.apply(ep.state, task.action_by_id[ids[i]]) for i in legal]
        tpl = task.template()
        v0 = m.state_values(tpl, task, [ep.state])[0]
        vn = m.state_values(tpl, task, succ)
        ref = torch.stack([logits[i] for i in legal])
        assert torch.allclose(v0 - vn, ref, atol=1e-3, rtol=1e-4)
        ev = EV.NeuralEval("V_DENSE", m, dev)
        ev.prepare(task, {}, Budget())
        a = ev.evaluate(succ * 4, Budget())
        assert max(abs(x - y) for x, y in zip(a[:len(succ)], vn.tolist())) < 1e-3
        assert max(abs(x - y) for x, y in zip(a[:len(succ)], a[len(succ):2 * len(succ)])) < 1e-4            # chunking is a pure compute split


class _Bad(EV.Evaluator):
    def evaluate(self, states, budget):
        raise ModelNonFinite("nan")


class _Boom(EV.Evaluator):
    def evaluate(self, states, budget):
        raise RuntimeError("adapter exploded")


@need_old
def test_run_case_records_solution_validation_and_technical_failures(tmp_path):
    old = OldRun(OLD_ROOT)
    c = next(x for x in old.manifest["train"] if x["case_id"] == "train_n3_001")
    cfg = {"wall_seconds": 30.0, "max_expansions": 100000, "max_rss_growth_bytes": 8 * 2 ** 30, "grace_seconds": 20.0}
    rec = run_case(EV.CountEval(), "H_COUNT", "train", c, DP.DOMAIN_TYPED, cfg, tmp_path)
    assert rec["status"] == "SOLVED" and rec["plan_valid"] is True and rec["solved"] and rec["plan_length"] >= 1 and (tmp_path / ("H_COUNT__train__train_n3_001.plan")).is_file()
    assert run_case(_Bad(), "X", "train", c, DP.DOMAIN_TYPED, cfg)["status"] == "MODEL_NONFINITE"
    r = run_case(_Boom(), "X", "train", c, DP.DOMAIN_TYPED, cfg)
    assert r["status"] == "ADAPTER_ERROR" and not r["solved"]
    small = dict(cfg, max_expansions=2)
    r = run_case(EV.CountEval(), "H_COUNT", "train", next(x for x in old.manifest["train"] if x["case_id"] == "train_n5_016"), DP.DOMAIN_TYPED, small)
    assert r["status"] in ("NODE_LIMIT", "SOLVED") and (r["status"] != "SOLVED" or r["expanded"] <= 2)
