"""Fixtures of card C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3: the two-open-list ALT engine on synthetic graphs, the observer hook of the single-queue engine, FAST packing / edge / value equivalence (CPU, where
the arithmetic is deterministic), and block order. Only synthetic graphs and Train96 problems. CPU only."""
import random
from pathlib import Path

import pytest
import torch

from cp_disr.pddl import depots as DP
from cp_disr.pddl import task as T
from cp_disr.pddl.fast_alt.alt_engine import gbfs_alt
from cp_disr.pddl.fast_alt.evaluators import FastNeuralEval, RefNeuralEval
from cp_disr.pddl.fast_alt.runner import Roll, block_order
from cp_disr.pddl.search_match import evaluators as EV
from cp_disr.pddl.search_match.budget import Budget
from cp_disr.pddl.search_match.engine import ModelNonFinite, SuccessorIndex, gbfs
from cp_disr.pddl.search_match.runner import OldRun

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e"
HAVE_OLD = (OLD_ROOT / "data" / "manifest.json").is_file()
INF = float("inf")
DOMAIN = """(define (domain graph) (:predicates (at ?n) (edge ?a ?b))
 (:action move :parameters (?a ?b) :precondition (and (at ?a) (edge ?a ?b)) :effect (and (at ?b) (not (at ?a)))))"""


def graph_task(edges, start="a", goal="f"):
    nodes = sorted({x for e in edges for x in e} | {start, goal})
    prob = "(define (problem g) (:domain graph) (:objects %s) (:init (at %s) %s) (:goal (at %s)))" % (" ".join(nodes), start, " ".join("(edge %s %s)" % e for e in edges), goal)
    return T.StripsTask(DOMAIN, prob)


class DictEval(EV.Evaluator):
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


def setup(edges, hm, ha, **kw):
    task = graph_task(edges)
    m, a = DictEval(hm), DictEval(ha)
    m.prepare(task, {}, None)
    a.prepare(task, {}, None)
    b = Budget(kw.pop("wall", 30), kw.pop("max_exp", 100_000), 8 * 2 ** 30)
    order = []
    res = gbfs_alt(task, SuccessorIndex(task), m, a, b, kw.pop("milestones", (1000, 10000, 100000)), observer=lambda s, n: order.append(m.name_of[EV.bits(s)[0]]))
    names = lambda ids: [task.actions[i].args for i in ids]
    return task, m, a, res, order, names


def test_alt_with_identical_scores_expands_like_the_single_queue_engine():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e"), ("b", "e"), ("e", "f")]
    h = {"a": 6, "b": 3, "c": 2, "d": 1, "e": 0}
    task, m, a, res, order, names = setup(edges, h, h)
    t2 = graph_task(edges)
    ev = DictEval(h)
    ev.prepare(t2, {}, None)
    single = []
    r1 = gbfs(t2, SuccessorIndex(t2), ev, Budget(30, 10 ** 5, 2 ** 33), observer=lambda s, n: single.append(ev.name_of[EV.bits(s)[0]]))
    assert order == single and names(res.plan) == [t2.actions[i].args for i in r1.plan] and res.expanded == r1.expanded
    assert sum(res.extra["valid_pops"]) == res.expanded                                 # stale copies (below the top) never use a turn


def test_a_state_is_expanded_once_and_stale_entries_do_not_use_turns():
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "f")]
    hm = {"a": 0, "b": 5, "c": 1, "d": 3}
    ha = {"a": 0, "b": 1, "c": 5, "d": 3}
    task, m, a, res, order, names = setup(edges, hm, ha)
    assert len(order) == len(set(order)) == res.expanded
    assert order[0] == "a" and order[1] == "b"                                          # turn 1 (add heap) takes b although the main heap prefers c
    assert res.extra["valid_pops"][0] + res.extra["valid_pops"][1] == res.expanded
    assert names(res.plan) == [("a", "b"), ("b", "f")]


def test_shorter_open_path_updates_in_place_and_both_heaps_use_it():
    edges = [("a", "p"), ("a", "z"), ("p", "r"), ("r", "q"), ("z", "q"), ("q", "f")]
    h = {"a": 9, "p": 1, "r": 1, "z": 5, "q": 9}
    task, m, a, res, order, names = setup(edges, h, h)
    assert res.status == "SOLVED" and res.path_updates == 1 and names(res.plan) == [("a", "z"), ("z", "q"), ("q", "f")]
    assert m.log.count("q") == 1 and a.log.count("q") == 1                              # each scorer scores a state exactly once


def test_dead_end_branch_and_exhaustion_fallback_terminate():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "f")]
    task, m, a, res, order, names = setup(edges, {"a": 5, "b": 1, "c": 5, "d": 0}, {"a": 5, "b": 9, "c": 1, "d": 9})
    assert res.status == "SOLVED" and names(res.plan) == [("a", "c"), ("c", "f")]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]                                      # goal unreachable: both heaps run empty
    task, m, a, res, order, names = setup(edges, {}, {})
    assert res.status == "OPEN_EXHAUSTED" and res.expanded == 4 and res.extra["fallback_pops"] == 0       # both heaps always hold the same valid states, so the fallback branch is defensive only


def test_fifo_ties_negative_and_infinite_scores_goal_generation_and_limits():
    edges = [("a", "d"), ("a", "b"), ("a", "c"), ("b", "f")]
    task, m, a, res, order, names = setup(edges, {"a": 3, "b": 1, "c": 1, "d": 1}, {"a": 3, "b": 2, "c": 2, "d": 2})
    assert order[:2] == ["a", "b"] and names(res.plan) == [("a", "b"), ("b", "f")] and m.log[1:4] == ["b", "c", "d"]
    task, m, a, res, order, names = setup([("a", "b"), ("b", "f")], {"a": -5, "b": -7}, {"a": INF, "b": 0})
    assert res.status == "SOLVED" and res.h_root == -5                                  # negative main values are ordinary keys, +inf of the second score only sorts last
    task, m, a, res, order, names = setup([("a", "b"), ("b", "f")], {"a": INF}, {})
    assert res.status == "MODEL_NONFINITE"                                              # +inf from the main (learned) score is a technical error
    task, m, a, res, order, names = setup([("a", "f")], {}, {})
    assert res.status == "SOLVED" and res.expanded == 1 and m.log == ["a"]              # goal successor is never scored
    chain = [("a", "n00")] + [("n%02d" % i, "n%02d" % (i + 1)) for i in range(30)] + [("n30", "f")]
    task, m, a, res, order, names = setup(chain, {}, {}, milestones=(5, 10, 100))
    assert res.status == "SOLVED" and set(res.snapshots) == {5, 10}
    task, m, a, res, order, names = setup(chain, {}, {}, max_exp=7, milestones=(5, 10))
    assert res.status == "NODE_LIMIT" and res.expanded == 7
    task = graph_task(chain)
    m2, a2 = DictEval({}), DictEval({})
    m2.prepare(task, {}, None)
    a2.prepare(task, {}, None)
    assert gbfs_alt(task, SuccessorIndex(task), m2, a2, Budget(0.0, 10, 2 ** 33)).status == "TIMEOUT"


def test_observer_hook_does_not_change_the_single_queue_search():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "f")]
    h = {"a": 5, "b": 1, "c": 5, "d": 0}
    out = []
    for obs in (None, lambda s, n: None):
        t = graph_task(edges)
        ev = DictEval(h)
        ev.prepare(t, {}, None)
        r = gbfs(t, SuccessorIndex(t), ev, Budget(30, 10 ** 5, 2 ** 33), observer=obs)
        out.append((r.status, r.plan, r.expanded, r.generated, tuple(ev.log)))
    assert out[0] == out[1]
    roll = Roll(1)
    t = graph_task(edges)
    ev = DictEval(h)
    ev.prepare(t, {}, None)
    gbfs(t, SuccessorIndex(t), ev, Budget(30, 10 ** 5, 2 ** 33), observer=roll)
    marks = roll.end()
    assert "1" in marks and marks["end"][0] == 4


def test_block_order_is_deterministic_and_alt_last():
    seen = set()
    for i in range(40):
        o = block_order("ipc_p%02d" % i)
        assert o[2] == "ALT_DENSE_ADD" and set(o[:2]) == {"REF_DENSE", "FAST_DENSE"} and o == block_order("ipc_p%02d" % i)
        seen.add(o[0])
    assert seen == {"REF_DENSE", "FAST_DENSE"}


# ---------------------------------------------------------------------------------------------------------- FAST (CPU: deterministic arithmetic)
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


def random_states(task, n, seed):
    rng = random.Random(seed)
    idx = SuccessorIndex(task)
    s, out = task.init_mask, []
    for _ in range(n):
        out.append(s)
        s = task.apply(s, rng.choice(idx.applicable(s)))
    return out


def test_fast_packing_edges_and_values_equal_the_reference_on_cpu(train_tasks):
    from cp_disr.pddl import model as PM
    dev = torch.device("cpu")
    m = PM.make_model("dense", dev)
    with torch.no_grad():
        for p in m.attn.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m.eval()
    from cp_disr.pddl.fast_alt.verify import reference_codes
    for cid, c, task in train_tasks:
        ref, fast = RefNeuralEval("REF", m, dev), FastNeuralEval("FAST", m, dev)
        b = Budget(120, 10 ** 9, 8 * 2 ** 30)
        ref.prepare(task, {}, b)
        fast.prepare(task, {}, b)
        states = random_states(task, 60, 1) + [1 << (len(task.dyn_atoms) - 1), (1 << len(task.dyn_atoms)) - 1]
        assert torch.equal(reference_codes(m, task.template(), task, states, dev), fast.fv.pack(states))        # also bits above 64 and the highest atom
        for B in (1, 7, 48, 62):
            r, f = ref.evaluate(states[:B], b), fast.evaluate(states[:B], b)
            assert max(abs(x - y) for x, y in zip(r, f)) <= 1e-5 + 1e-6 * max(abs(x) for x in r)
            assert sorted(range(B), key=lambda i: (r[i], i)) == sorted(range(B), key=lambda i: (f[i], i)) or max(abs(x - y) for x, y in zip(r, f)) > 0


def test_fast_does_not_change_the_search_on_cpu(train_tasks):
    from cp_disr.pddl import model as PM
    from cp_disr.pddl.fast_alt.evaluators import TracingEval
    dev = torch.device("cpu")
    m = PM.make_model("rel", dev)
    m.eval()
    for cid, c, task in train_tasks[:2]:
        idx = SuccessorIndex(task)
        runs = []
        for mk in (RefNeuralEval, FastNeuralEval):
            tr = TracingEval(mk("x", m, dev))
            b = Budget(60, 400, 8 * 2 ** 30)
            tr.prepare(task, {}, b)
            r = gbfs(task, idx, tr, b)
            runs.append((r.status, r.plan, r.expanded, tr.batches))
        assert runs[0] == runs[1]
