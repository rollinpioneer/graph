"""Fixtures of the public-Depots card (C1-PUBLIC-DEPOTS-REL-V1): PDDL transfer consistent with Fast Downward, legal actions / goals not rewritten, complete state equality, the generic relation mask
reproduces the Blocksworld mask, labels never enter the forward pass, gate behaviour with 0 / 1 / 2 root actions, h_add. CPU only; external-tool fixtures are skipped when the tools are absent."""
import os
import random
import subprocess
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import a04p_controls as AC
from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import scorer_control as SC
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld.environment import BwEpisode
from cp_disr.blocksworld.method_serial import lookahead as LA
from cp_disr.blocksworld.method_serial import model as MD
from cp_disr.pddl import bw_stage as BWS
from cp_disr.pddl import control as CT
from cp_disr.pddl import data as DT
from cp_disr.pddl import depots as DP
from cp_disr.pddl import model as PM
from cp_disr.pddl import stages as ST
from cp_disr.pddl import task as T
from cp_disr.pddl.parse import parse_domain, parse_problem
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
EXT = Path(os.environ.get("CPDISR_EXT", str(Path.home() / "ext")))
FD = EXT / "downward" / "fast-downward.py"
BENCH = EXT / "downward-benchmarks" / "depot"
EXACT = EXT / "bin" / "exact_dist"
HAVE_EXT = FD.is_file() and (BENCH / "domain.pddl").is_file() and EXACT.is_file() and DP.GEN.is_file()
need_ext = pytest.mark.skipif(not HAVE_EXT, reason="external planners / generator not on this machine")
HAVE_BW = (ROOT / SC.M1GOAL["path"]).is_file() and (ROOT / E.MODELS["B2"]["path"]).is_file()
need_bw = pytest.mark.skipif(not HAVE_BW, reason="Blocksworld checkpoints not on this machine")


def tiny_problem(n=3, seed=1):
    for s in range(seed, seed + 400):
        text, _cmd = DP.generate(n, s)
        a = DP.analyse(text)
        if a["goal_complete"] and not a["init_satisfies_goal"]:
            return text
    raise AssertionError("no tiny instance")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    text = tiny_problem()
    p = tmp_path_factory.mktemp("tiny") / "p.pddl"
    p.write_text(text)
    return p, T.depots_task(DP.DOMAIN_TYPED, p)


# ---------------------------------------------------------------------------------------------------------- PDDL transfer
@need_ext
def test_exact_distance_equals_lmcut_and_fd_plans_replay_in_our_semantics(tiny, tmp_path):
    p, task = tiny
    res = DT.build_case_data(DP.DOMAIN_TYPED, p, "c")
    assert res["status"] == "OK"
    fd = DT.run_fd(DP.DOMAIN_TYPED, p, tmp_path, "seq-opt-lmcut", 120, tag="lmcut")
    assert fd["solved"] and fd["best_length"] == res["optimal_length"]                 # exact BFS == A* + LM-cut
    ids = DT.plan_ids_from_file(fd["best_plan_file"])
    assert len(ids) == res["optimal_length"]
    assert task.goal_satisfied(task.replay(ids))
    sat = DT.run_fd(DP.DOMAIN_TYPED, p, tmp_path, "lama-first", 60, tag="lama")
    ids = DT.plan_ids_from_file(sat["best_plan_file"])
    assert task.goal_satisfied(task.replay(ids)) and len(ids) >= res["optimal_length"]


@need_ext
def test_ipc_instance_grounds_and_a_fd_plan_replays(tmp_path):
    task = T.depots_task(BENCH / "domain.pddl", BENCH / "p01.pddl")
    subprocess.run(["python3", str(FD), "--plan-file", str(tmp_path / "p.plan"), "--sas-file", str(tmp_path / "o.sas"), "--overall-time-limit", "60", str(BENCH / "domain.pddl"), str(BENCH / "p01.pddl"),
                    "--search", "lazy_greedy([ff()],preferred=[ff()])"], capture_output=True, cwd=str(tmp_path))
    ids = DT.plan_ids_from_file(tmp_path / "p.plan")
    assert ids and task.goal_satisfied(task.replay(ids))
    assert {t for t in task.obj_type.values()} <= {"crate", "pallet", "truck", "hoist", "place"}            # leaf types from static unary facts; depot / distributor merged into place


def _reference_legal(task, state):
    """Independent legality: enumerate typed bindings of the LIFTED domain against the atom set of the state (no bit masks, no reachability pruning)."""
    d, p = task.domain, task.problem
    atoms = {task.dyn_atoms[i] for i in task.state_list(state)} | set(a for a in p.init if a[0] not in task.dyn_preds)
    out = set()
    import itertools
    for act in d.actions.values():
        pools = [sorted(o for o, t in p.objects.items() if (d.subtype_of(t, pt) if d.typed else True)) for _v, pt in act.params]
        for b in itertools.product(*pools):
            sub = dict(zip((v for v, _t in act.params), b))
            g = lambda xs: [(pr, tuple(sub[v] for v in args)) for pr, args in xs]
            if all(a in atoms for a in g(act.pre_pos)) and not any(a in atoms for a in g(act.pre_neg)):
                add, dele = set(g(act.add)), set(g(act.delete)) - set(g(act.add))
                if add <= atoms and not (dele & atoms):
                    continue                                                       # no-op by the documented rule
                out.add("a:" + act.name + ":" + ":".join(b) + ":v1")
    return out


@need_ext
def test_legal_actions_and_goal_are_exactly_the_lifted_semantics_along_random_walks(tiny):
    _p, task = tiny
    rng = random.Random(0)
    goal_atoms = {a for a in task.problem.goal}
    assert {task.dyn_atoms[i] for i in task.state_list(task.goal_mask)} == {a for a in goal_atoms if a[0] in task.dyn_preds}
    s = task.init_mask
    for _ in range(60):
        legal = task.legal(s)
        assert {a.aid for a in legal} == _reference_legal(task, s)
        s = task.apply(s, rng.choice(legal))


@need_ext
def test_state_equality_includes_every_dynamic_atom(tiny):
    _p, task = tiny
    s = task.init_mask
    for a in task.legal(s):
        assert task.apply(s, a) != s                                                # no legal action is a no-op (the no-op ground actions were dropped)
    kinds = {a[0] for a in task.dyn_atoms}
    assert {"at", "in", "lifting", "available", "clear", "on"} <= kinds           # transport position, loaded crates, held crates, hoist state and stacks are all in the state


@need_ext
def test_labels_are_scoring_only_the_selected_actions_do_not_depend_on_them(tiny):
    p, task = tiny
    res = DT.build_case_data(DP.DOMAIN_TYPED, p, "c")
    case = T.PddlCase("c", "t", task, res["optimal_length"], T.step_cap_for(res["optimal_length"]))
    model = PM.make_model("rel", torch.device("cpu"))
    model.eval()
    pol = MD.SerialPolicy(model)
    runs = []
    for lab in (None, ST.ExactLabeler(task)):
        ep = T.PddlEpisode(case)
        r = CT.run_episode(pol, ep, CT.PddlOps(task, model), CT.choose_one_step, lab)
        runs.append([d["selected"] for d in r["decisions"]])
        if lab is not None:
            lab.close()
    assert runs[0] == runs[1]
    snap = T.PddlEpisode(case).snapshot()
    fields = list(getattr(snap, "__dataclass_fields__", {})) or list(snap._fields)
    assert not any("label" in k or "optimal" in k or "dist" in k for k in fields)


@need_ext
def test_generator_instances_are_complete_layouts_and_isomorphism_hash_ignores_names(tiny):
    p, _task = tiny
    text = p.read_text()
    a = DP.analyse(text)
    assert a["goal_complete"] and sum(a["goal_heights"]) == a["n_crates"]
    swapped = text.replace("crate0", "@@").replace("crate1", "crate0").replace("@@", "crate1")
    assert DP.canonical_hash(swapped) == DP.canonical_hash(text)
    other = tiny_problem(3, 500)
    assert DP.canonical_hash(other) != DP.canonical_hash(text) or other == text


# ---------------------------------------------------------------------------------------------------------- gate boundaries (Depots controller)
class _NoLeaf:
    def state_values(self, *a, **k):
        raise AssertionError("no leaf evaluation expected")


class _CountLeaf:
    def __init__(self):
        self.calls = 0

    def state_values(self, template, task, states, steps=None):
        self.calls += 1
        return torch.arange(len(states), dtype=torch.float32)


def _fake_snapshot(task, state):
    ep = T.PddlEpisode(T.PddlCase("c", "t", task, 10, 30))
    ep.state = state
    return ep, ep.snapshot()


@need_ext
def test_depots_gate_with_zero_one_and_two_roots(tiny):
    _p, task = tiny
    ops = CT.PddlOps(task, _NoLeaf())
    ep, snap = _fake_snapshot(task, task.init_mask)
    ids = snap.candidate_ids
    legal = [i for i, ok in enumerate(snap.mask) if ok]
    assert len(legal) >= 3
    succ = {i: ops.apply_id(ep.state, ids[i]) for i in legal}
    logits = torch.zeros(len(ids))
    for rank, i in enumerate(legal):
        logits[i] = -float(rank)                                                    # distinct, margin 1 between the first two
    # 0 roots: every successor visited
    mem = CT.Memory(ep.state)
    mem.visited |= set(succ.values())
    c = CT.Counters()
    sel, info = CT.choose_gated(ops, _NoLeaf(), ep, snap, logits, mem, 100.0, c)
    assert sel is None and info["trigger"] == "NO_UNVISITED_SUCCESSOR" and c.gate_expansions == 0
    # 1 root: even a huge tau does not start a look-ahead
    mem = CT.Memory(ep.state)
    mem.visited |= {s for i, s in succ.items() if i != legal[1]}
    c = CT.Counters()
    sel, info = CT.choose_gated(ops, _NoLeaf(), ep, snap, logits, mem, 100.0, c)
    assert sel == legal[1] and info["gate"] == "single_root" and c.gate_expansions == 0
    # 2+ roots, margin above tau and no local dead end: keep the one-step choice
    mem = CT.Memory(ep.state)
    c = CT.Counters()
    sel, info = CT.choose_gated(ops, _NoLeaf(), ep, snap, logits, mem, 0.5, c)
    assert sel == legal[0] and info["gate"] == "keep_one_step" and c.gate_expansions == 0 and c.gate_checks == 1
    # 2+ roots, margin <= tau: the shared two-step tree is evaluated
    leaf = _CountLeaf()
    c = CT.Counters()
    ops.model = leaf
    sel, info = CT.choose_gated(ops, leaf, ep, snap, logits, CT.Memory(ep.state), 1.0, c)
    assert info["gate"] == "low_margin" and c.gate_expansions == 1 and leaf.calls == 1
    # tau unavailable (None): the margin sub-gate is off
    c = CT.Counters()
    ops.model = _NoLeaf()
    sel, info = CT.choose_gated(ops, ops.model, ep, snap, logits, CT.Memory(ep.state), None, c)
    assert info["gate"] == "keep_one_step"
    # local dead end behind the first choice: every successor of it is already visited -> expand although the margin is large
    first = legal[0]
    s1 = succ[first]
    mem = CT.Memory(ep.state)
    mem.visited |= {ops.apply_id(s1, b) for b in ops.legal_ids(s1)}
    leaf, c = _CountLeaf(), CT.Counters()
    ops.model = leaf
    sel, info = CT.choose_gated(ops, leaf, ep, snap, logits, mem, 0.5, c)
    assert info["gate"] == "local_trap" and c.gate_expansions == 1 and c.trap_unfolds == len(ops.legal_ids(s1))


# ---------------------------------------------------------------------------------------------------------- Blocksworld fixtures
@pytest.fixture(scope="module")
def bw_cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


@pytest.fixture(scope="module")
def mg():
    return SC.load_scorer(ROOT, "MG_C3", torch.device("cpu")).model


@need_bw
def test_generic_relation_mask_reproduces_the_blocksworld_mask(bw_cases, mg):
    bw, gen = MD.GoalAttention("rel"), PM.GeneralGoalAttention("rel")
    n = 0
    for c in bw_cases[::9]:
        ep = BwEpisode(c)
        for _ in range(4):
            snap = ep.snapshot()
            st, gprop = mg.goal_free_static(snap.template)
            codes = mg.base._codes(st, snap.facts.values).unsqueeze(0)
            _f1, a1 = bw.pair_features(bw._goal_static(st, gprop, snap.template), codes, gprop)
            _f2, a2 = gen.pair_features(gen._goal_static(st, gprop, snap.template), codes, gprop)
            assert torch.equal(a1, a2)
            n += 1
            legal = [j for j, ok in enumerate(snap.mask) if ok]
            ep.step(snap.candidate_ids[legal[len(legal) // 2]])
    assert n > 20


@need_bw
def test_blocksworld_generic_two_step_and_native_look_ahead_choose_the_same(bw_cases, mg):
    model = MD.SerialModel(mg, "MG")
    model.eval()
    native = LA.make_chooser(model, "MG", LA.Counters())
    fixed = BWS.BwChooser(model, "fixed", BWS.GateCounters())
    for c in bw_cases[::40]:
        ep = BwEpisode(c)
        ops = BWS.BwOps(c.problem, model)
        mem = AC.Memory(ep.state)
        for _ in range(5):
            if ep.done:
                break
            snap = ep.snapshot()
            logits = model.eval_logits(snap)
            a, _ia = native("", ep, snap, logits, mem)
            b, _ib = fixed("", ep, snap, logits, mem)
            g, _ig = CT.choose_two_step(ops, model, ep, snap, logits, mem, CT.Counters())
            assert a == b == g
            if a is None:
                break
            ep.step(snap.candidate_ids[a])
            mem.visited.add(ep.state)


@need_bw
def test_blocksworld_gate_boundaries_and_no_leaf_for_single_root(bw_cases, mg):
    model = MD.SerialModel(mg, "MG")
    model.eval()
    case = next(c for c in bw_cases if c.problem.n >= 4)
    ep = BwEpisode(case)
    snap = ep.snapshot()
    ids = snap.candidate_ids
    legal = [i for i, ok in enumerate(snap.mask) if ok]
    succ = {i: S.apply(ep.state, ep._action_of[ids[i]]) for i in legal}
    logits = torch.zeros(len(ids))
    for r, i in enumerate(legal):
        logits[i] = -float(r) * 2.0
    mem = AC.Memory(ep.state)
    cnt = BWS.GateCounters()
    ch = BWS.BwChooser(_NoLeaf(), "gated", cnt, tau=1.0)
    mem.visited |= set(succ.values())
    sel, info = ch("", ep, snap, logits, mem)
    assert sel is None and cnt.gate_expansions == 0
    mem = AC.Memory(ep.state)
    mem.visited |= {s for i, s in succ.items() if i != legal[-1]}
    sel, info = ch("", ep, snap, logits, mem)
    assert sel == legal[-1] and ch.log[-1]["gate"] == "single_root" and cnt.gate_expansions == 0
    mem = AC.Memory(ep.state)
    sel, info = ch("", ep, snap, logits, mem)                                      # margin 2 > tau 1, not a dead end
    assert ch.log[-1]["gate"] == "keep_one_step" and cnt.gate_expansions == 0 and sel == legal[0]
    ch2 = BWS.BwChooser(model, "gated", BWS.GateCounters(), tau=5.0)
    sel, info = ch2("", ep, snap, logits, AC.Memory(ep.state))                     # margin 2 <= tau 5: expansion
    assert ch2.log[-1]["gate"] == "low_margin" and ch2.cnt.gate_expansions == 1 and ch2.log[-1]["expanded"]


def test_h_add_of_a_two_block_problem_is_hand_computable():
    p = S.Problem(S.default_names(2), (0, 1), (S.TABLE, S.TABLE), (1, S.TABLE))                 # goal: b0 on b1, b1 on the table
    h = BWS.HAdd(p)
    assert h(p.init) == 2                                                          # OnTable(b1) = 0 ; On(b0, b1) = 1 + Holding(b0) (1) + Clear(b1) (0)
    assert h((S.HELD, S.TABLE)) == 1
    assert h((1, S.TABLE)) == 0


def test_select_checkpoint_rule_and_percentile():
    rows = {"820": {"success": 5, "mean_penalised_ratio": 2.0}, "1640": {"success": 7, "mean_penalised_ratio": 2.5}, "2460": {"success": 7, "mean_penalised_ratio": 2.4},
            "3280": {"success": 7, "mean_penalised_ratio": 2.4}, "4100": {"success": 6, "mean_penalised_ratio": 1.0}}
    assert ST.select_checkpoint(rows) == "2460"                                    # most successes, then the smaller penalised cost, then the earlier checkpoint
    assert BWS.percentile([1, 2, 3, 4, 5], 10) == pytest.approx(1.4)


@need_ext
def test_three_models_start_from_identical_tensors_and_agree_at_initialisation(tiny):
    p, task = tiny
    ms = {m: PM.make_model(m, torch.device("cpu")) for m in ("mg", "dense", "rel")}
    base = ms["mg"].state_dict()
    for m in ("dense", "rel"):
        sd = ms[m].state_dict()
        assert all(torch.equal(sd[k], v) for k, v in base.items())                           # shared encoder / heads: identical initial tensors
    a, b = ms["dense"].attn.state_dict(), ms["rel"].attn.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)                                         # DENSE and REL add the same new layer
    case = T.PddlCase("c", "t", task, 10, 24)
    snap = T.PddlEpisode(case).snapshot()
    for m in ms.values():
        m.eval()
    out = {m: ms[m].eval_logits(snap) for m in ms}
    assert torch.allclose(out["mg"], out["dense"]) and torch.allclose(out["mg"], out["rel"])  # zero-initialised output projection: the same function at step 0
    assert ms["rel"].mask_density(snap) <= 1.0 == ms["dense"].mask_density(snap)               # a tiny 3-crate task may have every goal pair related; larger tasks are measured in the run