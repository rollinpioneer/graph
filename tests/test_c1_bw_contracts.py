"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: contract correctness (runbook 17.1). Offline, pure symbolic."""
import random

import pytest

from cp_disr.blocksworld import contracts as C
from cp_disr.blocksworld import generator as G
from cp_disr.blocksworld import state as S
from cp_disr.contracts import nominal_overlay, precondition_value
from cp_disr.facts import Truth


def _facts(problem, state):
    return {k: (Truth.TRUE if v else Truth.FALSE) for k, v in S.fact_values(problem, state).items()}


def _walk(seed, steps=25):
    rng = random.Random(seed)
    n = rng.choice([3, 4, 5, 6])
    colors = tuple(rng.randint(0, 1) for _ in range(n))
    st = G.random_initial(n, list(colors), rng)
    prob = S.Problem(S.default_names(n), colors, st, st)
    out = [st]
    for _ in range(steps):
        st = S.apply(st, rng.choice(S.legal_actions(st)))
        out.append(st)
    return prob, out


@pytest.mark.parametrize("seed", range(12))
def test_preconditions_and_effects_match_the_engine_on_random_walks(seed):
    prob, states = _walk(seed)
    cons = {c.id: c for c in C.grounded_contracts(prob)}
    for st in states:
        facts = _facts(prob, st)
        legal = set(S.legal_actions(st))
        for a in C.all_actions(prob.n):
            c = cons[S.action_id(prob.names, a)]
            ok = precondition_value(c, facts) == Truth.TRUE
            assert ok == (a in legal)
            if ok:
                assert dict(nominal_overlay(c, facts).items()) == _facts(prob, S.apply(st, a))


@pytest.mark.parametrize("seed", range(6))
def test_invariants_hold_along_walks_and_illegal_states_are_rejected(seed):
    _prob, states = _walk(seed, 40)
    assert all(S.is_valid(s) for s in states)
    assert not S.is_valid((0, S.TABLE))                       # a block on itself
    assert not S.is_valid((1, 0))                             # cycle
    assert not S.is_valid((S.HELD, S.HELD, S.TABLE))          # two held blocks
    assert not S.is_valid((2, 2, S.TABLE))                    # two blocks on one support
    assert not S.is_valid((1, S.HELD))                        # a block resting on a held block


def test_masks_exclude_illegal_actions_and_stack_has_distinct_arguments():
    prob, states = _walk(3)
    cons = C.grounded_contracts(prob)
    assert all(not (c.name in ("STACK", "UNSTACK") and c.bound_arguments[0] == c.bound_arguments[1]) for c in cons)
    for st in states:
        legal = {S.action_id(prob.names, a) for a in S.legal_actions(st)}
        assert all(a in {c.id for c in cons} for a in legal)
        held = S.held(st)
        if held >= 0:
            assert not any(a[0] in ("PICK_UP", "UNSTACK") for a in S.legal_actions(st))
        else:
            assert not any(a[0] in ("PUT_DOWN", "STACK") for a in S.legal_actions(st))


def test_colours_enter_through_static_precondition_edges_not_names():
    prob = S.Problem(S.default_names(3), (S.RED, S.BLUE, S.RED), (S.TABLE, S.TABLE, S.TABLE), (S.TABLE, 0, 1))
    t = C.template_for(prob)
    edges = {(a, b, r) for a, b, r in t.edges}
    for c in t.contracts:
        for blk in c.bound_arguments:
            idx = prob.names.index(blk)
            atom = "p:%s:%s" % ("Red" if prob.colors[idx] == S.RED else "Blue", blk)
            assert (atom, c.id, "PRE_POS") in edges
    facts = S.fact_values(prob, prob.init)
    assert facts["p:Red:b0"] and not facts["p:Blue:b0"] and facts["p:Blue:b1"]
    assert {n.id for n in t.nodes if n.kind == "PROPOSITION"} == set(facts)


def test_goal_atoms_and_success_condition():
    prob = S.Problem(S.default_names(3), (0, 1, 0), (S.TABLE, S.TABLE, S.TABLE), (S.TABLE, 0, 1))
    assert S.goal_atoms(prob) == ("p:On:b1:b0", "p:On:b2:b1", "p:OnTable:b0")
    s = prob.init
    for a in (("PICK_UP", 1), ("STACK", 1, 0), ("PICK_UP", 2), ("STACK", 2, 1)):
        assert not S.goal_satisfied(prob, s)
        s = S.apply(s, a)
    assert S.goal_satisfied(prob, s)


def test_reward_discount_and_cap_follow_the_runbook(monkeypatch):
    from cp_disr.blocksworld.environment import BwEpisode, Case, step_cap_for
    from cp_disr.rl import set_suite_half_life
    set_suite_half_life(6.0)
    prob = S.Problem(S.default_names(3), (0, 1, 0), (S.TABLE, S.TABLE, S.TABLE), (S.TABLE, 0, 1))
    case = Case("t", "train", prob, 4, step_cap_for(4))
    assert case.step_cap == 12
    ep = BwEpisode(case)
    rewards = []
    for a in ("PICK_UP:b1", "STACK:b1:b0", "PICK_UP:b2", "STACK:b2:b1"):
        rewards.append(ep.step("a:%s:v1" % a))
    assert [r[0] for r in rewards] == [0.0, 0.0, 0.0, 1.0] and rewards[-1][1] and ep.success
    ep2 = BwEpisode(case)
    reasons = []
    while not ep2.done:
        reasons.append(ep2.step(sorted(ep2.legal_ids())[0])[3])
    assert len(reasons) <= case.step_cap and reasons[-1] in ("TASK_SUCCESS", "DEADLINE")
    with pytest.raises(RuntimeError):
        ep2.step("a:PICK_UP:b0:v1")                       # the episode is finished
    ep3 = BwEpisode(case)
    with pytest.raises(RuntimeError):
        ep3.step("a:STACK:b0:b1:v1")                      # nothing is held: the action is masked and must never execute
