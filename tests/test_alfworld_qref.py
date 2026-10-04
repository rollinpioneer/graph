"""Multi-holder Q_ref: probabilities, exchange-optimal order, public-state-only dependence."""
import itertools

import pytest

from cp_disr.analysis import alfworld_qref as Q
from cp_disr.platforms.alfworld.adapter import PublicState


class StubTables:
    RATES = {"drawer": 0.10, "shelf": 0.25, "toilet": 0.45, "countertop": 0.30, "bed": 0.05}
    OPEN = {"drawer": 1.0, "shelf": 0.0, "toilet": 0.0, "countertop": 0.0, "bed": 0.0}

    def holder_rate(self, otype, rtype):
        return self.RATES[rtype]

    def p_open(self, rtype):
        return self.OPEN.get(rtype, 0.0)


def pub(unchecked, raw_steps=0):
    return PublicState(goal_otype="x", goal_rtype="desk", receptacles=tuple(unchecked) + ("desk 1",), feasible=tuple(unchecked), unchecked=tuple(unchecked),
                       goal_instance="desk 1", raw_steps=raw_steps, max_steps=50, checked=(), target_in=(), target_found=False, holding=None,
                       placed=False, legal=tuple(("CHECK", r) for r in unchecked), done=False)


U = ("drawer 1", "shelf 1", "toilet 1", "countertop 1", "bed 1")


def test_first_holder_probabilities_sum_to_one(monkeypatch):
    monkeypatch.setattr(Q, "gamma", lambda s: 1.0)  # no discounting: value == total probability mass
    p = pub(U)
    for order in itertools.islice(itertools.permutations(U), 12):
        assert Q.order_value(StubTables(), p, list(order)) == pytest.approx(1.0)


def test_greedy_index_order_is_the_optimal_order_by_brute_force():
    t, p = StubTables(), pub(U)
    best = max(itertools.permutations(U), key=lambda o: Q.order_value(t, p, list(o)))
    assert Q.order_value(t, p, Q.greedy_order(t, p)) == pytest.approx(Q.order_value(t, p, list(best)))
    q = Q.q_ref(t, p)
    # Q_ref(first=a) is the value of visiting a first then mu_ref; the best first action is the index-greedy head
    assert max(q, key=q.get) == ("CHECK", Q.greedy_order(t, p)[0])


def test_q_ref_uses_public_state_only_and_respects_the_step_cap():
    t = StubTables()
    q0 = Q.q_ref(t, pub(U))
    assert Q.q_ref(t, pub(U)) == q0
    late = Q.q_ref(t, pub(U, raw_steps=46))
    assert all(v < q0[k] for k, v in late.items())  # later start => lower discounted value
    assert Q.q_ref(t, pub(U, raw_steps=49))[("CHECK", "toilet 1")] == 0.0  # cannot finish within the cap


def test_marginal_belief_is_a_probability_conditional_on_a_holder_existing():
    b = Q.belief(StubTables(), pub(U))
    assert all(0 < v <= 1 for v in b.values()) and sum(b.values()) >= 1.0  # several holders can exist
    assert b["toilet 1"] > b["drawer 1"] > b["bed 1"]
