"""C01-C08: the frozen contract cannot rank the two object orders (offline; no environment, no simulator)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reach():
    """The offline reachability document, produced by the same code path freeze-configs uses."""
    import os
    import tempfile

    out = Path(tempfile.mkdtemp())
    (out / "geometry").mkdir(parents=True, exist_ok=True)
    return m._contract_reachability(ROOT, out)


@pytest.fixture(scope="module")
def contracts():
    contracts, template, objects = m._build_offline_template(ROOT)
    return contracts, template, objects


def test_C01_exactly_two_initial_pick_candidates(contracts):
    _contracts, template, _objects = contracts
    facts = m._initial_fact_store()
    initial = sorted(c.id for c in template.contracts if m._precond_true(c, facts))
    assert initial == sorted(m.INITIAL_CANDIDATES)
    assert len(initial) == 2
    assert all(cid.split(":")[1] == "PICK" for cid in initial)


def test_C02_target_first_route_reachable(reach):
    assert reach["routes"][m.ROUTE_TARGET_FIRST]["reachable"] is True
    assert tuple(reach["routes"][m.ROUTE_TARGET_FIRST]["plan"]) == m.ROUTES[m.ROUTE_TARGET_FIRST]


def test_C03_second_first_route_reachable(reach):
    assert reach["routes"][m.ROUTE_SECOND_FIRST]["reachable"] is True
    assert tuple(reach["routes"][m.ROUTE_SECOND_FIRST]["plan"]) == m.ROUTES[m.ROUTE_SECOND_FIRST]


def test_C04_route_length_equal(reach):
    lengths = {name: r["length"] for name, r in reach["routes"].items()}
    assert lengths[m.ROUTE_TARGET_FIRST] == lengths[m.ROUTE_SECOND_FIRST] == 4
    assert reach["route_lengths_equal"] is True


def test_C05_skill_multiset_equal(reach):
    assert reach["skill_multisets_equal"] is True
    assert sorted(reach["skill_multiset"]) == ["PICK", "PICK", "PLACE", "PLACE"]
    tf = sorted(c.split(":")[1] for c in m.ROUTES[m.ROUTE_TARGET_FIRST])
    sf = sorted(c.split(":")[1] for c in m.ROUTES[m.ROUTE_SECOND_FIRST])
    assert tf == sf


def test_C06_b_plan_nominal_cost_equal(reach):
    costs = reach["b_plan_nominal_costs_seconds"]
    assert set(costs) == {m.ROUTE_TARGET_FIRST, m.ROUTE_SECOND_FIRST}
    assert abs(costs[m.ROUTE_TARGET_FIRST] - costs[m.ROUTE_SECOND_FIRST]) < 1e-12
    assert reach["b_plan_nominal_costs_equal"] is True
    assert reach["both_routes_minimal_cost"] is True
    assert abs(costs[m.ROUTE_TARGET_FIRST] - 4 * m.D_REF) < 1e-9


def test_C07_no_open_buffer_or_move_action(contracts, reach):
    _contracts, template, _objects = contracts
    ids = {c.id for c in template.contracts}
    assert ids == set(m.GROUNDED_ACTIONS)
    for cid in ids:
        assert not cid.startswith(("a:OPEN:", "a:PLACE_BUFFER:", "a:MOVE:"))
    assert reach["no_hard_fact_ranks_routes"] is True


def test_C08_no_geometry_hard_fact(contracts):
    _contracts, template, _objects = contracts
    props = {n.id for n in template.nodes if n.kind == "PROPOSITION"}
    assert props == set(m.VERIFIER_FACT_IDS)
    for forbidden in ("ClearPath", "Occluded", "Blocked", "EasyGrasp", "SafeToPick"):
        assert not any(forbidden in p for p in props)


def test_C08b_canonical_tie_break_reads_no_geometry(reach):
    assert reach["canonical_tie_break_reads_geometry"] is False
    assert reach["canonical_first_action"] in m.INITIAL_CANDIDATES
    assert reach["canonical_first_action"] in tuple(m.ROUTES[m.ROUTE_SECOND_FIRST])