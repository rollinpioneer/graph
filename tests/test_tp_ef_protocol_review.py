"""Offline tests for the protocol-review card (symbolic layer only; no simulator)."""
from pathlib import Path

import pytest

from cp_disr.analysis import tp_ef_protocol_review as r
from cp_disr.facts import Truth

ROOT = Path(__file__).resolve().parents[1]

INITIAL = {"p:AtBuffer:second_object:buffer": "FALSE", "p:AtBuffer:target:buffer": "FALSE", "p:GripperEmpty": "TRUE",
           "p:Held:second_object": "FALSE", "p:Held:target": "FALSE", "p:Inside:second_object:container": "FALSE",
           "p:Inside:target:container": "FALSE", "p:OnTable:second_object": "TRUE", "p:OnTable:target": "TRUE", "p:Open:container": "FALSE"}


@pytest.fixture(scope="module")
def tpl():
    return r.build_template(ROOT)


def test_zero_env_guard_installed(tpl):
    from cp_disr.platforms.libero import runtime_factory as rf
    with pytest.raises(RuntimeError, match="ZERO_ENV_GUARD"):
        rf.make_env(None)


def test_open_vs_pick_second_is_contract_tied_and_scriptable(tpl):
    v = {k: Truth(x) for k, x in INITIAL.items()}
    ra = r.min_plan_after(tpl, v, r.IDS["open"])
    rb = r.min_plan_after(tpl, v, r.IDS["ps"])
    assert ra[:2] == (True, 5) and rb[:2] == (True, 5)
    sa, _ = r.script_legality(tpl, v, r.SCRIPTS["OPEN_vs_PICK_second"]["A"])
    sb, _ = r.script_legality(tpl, v, r.SCRIPTS["OPEN_vs_PICK_second"]["B"])
    assert r.classify_pair(True, True, ra[:2], rb[:2], sa, sb) == "CONTRACT_TIED_SCRIPTABLE"


def test_pick_target_first_is_contract_dead_end(tpl):
    v = {k: Truth(x) for k, x in INITIAL.items()}
    ok, steps = r.script_legality(tpl, v, r.SCRIPTS["PICK_target_vs_PICK_second"]["A"])
    assert not ok and steps[1]["action"] == r.IDS["plt"] and not steps[1]["legal_at_step"]
    assert r.min_plan_after(tpl, v, r.IDS["pt"])[0] is False


def test_classification_rules():
    assert r.classify_pair(True, True, (True, 5), (False, None), True, False) == "CONTRACT_REVEALED_PAIR"
    assert r.classify_pair(True, True, (True, 5), (True, 7), True, True) == "CONTRACT_REVEALED_PAIR"
    assert r.classify_pair(True, False, (True, 5), (True, 5), True, True) == "NOT_BOTH_LEGAL"
    assert r.classify_pair(True, True, (False, None), (False, None), False, False) == "NEITHER_ROUTE_REACHABLE"
    assert r.classify_pair(True, True, (True, 5), (True, 5), True, False) == "CONTRACT_TIED_NEEDS_DYNAMIC_PLANNER"
