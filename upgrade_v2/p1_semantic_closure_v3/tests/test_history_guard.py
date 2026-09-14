from upgrade_v2.p1_semantic_closure_v3 import history_guard as hg


def test_required_sequences_pass():
    rows = hg.evaluate_required_sequences()
    assert all(r["passed"] for r in rows)
    assert len(rows) == 10


def test_unknown_not_true():
    out = hg.guard_events([dict(available_at_ns=1, a_valid=True, b_valid=None, goal_verified=True)])
    assert out[-1]["success_eligible"] is None


def test_oracle_not_overlay():
    events = [dict(available_at_ns=1, a_valid=True, b_valid=False, goal_verified=True)]
    overlay = hg.guard_events(events, rule="overlay")[-1]["success_eligible"]
    oracle = hg.guard_events(events, rule="oracle")[-1]["success_eligible"]
    assert overlay is False
    assert oracle is True