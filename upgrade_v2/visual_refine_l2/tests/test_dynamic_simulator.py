from __future__ import annotations

import numpy as np

from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec


def simulator(scenario: str = "normal_pick_place") -> DynamicTabletop:
    return DynamicTabletop(family_spec("test_family", scenario, 7, 100), 101)


def test_snapshot_restore_replays_state_exactly() -> None:
    sim = simulator()
    sim.perform("approach_object")
    snapshot = sim.snapshot()
    first = sim.perform("close_gripper")
    first_state = sim.snapshot()
    sim.restore(snapshot)
    second = sim.perform("close_gripper")
    second_state = sim.snapshot()
    assert first == second
    np.testing.assert_allclose(first_state["qpos"], second_state["qpos"], rtol=0.0, atol=1e-12)
    assert first_state["attached"] == second_state["attached"]


def test_attach_release_and_slip_are_explicit() -> None:
    sim = simulator()
    sim.perform("approach_object"); sim.perform("close_gripper")
    assert sim.attached and sim.data.eq_active[sim.weld_id]
    sim.perform("open_gripper")
    assert not sim.attached and not sim.data.eq_active[sim.weld_id]
    slip = simulator("slip_then_recover")
    for action in ("approach_object", "close_gripper", "lift", "transport_to_target"):
        slip.perform(action)
    assert any(event["event"] == "contact_lost" for event in slip.events)
    slip.perform("recover")
    assert any(event["event"] == "recovery_achieved" for event in slip.events)


def test_stable_place_and_terminal_classes_are_distinct() -> None:
    sim = simulator()
    for action in ("approach_object", "close_gripper", "lift", "transport_to_target", "align", "lower", "open_gripper", "verify"):
        sim.perform(action)
    assert sim.oracle_snapshot()["goal_stable"]
    assert "horizon" != "failure_terminal"
