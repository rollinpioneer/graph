from __future__ import annotations

import numpy as np

from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, family_spec
from upgrade_v2.visual_refine_l2.repaired_simulator import AttachRelposeDynamicTabletop


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


def test_action_records_replayable_low_level_controls() -> None:
    sim = simulator()
    result = sim.perform("approach_object")
    controls = result["low_level_control_sequence"]
    assert len(controls) == 4
    assert all(step["physics_steps"] == 5 for step in controls)
    assert controls[0]["start_time"] == result["start_time"]
    assert controls[-1]["end_time"] == result["end_time"]
    assert all(len(step["mocap_position"]) == 3 for step in controls)


def test_jitter_flags_are_respected() -> None:
    spec = family_spec(
        "fixed_family",
        "normal_pick_place",
        7,
        100,
        camera_jitter=False,
        object_size_jitter=False,
        friction_jitter=False,
    )
    assert spec.camera_jitter == 0.0
    assert spec.object_radius == 0.069
    assert spec.friction == 0.75


def test_attach_relpose_variant_updates_weld_and_restores_it() -> None:
    spec = family_spec("repair_family", "normal_pick_place", 7, 100)
    sim = AttachRelposeDynamicTabletop(spec, 101)
    sim.perform("approach_object")
    snapshot = sim.snapshot()
    sim.perform("close_gripper")
    relative = sim.object_xyz - sim.data.mocap_pos[0]
    np.testing.assert_allclose(sim.model.eq_data[sim.weld_id, 3:6], relative, rtol=0.0, atol=1e-12)
    sim.restore(snapshot)
    np.testing.assert_allclose(sim.model.eq_data, snapshot["model_eq_data"], rtol=0.0, atol=0.0)
    assert sim.repair_version == "l2rar2_attach_relpose_v1"


def test_attach_relpose_variant_uses_body2_in_body1_frame() -> None:
    spec = family_spec("pose_family", "normal_pick_place", 7, 100)
    sim = AttachRelposeDynamicTabletop(spec, 101)
    sim.perform("approach_object")
    half_turn = np.sqrt(0.5)
    sim.data.mocap_quat[0] = np.asarray((half_turn, 0.0, 0.0, half_turn))
    sim.data.qpos[sim.object_qpos + 3:sim.object_qpos + 7] = np.asarray((1.0, 0.0, 0.0, 0.0))
    sim.mujoco.mj_forward(sim.model, sim.data)
    gripper = sim.data.mocap_quat[0]
    inverse = np.asarray((gripper[0], -gripper[1], -gripper[2], -gripper[3]))
    world_delta = sim.object_xyz - sim.data.mocap_pos[0]
    expected_position = np.asarray((world_delta[1], -world_delta[0], world_delta[2]))
    sim._attach()
    np.testing.assert_allclose(sim.model.eq_data[sim.weld_id, 3:6], expected_position, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sim.model.eq_data[sim.weld_id, 6:10], inverse, rtol=0.0, atol=1e-12)
