from upgrade_v2.visual_refine_l2.predicates import FALSE, TRUE, PREDICATE_NAMES, _baseline_summary


def frame(**values: str):
    predicates = {name: FALSE for name in PREDICATE_NAMES}
    predicates.update(values)
    return {"predicates": predicates}


def test_baselines_use_only_their_declared_observation_subset() -> None:
    predictions = [
        frame(object_moves_with_gripper=TRUE, object_moving=TRUE),
        frame(object_moves_with_gripper=FALSE, object_moving=TRUE),
        frame(object_moves_with_gripper=TRUE, object_stable_on_target=TRUE, object_centered_on_target=TRUE),
    ]
    b0 = _baseline_summary(predictions, "B0_text_coarse_assumption")
    b1 = _baseline_summary(predictions, "B1_current_frame_rgb")
    b2 = _baseline_summary(predictions, "B2_single_view_temporal")

    assert b0 == {"goal": False, "stable_hold": False, "failure": False, "recovery": False, "unknown_rate": 1.0}
    assert b1["goal"] is True
    assert b1["stable_hold"] is False
    assert b1["failure"] is False
    assert b2["stable_hold"] is True
    assert b2["failure"] is True
    assert b2["recovery"] is True
