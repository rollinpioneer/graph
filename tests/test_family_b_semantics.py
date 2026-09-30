"""Pure contract and measurement semantics; no environment construction."""
from types import SimpleNamespace
import numpy as np
import pytest

from cp_disr.contracts import nominal_overlay, precondition_value
from cp_disr.facts import Truth
from cp_disr.platforms.libero.family_b_runtime import (
    ACTION_IDS, GOALS, VERIFIER_FACT_IDS, build_task_template)
from cp_disr.platforms.libero.family_b_adapters import (
    FamilyBEvaluator, FamilyBVerifier, _palette_mask)
from cp_disr.platforms.libero.family_b_env import FamilyBEnv, PALETTE

REGISTRY = "configs/runtime/tp_fb_contract_registry.yaml"
PREFIXES = {
    "B_PENDING": ("a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1",
                  "a:PICK:carrier:v1"),
    "C_PENDING": ("a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1",
                  "a:PICK:carrier:v1"),
    "BOTH_PENDING": ("a:PICK:carrier:v1",),
}
PADS = ("a:PLACE_BUFFER:carrier:pad_u:v1",
        "a:PLACE_BUFFER:carrier:pad_v:v1")


def initial_values():
    values = {fid: Truth.FALSE for fid in VERIFIER_FACT_IDS}
    for fid in ("p:GripperEmpty", "p:Open:receiver",
                "p:OnTable:carrier", "p:OnTable:obj_b", "p:OnTable:obj_c"):
        values[fid] = Truth.TRUE
    return values


@pytest.mark.parametrize("context", tuple(PREFIXES))
def test_production_nominal_contracts_have_two_legal_pads_and_reachable_goals(context):
    template = build_task_template(REGISTRY)
    assert {c.id for c in template.contracts} == set(ACTION_IDS)
    assert {g.fact_id for g in template.goals} == {g.fact_id for g in GOALS}
    by_id = {c.id: c for c in template.contracts}
    values = initial_values()
    for aid in PREFIXES[context]:
        assert precondition_value(by_id[aid], values) is Truth.TRUE
        values = dict(nominal_overlay(by_id[aid], values))
    legal = {aid for aid, c in by_id.items()
             if precondition_value(c, values) is Truth.TRUE}
    assert legal == set(PADS)
    for pad in PADS:
        after = dict(nominal_overlay(by_id[pad], values))
        assert after["p:GripperEmpty"] is Truth.TRUE
        assert after["p:AtBuffer:carrier:" + pad.split(":")[3]] is Truth.TRUE
        suffix = (
            ("a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1")
            if context == "B_PENDING" else
            ("a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1")
            if context == "C_PENDING" else
            ("a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1",
             "a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1")
        )
        for aid in suffix:
            assert precondition_value(by_id[aid], after) is Truth.TRUE
            after = dict(nominal_overlay(by_id[aid], after))
        assert all(after[g.fact_id] is Truth.TRUE for g in GOALS)
        assert len(suffix) + 1 == (3 if context != "BOTH_PENDING" else 5)


def test_two_destinations_and_receiver_slots_are_not_aliased():
    env = FamilyBEnv.__new__(FamilyBEnv)
    env.pad_centers = {"pad_u": np.array([-.2, .12, .829]),
                       "pad_v": np.array([.2, -.02, .829])}
    env.receiver_slots = {"obj_b": np.array([.145, .12, .829]),
                          "obj_c": np.array([.215, .12, .829])}
    u = env.resolve_skill_destination("PLACE_BUFFER", "carrier", "pad_u")
    v = env.resolve_skill_destination("PLACE_BUFFER", "carrier", "pad_v")
    assert not np.array_equal(u, v)
    assert not np.array_equal(
        env.resolve_skill_destination("PLACE", "obj_b", "receiver"),
        env.resolve_skill_destination("PLACE", "obj_c", "receiver"))
    with pytest.raises(ValueError, match="STOPPED_DESTINATION_BINDING"):
        env.resolve_skill_destination("PLACE_BUFFER", "obj_b", "pad_u")


def test_task_local_palette_covers_all_three_movable_objects():
    rgb = np.stack([PALETTE[x][:3] for x in ("carrier", "obj_b", "obj_c")])[None, :, :]
    for j, name in enumerate(("carrier", "obj_b", "obj_c")):
        mask = _palette_mask(rgb, name)
        assert mask.sum() == 1 and mask[0, j]


def _fake_env(b_inside=True, c_inside=False):
    center = np.array([.18, .12, .829])
    truth = {
        "receiver": center, "obj_b": center + np.array([-.035, 0, .02]),
        "obj_c": center + np.array([.035, 0, .02]) if c_inside
        else np.array([-.17, -.02, .85]),
        "table_top_z": .825, "gripper_qpos": np.array([.04, -.04]),
    }
    return SimpleNamespace(
        task_manifest=SimpleNamespace(task_id="T_P_FB"),
        task_contracts=(),
        public_layout=lambda: {
            "receiver": center, "pad_u": np.array([-.2, .12, .829]),
            "pad_v": np.array([.2, -.02, .829]), "table_top_z": .825,
        },
        hidden_truth=lambda: truth,
    )


def test_evaluator_requires_both_objects_and_empty_gripper():
    env = _fake_env()
    e = FamilyBEvaluator(env, 60, "T_P_FB")
    assert e.goal_true() is False
    env.hidden_truth()["obj_c"] = np.array([.215, .12, .85])
    assert e.goal_true() is True
    env.hidden_truth()["gripper_qpos"] = np.array([.005, -.005])
    assert e.goal_true() is False
    with pytest.raises(ValueError):
        FamilyBEvaluator(env, 60, "T_OTHER")


def test_verifier_never_fills_missing_dynamic_blob():
    env = _fake_env()
    measurement = SimpleNamespace(measurements={
        "blobs": {"carrier": None, "obj_b": None, "obj_c": None},
        "eef_pos": [.0, .0, 1.0], "gripper_qpos": [.04, -.04],
    })
    recs = {r.fact_id: r for r in FamilyBVerifier(env).verify(measurement)}
    assert recs["p:Open:receiver"].value is Truth.TRUE
    assert recs["p:Held:carrier"].value is Truth.UNKNOWN
    assert recs["p:Inside:obj_b:receiver"].value is Truth.UNKNOWN
    assert set(recs) == set(VERIFIER_FACT_IDS)


def test_reverse_mutation_aliases_two_pads_and_is_detected(monkeypatch):
    def wrong_resolver(self, skill, obj, destination):
        if skill == "PLACE_BUFFER":
            return self.pad_centers["pad_u"].copy()
        return self.receiver_slots[obj].copy()
    monkeypatch.setattr(FamilyBEnv, "resolve_skill_destination", wrong_resolver)
    with pytest.raises(AssertionError):
        test_two_destinations_and_receiver_slots_are_not_aliased()
