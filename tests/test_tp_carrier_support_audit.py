import math
import subprocess
import sys

import numpy as np
import pytest

from cp_disr.analysis import tp_carrier_support_audit as m
from cp_disr.analysis import tp_libero_audit as la

GRIP = {"finger_inner_face_open": 0.0399, "finger_outer_face_open": 0.0664, "finger_half_width_orthogonal": 0.0105, "finger_lowest_rel_site_z": -0.0092, "finger_highest_rel_site_z": 0.04, "pad_lowest_rel_site_z": -0.0044,
        "pad_height_m": 0.016, "hand_lowest_rel_site_z": 0.031, "eef_site_hand_z": 0.097, "hand_mesh_bounds": {"min": [-0.0316, -0.104, -0.026], "max": [0.0316, 0.1, 0.066]}}


def box(c, half, R=None):
    return {"center": np.array(c, float), "R": np.eye(3) if R is None else R, "half": np.array(half, float)}


def rec_from_boxes(name, boxes, mass=0.1, mu=1.0, com=None):
    lo, hi = np.min([b["center"] - b["half"] for b in boxes], axis=0), np.max([b["center"] + b["half"] for b in boxes], axis=0)
    return {"name": name, "ok": True, "boxes": boxes, "lo": lo.tolist(), "hi": hi.tolist(), "height": float(hi[2]), "mass_hi": mass, "mass_lo": mass / 2, "mass_mid": 0.75 * mass, "mu": mu,
            "com": com or [0.0, 0.0, float(hi[2] / 2)], "extent": (hi - lo).tolist(), "footprint_rect": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]}


def tray():
    bs = [box([0, 0, 0.005], [0.06, 0.06, 0.005]), box([0.055, 0, 0.02], [0.005, 0.06, 0.015]), box([-0.055, 0, 0.02], [0.005, 0.06, 0.015]), box([0, 0.055, 0.02], [0.06, 0.005, 0.015]), box([0, -0.055, 0.02], [0.06, 0.005, 0.015])]
    return rec_from_boxes("tray", bs)


def block(name, hx, hy, hz, mass=0.02):
    return rec_from_boxes(name, [box([0, 0, hz], [hx, hy, hz])], mass=mass)


def test_obb_overlap_basic_and_rotated():
    a, b = box([0, 0, 0], [0.1, 0.1, 0.1]), box([0.15, 0, 0], [0.1, 0.1, 0.1])
    assert m.obb_overlap(a, b) is True
    assert m.obb_overlap(a, box([0.25, 0, 0], [0.1, 0.1, 0.1])) is False
    th = math.radians(45)
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    thin = box([0.12, 0.12, 0], [0.2, 0.005, 0.1], R)
    assert m.obb_overlap(box([-0.1, -0.1, 0], [0.02, 0.02, 0.1]), thin) is False
    assert m.obb_overlap(box([0.12, 0.12, 0], [0.02, 0.02, 0.1]), thin) is True


def test_vertical_height_of_a_box_union():
    X, Y = np.meshgrid(np.array([0.0, 0.2]), np.array([0.0]), indexing="ij")
    z = m.vertical_height([box([0, 0, 0.01], [0.05, 0.05, 0.01]), box([0, 0, 0.03], [0.02, 0.02, 0.01])], X, Y)
    assert abs(z[0, 0] - 0.04) < 1e-9 and np.isnan(z[1, 0])


def test_support_field_of_a_tray_centre_margin_and_depth():
    f = m.support_field(tray())
    assert f["ok"] and abs(f["z_support"] - 0.01) < 1e-9 and abs(f["rim_z"] - 0.035) < 1e-9 and abs(f["cavity_depth_m"] - 0.025) < 1e-9
    assert 0.045 <= m.sample(f, 0.0, 0.0) <= 0.055
    assert m.sample(f, 0.0, 0.2) == 0.0


def test_carrier_grasp_finds_a_rim_pinch_on_a_tray_and_classifies_it():
    t = tray()
    g = m.carrier_grasp(t, GRIP)
    assert g["feasible"] > 0 and g["best"]["kind"] in ("RIM_WALL_PINCH", "BODY_PINCH") and g["best"]["torque_ratio"] < 0.5
    mob = {"movable": True, "reasons": []}
    assert m.classify_carrier(mob, g) == "GENERIC_MOVE_CARRIER_COMPATIBLE"
    assert m.classify_carrier({"movable": False, "reasons": ["x"]}, g) == "NOT_MOVABLE"
    assert m.classify_carrier(mob, {"best": None}) == "ASSET_SPECIFIC_CONTROLLER_REQUIRED"
    assert m.classify_carrier(mob, {"best": {"torque_ratio": 0.8}}) == "GENERIC_BINDING_REQUIRED"
    assert m.classify_carrier(mob, {"best": {"torque_ratio": 1.4}}) == "ASSET_SPECIFIC_CONTROLLER_REQUIRED"


def test_a_flat_slab_without_a_rim_or_pinchable_width_is_not_graspable():
    slab = rec_from_boxes("slab", [box([0, 0, 0.01], [0.09, 0.09, 0.01])])
    g = m.carrier_grasp(slab, GRIP)
    assert g["feasible"] == 0 and g["best"] is None


def test_carried_object_in_the_way_of_the_fingers_blocks_some_sites_but_not_all():
    t = tray()
    A = block("A", 0.035, 0.02, 0.01)
    f = m.support_field(t)
    g0 = m.carrier_grasp(t, GRIP)
    g1 = m.carrier_grasp(t, GRIP, A, (0.0, 0.0), f["z_support"])
    assert g1["feasible"] <= g0["feasible"] and g1["candidates"] == g0["candidates"]


def test_pair_classification_centre_vs_edge_on_a_tray():
    t, A = tray(), block("A", 0.02, 0.02, 0.01, mass=0.02)
    f = m.support_field(t)
    pe = m.pair_eval(t, A, f, GRIP)
    assert pe["centre"]["com_projection_margin_m"] > 0.04 and pe["edge"]["chosen"] is not None
    e = pe["edge"]["directions"][pe["edge"]["chosen"]]
    assert abs(e["com_margin_m"] - m.EDGE_MARGIN) < 0.006
    assert pe["acceleration_ratio"] == pytest.approx(m.SPEED_CAP / m.CONTROL_DT / (1.0 * m.G))


def test_route_model_carrier_first_is_never_strictly_worse_and_s2_is_a_tie():
    dur = {"PICK_s": 3.0, "PLACE_S": 3.0, "PLACE_s": 3.0, "source": "test"}
    dom = m.dominance(m.route_costs(dur))
    for s in ("S1_CO_TRANSPORT_STABLE", "S2_SEPARATE_DESTINATIONS", "S3_CO_TRANSPORT_UNSTABLE", "S4_NO_SUPPORT_NEUTRAL"):
        assert dom[s]["carrier_first_ever_strictly_more_skills"] is False and dom[s]["carrier_first_ever_strictly_more_skills_vs_card_route"] is False
    assert dom["S2_SEPARATE_DESTINATIONS"]["orders_differ_by_skill_count"] is False
    assert dom["S1_CO_TRANSPORT_STABLE"]["object_first_ever_strictly_more_skills"] is True


def test_rule_attack_refutes_only_rules_that_prescribe_object_first():
    dur = {"PICK_s": 3.0, "PLACE_s": 3.0}
    att = m.rule_attack(m.dominance(m.route_costs(dur)))
    assert att["R4"]["refuted"] and att["R5"]["refuted"]
    assert not att["R1"]["refuted"] and not att["R2"]["refuted"] and not att["R3"]["refuted"] and not att["R6"]["refuted"]


def test_contract_only_plan_does_not_credit_co_transport():
    assert m.mini_bplan("S1_CO_TRANSPORT_STABLE", {"PICK_s": 3, "PLACE_s": 3})["skills"] == 5
    assert m.mini_bplan("S2_SEPARATE_DESTINATIONS", {"PICK_s": 3, "PLACE_s": 3})["skills"] == 3


def test_relation_check_accepts_real_ids_and_flags_redundant_or_invalid():
    res = m.relation_check()
    assert all(r["expressible"] for r in res) and not any(r["contract_redundant"] for r in res)
    bad = dict(m.RELATIONS[0], type="HARD_PRE")
    saved = list(m.RELATIONS)
    try:
        m.RELATIONS[:] = [bad, dict(m.RELATIONS[0], target_ref="p:AtRegion:B:D_B")]
        out = m.relation_check()
        assert "type outside the schema enum" in out[0]["issues"] and out[1]["contract_redundant"] is True
    finally:
        m.RELATIONS[:] = saved


BDDL = """
(define (problem T) (:domain robosuite) (:language put the cup on the tray)
  (:regions (cup_region (:target main_table) (:ranges ((-0.1 -0.1 -0.05 -0.05)))) (tray_region (:target main_table) (:ranges ((0.0 -0.1 0.05 -0.05))))
            (far_region (:target main_table) (:ranges ((0.5 0.5 0.55 0.55)))))
  (:fixtures main_table - table)
  (:objects cup_1 - cup tray_1 - wooden_tray far_1 - cube)
  (:obj_of_interest cup_1)
  (:init (On cup_1 main_table_cup_region) (On tray_1 main_table_tray_region) (On far_1 main_table_far_region))
  (:goal (And (On cup_1 tray_1)))
)
"""


def test_universe_rules_on_a_synthetic_task():
    p = la.parse_bddl(BDDL)
    inst, sect = la.inst_maps(p)
    task = {"suite": "s", "index": 0, "name": "t", "parsed": p, "inst": inst, "sect": sect}
    reg = {"cup": {"class": "Cup", "articulated": False}, "wooden_tray": {"class": "WoodenTray", "articulated": False}, "cube": {"class": "Cube", "articulated": False}}
    carriers, carried = m.build_universes([task], reg)
    assert "wooden_tray" in carriers and carriers["wooden_tray"]["classification"] == "NATIVE_MOVABLE_OBJECT" and carriers["wooden_tray"]["name_matches_rule"]
    assert "cup" in carried and "PLACED_ON_OR_IN_CARRIER" in carried["cup"]["reasons"]
    assert "cube" not in carried                        # far away: not placed on, not near
    assert "main_table" not in carriers and "table" not in carriers


def test_universe_marks_fixture_only_types_as_official_fixtures():
    text = BDDL.replace("tray_1 - wooden_tray", "tray_1 - wooden_tray").replace(":fixtures main_table - table", ":fixtures main_table - table shelf_1 - shelf").replace("(On cup_1 tray_1)", "(On cup_1 shelf_1_top_region)")
    text = text.replace("(:objects cup_1 - cup", "(:regions2) (:objects cup_1 - cup")
    p = la.parse_bddl(text.replace("(:regions2) ", "").replace("(far_region", "(top_region (:target shelf_1)) (far_region"))
    inst, sect = la.inst_maps(p)
    task = {"suite": "s", "index": 0, "name": "t", "parsed": p, "inst": inst, "sect": sect}
    reg = {"cup": {"class": "Cup", "articulated": False}, "wooden_tray": {"class": "WoodenTray", "articulated": False}, "cube": {"class": "Cube", "articulated": False}, "shelf": {"class": "Shelf", "articulated": False}}
    carriers, _ = m.build_universes([task], reg)
    assert carriers["shelf"]["classification"] == "OFFICIAL_FIXTURE"


def test_module_import_pulls_no_simulator_stack():
    code = "import sys; import cp_disr.analysis.tp_carrier_support_audit as x; print([k for k in ('robosuite','mujoco','libero','torch','gym') if k in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_protected_hash_excludes_own_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("proot")
    own = root / "runs/final_master/S4/tp_carrier_support_design_audit/run1"
    other = root / "runs/final_master/S4/tp_vis_preflight/run1"
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    (own / "a.json").write_text("{}")
    (other / "b.json").write_text("{}")
    assert list(m.la_protected(root)) == ["runs/final_master/S4/tp_vis_preflight/run1/b.json"]
