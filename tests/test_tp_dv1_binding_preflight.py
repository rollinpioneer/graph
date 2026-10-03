import math
import struct
import subprocess
import sys

import numpy as np
import pytest

from cp_disr.analysis import tp_dv1_binding_preflight as m

GRIP = {"finger_inner_face_open": 0.0399, "finger_outer_face_open": 0.0664, "finger_half_width_orthogonal": 0.01, "finger_lowest_rel_site_z": -0.0085, "pad_lowest_rel_site_z": -0.0085, "pad_height_m": 0.02,
        "hand_lowest_rel_site_z": 0.031, "eef_site_hand_z": 0.1, "hand_mesh_bounds": {"min": [-0.1, -0.03, 0.0], "max": [0.1, 0.03, 0.07]}}


def box(c, half, R=None):
    return {"center": np.array(c, float), "R": np.eye(3) if R is None else R, "half": np.array(half, float)}


def obj(name, half, height=None, spawn="DETERMINED_BY_SOURCE"):
    b = box([0, 0, half[2]], half)
    return {"name": name, "boxes": [b], "height": 2 * half[2] if height is None else height, "com_inside_support": True, "spawn": {"status": spawn}, "extent_world_xyz": [2 * h for h in half]}


def test_obb_hit_identity_matches_axis_aligned_slab():
    o = np.array([0.0, 0.0, 1.0])
    d = np.array([[[0.0, 0.0, -1.0]], [[0.05, 0.0, -1.0]]])
    t = m.obb_hit(o, d, np.array([0.0, 0.0, 0.0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert abs(t[0, 0] - 0.9) < 1e-9 and abs(t[1, 0] - 0.9) < 1e-9
    miss = m.obb_hit(o, np.array([[[1.0, 0.0, -0.1]]]), np.array([0.0, 0.0, 0.0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert np.isinf(miss[0, 0])


def test_obb_hit_respects_rotation():
    th = math.radians(45)
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    o = np.array([0.0, 0.0, 1.0])
    d = np.array([[[0.0, 0.0, -1.0]]])
    # a thin long box rotated 45 deg still covers the origin and is hit from above at its top face z=0.1
    t = m.obb_hit(o, d, np.zeros(3), R, np.array([0.2, 0.01, 0.1]))
    assert abs(t[0, 0] - 0.9) < 1e-9


def test_pinch_width_follows_yaw():
    bx = [box([0, 0, 0.01], [0.04, 0.02, 0.01])]            # 0.08 along x, 0.04 along y
    assert abs(m.pinch_width(bx, 0.0) - 0.04) < 1e-9
    assert abs(m.pinch_width(bx, math.pi / 2) - 0.08) < 1e-9
    assert m.pinch_width(bx, math.pi / 4) > 0.04


def test_pinch_audit_flat_box_needs_no_yaw_and_big_bowl_is_eliminated():
    ok = m.pinch_audit(obj("butter", [0.038, 0.0198, 0.0087]), GRIP)
    assert ok["status"] == "GENERIC_PICK_YAW0" and ok["required_yaw_deg"] == 0 and ok["pinch_width_yaw0_m"] < ok["pinch_limit_m"]
    long_y = m.pinch_audit(obj("bbq", [0.0235, 0.0535, 0.0145]), GRIP)
    assert long_y["status"] == "GENERIC_PICK_YAW90" and long_y["needs_yaw_extension"] is True
    bowl = m.pinch_audit(obj("bowl", [0.053, 0.053, 0.025]), GRIP)
    assert bowl["status"] == "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED" and any("no yaw" in r for r in bowl["reasons"])


def test_existing_grasp_formula_would_drive_fingers_into_the_table_for_thin_objects():
    p = m.pinch_audit(obj("thin", [0.038, 0.0198, 0.0087]), GRIP)
    assert p["existing_formula_finger_low_at_press_m"] < 0 <= p["finger_low_abs_m"]


def test_unverified_spawn_is_not_usable():
    p = m.pinch_audit(obj("x", [0.038, 0.0198, 0.0087], spawn="UNVERIFIED"), GRIP)
    assert p["status"] == "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED" and any("UNVERIFIED" in r for r in p["reasons"])


def objects_dir(tmp_path_factory, text):
    root = tmp_path_factory.mktemp("liblike")
    d = root / "envs" / "objects"
    d.mkdir(parents=True)
    (d / "o.py").write_text(text)
    return root


CLASSES = '''
import numpy as np
class Base:
    def __init__(self):
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "x"
class Flat(Base):
    def __init__(self):
        super().__init__()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
class Upright(Base):
    pass
class Ranged(Base):
    def __init__(self):
        super().__init__()
        self.rotation = (0.0, 1.0)
class WithQuat(Base):
    def __init__(self):
        super().__init__()
        self.init_quat = [1, 0, 0, 0]
'''


def test_spawn_orientation_from_class_chain(tmp_path_factory):
    root = objects_dir(tmp_path_factory, CLASSES)
    flat = m.spawn_orientation(root, "Flat")
    assert flat["status"] == "DETERMINED_BY_SOURCE" and np.allclose(flat["quat_wxyz"], [1, 0, 0, 0]) and flat["rotation_source"]["defined_in"] == "Flat"
    up = m.spawn_orientation(root, "Upright")
    assert up["status"] == "DETERMINED_BY_SOURCE" and np.allclose(up["quat_wxyz"], [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]) and up["rotation_source"]["defined_in"] == "Base"
    assert m.spawn_orientation(root, "Ranged")["status"] == "UNVERIFIED"
    assert m.spawn_orientation(root, "WithQuat")["status"] == "UNVERIFIED"


def test_msh_volume_of_a_unit_cube(tmp_path_factory):
    v = np.array(list(__import__("itertools").product((0, 1), repeat=3)), dtype="<f4")
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype="<i4")
    p = tmp_path_factory.mktemp("mesh") / "cube.msh"
    p.write_bytes(struct.pack("<4i", 8, 0, 0, 12) + v.tobytes() + f.tobytes())
    vol, verts = m.mesh_volume(p, [0.1, 0.1, 0.1])
    assert abs(vol - 0.001) < 1e-9 and verts.shape == (8, 3)


def test_mass_and_inertia_of_one_box():
    parsed = {"boxes": [{"center": np.array([0.0, 0.0, 0.0]), "R": np.eye(3), "half": np.array([0.01, 0.02, 0.03]), "density": 100.0, "solref": [0.001, 1.0], "friction": [1, 1, 1]}], "visual_meshes": [], "sites": {}}
    mi = m.mass_inertia(parsed, ".")
    assert abs(mi["mass_boxes_only_kg"] - 100 * 8 * 0.01 * 0.02 * 0.03) < 1e-12 and mi["mass_ratio_high_over_low"] == 1.0
    I = np.array(mi["inertia_boxes_only_about_com_kgm2"])
    assert abs(I[0, 0] - mi["mass_boxes_only_kg"] / 12 * (0.04 ** 2 + 0.06 ** 2)) < 1e-12


def test_evaluator_generalisation_reproduces_d0_predicate():
    eq = m.evaluator_equivalence(n=20000)
    assert eq["mismatches"] == 0
    assert m.registered_inside(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, (0.06, 0.06), 0.14) is True
    assert m.registered_inside(np.array([0.07, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, (0.06, 0.06), 0.14) is False


def test_show_path_blocked_by_a_basket_on_the_route():
    o = obj("butter", [0.038, 0.0198, 0.0087])
    assert m.show_path_clear(o, GRIP) is True
    assert m.show_path_clear(o, GRIP, basket_rect=(-0.05, 0.1, -0.15, -0.05)) is False


def test_fixed_rule_names_are_frozen():
    assert len(m.RULES) == 10 and len(m.PAIRS) == 6 and m.OBJECTS == ("bbq_sauce", "butter", "chocolate_pudding", "cream_cheese")
    assert m.FAIL_ORDER.index("DV1_FAIL_VERIFIER_NOT_PUBLICLY_IMPLEMENTABLE") < m.FAIL_ORDER.index("DV1_FAIL_BASKET_GEOMETRY")


def test_protected_hash_excludes_own_output_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("proot")
    own = root / "runs/final_master/S4/tp_dv1_binding_preflight/run1"
    other = root / "runs/final_master/S4/tp_vis_preflight/run1"
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    (own / "a.json").write_text("{}")
    (other / "b.json").write_text("{}")
    assert list(m.protected_hashes(root)) == ["runs/final_master/S4/tp_vis_preflight/run1/b.json"]


def test_module_import_pulls_no_simulator_stack():
    code = "import sys; import cp_disr.analysis.tp_dv1_binding_preflight as x; print([k for k in ('robosuite','mujoco','libero','torch','gym') if k in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
