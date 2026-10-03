"""Offline tests for the R2 restore card: metrics, thresholds, static compatibility gate. No simulator."""
from pathlib import Path

import numpy as np

from cp_disr.analysis import tp_ef_post_open_restore_r2 as r2

ROOT = Path(__file__).resolve().parents[1]


def test_rgb_identical_passes_and_reports_zero():
    a = np.random.RandomState(0).randint(0, 255, (16, 16, 3)).astype(np.uint8)
    m = r2.rgb_metrics(a, a.copy())
    assert m["pass_"] and m["changed_pixel_count"] == 0 and m["max_abs"] == 0 and m["bounding_boxes"]["overall"] is None


def test_rgb_within_three_passes_but_four_fails():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = a.copy()
    b[2:4, 3:6, 0] = 3
    m = r2.rgb_metrics(a, b)
    assert m["pass_"] and m["max_abs"] == 3 and m["changed_pixel_count"] == 6 and m["changed_channel_count"] == 6
    assert m["bounding_boxes"]["overall"] == {"row_min": 2, "row_max": 3, "col_min": 3, "col_max": 5}
    assert m["per_channel_changed"] == [6, 0, 0] and m["value_histogram"]["3"] == 6
    b[0, 0, 1] = 4
    m2 = r2.rgb_metrics(a, b)
    assert not m2["pass_"] and m2["max_abs"] == 4


def test_rgb_dtype_or_shape_mismatch_fails():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert not r2.rgb_metrics(a, a.astype(np.int16))["pass_"]
    assert not r2.rgb_metrics(a, np.zeros((5, 4, 3), dtype=np.uint8))["pass_"]


def test_quantiles_and_components_reported():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[1, 1] = 2
    b[7:9, 7:9] = 1
    m = r2.rgb_metrics(a, b)
    assert m["changed_pixel_count"] == 5 and set(m["quantiles_changed_elements"]) == {"50", "90", "99", "100"}
    assert m["bounding_boxes"].get("component_count") in (2, None)


def test_depth_and_proprio_thresholds():
    d = np.ones((4, 4, 1), dtype=np.float32)
    assert r2.depth_metrics(d, d.copy(), "x")["pass_"]
    e = d.copy()
    e[0, 0, 0] += 1e-5
    assert not r2.depth_metrics(d, e, "x")["pass_"]
    assert r2.vector_metrics(np.zeros(3), np.full(3, 5e-10), 1e-9)["pass_"]
    assert not r2.vector_metrics(np.zeros(3), np.full(3, 5e-9), 1e-9)["pass_"]


def _summary(xyz=(0.1, 0.2, 0.3), present=True, reasons=None):
    return {"blobs": {"target": {"xyz": list(xyz), "pixels": 40, "std": [0, 0, 0], "source": None} if present else None}, "unknown_reasons": reasons or {}, "depth_stats": {"min": 1.0, "max": 2.0, "mean": 1.5}}


def test_perception_comparison_is_exact_on_decisions_and_tolerant_on_xyz():
    assert r2.compare_perception(_summary(), _summary())["pass_"]
    assert r2.compare_perception(_summary(), _summary(xyz=(0.1 + 5e-5, 0.2, 0.3)))["pass_"]
    assert not r2.compare_perception(_summary(), _summary(xyz=(0.1 + 5e-4, 0.2, 0.3)))["pass_"]
    assert not r2.compare_perception(_summary(), _summary(present=False))["pass_"]
    assert not r2.compare_perception(_summary(), _summary(reasons={"target": "x"}))["pass_"]


def test_frozen_thresholds_not_looser_than_family_b_contract():
    gate = r2.compat_gate(ROOT)
    assert gate["checks"]["contract_clause"]["frozen_here_is_stricter_or_equal"]


def test_static_compat_gate_binds_t_b_render_path():
    gate = r2.compat_gate(ROOT)
    ck = gate["checks"]
    assert ck["constructor_render_args"]["identical_except_camera_set"], ck["constructor_render_args"]
    assert ck["constructor_render_args"]["camera_static_ok"]
    assert all(ck["observation_source_identity"]["shared_modules_unchanged_since_family_b_commits"].values())
    assert ck["observation_source_identity"]["equals_family_b_recorded_perception"]
    assert ck["renderer_stack"]["matches_manifest"]
    assert gate["applicable"], gate["problems"]


def test_obs_dict_from_arrays_layout():
    p = np.arange(23, dtype=float)
    d = r2.obs_dict_from_arrays({"rgb": np.zeros((2, 2, 3)), "depth": np.zeros((2, 2, 1)), "proprio": p})
    assert list(d["eef_pos"]) == [0, 1, 2] and list(d["gripper_qpos"]) == [7, 8]


def test_budget_design():
    assert r2.CAPS["environment_constructions"] == 3 and r2.CAPS["start_case_calls"] == 3 and r2.CAPS["dev05_open_skill_calls"] == 1
    assert r2.CAPS["skill_retries"] == 0 and r2.CAPS["provider_requests"] == 0
