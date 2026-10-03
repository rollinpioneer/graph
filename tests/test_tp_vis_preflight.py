"""Offline tests for the different-height visual preflight (pure numpy / source analysis; no environment is constructed)."""
import os
from pathlib import Path

import numpy as np
import pytest

from cp_disr.analysis import tp_occ_preflight as o
from cp_disr.analysis import tp_vis_preflight as v

ROOT = Path(__file__).resolve().parents[1]
HAVE_DEV = os.path.isdir(o.SNAP_DEV03) and os.path.isdir(o.SNAP_DEV05)


@pytest.fixture(scope="module")
def cam():
    return o.camera(ROOT)


@pytest.fixture(scope="module")
def grip():
    return v.bsi.gripper_geometry()


def test_frozen_grids_are_the_declared_ones():
    assert v.TALL_TOTAL_HEIGHTS == (0.06, 0.08, 0.10, 0.12) and v.CUBE_H == 0.04
    assert list(v.COLOR_CANDIDATES) == sorted(v.COLOR_CANDIDATES) and len(v.COLOR_CANDIDATES) == 7


def test_only_the_0p06_height_survives_the_unmodified_verifier(grip):
    geom = v.bsi.Geometry(ROOT, grip)
    rows = v.height_audit(geom, grip)
    assert [r["total_height"] for r in rows if r["survives_unmodified_production"]] == [0.06]
    assert all("REQUIRES_VERIFIER_CHANGE_ONTABLE" in r["eliminated_by"] for r in rows if r["total_height"] >= 0.08)


def test_production_surface_has_no_unknown_and_flags_the_verifier():
    s = v.production_surface(ROOT)
    assert not s["unknown_rows"]
    assert any(r["file"].endswith("verifier.py") and r["class"] == "SEMANTIC_CHANGE_REQUIRED" for r in s["rows"])
    assert any(r["class"] == "PER_OBJECT_GEOMETRY_LOOKUP_REQUIRED" for r in s["rows"]) and any(r["class"] == "NEW_COLOR_BINDING_REQUIRED" for r in s["rows"])


def test_equal_height_regression_and_machinery(cam):
    r = v.regression(cam)
    assert r["reproduces_occ_zero"] and r["machinery_can_occlude"] and r["equal_height_layouts_checked"] > 50


def test_taller_occluder_shifts_the_cube_centroid(cam):
    d = o.pixel_rays(cam)
    m = v.metrics_vis(cam, d, (-0.2, -0.10), (-0.2 + 0.065, -0.10), 0.12)
    assert m["visible_fraction_top"] < 1.0 and m["centroid_shift_world_m"] > 0


def test_classification_and_bins():
    base = {"visible_top_px": 40, "centroid_shift_world_m": 0.0}
    assert v.classify_vis(base, 0.82, 0.0119, 0.028) == "CLEAR"
    assert v.classify_vis(dict(base, centroid_shift_world_m=0.013), 0.82, 0.0119, 0.028) == "SOFT_OCCLUSION_CANDIDATE"
    assert v.classify_vis(dict(base, centroid_shift_world_m=0.03), 0.82, 0.0119, 0.028) == "HARD_GRASP_MISS"
    assert v.classify_vis(dict(base, visible_top_px=4), 0.82, 0.0119, 0.028) == "HARD_DETECTION_LOSS"
    assert v.severity_bin(0.013, 0.0119, 0.0199) == "WEAK" and v.severity_bin(0.022, 0.0119, 0.0199) == "MODERATE"


@pytest.mark.skipif(not HAVE_DEV, reason="saved development snapshots are not on this machine")
def test_hand_body_forbids_a_tall_object_right_in_front(cam, grip):
    w = v.WorldH(ROOT, grip, 1)
    near = w.safe_h((-0.10, -0.10), (-0.10 + 0.05, -0.10), 0.06)
    assert not near["safe"] and any("HAND_BODY" in r for r in near["reasons"])
    cube_like = w.safe_h((-0.10, -0.10), (-0.10 + 0.05, -0.10), 0.04)
    assert not any("HAND_BODY" in r for r in cube_like["reasons"])
    far = w.safe_h((-0.15, -0.10), (0.05, -0.10), 0.06)
    assert far["safe"], far["reasons"]


@pytest.mark.skipif(not HAVE_DEV, reason="saved development snapshots are not on this machine")
def test_evidence_binding_reproduces_occ():
    b, _ = v.evidence_binding(ROOT)
    assert b["status"] == "PASS", b["problems"]


@pytest.mark.skipif(not HAVE_DEV, reason="saved development snapshots are not on this machine")
def test_colour_audit_is_deterministic_and_finite():
    srcs, rows = o.calibration_sources(ROOT, v.bsi.Guard())
    a, b = v.color_audit(srcs, rows), v.color_audit(srcs, rows)
    assert a["selected"] == b["selected"] and set(a["candidates"]) == set(v.COLOR_CANDIDATES)
    if a["selected"]:
        assert a["selected"] == sorted(a["passing"])[0]
