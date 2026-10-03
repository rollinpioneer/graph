"""Offline tests for the BSI preflight card (static analysis only; no environment is ever constructed)."""
from pathlib import Path

import pytest

from cp_disr.analysis import tp_bsi_preflight as b

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def geom():
    return b.Geometry(ROOT, b.gripper_geometry())


def test_stats_and_discrete_bound():
    s = b.stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["median"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0
    summ = {k: b.stats(v) for k, v in {"PICK_target": [5.0, 5.0], "PLACE_target": [3.0, 3.0], "PICK_second": [4.0, 4.0], "PLACE_BUFFER_second": [3.5, 3.5]}.items()}
    out = b.discrete_bound(summ)
    bc = out["modes"]["BLOCK_CONTAINER"]
    assert bc["delta_T_if_recovery_mean"] == 8.0 and abs(bc["p_min_for_2p1s_mean"] - 2.1 / 8.0) < 1e-12 and out["status"] == "PASS"
    weak = {k: b.stats([0.5, 0.5]) for k in summ}
    assert b.discrete_bound(weak)["status"] == "FAIL" and b.discrete_bound(weak)["category"] == "PREFLIGHT_FAIL_INSUFFICIENT_DISCRETE_EFFECT"
    assert out["median_gate_note"]["P_median_pass_n5_by_failure_probability"]["0.5"] == pytest.approx(0.5)


def test_guards():
    g = b.Guard()
    with pytest.raises(PermissionError):
        g.read("runs/x/tb_indep_holdout_1/f.json")
    assert g.opened == 0 and len(g.refused) == 1


def test_geometry_constants_come_from_sources(geom):
    assert geom.cube == 0.02 and geom.inner == 0.03 and geom.cont_h == 0.048 and geom.wall == 0.012 and geom.buffer_half == 0.08
    assert abs(geom.drop_container - 0.078) < 1e-9 and abs(geom.drop_buffer - 0.07) < 1e-9 and abs(geom.grasp_z - 0.022) < 1e-9


def test_gripper_open_fingers_span_the_container_wall(geom):
    g = geom.grip
    assert g["finger_inner_face_open"] < geom.inner + 2 * geom.wall and g["finger_outer_face_open"] > geom.inner
    assert g["finger_lowest_point_rel_site_z"] < 0


def test_blocker_inside_container_is_a_certain_pick_collision(geom):
    pose = geom.blocker_pose("BLOCK_CONTAINER", (0.0, 0.0))
    assert pose["rest"] == "ON_CONTAINER_FLOOR"
    ps = geom.pick_sweep("BLOCK_CONTAINER", pose)
    assert {c["axis"] for c in ps["descent_collisions"]} == {0, 1} and all(c["solid"] == "container_wall" for c in ps["descent_collisions"])


def test_blocker_overlapping_the_wall_is_invalid_and_far_blocker_is_clear(geom):
    assert geom.blocker_pose("BLOCK_CONTAINER", (0.03, 0.0))["rest"] == "INVALID_OVERLAPS_WALL"
    far = geom.blocker_pose("BLOCK_CONTAINER", (0.10, 0.0))
    assert far["rest"] == "ON_TABLE_OUTSIDE" and geom.score("BLOCK_CONTAINER", (0.10, 0.0)) == 0.0


def test_buffer_blocker_on_pad_is_below_the_proven_clearance(geom):
    pose = geom.blocker_pose("BLOCK_BUFFER", (0.02, 0.0))
    assert pose["rest"] == "ON_BUFFER_PAD"
    ps = geom.pick_sweep("BLOCK_BUFFER", pose)
    assert not ps["descent_collisions"] and 0 < ps["min_vertical_clearance"] < b.PROVEN_CLEARANCE(geom)


def test_score_bins_and_selection_is_deterministic():
    assert [b.Geometry.bin_of(s) for s in (0.0, 0.1, 0.5, 0.9)] == ["CLEAR", "WEAK", "MODERATE", "HARD_BLOCKING_RISK"]
    cands = []
    for mode in b.PICK_MODES:
        for bin_name in ("WEAK", "MODERATE"):
            for k in range(5):
                cands.append({"mode": mode, "offset": [0.01 * k, 0.0], "score_bin": bin_name, "safety": "SAFE" if k % 2 == 0 else "UNCERTIFIED"})
    a, c = b.select_pilot_scenes(cands), b.select_pilot_scenes(list(reversed(cands)))
    assert set(a) == {"BC-WEAK", "BC-MODERATE", "BB-WEAK", "BB-MODERATE"} and a == c and all(v["safety"] == "SAFE" for v in a.values())


def test_real_grid_has_no_safe_weak_or_moderate_band(geom):
    cam = b.json.loads((ROOT / "configs/final_master/family_b_obs_v2/observation_profile_v2.json").read_text())["camera_manifest"]["agentview"]
    cands = b.candidate_grid(geom, cam, 0.825)
    cat, per = b.category_from_candidates(cands)
    assert cat == "PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND"
    assert per["BLOCK_CONTAINER"]["all_failures_are_collisions"] and per["BLOCK_CONTAINER"]["safe_weak_or_moderate"] == 0 and per["BLOCK_BUFFER"]["safe_weak_or_moderate"] == 0
    assert per["BLOCK_BUFFER"]["weak_or_moderate_valid"] > 0 and not per["BLOCK_BUFFER"]["all_failures_are_collisions"]


def test_canary_documents_on_synthetic_scenes():
    sc = {n: {"mode": "BLOCK_CONTAINER" if n.startswith("BC") else "BLOCK_BUFFER", "offset": [0.0, 0.0], "blocker_center": [0.0, 0.0], "rest": "X", "score": 0.5, "score_bin": "MODERATE", "safety": "SAFE", "camera": {"pass": True}}
          for n in ("BC-WEAK", "BC-MODERATE", "BB-WEAK", "BB-MODERATE")}
    reg, spec = b.canary_documents(sc)
    assert [e["id"] for e in reg["episodes"]] == ["C1", "C2", "C3", "C4"] and [e["first_route"] for e in reg["episodes"]] == ["S", "T", "T", "S"]
    assert "never retry" in spec["recovery_policy"]["BLOCK_CONTAINER_bad"]


def test_zero_env_guard_raises():
    zg = b.ZeroEnvGuard().install()
    from cp_disr.platforms.libero import runtime_factory as rf
    with pytest.raises(RuntimeError, match="ZERO_ENV_GUARD"):
        rf.make_env(None)
    assert zg.calls == 1
