"""G01-G10: measured gripper geometry, sweep model and the bar-interferer design (saved wave D evidence only; no environment)."""
from __future__ import annotations

import copy
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m
from cp_disr.analysis import s4_tp_sr_pilot_v2_geom as G
from cp_disr.analysis import s4_tp_sr_pilot_v2_proto as P

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"
_hits = sorted(glob.glob(str(ROOT / "runs/final_master/S4/tp_soft_relocation_pilot_v2/*/captures/094286b08fcf5904")))
pytestmark = pytest.mark.skipif(not _hits, reason="wave D evidence not present")
OUT = Path(_hits[0]).parents[1] if _hits else None


@pytest.fixture(scope="module")
def geo():
    return G.measure_gripper(OUT)


@pytest.fixture(scope="module")
def layouts():
    return {r["case_id"]: r for r in json.loads((OUT / "spec/T_P_SR_V2_diag_split.json").read_text())["dev"]}


@pytest.fixture(scope="module")
def designs(geo, layouts):
    return {fam: G.design_scene(geo, layouts[case], P.SIDES[fam]) for fam, case in P.LAYOUT_CASE.items()}


def test_G01_opening_is_measured_from_saved_pad_geometry_and_limits_bar_width(geo):
    assert 0.070 < geo["usable_opening_open_m"] < 0.085
    inner = geo["usable_opening_open_m"]
    assert abs(inner - (geo["pad_centre_spacing_open_m"] - geo["pad_thickness_along_finger_axis_m"])) < 5e-4
    assert 2 * G.BAR["hy"] <= 0.8 * inner
    assert geo["source"]["open_state"]["sha256"] and geo["source"]["closed_state"]["sha256"]


def test_G02_bar_matches_cube_height_and_has_a_distinct_long_axis():
    assert G.BAR["hz"] == G.OBJECT_HALF and G.BAR["hx"] > G.BAR["hy"]


def test_G03_at_least_21_samples_per_linear_segment_over_all_five_phases(geo):
    pts = G.sweep_points(geo["eef_start"], (0.0, 0.0), 21)
    assert pts.shape[0] == 5 * 21
    for ph in range(5):
        assert int(np.sum(pts[:, 0] == ph)) >= 21
    assert list(G.PHASES) == ["approach", "descend", "press", "close", "lift_show"]


def test_G04_palm_stays_above_the_object_slab_at_every_skill_pose(geo):
    lowest_eef_z = G.GRASP_Z - G.PRESS_DZ
    assert lowest_eef_z + geo["palm_bottom_above_eef_m"] > G.OBJ_ZRANGE[1]


def test_G05_interference_configs_overlap_the_target_sweep_yet_keep_own_grasp_clear(designs):
    for fam, d in designs.items():
        assert d["status"] == "OK"
        i = d["interference"]
        assert i["target_sweep_clearance_m"] <= 0 and i["target_sweep_overlap_samples"] > 0
        assert i["initial_gap_to_target_m"] >= G.MIN_INITIAL_GAP - 1e-9
        assert i["own_grasp_sweep_clearance_m"] > 0 and i["target_sweep_clearance_after_relocation_m"] > 0


def test_G06_controls_only_shift_laterally_and_clear_the_sweep(designs):
    for d in designs.values():
        c, i = d["control"], d["interference"]
        assert c["target_sweep_clearance_m"] >= 0.020 and abs(c["bar_xy"][0] - i["bar_xy"][0]) < 1e-9
        assert c["lateral_shift_m"] > 0 and abs(abs(c["bar_xy"][1] - i["bar_xy"][1]) - c["lateral_shift_m"]) < 1e-9


def test_G07_two_scenes_are_spatially_distinct(layouts):
    a, b = (np.array(layouts[P.LAYOUT_CASE[k]]["target_xy"]) for k in ("A", "B"))
    assert np.linalg.norm(a - b) >= 0.10 and P.SIDES["A"] == -P.SIDES["B"]


def test_G08_unsatisfiable_geometry_stops_with_the_named_status(geo, layouts):
    wide = dict(G.BAR, hy=0.05)
    r = G.design_scene(geo, layouts["T_P_SR_pool_33"], 1, bar=wide)
    assert r["status"] == "STOPPED_ASYMMETRIC_INTERFERER_NOT_GRASPABLE"
    tiny_geo = copy.copy(geo)
    tiny_geo["usable_opening_open_m"] = 0.05
    assert G.design_scene(tiny_geo, layouts["T_P_SR_pool_33"], 1)["status"] == "STOPPED_ASYMMETRIC_INTERFERER_NOT_GRASPABLE"


def test_G09_contract_gate_both_first_actions_legal_both_routes_reachable_no_hard_blockage():
    r = P.contract_reachability(ROOT, OUT, m.load_config(CFG))
    assert r["both_first_actions_legal"] and r["both_routes_reachable"] and r["no_hard_blockage_fact"] and r["pass"]
    assert [s["action"] for s in r["routes"]["relocation"]["steps"]] == list(P.ROUTE_ACTIONS["relocation"]) and r["variants"]


def test_G10_freeze_registers_sixteen_paired_branches_and_freezes_read_only(tmp_path):
    out = tmp_path / "o"
    for sub in ("diagnostics", "spec", "decision"):
        (out / sub).mkdir(parents=True)
    (out / "captures").symlink_to(OUT / "captures")
    shutil.copy(OUT / "spec/T_P_SR_V2_diag_split.json", out / "spec/T_P_SR_V2_diag_split.json")
    (out / "diagnostics/wave_d_gate_amended.json").write_text(json.dumps({"wave_d_status": "PASS", "wave_p_released": True}))
    (out / "stage_manifest.json").write_text(json.dumps({"phases_done": []}))
    res = P.freeze_prototypes(ROOT, CFG, out)
    reg = json.loads((out / "prototype/prototype_branch_registration.json").read_text())["branches"]
    assert res["status"] == "FROZEN" and len(reg) == 16 and len({b["branch_id"] for b in reg}) == 16
    assert {b["case_id"] for b in reg} == set(P.CONFIG_IDS) and {b["route"] for b in reg} == {"direct", "relocation"}
    seeds = {}
    for b in reg:
        seeds.setdefault((b["case_id"], b["repeat"]), set()).add(b["restore_seed"])
    assert len(seeds) == 8 and all(len(v) == 1 for v in seeds.values())
    for f in ("prototype_configs.json", "prototype_geometry.csv", "prototype_contract_reachability.json", "prototype_branch_registration.json", "gripper_geometry.json",
              "gripper_sweep_model.json", "gripper_sweep_points.npy"):
        assert (out / "prototype" / f).is_file() and not ((out / "prototype" / f).stat().st_mode & 0o222)
    with pytest.raises(m.StopRun):
        P.freeze_prototypes(ROOT, CFG, out)                                       # one-shot
    assert np.load(out / "prototype/gripper_sweep_points.npy").shape[0] == 4 * 2 * 5 * 21
