"""R-01..: identity, reach-safe geometry, role-based camera gate, staged dispatch, wave gates and final gate for V3-R1.
CPU only: no MuJoCo environment, no EGL context; every ledger lives in pytest tmp dirs."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2
from cp_disr.analysis import family_b_staging_v2 as s
from cp_disr.analysis import family_b_v3r1_reach_safe_mirror as v
from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / v.CFG).read_text())
SRC = ROOT / "src/cp_disr/analysis/family_b_v3r1_reach_safe_mirror.py"
_MUT = {}
CELLS = tuple((l, c) for l in ("layout_0", "layout_1") for c in v.CONTEXTS)


@pytest.fixture(autouse=True)
def _no_real_simulator(monkeypatch):
    from robosuite.utils import binding_utils

    def boom(*a, **k):
        raise AssertionError("V3-R1 tests must not construct a MuJoCo sim / EGL render context")
    for attr in ("MjSim", "MjRenderContext"):
        if hasattr(binding_utils, attr):
            monkeypatch.setattr(getattr(binding_utils, attr), "__init__", boom)


# ------------------------------------------------------------------------------------ fixtures
def make_out(tmp_path, mod=v):
    out = tmp_path / "out"
    manifest = out / "spec/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("runtime: {}\n")
    prefixes = {k: x["prefix"] for k, x in CFG["contexts"].items()}
    branches = mod.build_branches(v.FROZEN_SHA, "c" * 40, "s" * 64, prefixes, set(), manifest, "f" * 16)
    reg = {"card_id": mod.CARD, "branches": branches, "manifest_sha256": m.sha(manifest)}
    phys = out / "physical"
    m.write(phys / "witnesses/e4_branch_registration.json", reg)
    m.write(phys / "registration.json", reg)
    mod.init_state(out, branches)
    return out, branches


class FakeProc:
    def __init__(self, polls=0, log=None, bid=None):
        self.pid = 0
        self.polls, self.log, self.bid = polls, log, bid

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        if self.log is not None:
            self.log.finish(self.bid)
        return 0


class FakeLauncher:
    """Stands in for the worker subprocess: claims, writes a terminal result, finishes the attempt."""

    def __init__(self, polls=0):
        self.calls, self.active, self.max_active, self.finished, self.polls = [], set(), 0, [], polls

    def finish(self, bid):
        if bid in self.active:
            self.active.discard(bid)
            self.finished.append(bid)

    def __call__(self, root, out, b, gpu):
        self.calls.append((b["branch_id"], gpu, b["step_name"], b["restore_seed"]))
        self.active.add(b["branch_id"])
        self.max_active = max(self.max_active, len(self.active))
        phys = Path(out) / "physical"
        claim_branch_attempt(phys, b["branch_id"])
        acts = [{"action_id": f"a{i}", "controller_exit": "NORMAL_TERMINATION"} for i in range(6)]
        m.write(phys / "branch_results" / f"{b['branch_id']}.json", {
            "branch_id": b["branch_id"], "status": "TASK_SUCCESS", "task_success": True, "actions": acts,
            "env_counts": {"bootstrap_constructions": 0, "constructions": 1, "internal_resets": 2, "reset_calls": 1}})
        finish_branch_attempt(phys, b["branch_id"], "COMPLETED", execution_status="TASK_SUCCESS")
        return FakeProc(self.polls, self, b["branch_id"])


def fake_evaluate(fail_keys=(), support=20):
    def evaluate(out, b, attempts=None):
        ok = v.branch_key(b) not in fail_keys
        per = {v.REMAINING[b["context"]]: {"agentview": {"mask_pixels": 30, "depth_support": support, "blob_present": True}}}
        return {"branch_id": b["branch_id"], "complete": ok, "task_success": ok, "checks": {}, "layout": b["layout"],
                "context": b["context"], "candidate": b["candidate"], "per_view": per, "status": "TASK_SUCCESS",
                "problems": [] if ok else [f"{b['branch_id']}:forced"]}
    return evaluate


def fake_initial(ok=True, carrier=49, setup=120):
    def initial(out, b):
        return {"present": True, "carrier": carrier, "setup_target": setup, "setup_target_object": mod_setup(b),
                "per_object_max": {}, "min_required": 12, "ok": ok and carrier >= 12 and setup >= 12}
    return initial


def mod_setup(b):
    return v.SETUP_TARGET[b["context"]]


def patch_waves(mod, monkeypatch, fail_keys=(), wave_status=None, support=20, initial_ok=True):
    wave_status = wave_status or {}
    monkeypatch.setattr(mod, "initial_support", fake_initial(initial_ok))
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(fail_keys, support))
    monkeypatch.setattr(mod, "wave_gate", lambda out, wave: {"wave": wave, "status": wave_status.get(wave, "PASS"),
                                                             "checks": {"forced": wave_status.get(wave, "PASS") == "PASS"}})


def no_identity(root, out):
    return None


def ledger(out):
    return json.loads((Path(out) / "budget_ledger.json").read_text())


def write_branch(out, b, cost, dur, steps, tail=3):
    """Terminal result + controller trace of the candidate action (the evidence the fairness gate reads)."""
    n = len(b["prefix"])
    acts = [{"action_id": a, "duration_sim": 1.0, "controller_exit": "NORMAL_TERMINATION", "raw_skill": {"sim_duration": 1.0}}
            for a in b["prefix"]]
    acts.append({"action_id": b["candidate_id"], "duration_sim": dur, "controller_exit": "NORMAL_TERMINATION",
                 "raw_skill": {"sim_duration": dur}})
    rest = (cost - dur) / tail
    acts += [{"action_id": f"t{i}", "duration_sim": rest, "controller_exit": "NORMAL_TERMINATION",
              "raw_skill": {"sim_duration": rest}} for i in range(tail)]
    phys = Path(out) / "physical"
    m.write(phys / "branch_results" / f"{b['branch_id']}.json", {
        "branch_id": b["branch_id"], "status": "TASK_SUCCESS", "task_success": True, "time_to_task_success": cost,
        "actions": acts, "env_counts": {"constructions": 1, "reset_calls": 1, "internal_resets": 2}})
    d = Path(out) / "captures" / b["branch_id"] / f"action_{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"event": "skill_begin", "eef_pos": [0.02, -0.08, 1.0]}, {"kind": "step", "qpos_sha256": "q", "qvel_sha256": "w"},
             {"event": "skill_end", "steps": steps, "eef_pos": [0.1, 0.1, 1.0]}]
    (d / "controller_trace.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def table(out, branches, costs, durs=None, steps=None):
    """costs/durs/steps: {(layout, context): (u, v)}."""
    durs = durs or {k: (6.0, 6.0) for k in CELLS}
    steps = steps or {k: (300, 300) for k in CELLS}
    by = {(b["layout"], b["context"], b["candidate"]): b for b in branches}
    for k in CELLS:
        for i, pad in enumerate(v.PADS):
            write_branch(out, by[(*k, pad)], costs[k][i], durs[k][i], steps[k][i])


PASS_COSTS = {("layout_0", "B_PENDING"): (10.0, 11.0), ("layout_0", "C_PENDING"): (11.0, 10.0),
              ("layout_1", "B_PENDING"): (11.0, 10.0), ("layout_1", "C_PENDING"): (10.0, 11.0)}
TAIL = {"B_PENDING": ["a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1", "a:PICK:carrier:v1", "a:PLACE:carrier:receiver:v1"],
        "C_PENDING": ["a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1", "a:PICK:carrier:v1", "a:PLACE:carrier:receiver:v1"]}


def verdicts(branches, **over):
    rows = {}
    for b in branches:
        rows[b["step"]] = {"branch_id": b["branch_id"], "layout": b["layout"], "context": b["context"],
                           "candidate": b["candidate"], "status": "TASK_SUCCESS", "task_success": True, "complete": True,
                           "problems": [], "recorder_errors": [], "key_target_depth_support_max": 20, "initial_support_ok": True,
                           "checks": {"both_pads_legal": True}, "sequence": b["prefix"] + [b["candidate_id"]] + TAIL[b["context"]]}
    for step, patch in over.items():
        rows[int(step)].update(patch)
    return rows


def pairs_of(out, branches):
    return {k: v.pair_stats(out, branches, *k) for k in CELLS}


def gate(tmp_path, costs=None, durs=None, steps=None, restore=True, zero=True, protected=True, **over):
    out, br = make_out(tmp_path)
    table(out, br, costs or PASS_COSTS, durs, steps)
    return v.final_gate(pairs_of(out, br), verdicts(br, **over), restore, zero, protected, {"A": "PASS", "B": "PASS", "C": "PASS"})


# ------------------------------------------------------------------- geometry and identity
def test_V01_frozen_scene_constants_and_layouts_file():
    assert v.SHOW == (0.02, -0.08) and v.CARRIER == (-0.08, -0.12) and v.BASE_PROXY_XY == (-0.56, 0.0)
    assert v.RECEIVER == (0.18, 0.12) and v.PAD_U == (0.03175082, 0.00519342) and v.PAD_V == (0.00824918, -0.16519342)
    assert v.ANCHOR_U == (0.04350163, 0.09038683) and v.ANCHOR_V == (-0.00350163, -0.25038683) and v.MIRROR_D == 0.086
    doc = json.loads((ROOT / v.LAYOUTS_PATH).read_text())
    assert doc == json.loads(json.dumps(v.layouts_document()))
    l0, l1 = doc["layouts"]
    assert (l0["obj_b_xy"], l0["obj_c_xy"]) == ([0.04350163, 0.09038683], [-0.00350163, -0.25038683])
    assert (l1["obj_b_xy"], l1["obj_c_xy"]) == ([-0.00350163, -0.25038683], [0.04350163, 0.09038683])
    assert [set(x) for x in doc["layouts"]] == [{"layout_id", "carrier_xy", "obj_b_xy", "obj_c_xy", "receiver_xy", "pad_u_xy",
                                                 "pad_v_xy"}] * 2
    for k in ("carrier_xy", "receiver_xy", "pad_u_xy", "pad_v_xy"):
        assert l0[k] == l1[k]


def test_V02_analytic_point_symmetry_identities():
    inv = v.invariants()
    assert inv["all_ok"] and set(inv["exact_identities"]) == {"U_plus_V_equals_2S", "AU_plus_AV_equals_2S", "S_to_U_minus_S_to_V",
                                                              "U_to_AU_minus_V_to_AV", "U_to_AV_minus_V_to_AU"}
    assert all(e <= 1e-9 for e in inv["exact_identities"].values())
    assert set(inv["base_radius_fairness"]) == {"B_to_U_minus_B_to_V", "B_to_AU_minus_B_to_AV"}
    assert all(e <= 1e-6 for e in inv["base_radius_fairness"].values())
    assert abs(inv["values"]["S_to_pad"] - 0.086) < 1e-6 and abs(inv["values"]["pad_to_far_anchor"] - 0.258) < 1e-6
    assert abs(inv["base_radii"]["B_to_U"] - 0.591774) < 1e-5 and abs(inv["base_radii"]["B_to_AU"] - 0.610233) < 1e-5


def test_V02b_actual_panda_base_is_read_from_the_robot_model(tmp_path):
    base, detail = v.actual_base_xy(ROOT)
    assert all(abs(a - b) <= 1e-6 for a, b in zip(base, v.BASE_PROXY_XY)) and detail["table_length_m"] == 0.8
    fake = tmp_path / "src/cp_disr/platforms/libero"
    fake.mkdir(parents=True)
    for name in ("d0_env.py", "family_b_obs_v2.py"):
        text = (ROOT / "src/cp_disr/platforms/libero" / name).read_text()
        (fake / name).write_text(text.replace("self.table_full_size = (0.8,", "self.table_full_size = (0.9,"))
    moved, _ = v.actual_base_xy(tmp_path)
    assert abs(moved[0] - (-0.61)) < 1e-9 and not all(abs(a - b) <= 1e-6 for a, b in zip(moved, v.BASE_PROXY_XY))


def test_V03_reach_cap_and_aabb_with_real_sizes():
    geo = v.geometry_checks()
    assert geo["all_ok"], [k for k, x in geo["checks"].items() if not x]
    names = set(geo["checks"])
    for lid in ("layout_0", "layout_1"):
        for pt in ("carrier", "obj_b", "obj_c", "pad_u", "pad_v"):
            assert f"{lid}:{pt}_radius_le_0.65" in names
        for need in ("inside_table_safe_boundary", "receiver_slots_hold_b_and_c"):
            assert f"{lid}:{need}" in names
    assert geo["checks"]["destinations_and_anchors_do_not_alias"]
    assert max(r["base_radius_m"] for r in geo["reach_rows"]) <= 0.6103 and v.REACH_CAP == 0.65
    assert abs(v.demonstrated_radius(ROOT) - 0.7602631123499285) < 1e-9
    # all 15 body pairs per layout carry an AABB clearance >= 8 mm (cube/pad half sizes, receiver outer walls)
    for lid in ("layout_0", "layout_1"):
        rows = [r for r in geo["aabb_rows"] if r["layout"] == lid]
        assert len(rows) == 15 and all(r["aabb_gap_m"] >= 0.008 for r in rows)
    assert v.AABB_MIN == 0.008 and v.CUBE_HALF == 0.02 and v.PAD_HALF_XY == 0.045


def test_V04_role_based_camera_gate_and_no_hidden_pose():
    cam = v.camera_audit()
    assert cam["all_ok"] and cam["rows"] and all(not r["hidden_pose_used_in_runtime"] for r in cam["rows"])
    gated = [r for r in cam["rows"] if r["gated"]]
    assert {r["object"] for r in gated} == {"carrier", "obj_b", "obj_c", "receiver", "pad_u", "pad_v"}
    need = {"carrier": 8.0, "target_anchor": 8.5, "pad": 8.5, "receiver": 8.5}
    for r in gated:
        assert r["required_edge_px"] == need[r["role"]] and r["required_margin_px"] == 9.0
        assert r["agentview_center_edge_margin_px"] >= 9.0 and r["agentview_px_per_cube_edge"] >= r["required_edge_px"]
    weak = {r["object"]: r for r in gated if r["layout"] == "layout_0"}
    assert abs(weak["carrier"]["agentview_px_per_cube_edge"] - 8.062) < 1e-3
    assert abs(weak["obj_c"]["agentview_px_per_cube_edge"] - 8.675) < 1e-3 and abs(weak["obj_c"]["agentview_center_edge_margin_px"] - 9.2) < 0.05
    assert cam["thresholds"]["predicted_cube_edge_px_min"] == need and cam["thresholds"]["center_edge_margin_px_min"] == 9.0


def test_V04b_carrier_gate_regression_historical_carrier_passes_uniform_8_5_would_reject():
    hist = v.read(ROOT / "configs/final_master/family_b_layouts.json")["layouts"][0]["carrier_xy"]
    lay = v.layout_specs()
    for l in lay:
        l["carrier_xy"] = list(hist)
    cam = v.camera_audit(lay)
    row = next(r for r in cam["rows"] if r["object"] == "carrier")
    assert abs(row["agentview_px_per_cube_edge"] - 8.256) < 1e-3 and row["qualifies"] and cam["all_ok"]
    frozen = next(r for r in v.camera_audit()["rows"] if r["object"] == "carrier")
    assert 8.0 <= frozen["agentview_px_per_cube_edge"] < 8.5 and frozen["qualifies"]
    # the targets keep the stricter 8.5 px gate: pull the far target outward until it fails
    far = v.layout_specs()
    for l in far:
        l["obj_c_xy"] = [-0.0035, -0.31] if l["layout_id"] == "layout_0" else l["obj_c_xy"]
    assert next(r for r in v.camera_audit(far)["rows"] if r["layout"] == "layout_0" and r["object"] == "obj_c")["qualifies"] is False


def test_V05_layouts_differ_only_by_bc_swap_and_no_fixed_rule():
    f = v.task_feature_checks()
    assert f["layouts_differ_only_by_bc_swap"] and f["same_object_id_changes_best_pad"] and f["geometric_best_matches_prediction"]
    assert f["no_fixed_rule_solves_all"] and f["always_u_correct"] == 2 and f["always_v_correct"] == 2
    assert v.PREDICTED_WINNER == {("layout_0", "B_PENDING"): "pad_u", ("layout_0", "C_PENDING"): "pad_v",
                                  ("layout_1", "B_PENDING"): "pad_v", ("layout_1", "C_PENDING"): "pad_u"}


def test_V06_eight_branches_candidate_set_is_exactly_u_v_same_seed_per_pair(tmp_path):
    out, br = make_out(tmp_path)
    assert [v.branch_key(b) for b in br] == list(v.ORDER) and [b["step_name"] for b in br] == list(v.STEP_NAMES)
    assert [(b["layout"], b["context"], b["candidate"]) for b in br] == [
        ("layout_0", "B_PENDING", "pad_u"), ("layout_0", "B_PENDING", "pad_v"), ("layout_0", "C_PENDING", "pad_v"),
        ("layout_0", "C_PENDING", "pad_u"), ("layout_1", "B_PENDING", "pad_v"), ("layout_1", "B_PENDING", "pad_u"),
        ("layout_1", "C_PENDING", "pad_u"), ("layout_1", "C_PENDING", "pad_v")]
    assert {b["candidate_id"] for b in br} == {"a:PLACE_BUFFER:carrier:pad_u:v1", "a:PLACE_BUFFER:carrier:pad_v:v1"}
    assert all(b["repeat"] == 0 and b["context"] != "BOTH_PENDING" for b in br)
    for k in CELLS:
        pair = [b for b in br if (b["layout"], b["context"]) == k]
        assert sorted(b["candidate"] for b in pair) == ["pad_u", "pad_v"] and len({b["restore_seed"] for b in pair}) == 1
    assert len({b["restore_seed"] for b in br}) == 4 and v.check_uniform({"branches": br}) == []
    assert [b["phase"] for b in br] == list("AABBCCCC")


def test_V07_new_ids_do_not_collide_with_history(tmp_path):
    out, br = make_out(tmp_path)
    old = s.historical_ids(ROOT) | {b["branch_id"] for b in s.r2_branches(ROOT).values()}
    v3_ids = {b["branch_id"] for b in json.loads((ROOT / v.V3_EVIDENCE / "registration.json").read_text())["branches"]}
    assert len(v3_ids) == 8 and v3_ids <= old and len(old) >= 36 and not {b["branch_id"] for b in br} & old
    assert all("V3_R1" in b["attempt_identity"] and b["card_id"] == v.CARD for b in br)
    manifest = out / "spec/manifest.yaml"
    prefixes = {k: x["prefix"] for k, x in CFG["contexts"].items()}
    with pytest.raises(ValueError, match="collide"):
        v.build_branches(v.FROZEN_SHA, "c" * 40, "s" * 64, prefixes, {br[0]["branch_id"]}, manifest, "f" * 16)


def test_V08_check_uniform_flags_broken_registration(tmp_path):
    out, br = make_out(tmp_path)
    bad = [dict(b) for b in br]
    bad[1]["restore_seed"] += 1
    assert any("seeds differ" in p for p in v.check_uniform({"branches": bad}))
    bad = [dict(b) for b in br]
    bad[2]["observation_profile_sha256"] = "x"
    assert v.check_uniform({"branches": bad})
    assert v.check_uniform({"branches": br[:7]})


def test_V09_frozen_profile_and_sources_unchanged():
    for rel in v.FROZEN_FILES:
        want = subprocess.run(["git", "-C", str(ROOT), "show", f"{r2.PROFILE_ORIGIN}:{rel}"], capture_output=True, check=True).stdout
        assert (ROOT / rel).read_bytes() == want, rel
    assert not r2.frozen_bytes_ok(ROOT)
    assert json.loads((ROOT / v.FROZEN_FILES[0]).read_text())["profile_sha256"] == v.FROZEN_SHA == CFG["observation_profile_sha256"]
    for rel in v.UNCHANGED_SOURCES:
        base = subprocess.run(["git", "-C", str(ROOT), "show", f"{v.BASE}:{rel}"], capture_output=True, check=True).stdout
        assert (ROOT / rel).read_bytes() == base, rel


def test_V10_show_pose_and_place_buffer_logic_match_base(tmp_path):
    assert v.show_pose_in_source(ROOT) == v.SHOW
    src = v.source_checks(ROOT)
    assert src["checks"]["show_pose_is_0.02_-0.08"] and src["checks"]["place_buffer_logic_equals_base"]
    tmp = tmp_path / "t/src/cp_disr/platforms/libero"
    tmp.mkdir(parents=True)
    text = (ROOT / "src/cp_disr/platforms/libero/skill_executor.py").read_text()
    (tmp / "skill_executor.py").write_text(text.replace("np.array([0.02, -0.08,", "np.array([0.03, -0.08,"))
    assert v.show_pose_in_source(tmp_path / "t") == (0.03, -0.08) != v.SHOW


def test_V11_gate0_passes_on_this_worktree(tmp_path):
    res = v.gate0(ROOT, tmp_path / "g0")
    assert res["status"] == "GATE0_PASS" and (tmp_path / "g0/geometry_invariants.json").is_file()
    geo = json.loads((tmp_path / "g0/geometry_invariants.json").read_text())
    assert geo["layouts_file_matches_frozen_generator"]
    reach = json.loads((tmp_path / "g0/reach_envelope_audit.json").read_text())
    aabb = json.loads((tmp_path / "g0/aabb_clearance_audit.json").read_text())
    assert reach["all_ok"] and reach["cap_m"] == 0.65 and reach["margin_to_demonstrated_max_m"] > 0.14
    assert aabb["all_ok"] and aabb["min_gap_m"] >= 0.008 and len(aabb["pairs"]) == 30
    assert reach["base_from_robot_model"]["table_length_m"] == 0.8


def test_V12_no_provider_rl_optimizer_entry():
    forbidden = ("provider", "torch", "optim", "representation", "rl", "s2", "s3", "train", "elastic")
    for path in (SRC, ROOT / "scripts/family_b_v3r1_reach_safe_mirror.py"):
        tree = ast.parse(path.read_text())
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods += [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        for mod in mods:
            for part in mod.split("."):
                assert not any(part == f or part.startswith(f + "_") for f in forbidden), (path.name, mod)
    a = CFG["authorization"]
    assert a["provider"] == a["representation_forwards"] == a["rl_transitions"] == a["optimizer_steps"] == a["elastic"] == 0
    assert a["formal_test"] == 0 and a["s2"] is False and a["s3"] is False and a["tp_training"] is False and a["skill_retries"] == 0
    assert (a["branch_attempts"], a["environment_construction_attempts"], a["successful_explicit_resets"],
            a["live_skill_calls"]) == (8, 8, 8, 48)
    assert CFG["card_id"] == v.CARD and CFG["base_commit"] == v.BASE and CFG["observation_profile_sha256"] == v.FROZEN_SHA
    assert CFG["branches"]["both_pending"] is False and "BOTH_PENDING" not in CFG["contexts"]


# ------------------------------------------------------------------- ledger and staged order
def test_V13_ledger_caps_and_zero_budgets(tmp_path):
    out, br = make_out(tmp_path)
    led = ledger(out)
    assert (led["attempts"]["cap"], led["environment_constructions"]["cap"], led["explicit_resets"]["cap"],
            led["live_skill_calls"]["cap"]) == (8, 8, 8, 48)
    assert all(led[k]["cap"] == 0 and led[k]["used"] == 0 for k in (
        "skill_retries", "standalone_capture_resets", "provider_calls", "representation_forwards", "rl_transitions",
        "optimizer_steps", "elastic", "formal_test_episodes"))
    assert {k: x["cap"] for k, x in led["phases"].items()} == {"A": 2, "B": 2, "C": 4}
    assert led["history"]["v3_original"]["attempts_charged"] == 1 and led["history"]["v3_original"]["mechanism_measured"] is False


def test_V14_order_gate_blocks_out_of_sequence_reservation(tmp_path):
    out, br = make_out(tmp_path)
    before = ledger(out)
    for step in (1, 2, 3, 4, 5, 6, 7):
        with pytest.raises(ValueError, match="STOPPED_V3_ORDER"):
            v.reserve_and_charge(out, br[step])
    assert ledger(out) == before
    v.reserve_and_charge(out, br[0])
    with pytest.raises(ValueError, match="STOPPED_DUPLICATE_ATTEMPT"):
        v.reserve_and_charge(out, br[0])
    assert ledger(out)["attempts"]["used"] == 1


def test_V15_wave_gate_status_blocks_next_wave_even_if_steps_passed(tmp_path):
    out, br = make_out(tmp_path)

    def mark(i, status):
        v._set(out, lambda st: st["steps"][str(i)].update(status=status))
    for i in (0, 1):
        mark(i, "PASS")
    with pytest.raises(ValueError, match="STOPPED_V3_ORDER:step B1 needs wave A"):
        v.reserve_and_charge(out, br[2])
    v._set(out, lambda st: st["waves"]["A"].update(status="PASS"))
    v.reserve_and_charge(out, br[2])
    for i in (2, 3):
        mark(i, "PASS")
    with pytest.raises(ValueError, match="STOPPED_V3_ORDER:step C1 needs wave B"):
        v.reserve_and_charge(out, br[4])
    v._set(out, lambda st: st["waves"]["B"].update(status="PASS"))
    with pytest.raises(ValueError, match="STOPPED_V3_ORDER:step C2 needs C1"):
        v.reserve_and_charge(out, br[5])


def test_V16_full_pass_dispatches_in_fixed_order_with_pair_gpus_and_caps(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch)
    launcher = FakeLauncher(polls=1)
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] is None and v.state_of(out)["status"] == "COMPLETE"
    names = [c[2] for c in launcher.calls]
    assert names[:4] == ["A1", "A2", "B1", "B2"] and sorted(names[4:]) == ["C1", "C2", "C3", "C4"]
    assert names.index("C1") < names.index("C2") and names.index("C3") < names.index("C4")
    gpu = {c[2]: c[1] for c in launcher.calls}
    assert {gpu[n] for n in ("A1", "A2", "B1", "B2")} == {3} and gpu["C1"] == gpu["C2"] and gpu["C3"] == gpu["C4"]
    assert gpu["C1"] != gpu["C3"] and launcher.max_active == 2
    seed = {c[2]: c[3] for c in launcher.calls}
    assert seed["A1"] == seed["A2"] and seed["B1"] == seed["B2"] and seed["C1"] == seed["C2"] and seed["C3"] == seed["C4"]
    led = ledger(out)
    assert (led["attempts"]["used"], led["environment_constructions"]["used"], led["explicit_resets"]["used"]) == (8, 8, 8)
    assert led["live_skill_calls"]["used"] == 48 and led["actual"]["constructions_attempted"] == 8
    assert len(set(led["reserved_branches"])) == 8 and (out / "physical/throughput_v3.json").is_file()
    assert json.loads((out / "physical/throughput_v3.json").read_text())["speedup_claim"].startswith("NONE")
    with pytest.raises(ValueError, match="STOPPED_V3_ALREADY_RUN"):
        v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert len(launcher.calls) == 8


def test_V17_single_gpu_runs_everything_sequentially(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch)
    launcher = FakeLauncher()
    v.run_waves(ROOT, out, [3], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert len(launcher.calls) == 8 and {c[1] for c in launcher.calls} == {3} and launcher.max_active == 1


def test_V18_a1_failure_stops_everything(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, fail_keys={v.ORDER[0]})
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "A1" and [c[2] for c in launcher.calls] == ["A1"]
    with pytest.raises(ValueError, match="STOPPED_V3_WAVE_FAILED"):
        v.reserve_and_charge(out, br[1])
    assert ledger(out)["attempts"]["used"] == 1 and v.state_of(out)["status"] == "STOPPED"


def test_V19_low_support_stops_like_a_failure(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, support=11)
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "A1" and len(launcher.calls) == 1
    out2, br2 = make_out(tmp_path / "ok12")
    patch_waves(v, monkeypatch, support=12)
    launcher2 = FakeLauncher()
    assert v.run_waves(ROOT, out2, [3, 4], launcher=launcher2, identity_check=no_identity, sleep=lambda x: None)["stopped"] is None


def test_V20_wave_a_gate_failure_blocks_b_and_c(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, wave_status={"A": "FAIL"})
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "wave_A" and [c[2] for c in launcher.calls] == ["A1", "A2"]
    with pytest.raises(ValueError, match="STOPPED_V3_WAVE_FAILED"):
        v.reserve_and_charge(out, br[2])
    assert ledger(out)["attempts"]["used"] == 2


def test_V21_wave_b_failure_means_layout_1_does_not_run(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, wave_status={"B": "FAIL"})
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "wave_B" and [c[2] for c in launcher.calls] == ["A1", "A2", "B1", "B2"]
    assert ledger(out)["attempts"]["used"] == 4
    assert all(v.state_of(out)["steps"][str(i)]["status"] == "NOT_RUN" for i in range(4, 8))


def test_V22_wave_c_first_of_pair_failure_blocks_its_second_and_new_dispatch(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, fail_keys={v.ORDER[4]})
    launcher = FakeLauncher(polls=1)
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    names = [c[2] for c in launcher.calls]
    assert res["stopped"] == "C1" and "C2" not in names and "C4" not in names and names.count("C3") == 1
    assert ledger(out)["attempts"]["used"] == len(names) <= 6 and v.state_of(out)["status"] == "STOPPED"


def test_V23_wave_c_gate_failure_is_final(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, wave_status={"C": "FAIL"})
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "wave_C" and len(launcher.calls) == 8 and v.state_of(out)["status"] == "STOPPED"
    assert ledger(out)["attempts"]["used"] == 8
    with pytest.raises(ValueError):
        v.reserve_and_charge(out, br[0])


def test_V24_ninth_attempt_is_impossible(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch)
    v.run_waves(ROOT, out, [3, 4], launcher=FakeLauncher(), identity_check=no_identity, sleep=lambda x: None)
    extra = dict(br[0], branch_id="9" * 64)
    with pytest.raises(ValueError):
        v.reserve_and_charge(out, extra)
    assert ledger(out)["attempts"]["used"] == 8


def test_V25_worker_refuses_unreserved_branch(tmp_path):
    out, br = make_out(tmp_path)
    with pytest.raises(RuntimeError, match="STOPPED_BUDGET"):
        v.worker(ROOT, out, br[0]["branch_id"])


# ------------------------------------------------------------------- wave gates on real evidence files
def wave(tmp_path, wave_name, costs=None, durs=None, steps=None):
    out, br = make_out(tmp_path)
    table(out, br, costs or PASS_COSTS, durs, steps)
    return v.wave_gate(out, wave_name)


def test_V26_wave_gates_pass_on_the_predicted_pattern(tmp_path):
    for w in "ABC":
        g = wave(tmp_path / w, w)
        assert g["status"] == "PASS", g["checks"]
    assert any("sign_delta" in k for k in wave(tmp_path / "B2", "B")["checks"])


def test_V27_duration_diff_gate_has_a_strict_0_15_edge(tmp_path):
    cell = ("layout_0", "B_PENDING")
    ok = wave(tmp_path / "ok", "A", durs={**{k: (6.0, 6.0) for k in CELLS}, cell: (6.0, 6.14)})
    bad = wave(tmp_path / "bad", "A", durs={**{k: (6.0, 6.0) for k in CELLS}, cell: (6.0, 6.2)})
    assert ok["status"] == "PASS" and bad["status"] == "FAIL"
    assert bad["checks"]["layout_0/B_PENDING:candidate_duration_diff_le_0.15"] is False


def test_V28_steps_diff_gate(tmp_path):
    cell = ("layout_0", "B_PENDING")
    base = {k: (300, 300) for k in CELLS}
    assert wave(tmp_path / "ok", "A", steps={**base, cell: (300, 303)})["status"] == "PASS"
    bad = wave(tmp_path / "bad", "A", steps={**base, cell: (300, 304)})
    assert bad["status"] == "FAIL" and bad["checks"]["layout_0/B_PENDING:candidate_steps_diff_le_3"] is False


def test_V29_preference_margin_is_strict_and_directional(tmp_path):
    cell = ("layout_0", "B_PENDING")
    assert wave(tmp_path / "a", "A", costs={**PASS_COSTS, cell: (10.0, 10.1)})["status"] == "FAIL"      # inside the margin
    assert wave(tmp_path / "b", "A", costs={**PASS_COSTS, cell: (10.0, 10.2)})["status"] == "PASS"
    assert wave(tmp_path / "c", "A", costs={**PASS_COSTS, cell: (11.0, 10.0)})["status"] == "FAIL"      # wrong direction
    cell_c = ("layout_0", "C_PENDING")
    assert wave(tmp_path / "d", "B", costs={**PASS_COSTS, cell_c: (10.0, 11.0)})["status"] == "FAIL"
    g = wave(tmp_path / "e", "B", costs={**PASS_COSTS, cell_c: (10.0, 11.0)})
    assert g["status"] == "FAIL" and g["checks"]["layout_0:sign_delta_B_differs_from_sign_delta_C"] is False


def test_V30_wave_c_requires_both_layout_1_cells(tmp_path):
    assert wave(tmp_path / "ok", "C")["status"] == "PASS"
    bad = wave(tmp_path / "bad", "C", costs={**PASS_COSTS, ("layout_1", "C_PENDING"): (11.0, 10.0)})
    assert bad["status"] == "FAIL" and bad["checks"]["layout_1/C_PENDING:pad_u_faster_by_more_than_0.15"] is False


def test_V31_missing_evidence_fails_closed(tmp_path):
    out, br = make_out(tmp_path)
    g = v.wave_gate(out, "A")
    assert g["status"] == "FAIL" and not any(g["checks"].values())
    assert v.prefers({"cost_u": None, "cost_v": 1.0}, "pad_u") is False


def test_V32_support_threshold_is_12_while_production_stays_8():
    from cp_disr.platforms.libero.perception import THRESHOLDS
    assert v.SUPPORT_MIN == 12 and THRESHOLDS["min_pixels"] == 8
    row = {"context": "B_PENDING", "per_view": {"obj_b": {"agentview": {"depth_support": 11}, "sideview": {"depth_support": 12}}}}
    assert v.target_support(row) == 12
    assert v.target_support({"context": "C_PENDING", "per_view": {}}) == 0


def test_V33_step_verdict_applies_support_margin(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    monkeypatch.setattr(v, "initial_support", fake_initial())
    for support, expect in ((11, False), (12, True)):
        monkeypatch.setattr(v, "evaluate_branch", fake_evaluate(support=support))
        ok, row = v.step_verdict(out, br[0])
        assert ok is expect and row["key_target_depth_support_max"] == support and row["support_margin_vs_12"] == support - 12
    monkeypatch.setattr(v, "evaluate_branch", fake_evaluate(fail_keys={v.ORDER[0]}))
    assert v.step_verdict(out, br[0])[0] is False


def _initial_file(out, b, carrier, obj_b, obj_c, side=0):
    d = Path(out) / "captures" / b["branch_id"]
    d.mkdir(parents=True, exist_ok=True)
    def blk(a, s):
        return {o: {"depth_support": x, "mask_pixels": x} for o, x in (("carrier", a[0]), ("obj_b", a[1]), ("obj_c", a[2]))}
    (d / "initial_view_analysis.json").write_text(json.dumps({"per_view": {"agentview": blk((carrier, obj_b, obj_c), 0),
                                                                          "sideview": blk((side, side, side), 0)}}))


def test_V33b_initial_public_support_gate_for_carrier_and_setup_target(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    a1 = br[0]
    assert a1["context"] == "B_PENDING" and v.SETUP_TARGET["B_PENDING"] == "obj_c"
    assert v.initial_support(out, a1)["present"] is False and v.initial_support(out, a1)["ok"] is False
    _initial_file(out, a1, carrier=49, obj_b=35, obj_c=120)
    ok = v.initial_support(out, a1)
    assert ok["ok"] and ok["carrier"] == 49 and ok["setup_target"] == 120 and ok["per_object_max"]["obj_b"] == 35
    _initial_file(out, a1, carrier=11, obj_b=35, obj_c=120)
    assert v.initial_support(out, a1)["ok"] is False
    _initial_file(out, a1, carrier=49, obj_b=35, obj_c=11)          # setup target (obj_c) below 12 even though obj_b is fine
    assert v.initial_support(out, a1)["ok"] is False
    c_ctx = next(b for b in br if b["context"] == "C_PENDING")
    _initial_file(out, c_ctx, carrier=49, obj_b=11, obj_c=120)       # C_PENDING setup target is obj_b
    assert v.initial_support(out, c_ctx)["ok"] is False
    _initial_file(out, c_ctx, carrier=49, obj_b=12, obj_c=0)
    assert v.initial_support(out, c_ctx)["ok"] is True
    monkeypatch.setattr(v, "evaluate_branch", fake_evaluate(support=20))
    _initial_file(out, a1, carrier=11, obj_b=35, obj_c=120)
    ok_step, row = v.step_verdict(out, a1)
    assert ok_step is False and row["initial_support_ok"] is False and row["problems"] == []


def test_V33c_low_initial_support_stops_a1_before_anything_else(tmp_path, monkeypatch):
    out, br = make_out(tmp_path)
    patch_waves(v, monkeypatch, initial_ok=False)
    launcher = FakeLauncher()
    res = v.run_waves(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity, sleep=lambda x: None)
    assert res["stopped"] == "A1" and [c[2] for c in launcher.calls] == ["A1"]
    assert v.state_of(out)["steps"]["0"]["initial_support_ok"] is False and ledger(out)["attempts"]["used"] == 1


# ------------------------------------------------------------------- final gate
def test_V34_final_gate_pass(tmp_path):
    g = gate(tmp_path)
    assert g["status"] == "PASS" and g["label"] == v.GATE_PASS and g["next_action"] == v.NEXT_PASS and all(g["conditions"].values())
    assert len(g["conditions"]) == 15 and g["always_u_accuracy"] == "2/4" and g["always_v_accuracy"] == "2/4"
    assert g["state_conditioned_preference_reversal"] is True and g["claims"]["provider_or_representation_started"] is False
    assert g["measured_winner"] == {"layout_0/B_PENDING": "pad_u", "layout_0/C_PENDING": "pad_v",
                                    "layout_1/B_PENDING": "pad_v", "layout_1/C_PENDING": "pad_u"}


def _not_established(g, cond):
    assert g["status"] == "NOT_ESTABLISHED" and g["label"] == v.GATE_FAIL and g["next_action"] == v.NEXT_FAIL
    assert g["conditions"][cond] is False


def test_V35_every_single_failed_condition_blocks_the_pass(tmp_path):
    cell = ("layout_1", "B_PENDING")
    always_u = {k: (10.0, 11.0) for k in CELLS}
    always_v = {k: (11.0, 10.0) for k in CELLS}
    d = {k: (6.0, 6.0) for k in CELLS}
    st = {k: (300, 300) for k in CELLS}
    _not_established(gate(tmp_path / "1", **{"4": {"status": "TASK_FAILED", "task_success": False}}), "1_8_of_8_task_success")
    _not_established(gate(tmp_path / "2", **{"2": {"complete": False}}), "2_8_of_8_complete_records")
    _not_established(gate(tmp_path / "3", restore=False), "3_four_pair_restores_pass")
    _not_established(gate(tmp_path / "4", durs={**d, cell: (6.0, 6.3)}), "4_place_buffer_duration_diff_le_0.15")
    _not_established(gate(tmp_path / "5", steps={**st, cell: (300, 310)}), "5_controller_steps_diff_le_3")
    _not_established(gate(tmp_path / "6", **{"3": {"key_target_depth_support_max": 11}}), "6_public_depth_support_ge_12_initial_and_post_candidate")
    _not_established(gate(tmp_path / "6b", **{"0": {"initial_support_ok": False}}), "6_public_depth_support_ge_12_initial_and_post_candidate")
    _not_established(gate(tmp_path / "7", **{"5": {"recorder_errors": ["x"]}}), "7_no_noplan_timeout_exception_or_recorder_error")
    _not_established(gate(tmp_path / "7b", **{"5": {"problems": ["NO_PLAN"]}}), "7_no_noplan_timeout_exception_or_recorder_error")
    _not_established(gate(tmp_path / "8", **{"6": {"checks": {"both_pads_legal": False}}}), "8_candidates_equally_legal")
    bad_seq = verdicts(make_out(tmp_path / "x")[1])[7]["sequence"][:-1]
    _not_established(gate(tmp_path / "9", **{"7": {"sequence": bad_seq}}), "9_plan_length_and_skill_multiset_match")
    _not_established(gate(tmp_path / "10", costs={**PASS_COSTS, ("layout_0", "B_PENDING"): (10.0, 10.1)}), "10_layout_0_preference_reversal")
    _not_established(gate(tmp_path / "11", costs={**PASS_COSTS, ("layout_1", "C_PENDING"): (10.0, 10.1)}), "11_layout_1_preference_reversal")
    g = gate(tmp_path / "12", costs=always_u)
    _not_established(g, "12_same_object_best_pad_changes_with_layout")
    assert g["conditions"]["13_neither_always_u_nor_always_v_reaches_4_of_4"] is False and g["always_u_accuracy"] == "4/4"
    _not_established(gate(tmp_path / "13", costs=always_v), "13_neither_always_u_nor_always_v_reaches_4_of_4")
    _not_established(gate(tmp_path / "14", zero=False), "14_provider_representation_rl_optimizer_zero")
    _not_established(gate(tmp_path / "15", protected=False), "15_old_evidence_and_frozen_config_unchanged")


def test_V36_partial_run_is_not_established_without_crashing(tmp_path):
    out, br = make_out(tmp_path)
    table(out, br, PASS_COSTS)
    rows = verdicts(br)
    rows[7] = None
    g = v.final_gate(pairs_of(out, br), rows, False, True, True, {"A": "PASS", "B": "PASS", "C": "FAIL"})
    assert g["status"] == "NOT_ESTABLISHED" and g["conditions"]["1_8_of_8_task_success"] is False


# ------------------------------------------------------------------- evidence arithmetic
def test_V37_candidate_evidence_reads_duration_steps_and_eef(tmp_path):
    out, br = make_out(tmp_path)
    write_branch(out, br[0], cost=12.0, dur=6.2, steps=311)
    ev = v.candidate_evidence(out, br[0])
    assert ev["present"] and ev["duration"] == 6.2 and ev["steps"] == 311 and ev["start_eef"] == [0.02, -0.08, 1.0]
    assert v.candidate_evidence(out, br[1])["present"] is False


def test_V38_cost_decomposition_sums_to_total(tmp_path):
    out, br = make_out(tmp_path)
    table(out, br, PASS_COSTS, durs={k: (6.0, 5.0) for k in CELLS})
    for k in CELLS:
        p = v.pair_stats(out, br, *k)
        d = v.decomposition(out, br, p)
        parts = d["candidate_action_effect"] + d["first_pick_effect"] + d["remaining_tail_effect"]
        assert abs(d["decomposition_error"]) < 1e-9 and abs(parts - p["delta_u_minus_v"]) < 1e-9
        assert d["total_candidate_to_success_effect"] == p["delta_u_minus_v"] and abs(d["candidate_action_effect"] - 1.0) < 1e-9


def test_V39_primary_metric_is_total_not_tail_only(tmp_path):
    g = gate(tmp_path)
    assert "tail-only is decomposition only" in g["primary_metric"] and "failure_policy" in g
    assert CFG["mechanism_gate"]["primary_metric"] == "candidate_start_to_first_verified_task_success"
    assert CFG["mechanism_gate"]["tolerance_seconds"] == v.TOL_T == 0.15


def test_V40_evaluate_branch_is_the_staging_evaluation_with_this_card_identity():
    assert v.evaluate_branch.__globals__["CARD"] == v.CARD != s.CARD
    assert callable(s.build_worker()) and callable(s.build_paired_restore())


def test_V41_modifying_old_raw_evidence_fails_protection(tmp_path, monkeypatch):
    old = tmp_path / "old/captures/b1"
    old.mkdir(parents=True)
    (old / "x.npy").write_bytes(b"raw")
    monkeypatch.setattr(v, "protected_paths", lambda root: [tmp_path / "old"])
    out, br = make_out(tmp_path)
    v.inventory_before(ROOT, out)
    assert v.verify(ROOT, out)["checks"]["protected_unchanged"] is True
    (old / "x.npy").write_bytes(b"tampered")
    res = v.verify(ROOT, out)
    assert res["checks"]["protected_unchanged"] is False and res["status"] == "FAIL"


def test_V42_layouts_file_is_the_only_runtime_layout_source():
    base = yaml.safe_load((ROOT / v.STAGING_EVIDENCE / "spec/runtime_manifest_T_P_FB_staging_v2.yaml").read_text())
    assert base["runtime"].get("layouts_path") != v.LAYOUTS_PATH
    layouts = json.loads((ROOT / v.LAYOUTS_PATH).read_text())["layouts"]
    assert [l["layout_id"] for l in layouts] == ["layout_0", "layout_1"]


# ------------------------------------------------------------------- mutations
def _mutant(tmp_path, name, old, new):
    src = SRC.read_text()
    assert src.count(old) == 1, name
    path = tmp_path / f"{name}.py"
    path.write_text(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(f"cp_disr.analysis.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(name, desc, caught, scenario):
    _MUT[name] = {"mutation": desc, "killed": caught, "scenario": scenario}
    target = os.environ.get("CP_DISR_V3R1_DIR")
    if target:
        Path(target, "mutation_receipt.json").write_text(json.dumps(_MUT, indent=1, sort_keys=True))


def test_V43_mutation_uv_base_radius_differs_by_more_than_1cm_fails_fairness():
    assert v.invariants()["all_ok"]
    shifted = v.layout_specs(pad_v=(0.00824918 + 0.012, -0.16519342))
    inv = v.invariants(shifted)
    assert inv["all_ok"] is False and inv["base_radius_fairness"]["B_to_U_minus_B_to_V"] > 0.01
    assert inv["exact_identities"]["U_plus_V_equals_2S"] > 1e-9
    _record("R-43_uv_base_radius_diff_gt_1cm", "pad_v moved 1.2 cm radially", True, "base-radius fairness identity fails")


def _target_at_radius(radius):
    lay = v.layout_specs()
    for l in lay:
        l["obj_b_xy"] = [-0.56 + radius, 0.0] if l["layout_id"] == "layout_0" else l["obj_b_xy"]
    return lay


def test_R44_mutation_target_radius_0_66_fails_gate0_geometry():
    assert v.geometry_checks(_target_at_radius(0.64))["checks"]["layout_0:obj_b_radius_le_0.65"] is True
    geo = v.geometry_checks(_target_at_radius(0.66))
    assert geo["checks"]["layout_0:obj_b_radius_le_0.65"] is False and geo["all_ok"] is False
    _record("R-44_target_radius_0.66", "obj_b anchor at 0.66 m", True, "reach cap rejects it")


def _radius_gate_rejects(mod):
    geo = mod.geometry_checks(_target_at_radius(0.66))
    assert geo["checks"]["layout_0:obj_b_radius_le_0.65"] is False


def test_R45_mutation_restoring_the_demonstrated_plus_margin_rule_is_detected(tmp_path):
    _radius_gate_rejects(v)
    mut = _mutant(tmp_path, "mut_old_reach_rule", "REACH_CAP = 0.65 ", "REACH_CAP = 0.7603 + 0.05 ")
    with pytest.raises(BaseException) as caught:
        _radius_gate_rejects(mut)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-45_old_reach_rule", "cap = demonstrated + 0.05", True, "0.66 m target must fail")


def test_R46_mutation_deleting_the_dynamic_target_reach_check_is_detected(tmp_path):
    mut = _mutant(tmp_path, "mut_no_target_reach", 'REACH_POINTS = ("carrier", "obj_b", "obj_c", "pad_u", "pad_v")',
                  'REACH_POINTS = ("carrier", "pad_u", "pad_v")')
    with pytest.raises(BaseException) as caught:
        _radius_gate_rejects(mut)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-46_no_target_reach_check", "obj_b/obj_c removed from REACH_POINTS", True, "0.66 m target must fail")


def _overlap_fixture():
    lay = v.layout_specs()
    for l in lay:
        l["carrier_xy"] = [v.PAD_U[0] + 0.06, v.PAD_U[1] + 0.06]       # AABB overlap with pad_u (-5 mm), centre distance 0.085
    return lay


def _aabb_flags_overlap(mod):
    geo = mod.geometry_checks(_overlap_fixture())
    assert geo["checks"]["layout_0:aabb_clearance:carrier|pad_u"] is False


def test_R47_mutation_euclidean_distance_instead_of_aabb_is_detected(tmp_path):
    _aabb_flags_overlap(v)
    mut = _mutant(tmp_path, "mut_euclid_not_aabb",
                  "    return max(abs(c1[0] - c2[0]) - h1[0] - h2[0], abs(c1[1] - c2[1]) - h1[1] - h2[1])",
                  "    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1])) - h1[0] - h2[0]")
    with pytest.raises(BaseException) as caught:
        _aabb_flags_overlap(mut)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-47_euclid_instead_of_aabb", "_box_gap uses centre distance", True, "diagonal carrier/pad overlap fixture")


def _carrier_roles(mod):
    hist = v.read(ROOT / "configs/final_master/family_b_layouts.json")["layouts"][0]["carrier_xy"]
    lay = mod.layout_specs()
    for l in lay:
        l["carrier_xy"] = list(hist)
    assert mod.camera_audit(lay)["all_ok"] is True and mod.camera_audit()["all_ok"] is True


def test_R48_mutation_uniform_8_5_px_camera_gate_is_detected(tmp_path):
    _carrier_roles(v)
    mut = _mutant(tmp_path, "mut_uniform_camera_gate", 'CAM_EDGE_MIN = {"carrier": 8.0,', 'CAM_EDGE_MIN = {"carrier": 8.5,')
    with pytest.raises(BaseException) as caught:
        _carrier_roles(mut)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-48_uniform_camera_gate", "carrier edge gate 8.0 -> 8.5", True, "carrier regression")


def _initial_gate_blocks(mod, tmp_path, monkeypatch):
    out, br = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(support=20))
    monkeypatch.setattr(mod, "initial_support", fake_initial(ok=False))
    assert mod.step_verdict(out, br[0])[0] is False


def test_R49_mutation_dropping_the_initial_support_gate_is_detected(tmp_path, monkeypatch):
    _initial_gate_blocks(v, tmp_path / "prod", monkeypatch)
    mut = _mutant(tmp_path, "mut_initial_support_dropped", ' and support >= SUPPORT_MIN and init["ok"])', ' and support >= SUPPORT_MIN)')
    with pytest.raises(BaseException) as caught:
        _initial_gate_blocks(mut, tmp_path / "mut", monkeypatch)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-49_initial_support_dropped", "init['ok'] removed from the verdict", True, "carrier/setup-target support 11 must stop")


def test_V44_mutation_layout_1_not_swapped_fails_the_task_feature():
    assert v.task_feature_checks()["layouts_differ_only_by_bc_swap"]
    f = v.task_feature_checks(v.layout_specs(swap_layout_1=False))
    assert f["layouts_differ_only_by_bc_swap"] is False and f["same_object_id_changes_best_pad"] is False
    _record("R-44b_layout_1_not_swapped", "layout_1 identical to layout_0", True, "task-feature gate fails")


def _order_blocked(mod, tmp_path):
    out, br = make_out(tmp_path, mod)
    with pytest.raises(ValueError, match="STOPPED_V3_ORDER"):
        mod.reserve_and_charge(out, br[1])


def test_V45_mutation_deleting_the_staged_order_gate_is_detected(tmp_path):
    _order_blocked(v, tmp_path / "prod")
    old = '        _gate_ok(state_of(out), b["step"])\n'
    mut = _mutant(tmp_path, "mut_order_gate_deleted", old, "        pass\n")
    with pytest.raises(BaseException) as caught:
        _order_blocked(mut, tmp_path / "mut")
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-45b_order_gate_deleted", "_gate_ok call removed", True, "A2 reservable before A1 passes")


def _duration_gate_blocks(mod, tmp_path):
    out, br = make_out(tmp_path, mod)
    base = {k: (6.0, 6.0) for k in CELLS}
    table(out, br, PASS_COSTS, durs={**base, ("layout_0", "B_PENDING"): (6.0, 6.5)})
    g = mod.wave_gate(out, "A")
    assert g["status"] == "FAIL" and g["checks"]["layout_0/B_PENDING:candidate_duration_diff_le_0.15"] is False


def test_V46_mutation_loosening_the_0_15_gate_is_detected(tmp_path):
    _duration_gate_blocks(v, tmp_path / "prod")
    mut = _mutant(tmp_path, "mut_tol_loosened", "TOL_T = 0.15 ", "TOL_T = 5.0 ")
    with pytest.raises(BaseException) as caught:
        _duration_gate_blocks(mut, tmp_path / "mut")
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-46b_tolerance_loosened", "TOL_T 0.15 -> 5.0", True, "0.5 s PLACE_BUFFER mismatch must fail wave A")


def _support_blocks(mod, tmp_path, monkeypatch):
    out, br = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "initial_support", fake_initial())
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(support=11))
    assert mod.step_verdict(out, br[0])[0] is False


def test_V47_mutation_dropping_the_support_margin_is_detected(tmp_path, monkeypatch):
    _support_blocks(v, tmp_path / "prod", monkeypatch)
    mut = _mutant(tmp_path, "mut_support_dropped", "\nSUPPORT_MIN = 12 ", "\nSUPPORT_MIN = 8 ")
    with pytest.raises(BaseException) as caught:
        _support_blocks(mut, tmp_path / "mut", monkeypatch)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-47b_support_margin_dropped", "SUPPORT_MIN 12 -> 8", True, "support 11 must not qualify")


def _always_u_not_established(mod, tmp_path):
    out, br = make_out(tmp_path, mod)
    table(out, br, {k: (10.0, 11.0) for k in CELLS})
    g = mod.final_gate({k: mod.pair_stats(out, br, *k) for k in CELLS}, verdicts(br), True, True, True, {})
    assert g["status"] == "NOT_ESTABLISHED"


def test_V48_mutation_forcing_reversal_conditions_true_is_detected(tmp_path):
    _always_u_not_established(v, tmp_path / "prod")
    src = SRC.read_text()
    olds = ['"10_layout_0_preference_reversal": bool(c10)', '"11_layout_1_preference_reversal": bool(c11)',
            '"12_same_object_best_pad_changes_with_layout": bool(c12)',
            '"13_neither_always_u_nor_always_v_reaches_4_of_4": bool(c13)']
    for old in olds:
        assert src.count(old) == 1
        src = src.replace(old, old.split(": bool")[0] + ": True")
    path = tmp_path / "mut_all.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("cp_disr.analysis.mut_all", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(BaseException) as caught:
        _always_u_not_established(mod, tmp_path / "mut")
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R-48b_reversal_forced_true", "conditions 10-13 forced True", True, "always-U data must be NOT_ESTABLISHED")
