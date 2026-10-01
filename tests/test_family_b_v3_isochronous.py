"""V-01..: identity, geometry, staged dispatch, wave gates and final gate for the V3 isochronous cross-staging pilot.
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
from cp_disr.analysis import family_b_v3_isochronous as v
from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / v.CFG).read_text())
SRC = ROOT / "src/cp_disr/analysis/family_b_v3_isochronous.py"
_MUT = {}
CELLS = tuple((l, c) for l in ("layout_0", "layout_1") for c in v.CONTEXTS)


@pytest.fixture(autouse=True)
def _no_real_simulator(monkeypatch):
    from robosuite.utils import binding_utils

    def boom(*a, **k):
        raise AssertionError("V3 tests must not construct a MuJoCo sim / EGL render context")
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


def patch_waves(mod, monkeypatch, fail_keys=(), wave_status=None, support=20):
    wave_status = wave_status or {}
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
                           "problems": [], "recorder_errors": [], "key_target_depth_support_max": 20,
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
    assert v.SHOW == (0.02, -0.08) and v.CARRIER == (-0.05465481022352864, -0.12557790016547377)
    assert v.RECEIVER == (0.18, 0.12) and v.PAD_U == (-0.08, 0.02) and v.PAD_V == (0.12, -0.18)
    assert v.ANCHOR_U == (-0.18, 0.02) and v.ANCHOR_V == (0.22, -0.18)
    doc = json.loads((ROOT / v.LAYOUTS_PATH).read_text())
    assert doc == json.loads(json.dumps(v.layouts_document()))
    l0, l1 = doc["layouts"]
    assert (l0["obj_b_xy"], l0["obj_c_xy"]) == ([-0.18, 0.02], [0.22, -0.18])
    assert (l1["obj_b_xy"], l1["obj_c_xy"]) == ([0.22, -0.18], [-0.18, 0.02])
    assert [set(x) for x in doc["layouts"]] == [{"layout_id", "carrier_xy", "obj_b_xy", "obj_c_xy", "receiver_xy", "pad_u_xy",
                                                 "pad_v_xy"}] * 2
    for k in ("carrier_xy", "receiver_xy", "pad_u_xy", "pad_v_xy"):
        assert l0[k] == l1[k]


def test_V02_analytic_point_symmetry_identities():
    inv = v.invariants()
    assert inv["all_ok"] and set(inv["errors"]) == {"U_plus_V_equals_2S", "AU_plus_AV_equals_2S", "S_to_U_minus_S_to_V",
                                                    "U_to_AU_minus_V_to_AV", "U_to_AV_minus_V_to_AU"}
    assert all(e <= 1e-9 for e in inv["errors"].values()) and inv["tolerance"] == 1e-9


def test_V03_geometry_contract_with_real_sizes():
    geo = v.geometry_checks(demonstrated_radius=v.demonstrated_radius(ROOT))
    assert geo["all_ok"], [k for k, x in geo["checks"].items() if not x]
    names = set(geo["checks"])
    for lid in ("layout_0", "layout_1"):
        for need in ("movables_do_not_touch", "movables_clear_of_statics", "statics_do_not_touch", "inside_table_safe_boundary",
                     "receiver_slots_hold_b_and_c", "pads_reachable"):
            assert f"{lid}:{need}" in names
    assert geo["checks"]["destinations_and_anchors_do_not_alias"]


def test_V04_camera_projection_audit_and_no_hidden_pose():
    cam = v.camera_audit()
    assert cam["all_ok"] and cam["rows"] and all(not r["hidden_pose_used_in_runtime"] for r in cam["rows"])
    assert all(r["agentview_px_per_cube_edge"] >= 7.0 and r["inside_margin_8px"]["agentview"] for r in cam["rows"])


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
    assert len(old) >= 28 and not {b["branch_id"] for b in br} & old
    assert all("V3_ISO" in b["attempt_identity"] and b["card_id"] == v.CARD for b in br)
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
    assert geo["layouts_file_matches_frozen_generator"] and "reach" in geo


def test_V12_no_provider_rl_optimizer_entry():
    forbidden = ("provider", "torch", "optim", "representation", "rl", "s2", "s3", "train", "elastic")
    for path in (SRC, ROOT / "scripts/family_b_v3_isochronous.py"):
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
    for support, expect in ((11, False), (12, True)):
        monkeypatch.setattr(v, "evaluate_branch", fake_evaluate(support=support))
        ok, row = v.step_verdict(out, br[0])
        assert ok is expect and row["key_target_depth_support_max"] == support and row["support_margin_vs_12"] == support - 12
    monkeypatch.setattr(v, "evaluate_branch", fake_evaluate(fail_keys={v.ORDER[0]}))
    assert v.step_verdict(out, br[0])[0] is False


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
    _not_established(gate(tmp_path / "6", **{"3": {"key_target_depth_support_max": 11}}), "6_key_target_support_ge_12")
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
    target = os.environ.get("CP_DISR_V3_DIR")
    if target:
        Path(target, "mutation_receipt.json").write_text(json.dumps(_MUT, indent=1, sort_keys=True))


def test_V43_mutation_v_offset_by_1cm_fails_the_fairness_geometry():
    assert v.invariants()["all_ok"]
    shifted = v.layout_specs(pad_v=(0.12, -0.17))
    inv = v.invariants(shifted)
    assert inv["all_ok"] is False
    assert inv["errors"]["U_plus_V_equals_2S"] > 1e-9 and inv["errors"]["S_to_U_minus_S_to_V"] > 1e-9
    assert json.loads(json.dumps(shifted)) != json.loads((ROOT / v.LAYOUTS_PATH).read_text())["layouts"]
    _record("V-43_v_offset_1cm", "pad_v shifted by 1 cm", True, "analytic fairness identities fail")


def test_V44_mutation_layout_1_not_swapped_fails_the_task_feature():
    assert v.task_feature_checks()["layouts_differ_only_by_bc_swap"]
    f = v.task_feature_checks(v.layout_specs(swap_layout_1=False))
    assert f["layouts_differ_only_by_bc_swap"] is False and f["same_object_id_changes_best_pad"] is False
    _record("V-44_layout_1_not_swapped", "layout_1 identical to layout_0", True, "task-feature gate fails")


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
    _record("V-45_order_gate_deleted", "_gate_ok call removed", True, "A2 reservable before A1 passes")


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
    _record("V-46_tolerance_loosened", "TOL_T 0.15 -> 5.0", True, "0.5 s PLACE_BUFFER mismatch must fail wave A")


def _support_blocks(mod, tmp_path, monkeypatch):
    out, br = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(support=11))
    assert mod.step_verdict(out, br[0])[0] is False


def test_V47_mutation_dropping_the_support_margin_is_detected(tmp_path, monkeypatch):
    _support_blocks(v, tmp_path / "prod", monkeypatch)
    mut = _mutant(tmp_path, "mut_support_dropped", "SUPPORT_MIN = 12 ", "SUPPORT_MIN = 8 ")
    with pytest.raises(BaseException) as caught:
        _support_blocks(mut, tmp_path / "mut", monkeypatch)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("V-47_support_margin_dropped", "SUPPORT_MIN 12 -> 8", True, "support 11 must not qualify")


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
    _record("V-48_reversal_forced_true", "conditions 10-13 forced True", True, "always-U data must be NOT_ESTABLISHED")
