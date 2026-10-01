"""R2-01..R2-15: identity, staged dispatch and canary gating for the v2-r2 validation.  CPU only: no MuJoCo
environment, no EGL context, no experiment ledger (every ledger here lives in pytest tmp dirs)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import types
from pathlib import Path

import pytest
import yaml

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2
from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = yaml.safe_load((ROOT / m.CFG_DIR / "config.yaml").read_text())
R2_SRC = ROOT / "src/cp_disr/analysis/family_b_obs_v2_r2.py"
OLD_CARD = "CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1"
_MUT = {}


@pytest.fixture(autouse=True)
def _no_real_simulator(monkeypatch):
    from robosuite.utils import binding_utils

    def boom(*a, **k):
        raise AssertionError("R2 dispatch tests must not construct a MuJoCo sim / EGL render context")
    for attr in ("MjSim", "MjRenderContext"):
        if hasattr(binding_utils, attr):
            monkeypatch.setattr(getattr(binding_utils, attr), "__init__", boom)


# ------------------------------------------------------------------------------ fixtures
def make_out(tmp_path, mod=r2):
    out = tmp_path / "out"
    manifest = out / "spec/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("runtime: {}\n")
    branches = mod.r2_branches(mod.FROZEN_SHA, "c" * 40, "s" * 64, BASE_CFG["contexts"], set(), manifest, {})
    reg = {"card_id": mod.CARD, "branches": branches, "manifest_sha256": m.sha(manifest)}
    phys = out / "physical"
    m.write(phys / "witnesses/e4_branch_registration.json", reg)
    m.write(phys / "registration.json", reg)
    m.write(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": 4, "used": 0}})
    m.write(out / "budget_ledger_r2.json", {
        "attempts": {"cap": 4, "used": 0}, "environment_constructions": {"cap": 4, "used": 0},
        "explicit_resets": {"cap": 4, "used": 0}, "live_skill_calls": {"cap": 24, "used": 0},
        "phases": {"canary": {"cap": 1, "used": 0}, "remainder": {"cap": 3, "used": 0}},
        "reserved_branches": [], "remainder_slots_status": "PENDING_CANARY", "original_remaining_20": "NOT_RELEASED",
        "history": {"v2_r1": {"unused_slots_status": "RETIRED_NOT_TRANSFERABLE"}},
        "actual": {}})
    return out, branches


class FakeProc:
    def __init__(self):
        self.pid = 0

    def poll(self):
        return 0


class FakeLauncher:
    """Stands in for the worker subprocess: claims, writes a terminal result, finishes the attempt."""

    def __init__(self):
        self.calls = []

    def __call__(self, root, out, b, gpu):
        self.calls.append((b["branch_id"], gpu, (b["layout"], b["context"], b["repeat"], b["candidate"])))
        phys = Path(out) / "physical"
        claim_branch_attempt(phys, b["branch_id"])
        acts = [{"action_id": f"a{i}", "controller_exit": "NORMAL_TERMINATION"} for i in range(6)]
        m.write(phys / "branch_results" / f"{b['branch_id']}.json", {
            "branch_id": b["branch_id"], "status": "TASK_SUCCESS", "task_success": True, "actions": acts,
            "env_counts": {"bootstrap_constructions": 0, "constructions": 1, "internal_resets": 2, "reset_calls": 1}})
        finish_branch_attempt(phys, b["branch_id"], "COMPLETED", execution_status="TASK_SUCCESS")
        return FakeProc()


def fake_evaluate(ok):
    def evaluate(out, b, attempts=None):
        checks = {n: ok for names in r2.GROUPS.values() for n in names}
        return {"branch_id": b["branch_id"], "checks": checks, "problems": [] if ok else [f"{b['branch_id']}:forced"]}
    return evaluate


def no_identity(root, out):
    return None


def canary_run(mod, tmp_path, monkeypatch, ok):
    out, branches = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(ok))
    launcher = FakeLauncher()
    res = mod.run_canary(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    return out, branches, launcher, res


def ledger(out):
    return json.loads((Path(out) / "budget_ledger_r2.json").read_text())


# ------------------------------------------------------------------------------------ R2-01/02/04
def test_R2_01_identity_is_r2_everywhere(tmp_path):
    out = tmp_path / "reg"
    res = r2.register(ROOT, out)
    assert res["status"] == "REGISTERED"
    auth = json.loads((out / "authorization_r2.json").read_text())
    assert auth["card_id"] == r2.CARD and "R2" in r2.CARD and auth["original_remaining_20_released"] is False
    assert auth["r1_unused_slots_transferred"] is False
    reg = json.loads((out / "registration_r2.json").read_text())
    assert reg["card_id"] == r2.CARD
    for b in reg["branches"]:
        assert b["card_id"] == r2.CARD and "R2" in b["attempt_identity"] and b["wave"].startswith("obs_v2_r2")
    for rel in ("authorization_r2.json", "registration_r2.json", "budget_ledger_r2.json", "source_identity.json",
                "canary_registration.json", "r2_config_resolved.yaml", "spec/runtime_manifest_T_P_FB_obs_v2_r2.yaml"):
        assert OLD_CARD not in (out / rel).read_text(), rel
    ident = json.loads((out / "source_identity.json").read_text())
    assert ident["repair_base_commit"] == r2.REPAIR_BASE and ident["observation_profile_sha256"] == r2.FROZEN_SHA
    assert set(r2.SOURCE_FILES_R2) <= set(ident["sources"])


def test_R2_02_branch_ids_unique_vs_history(tmp_path):
    out = tmp_path / "reg"
    res = r2.register(ROOT, out)
    ids = set(res["branches"])
    old = {b["branch_id"] for b in json.loads((ROOT / m.OLD_EVIDENCE / "physical/registration.json").read_text())["branches"]}
    r1 = {b["branch_id"] for b in json.loads((ROOT / r2.R1_DIR / "registration_v2.json").read_text())["branches"]}
    assert len(ids) == 4 and not (ids & old) and not (ids & r1)
    seen = set()
    for p in (ROOT / "runs").rglob("registration*.json"):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("branches"), list):
            seen |= {b.get("branch_id") for b in doc["branches"] if isinstance(b, dict)}
    seen -= ids  # the registration under test lives in tmp, never under runs/
    assert not (ids & seen)


def test_R2_04_canary_is_layout0_pad_v(tmp_path):
    out, branches = make_out(tmp_path)
    assert r2.CANARY == ("layout_0", "B_PENDING", 0, "pad_v")
    c = r2.canary_of(branches)
    assert (c["layout"], c["context"], c["repeat"], c["candidate"]) == r2.CANARY and c["phase"] == "canary"
    assert [b["phase"] for b in branches].count("canary") == 1
    assert yaml.safe_load((ROOT / r2.CFG_R2).read_text())["canary"] == {
        "layout": "layout_0", "context": "B_PENDING", "repeat": 0, "candidate": "pad_v"}


# -------------------------------------------------------------------------- canary dispatch R2-03/08
def test_R2_03_run_canary_reserves_and_dispatches_exactly_one(tmp_path, monkeypatch):
    out, branches, launcher, res = canary_run(r2, tmp_path, monkeypatch, True)
    assert len(launcher.calls) == 1 and launcher.calls[0][2] == r2.CANARY
    assert res["dispatched"] == 1 and ledger(out)["reserved_branches"] == [launcher.calls[0][0]]
    assert len(list((out / "physical/branch_results").glob("*.json"))) == 1


def test_R2_08_canary_charges_one_attempt_one_construction_one_reset(tmp_path, monkeypatch):
    out, _, launcher, _ = canary_run(r2, tmp_path, monkeypatch, True)
    led = ledger(out)
    assert led["attempts"]["used"] == led["environment_constructions"]["used"] == led["explicit_resets"]["used"] == 1
    assert led["phases"] == {"canary": {"cap": 1, "used": 1}, "remainder": {"cap": 3, "used": 0}}
    assert led["actual"]["explicit_resets"] == 1 and led["actual"]["constructor_successful"] == 1
    assert led["live_skill_calls"]["used"] == 6
    assert json.loads((out / "physical/budget_ledger.json").read_text())["physical_witness_episodes"]["used"] == 1


# ------------------------------------------------------------------------ remainder R2-05/06/07/09
def remainder_scenario(mod, tmp_path, monkeypatch, canary_ok):
    out, branches, launcher, _ = canary_run(mod, tmp_path, monkeypatch, canary_ok)
    before = (out / "budget_ledger_r2.json").read_bytes()
    n_calls = len(launcher.calls)
    return out, branches, launcher, before, n_calls


def fail_blocks_remainder(mod, tmp_path, monkeypatch):
    out, branches, launcher, before, n_calls = remainder_scenario(mod, tmp_path, monkeypatch, False)
    with pytest.raises(ValueError, match="STOPPED_CANARY_NOT_PASS"):
        mod.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    assert len(launcher.calls) == n_calls == 1
    after = json.loads((out / "budget_ledger_r2.json").read_text())
    assert after["attempts"]["used"] == 1 and after["phases"]["remainder"]["used"] == 0
    assert after["remainder_slots_status"] == "NOT_RELEASED_AFTER_CANARY_FAILURE"
    assert json.loads((out / "canary_gate.json").read_text())["status"] == "CANARY_FAIL"


def test_R2_05_failed_canary_blocks_remainder_without_spending(tmp_path, monkeypatch):
    fail_blocks_remainder(r2, tmp_path, monkeypatch)


def test_R2_06_passed_canary_releases_exactly_three_others(tmp_path, monkeypatch):
    out, branches, launcher, before, n_calls = remainder_scenario(r2, tmp_path, monkeypatch, True)
    res = r2.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    rest = [c[2] for c in launcher.calls[1:]]
    assert sorted(rest) == sorted(r2.REMAINDER) and r2.CANARY not in rest and len(launcher.calls) == 4
    led = ledger(out)
    assert led["attempts"]["used"] == 4 and led["phases"]["remainder"]["used"] == 3 and not res["faults"]
    assert len(set(led["reserved_branches"])) == 4


def test_R2_07_no_branch_is_dispatched_twice(tmp_path, monkeypatch):
    out, branches, launcher, before, n_calls = remainder_scenario(r2, tmp_path, monkeypatch, True)
    r2.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    with pytest.raises(ValueError):
        r2.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    with pytest.raises(ValueError, match="STOPPED_CANARY_ALREADY_RUN"):
        r2.run_canary(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    with pytest.raises(ValueError, match="STOPPED_DUPLICATE_ATTEMPT"):
        r2.reserve_and_charge(out, branches[0], branches[0]["phase"])
    ids = [c[0] for c in launcher.calls]
    assert len(ids) == len(set(ids)) == 4


def test_R2_09_remainder_phase_cannot_exceed_three(tmp_path, monkeypatch):
    out, branches, launcher, before, n_calls = remainder_scenario(r2, tmp_path, monkeypatch, True)
    r2.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    extra = dict(branches[0], branch_id="x" * 16, phase="remainder")
    with pytest.raises(ValueError, match="STOPPED_BUDGET_EXHAUSTED"):
        r2.reserve_and_charge(out, extra, "remainder")
    assert ledger(out)["attempts"]["used"] == 4


# ---------------------------------------------------------------------------- GPU policy R2-10/11
def test_R2_10_and_11_gpu_queues(tmp_path):
    _, branches = make_out(tmp_path)
    rest = [b for b in branches if r2.branch_key(b) in r2.REMAINDER]
    q = r2.remainder_queues(rest, 3, [3, 4])
    assert [r2.branch_key(b) for b in q[3]] == [("layout_0", "B_PENDING", 0, "pad_u")]
    assert [r2.branch_key(b) for b in q[4]] == [("layout_1", "C_PENDING", 0, "pad_v"),
                                                ("layout_1", "C_PENDING", 0, "pad_u")]
    solo = r2.remainder_queues(rest, 3, [3])
    assert list(solo) == [3] and [r2.branch_key(b) for b in solo[3]] == list(r2.REMAINDER)
    assert list(r2.remainder_queues(rest, 4, [3, 4])) == [4, 3]


def test_R2_10_dispatch_uses_canary_gpu_for_layout0_pad_u(tmp_path, monkeypatch):
    out, branches, launcher, before, n_calls = remainder_scenario(r2, tmp_path, monkeypatch, True)
    canary_gpu = launcher.calls[0][1]
    r2.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    by = {c[2]: c[1] for c in launcher.calls}
    assert by[("layout_0", "B_PENDING", 0, "pad_u")] == canary_gpu == by[r2.CANARY]
    assert by[("layout_1", "C_PENDING", 0, "pad_v")] == by[("layout_1", "C_PENDING", 0, "pad_u")] != canary_gpu


# ------------------------------------------------------------------------------ R2-12 / R2-15
def test_R2_12_dispatch_tests_do_not_touch_a_simulator_or_real_ledger():
    from robosuite.utils import binding_utils
    with pytest.raises(AssertionError):
        binding_utils.MjSim(None)
    text = Path(__file__).read_text()
    forbidden = ["make_family_b_obs_v2_env" + "(", "robosuite." + "make(", "default_" + "launcher(",
                 "family_b_observation_v2" + "_r2/2"]
    assert not [f for f in forbidden if f in text]


def test_R2_15_frozen_profile_files_byte_identical():
    for rel in r2.FROZEN_FILES:
        want = subprocess.run(["git", "-C", str(ROOT), "show", f"{r2.PROFILE_ORIGIN}:{rel}"], capture_output=True,
                              check=True).stdout
        assert (ROOT / rel).read_bytes() == want, rel
    assert not r2.frozen_bytes_ok(ROOT)
    assert json.loads((ROOT / r2.FROZEN_FILES[0]).read_text())["profile_sha256"] == r2.FROZEN_SHA
    assert yaml.safe_load((ROOT / r2.CFG_R2).read_text())["observation_profile_sha256"] == r2.FROZEN_SHA


# ------------------------------------------------------------------------------------ R2-13/14
def _mutant(tmp_path, name, old, new):
    src = R2_SRC.read_text()
    assert src.count(old) == 1, name
    path = tmp_path / f"{name}.py"
    path.write_text(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(f"cp_disr.analysis.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE_BLOCK = '''    gate = check_canary(root, out, write_file=False)
    recorded = read(out / "canary_gate.json") if (out / "canary_gate.json").is_file() else {}
    if gate["status"] != PASS_CANARY or recorded.get("status") != PASS_CANARY:
        raise ValueError("STOPPED_CANARY_NOT_PASS")
'''
GATE_IF = 'if gate["status"] != PASS_CANARY or recorded.get("status") != PASS_CANARY:'


def _record(name, desc, caught):
    _MUT[name] = {"mutation": desc, "killed": caught, "scenario": "failed canary must block run_remainder"}
    target = os.environ.get("CP_DISR_R2_DIR")
    if target:
        Path(target, "mutation_receipt.json").write_text(json.dumps(_MUT, indent=1, sort_keys=True, default=str))


def test_R2_13_deleting_the_canary_gate_check_is_detected(tmp_path, monkeypatch):
    fail_blocks_remainder(r2, tmp_path / "prod", monkeypatch)  # production passes the scenario
    mut = _mutant(tmp_path, "mut_gate_check_deleted", GATE_BLOCK, "    pass\n")
    with pytest.raises(BaseException) as caught:
        fail_blocks_remainder(mut, tmp_path / "mut", monkeypatch)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("R2-13_gate_check_deleted", "canary gate check removed from run_remainder", True)


def test_R2_14_remainder_dispatch_after_canary_failure_is_detected(tmp_path, monkeypatch):
    mut = _mutant(tmp_path, "mut_dispatch_after_failure", GATE_IF, "if False:")
    with pytest.raises(BaseException):
        fail_blocks_remainder(mut, tmp_path / "mut", monkeypatch)
    out, branches, launcher, before, n_calls = remainder_scenario(mut, tmp_path / "probe", monkeypatch, False)
    mut.run_remainder(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    assert len(launcher.calls) == 4  # the mutant really dispatches the remainder after a FAILED canary
    _record("R2-14_dispatch_after_failure", "run_remainder proceeds although the canary gate is not PASS", True)


# --------------------------------------------------------------- extra guards for the worker path (offline)
def test_R2_16_worker_refuses_a_branch_the_coordinator_did_not_reserve(tmp_path):
    out, branches = make_out(tmp_path)
    with pytest.raises(RuntimeError, match="not reserved by the coordinator"):
        r2.worker(ROOT, out, branches[0]["branch_id"])
    assert ledger(out)["attempts"]["used"] == 0


def test_R2_17_camera_record_keeps_numeric_live_vs_frozen_values():
    import numpy as np
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    profile = json.loads((ROOT / m.PROFILE_PATH).read_text())
    refs = profile["camera_manifest"]
    names = list(obs.CAMERAS)

    def env(shift=0.0):
        model = types.SimpleNamespace(
            camera_name2id=lambda n: names.index(n),
            cam_mode=np.array([refs[c]["mode"] for c in names]), cam_bodyid=np.zeros(2, dtype=int),
            cam_pos=np.array([refs[c]["pos"] for c in names], dtype=float) + shift,
            cam_quat=np.array([refs[c]["quat"] for c in names], dtype=float),
            cam_fovy=np.array([refs[c]["fovy"] for c in names], dtype=float))
        return types.SimpleNamespace(sim=types.SimpleNamespace(model=model), camera_names=names,
                                     camera_widths=[128, 128], camera_heights=[128, 128])
    ok = r2._camera_record(env(), profile)
    assert ok["status"] == "PASS" and ok["fallback_used"] is False and ok["live_values_written_back"] is False
    for c in names:
        row = ok["cameras"][c]
        assert row["frozen"]["pos"] == refs[c]["pos"] and row["live"]["pos"] and row["tolerance"] == 1e-9
        assert row["abs_diff"]["pos_max"] == 0.0 and row["pass"] is True
    bad = r2._camera_record(env(1e-6), profile)
    assert bad["status"] == "FAIL" and bad["cameras"]["agentview"]["abs_diff"]["pos_max"] > 1e-9
