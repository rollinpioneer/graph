"""S-01..S-20: identity, staged dispatch, Gate M, combined analysis and mechanism gate for the staging continuation.
CPU only: no MuJoCo environment, no EGL context; every ledger lives in pytest tmp dirs."""
from __future__ import annotations

import difflib
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2
from cp_disr.analysis import family_b_staging_v2 as s
from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / s.CFG).read_text())
SRC = ROOT / "src/cp_disr/analysis/family_b_staging_v2.py"
_MUT = {}


@pytest.fixture(autouse=True)
def _no_real_simulator(monkeypatch):
    from robosuite.utils import binding_utils

    def boom(*a, **k):
        raise AssertionError("staging tests must not construct a MuJoCo sim / EGL render context")
    for attr in ("MjSim", "MjRenderContext"):
        if hasattr(binding_utils, attr):
            monkeypatch.setattr(getattr(binding_utils, attr), "__init__", boom)


# ------------------------------------------------------------------------------------ fixtures
def make_out(tmp_path, mod=s):
    out = tmp_path / "out"
    manifest = out / "spec/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("runtime: {}\n")
    prefixes = {k: v["prefix"] for k, v in CFG["contexts"].items()}
    branches = mod.build_branches(s.FROZEN_SHA, "c" * 40, "s" * 64, prefixes, set(), manifest, "f" * 16)
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
        self.calls.append((b["branch_id"], gpu, s.branch_key(b), b["restore_seed"]))
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


def fake_evaluate(fail_keys=()):
    def evaluate(out, b, attempts=None):
        ok = s.branch_key(b) not in fail_keys
        return {"branch_id": b["branch_id"], "complete": ok, "task_success": ok, "checks": {}, "layout": b["layout"],
                "problems": [] if ok else [f"{b['branch_id']}:forced"]}
    return evaluate


def no_identity(root, out):
    return None


def ledger(out):
    return json.loads((Path(out) / "budget_ledger.json").read_text())


def comp_run(mod, tmp_path, monkeypatch, fail_keys=(), gpus=(3, 4)):
    out, branches = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate(fail_keys))
    launcher = FakeLauncher()
    res = mod.run_complementary(ROOT, out, list(gpus), launcher=launcher, identity_check=no_identity)
    return out, branches, launcher, res


def by_key(branches):
    return {s.branch_key(b): b for b in branches}


# ------------------------------------------------------------------------------- S-01..S-04
def test_S01_twenty_logical_branches_complete_and_unique(tmp_path):
    out, branches = make_out(tmp_path)
    want = {("layout_0", "B_PENDING", 1), ("layout_0", "C_PENDING", 0), ("layout_0", "C_PENDING", 1),
            ("layout_0", "BOTH_PENDING", 0), ("layout_0", "BOTH_PENDING", 1),
            ("layout_1", "B_PENDING", 0), ("layout_1", "B_PENDING", 1), ("layout_1", "C_PENDING", 1),
            ("layout_1", "BOTH_PENDING", 0), ("layout_1", "BOTH_PENDING", 1)}
    keys = {s.branch_key(b) for b in branches}
    assert len(branches) == 20 and len(keys) == 20
    assert {k[:3] for k in keys} == want and all(sorted(k[3] for k in keys if k[:3] == w) == ["pad_u", "pad_v"] for w in want)
    assert not keys & set(s.R2_FOUR) and len(s.ALL24) == 24 and set(s.ALL24) == keys | set(s.R2_FOUR)
    assert s.check_uniform({"branches": branches}) == []
    assert {k for k in keys if k in s.COMPLEMENTARY} == set(s.COMPLEMENTARY) and len(s.REMAINING16) == 16
    prefixes = {b["context"]: b["prefix"] for b in branches}
    assert prefixes["BOTH_PENDING"] == ["a:PICK:carrier:v1"] and len(prefixes["B_PENDING"]) == 3
    assert sum(len(b["prefix"]) + 1 + 2 * len(s.REMAINING_OBJ[b["context"]]) for b in branches) == 20 * 6


def test_S02_no_id_collision_with_v1_r1_r2(tmp_path):
    out, branches = make_out(tmp_path)
    ids = {b["branch_id"] for b in branches}
    old = {b["branch_id"] for b in json.loads((ROOT / m.OLD_EVIDENCE / "physical/registration.json").read_text())["branches"]}
    r1 = {b["branch_id"] for b in json.loads((ROOT / r2.R1_DIR / "registration_v2.json").read_text())["branches"]}
    r2ids = {b["branch_id"] for b in s.r2_branches(ROOT).values()}
    assert len(ids) == 20 and not ids & (old | r1 | r2ids) and not ids & s.historical_ids(ROOT)
    assert all(s.CARD in json.dumps(b) and "STAGING_V2_CONT" in b["attempt_identity"] for b in branches)
    with pytest.raises(ValueError, match="STOPPED_SOURCE_IDENTITY"):
        make_collision = next(iter(ids))
        s.build_branches(s.FROZEN_SHA, "c" * 40, "s" * 64, {k: v["prefix"] for k, v in CFG["contexts"].items()},
                         {make_collision}, tmp_path / "x.yaml", "f" * 16)


def test_S03_r2_four_are_read_only_references(tmp_path, monkeypatch):
    before = m.inventory([ROOT / s.R2_DIR])
    ref = s.r2_reference_manifest(ROOT)
    assert len(ref["branches"]) == 4 and {b["role"] for b in ref["branches"]} == {"V2_TECHNICAL_INPUT_FROZEN"}
    for b in ref["branches"]:
        assert m.sha(ROOT / b["result_path"]) == b["result_sha256"]
    out, branches = make_out(tmp_path)
    reg = s.combined_registry(ROOT, out)
    assert len(reg) == 24 and all(reg[k][1] == ROOT / s.R2_DIR for k in s.R2_FOUR)
    assert all(reg[k][1] == out for k in s.NEW20)
    assert m.inventory([ROOT / s.R2_DIR]) == before


def test_S04_incompatibility_blocks_merging(tmp_path, monkeypatch):
    real = s.compatibility_manifest(ROOT, tmp_path / "compat")
    assert real["status"] == "COMPATIBLE", real["incompatible_components"]
    monkeypatch.setattr(s, "_git_blob", lambda root, commit, rel: b"different")
    bad = s.compatibility_manifest(ROOT, tmp_path / "compat_bad")
    assert bad["status"] == "STOPPED_RUNTIME_PROFILE_INCOMPATIBLE" and bad["incompatible_components"]
    monkeypatch.undo()
    monkeypatch.setattr(s, "compatibility_manifest", lambda root, out=None: {"status": "STOPPED_RUNTIME_PROFILE_INCOMPATIBLE",
                                                                             "incompatible_components": ["env"]})
    out = tmp_path / "reg"
    with pytest.raises(ValueError, match="STOPPED_RUNTIME_PROFILE_INCOMPATIBLE"):
        s.register(ROOT, out)
    assert not (out / "physical/registration.json").exists() and not (out / "combined_v2_24_manifest.json").exists()


# ------------------------------------------------------------------------------------ S-05 Gate M
def test_S05_gate_m_builds_no_environment_and_reports_margins(tmp_path, monkeypatch):
    from cp_disr.platforms.libero import family_b_obs_v2 as obs

    def boom(*a, **k):
        raise AssertionError("Gate M must not construct an environment")
    monkeypatch.setattr(obs, "make_family_b_obs_v2_env", boom)
    monkeypatch.setattr(obs.FamilyBObsV2Env, "__init__", boom)
    before = m.inventory([ROOT / s.R2_DIR])
    review = s.gate_m(ROOT, tmp_path)
    assert review["status"] == s.GATE_M_OK and review["new_samples"] == 0 and review["environments_constructed"] == 0
    assert review["replay"]["reproduces_saved_records"] and not review["replay"]["mismatches"]
    for name in ("observation_margin_review.json", "observation_margin_review.md", "observation_coverage_matrix.csv"):
        assert (tmp_path / name).is_file()
    st = review["statements"]
    assert st["r2_pass_revoked"] is False and st["untested_complementary_positions_inferred_from_r2"] is False
    assert st["min_pixels_lowered"] is False and st["palette_changed"] is False and st["carry_over"] is False
    side = [x for x in st["tested_pad_v_remaining_target_sideview_support"]]
    assert side and all(x["support"] == 8 and x["margin_to_min_pixels"] == 0 for x in side)
    assert review["margin_zero_cells"] and review["status"] == s.GATE_M_OK  # margin 0 alone does not stop
    assert st["margin_zero_alone_stops_the_run"] is False and st["margin_zero_forces_complementary_pad_v_canaries_first"]
    assert m.inventory([ROOT / s.R2_DIR]) == before


def test_S05b_gate_m_stops_when_saved_evidence_incomplete(tmp_path):
    fake = tmp_path / "r2"
    fake.mkdir()
    (fake / "registration_r2.json").write_text((ROOT / s.R2_DIR / "registration_r2.json").read_text())
    with pytest.raises(ValueError, match=s.GATE_M_STOP):
        s.gate_m(ROOT, tmp_path / "o", r2_dir=str(fake))
    assert json.loads((tmp_path / "o/observation_margin_review.json").read_text())["status"] == s.GATE_M_STOP


# ------------------------------------------------------------------------------ S-06..S-10 dispatch
def attempt_stage_blocked(out, branches, steps):
    k = by_key(branches)
    for step in steps:
        with pytest.raises(ValueError, match="STOPPED_COMPLEMENTARY"):
            s.reserve_and_charge(out, k[s.COMPLEMENTARY[step]], "complementary")


def scenario_a_fails(mod, tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(mod, tmp_path, monkeypatch, fail_keys={s.COMPLEMENTARY[0]})
    assert len(launcher.calls) == 1 and launcher.calls[0][2] == s.COMPLEMENTARY[0]
    assert res["gate"] == s.COMP_FAIL and ledger(out)["attempts"]["used"] == 1
    attempt_stage_blocked(out, branches, (1, 2, 3))
    assert ledger(out)["attempts"]["used"] == 1 and ledger(out)["remaining16_released"] is False
    assert json.loads((out / "complementary_canary_gate.json").read_text())["status"] == s.COMP_FAIL


def test_S06_a_failure_blocks_b_c_d(tmp_path, monkeypatch):
    scenario_a_fails(s, tmp_path, monkeypatch)


def test_S06b_b_failure_blocks_c_d(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch, fail_keys={s.COMPLEMENTARY[1]})
    assert [c[2] for c in launcher.calls] == list(s.COMPLEMENTARY[:2]) and res["failed_step"] == 1
    attempt_stage_blocked(out, branches, (2, 3))
    assert ledger(out)["attempts"]["used"] == 2 and ledger(out)["remaining16_released"] is False


def test_S06c_staged_order_and_gpu_sharing(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch)
    keys = [c[2] for c in launcher.calls]
    assert keys[:2] == list(s.COMPLEMENTARY[:2]) and set(keys[2:]) == set(s.COMPLEMENTARY[2:])
    gpu = {c[2]: c[1] for c in launcher.calls}
    assert gpu[s.COMPLEMENTARY[0]] == gpu[s.COMPLEMENTARY[2]] != gpu[s.COMPLEMENTARY[1]] == gpu[s.COMPLEMENTARY[3]]
    assert res["gate"] == s.COMP_PASS and ledger(out)["attempts"]["used"] == 4


def test_S07_complementary_pass_releases_only_the_remaining_16(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch)
    assert res["gate"] == s.COMP_PASS and ledger(out)["remaining16_released"] is True
    n = len(launcher.calls)
    rem = s.run_remaining16(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    new = launcher.calls[n:]
    assert len(new) == 16 and {c[2] for c in new} == set(s.REMAINING16)
    assert not {c[2] for c in new} & set(s.COMPLEMENTARY) and not {c[2] for c in new} & set(s.R2_FOUR)
    led = ledger(out)
    assert led["attempts"]["used"] == 20 and led["phases"]["remaining16"]["used"] == 16
    assert led["explicit_resets"]["used"] == led["environment_constructions"]["used"] == 20
    assert led["live_skill_calls"]["used"] == 120 and led["actual"]["skill_calls"] == 120
    assert rem["faults"] == [] and rem["stop_reason"] is None
    with pytest.raises(ValueError, match="STOPPED_BUDGET_EXHAUSTED|STOPPED_DUPLICATE_ATTEMPT"):
        s.reserve_and_charge(out, by_key(branches)[s.REMAINING16[0]], "remaining16")


def test_S08_complementary_failure_blocks_remaining_16(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch, fail_keys={s.COMPLEMENTARY[0]})
    n = len(launcher.calls)
    with pytest.raises(ValueError, match="STOPPED_REMAINING16_NOT_RELEASED"):
        s.run_remaining16(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    assert len(launcher.calls) == n
    with pytest.raises(ValueError, match="STOPPED_REMAINING16_NOT_RELEASED"):
        s.reserve_and_charge(out, by_key(branches)[s.REMAINING16[0]], "remaining16")
    assert ledger(out)["phases"]["remaining16"]["used"] == 0


def test_S09_pairs_same_seed_same_gpu_sequential_canonical(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch)
    n = len(launcher.calls)
    launcher.polls = 2
    s.run_remaining16(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    calls = launcher.calls[n:]
    assert launcher.max_active <= 2
    pos = {c[2]: i for i, c in enumerate(calls)}
    for pair in s.REMAINING16_PAIRS:
        u, v = pair
        assert u[3] == "pad_u" and v[3] == "pad_v" and pos[u] < pos[v]
        cu, cv = calls[pos[u]], calls[pos[v]]
        assert cu[1] == cv[1] and cu[3] == cv[3]
        assert launcher.finished.index(cu[0]) < pos_of_launch_after(launcher, cv[0], calls)
    firsts = [c[2][:3] for c in calls if c[2][3] == "pad_u"]
    assert firsts == [p[0][:3] for p in s.REMAINING16_PAIRS]
    order = [(s.LAYOUTS.index(p[0][0]), s.CONTEXTS.index(p[0][1]), p[0][2]) for p in s.REMAINING16_PAIRS]
    assert order == sorted(order)


def pos_of_launch_after(launcher, bid, calls):
    # the finished list records U before V launches because the coordinator only starts V after polling U to completion
    return launcher.finished.index(bid) if bid in launcher.finished else 10 ** 6


def test_S10_caps_20_20_20_120_and_zero_other(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch)
    s.run_remaining16(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    led = ledger(out)
    assert (led["attempts"]["cap"], led["environment_constructions"]["cap"], led["explicit_resets"]["cap"],
            led["live_skill_calls"]["cap"]) == (20, 20, 20, 120)
    assert all(led[k]["cap"] == 0 and led[k]["used"] == 0 for k in
               ("skill_retries", "standalone_capture_resets", "provider_calls", "representation_forwards", "rl_transitions",
                "optimizer_steps", "elastic", "formal_test"))
    assert len(list((out / "physical/branch_results").glob("*.json"))) == 20
    led["attempts"]["used"] = 20
    (out / "budget_ledger.json").write_text(json.dumps(led))
    with pytest.raises(ValueError, match="STOPPED_BUDGET_EXHAUSTED|STOPPED_DUPLICATE_ATTEMPT"):
        s.reserve_and_charge(out, branches[0], branches[0]["phase"])
    assert json.loads((out / "physical/budget_ledger.json").read_text())["physical_witness_episodes"]["cap"] == 20


def test_S10b_two_consecutive_common_exceptions_stop_new_dispatch(tmp_path, monkeypatch):
    out, branches, launcher, res = comp_run(s, tmp_path, monkeypatch)

    class Boom(FakeLauncher):
        def __call__(self, root, out, b, gpu):
            proc = super().__call__(root, out, b, gpu)
            p = Path(out) / "physical/branch_results" / f"{b['branch_id']}.json"
            d = json.loads(p.read_text())
            d.update(status="EXCEPTION", detail="RuntimeError:same")
            p.write_text(json.dumps(d))
            return proc
    boom = Boom()
    rem = s.run_remaining16(ROOT, out, [3, 4], launcher=boom, identity_check=no_identity)
    assert rem["stop_reason"] == "TWO_CONSECUTIVE_COMMON_ENGINEERING_EXCEPTIONS"
    assert len(boom.calls) < 16 and ledger(out)["attempts"]["used"] < 20
    thr = json.loads((out / "physical/throughput_remaining16.json").read_text())
    assert thr["not_dispatched"] and thr["speedup_claim"].startswith("NONE")


# ------------------------------------------------------------------------- S-11..S-14 mechanism gate
def synth(costs=None, fail=(), both=None, seq_override=None, tol_cost=None):
    """24 rows; cost[(layout, context)] = (cost_u, cost_v) per repeat 0/1 unless a dict by repeat is given."""
    base = {("layout_0", "B_PENDING"): (10.0, 10.4), ("layout_0", "C_PENDING"): (10.4, 10.0),
            ("layout_1", "B_PENDING"): (10.0, 10.4), ("layout_1", "C_PENDING"): (10.4, 10.0)}
    base.update(costs or {})
    rows = []
    for layout, context, repeat, pad in s.ALL24:
        n = s.PREFIX_LEN[context]
        rem = s.REMAINING_OBJ[context]
        tail = [x for o in rem for x in (f"a:PICK:{o}:v1", f"a:PLACE:{o}:receiver:v1")]
        seq = ["p"] * n + [f"a:PLACE_BUFFER:carrier:{pad}:v1"] + tail
        seq = [x if ":" in x else f"a:PICK:s{i}:v1" for i, x in enumerate(seq)]
        if (layout, context, repeat, pad) in (seq_override or ()):
            seq = seq[:-1] + ["a:PICK:other:v1"]
        if context == "BOTH_PENDING":
            c = (both or {}).get((layout, repeat, pad), 12.0 + (0.5 if pad == "pad_u" else 0.0))
        else:
            cu, cv = base[(layout, context)]
            if isinstance(cu, dict):
                cu, cv = cu[repeat], cv[repeat]
            c = cu if pad == "pad_u" else cv
        ok = (layout, context, repeat, pad) not in fail
        rows.append({"layout": layout, "context": context, "repeat": repeat, "candidate": pad, "present": True,
                     "branch_id": f"{layout}{context}{repeat}{pad}", "source": "X", "task_success": ok, "complete": ok,
                     "cost": c if ok else None, "sequence": seq, "engineering_exception": None,
                     "both_pads_legal": True, "problems": []})
    return rows


def test_S11_failed_branch_has_null_cost_and_stays_in_denominators():
    rows = synth(fail={("layout_0", "BOTH_PENDING", 0, "pad_v")})
    bad = next(r for r in rows if not r["task_success"])
    assert bad["cost"] is None and len(rows) == 24
    g = s.mechanism_gate(rows, True)
    assert g["status"] == "UNRESOLVED_ENGINEERING" and g["conditions"]["1_24_of_24_task_success"] is False
    assert g["next_action"] != "REQUEST_FAMILY_B_PROVIDER_AND_REPRESENTATION_AUTHORIZATION"
    pair = next(p for p in g["pairs"] if (p["layout"], p["context"], p["repeat"]) == ("layout_0", "BOTH_PENDING", 0))
    assert pair["delta_u_minus_v"] is None and pair["both_success"] is False


def test_S12_different_sequences_are_not_cost_evidence():
    rows = synth(seq_override={("layout_0", "B_PENDING", 0, "pad_u")})
    g = s.mechanism_gate(rows, True)
    pair = next(p for p in g["pairs"] if (p["layout"], p["context"], p["repeat"]) == ("layout_0", "B_PENDING", 0))
    assert pair["comparable"] is False and pair["delta_u_minus_v"] is None and pair["tails_equal"] is False
    assert g["status"] != "ESTABLISHED" and g["conditions"]["10_plan_length_skill_multiset_and_tail_match"] is False


def test_S13_gate_requires_success_reversal_threshold_and_repeats():
    ok = s.mechanism_gate(synth(), True)
    assert ok["status"] == "ESTABLISHED" and ok["next_action"] == "REQUEST_FAMILY_B_PROVIDER_AND_REPRESENTATION_AUTHORIZATION"
    assert all(ok["conditions"].values()) and ok["claims"]["significance"] is False
    assert s.mechanism_gate(synth(fail={("layout_1", "C_PENDING", 1, "pad_v")}), True)["status"] == "UNRESOLVED_ENGINEERING"
    assert s.mechanism_gate(synth(), False)["status"] == "UNRESOLVED_ENGINEERING"
    no_reversal = synth({("layout_0", "C_PENDING"): (10.0, 10.4), ("layout_1", "C_PENDING"): (10.0, 10.4)})
    g = s.mechanism_gate(no_reversal, True)
    assert g["status"] == "NOT_ESTABLISHED" and g["next_action"] == "S4_RESEARCH_DECISION"
    assert g["conditions"]["5_sign_reversal_within_layout"] is False and g["winner_set"] == ["pad_u"]
    small = synth({("layout_0", "B_PENDING"): (10.0, 10.1)})
    assert s.mechanism_gate(small, True)["status"] == "NOT_ESTABLISHED"
    edge = synth({("layout_0", "B_PENDING"): (10.0, 10.25)})  # exactly representable: |delta| == tol is not above it
    assert s.mechanism_gate(edge, True, tol=0.25)["conditions"]["6_each_abs_delta_above_threshold"] is False
    assert s.TOL == 0.15
    disagree = synth({("layout_0", "B_PENDING"): ({0: 10.0, 1: 10.5}, {0: 10.4, 1: 10.0})})
    gd = s.mechanism_gate(disagree, True)
    assert gd["status"] == "NOT_ESTABLISHED" and gd["conditions"]["7_repeats_agree_in_direction"] is False


def test_S14_both_pending_is_reported_not_hard_coded_neutral():
    both = {(l, r, p): (13.0 if p == "pad_u" else 12.0) for l in s.LAYOUTS for r in s.REPEATS for p in s.PADS}
    g = s.mechanism_gate(synth(both=both), True)
    assert g["status"] == "ESTABLISHED" and len(g["both_pending_reported_only"]) == 4
    assert all(p["delta_u_minus_v"] == pytest.approx(1.0) for p in g["both_pending_reported_only"])
    assert g["conditions"]["13_both_not_required_neutral"] is True
    incomplete = synth(fail={("layout_1", "BOTH_PENDING", 1, "pad_u")})
    assert s.mechanism_gate(incomplete, True)["conditions"]["12_both_pending_complete_and_reported"] is False
    text = SRC.read_text()
    assert "neutral" not in text.split("def mechanism_gate")[1].split("def _csv")[0].replace("13_both_not_required_neutral", "")


def test_S15_nothing_downstream_can_autostart():
    text = SRC.read_text() + (ROOT / "scripts/family_b_staging_v2.py").read_text()
    for forbidden in ("import torch", "optimizer.step", "provider.", "from cp_disr.providers", "representation(",
                      "train(", "family_b_provider"):
        assert forbidden not in text, forbidden
    cfg = CFG["authorization"]
    assert cfg["provider"] == cfg["representation_forwards"] == cfg["rl_transitions"] == cfg["optimizer_steps"] == 0
    assert cfg["formal_test"] == 0 and cfg["s2"] is False and cfg["s3"] is False and cfg["tp_training"] is False
    g = s.mechanism_gate(synth(), True)
    assert g["claims"]["provider_or_representation_started"] is False
    assert "REQUEST" in g["next_action"]  # a request for authorization, never an execution


def test_S16_frozen_profile_files_byte_identical():
    for rel in s.FROZEN_FILES:
        want = subprocess.run(["git", "-C", str(ROOT), "show", f"{r2.PROFILE_ORIGIN}:{rel}"], capture_output=True,
                              check=True).stdout
        assert (ROOT / rel).read_bytes() == want, rel
    assert not r2.frozen_bytes_ok(ROOT) and not s.frozen_changed(ROOT)
    assert json.loads((ROOT / s.FROZEN_FILES[0]).read_text())["profile_sha256"] == s.FROZEN_SHA
    assert CFG["observation_profile_sha256"] == s.FROZEN_SHA


def test_S17_modifying_old_raw_evidence_fails_protection(tmp_path, monkeypatch):
    old = tmp_path / "old/captures/b1"
    old.mkdir(parents=True)
    (old / "x.npy").write_bytes(b"raw")
    monkeypatch.setattr(s, "protected_paths", lambda root: [tmp_path / "old"])
    out, branches = make_out(tmp_path)
    s.inventory_before(ROOT, out)
    assert s.verify(ROOT, out)["checks"]["protected_unchanged"] is True
    (old / "x.npy").write_bytes(b"tampered")
    assert s.verify(ROOT, out)["checks"]["protected_unchanged"] is False
    assert s.verify(ROOT, out)["status"] == "FAIL"


# ------------------------------------------------------------------ S-18/19 worker and pair identity
def test_S18_worker_differs_from_v2_worker_only_in_the_both_mapping():
    old = inspect.getsource(m.worker).splitlines()
    new = s.worker_source().splitlines()
    diff = [d for d in difflib.unified_diff(old, new, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    assert len(diff) == 4, diff  # def line and the mapping line, one removed + one added each
    assert any("DONE_BY_CONTEXT" in d for d in diff) and s.DONE["BOTH_PENDING"] == ()
    with pytest.raises(KeyError):
        {"B_PENDING": ("obj_c",), "C_PENDING": ("obj_b",)}["BOTH_PENDING"]
    assert callable(s.build_worker()) and callable(s.build_paired_restore())


def test_S19_both_pending_generalised_semantics(tmp_path):
    tails = s.expected_tail("BOTH_PENDING")
    assert len(tails) == 2 and all(len(t) == 4 for t in tails) and tails[0] != tails[1]
    assert s.expected_tail("B_PENDING") == [["a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1"]]
    cap = tmp_path / "cap"
    acts = [{"action_id": f"a{i}"} for i in range(3)]
    for i in range(3):
        d = cap / f"action_{i:02d}"
        d.mkdir(parents=True)
    (cap / "action_01/planner.json").write_text(json.dumps({"status": "PLAN_FOUND", "plan": ["a2"]}))
    ev = {0: {"terminated": False, "task_success": False}, 1: {"terminated": False, "task_success": False},
          2: {"terminated": True, "task_success": True}}
    assert s.planner_semantics(cap, acts, ev, 1) == []          # setup action 0 skipped, 1 has a planner record
    assert s.planner_semantics(cap, acts, ev, 3) == []          # everything is setup
    (cap / "action_02/planner.json").write_text("{}")
    assert any("planner called after TASK_SUCCESS" in p for p in s.planner_semantics(cap, acts, ev, 1))
    assert s.PREFIX_LEN == {"B_PENDING": 3, "C_PENDING": 3, "BOTH_PENDING": 1}


def test_S20_pair_restore_runs_on_all_twelve_pairs(tmp_path):
    keys = {s.pair_key(k) for k in s.ALL24}
    assert len(keys) == 12 and len(s.ALL24) == 24
    assert len(s.pair_list(set(s.REMAINING16))) == 8
    assert all([k[3] for k in p] == ["pad_u", "pad_v"] for p in s.REMAINING16_PAIRS)


# ----------------------------------------------------------------------------------- mutations
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
    target = os.environ.get("CP_DISR_STAGING_DIR")
    if target:
        Path(target, "mutation_receipt.json").write_text(json.dumps(_MUT, indent=1, sort_keys=True))


STEP_GATE_DOC = '"""Raises if the staged order or a recorded failure forbids dispatching this branch."""\n'
STEP_VERDICT_OK = '    ok = row["complete"] and row.get("task_success") is True and not row["problems"]\n    return ok, row\n'


def _b_after_a_failure_blocked(mod, tmp_path, monkeypatch):
    out, branches = make_out(tmp_path, mod)
    monkeypatch.setattr(mod, "evaluate_branch", fake_evaluate({s.COMPLEMENTARY[0]}))
    launcher = FakeLauncher()
    mod.run_complementary(ROOT, out, [3, 4], launcher=launcher, identity_check=no_identity)
    assert len(launcher.calls) == 1
    with pytest.raises(ValueError, match="STOPPED_COMPLEMENTARY"):
        mod.reserve_and_charge(out, by_key(branches)[s.COMPLEMENTARY[1]], "complementary")


def test_S21_mutation_deleting_complementary_gate_is_detected(tmp_path, monkeypatch):
    _b_after_a_failure_blocked(s, tmp_path / "prod", monkeypatch)
    mut = _mutant(tmp_path, "mut_step_gate_deleted", "def _step_gate(out, b):\n    " + STEP_GATE_DOC,
                  "def _step_gate(out, b):\n    " + STEP_GATE_DOC + "    return None\n")
    with pytest.raises(BaseException) as caught:
        _b_after_a_failure_blocked(mut, tmp_path / "mut", monkeypatch)
    assert not isinstance(caught.value, KeyboardInterrupt)
    mut2 = _mutant(tmp_path, "mut_verdict_always_ok", STEP_VERDICT_OK,
                   '    return True, row\n')
    with pytest.raises(BaseException):
        _b_after_a_failure_blocked(mut2, tmp_path / "mut2", monkeypatch)
    _record("S-21_complementary_gate_deleted", "step gate removed / verdict forced True", True,
            "after an A failure B/C/D must not be dispatched or reservable")


def _always_u_faster_not_established(mod):
    rows = synth({k: (10.0, 10.4) for k in (("layout_0", "B_PENDING"), ("layout_0", "C_PENDING"),
                                            ("layout_1", "B_PENDING"), ("layout_1", "C_PENDING"))})
    g = mod.mechanism_gate(rows, True)
    assert g["status"] == "NOT_ESTABLISHED" and g["next_action"] == "S4_RESEARCH_DECISION"


def test_S22_mutation_same_candidate_always_faster_judged_established_is_detected(tmp_path):
    _always_u_faster_not_established(s)
    c5 = '"c5_sign_reversal_within_layout": bool(c5)'
    c8 = '"c8_not_same_candidate_always_wins": bool(c8)'
    # defence in depth: either direction check alone still stops a same-candidate-always-faster dataset
    _always_u_faster_not_established(_mutant(tmp_path, "mut_c5_only", c5, c5.replace("bool(c5)", "True")))
    _always_u_faster_not_established(_mutant(tmp_path, "mut_c8_only", c8, c8.replace("bool(c8)", "True")))
    src = SRC.read_text().replace(c5, c5.replace("bool(c5)", "True")).replace(c8, c8.replace("bool(c8)", "True"))
    path = tmp_path / "mut_both.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("cp_disr.analysis.mut_both", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(BaseException) as caught:
        _always_u_faster_not_established(mod)
    assert not isinstance(caught.value, KeyboardInterrupt)
    _record("S-22_same_candidate_always_faster", "direction checks (c5 and c8) forced True", True,
            "pad_u always faster must be NOT_ESTABLISHED")
