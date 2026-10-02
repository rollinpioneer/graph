"""CP-DISR-TB-E1-ELASTIC-01 offline tests (CPU only; engineering-only data; no simulator, no GPU, no provider).

They exercise the ACTUAL `final_tb` / `final_tb_e1` code.  The production Policy/PPO/train_job run through the same CPU
stubs as the original launch tests (only the simulator boundary is replaced).  The compatibility-rebind mutation tests
change real source text and require the rebind to fail.
"""
from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from cp_disr import final_tb, final_tb_e1 as e1
from cp_disr.common import BindingError
from tests.helpers import final_tb_checks as chk

ROOT = chk.ROOT
GIB = 1024 ** 3
FAKE_PREP = "e" * 40
FAKE_GIT = lambda r, *a: FAKE_PREP if a[0] == "rev-parse" else ""


def _have_old_prep() -> bool:
    return subprocess.run(["git", "cat-file", "-e", e1.OLD_PREP + "^{commit}"], cwd=str(ROOT), capture_output=True).returncode == 0


needs_old_prep = pytest.mark.skipif(not _have_old_prep(), reason="old smoke prep commit not in this checkout")
BASE_OUT = ROOT / e1.BASE_LAUNCH_REL
needs_base = pytest.mark.skipif(not (BASE_OUT / "launch_state.json").is_file() or not (BASE_OUT / "smoke_receipt.json").is_file(),
                                reason="base launch evidence not in this checkout")


# 1 ---------------------------------------------------------------------------------------------------
def test_01_e1_plan_method_seed_are_strictly_bound():
    assert final_tb.validate_assignment("R-TB-E-1", "B1-K+E", 1)["budget_slot"] == "ELASTIC-01"
    for method in ("B2", "B1-K", "B0", "B1"):
        with pytest.raises(BindingError):
            final_tb.validate_assignment("R-TB-E-1", method, 1)
    for seed in (0, 2, True, "1", 1.0, None):
        with pytest.raises(BindingError):
            final_tb.validate_assignment("R-TB-E-1", "B1-K+E", seed)
    with pytest.raises(BindingError):
        final_tb.validate_assignment("R-TB-E-2", "B1-K+E", 1)
    # the other +E seed/plan combinations stay as registered
    with pytest.raises(BindingError):
        final_tb.validate_assignment("R-TB-E-0", "B1-K+E", 1)


def test_01b_run_context_rejects_wrong_seed_method_task(tmp_path):
    ctx = chk.make_context(tmp_path, "R-TB-E-1", ROOT)
    assert (ctx.plan_id, ctx.method, ctx.training_seed) == ("R-TB-E-1", "B1-K+E", 1)
    ctx.assert_worker("T_B", "B1-K+E")
    with pytest.raises(BindingError):
        ctx.assert_worker("T_B", "B2")
    with pytest.raises(BindingError):
        chk.make_context(tmp_path / "s0", "R-TB-E-1", ROOT, training_seed=0)
    with pytest.raises(BindingError):
        chk.make_context(tmp_path / "m", "R-TB-E-1", ROOT, method="B1-K")
    with pytest.raises(BindingError):
        chk.make_context(tmp_path / "x", "R-TB-E-1", ROOT, init_checkpoint="/some/e0/final.pt")  # no checkpoint key can be smuggled in


# 2 ---------------------------------------------------------------------------------------------------
def test_02_old_three_plan_identities_are_unchanged():
    assert final_tb.BASE_PLAN_TABLE == {
        "R-TB-E-0": {"method": "B1-K+E", "seed": 0, "release_order": 1},
        "R-TB-DK-1": {"method": "B2", "seed": 1, "release_order": 2},
        "R-TB-K-1": {"method": "B1-K", "seed": 1, "release_order": 3},
    }
    for pid, row in final_tb.BASE_PLAN_TABLE.items():
        assert final_tb.PLAN_TABLE[pid] == row
    assert set(final_tb.PLAN_TABLE) == set(final_tb.BASE_PLAN_TABLE) | {"R-TB-E-1"}
    assert final_tb.ELASTIC_PLAN_TABLE == {"R-TB-E-1": {"method": "B1-K+E", "seed": 1, "release_order": 4, "budget_slot": "ELASTIC-01"}}
    assert final_tb.METHODS == ("B1-K", "B1-K+E", "B2")
    assert e1.assert_plan_identities()["cumulative_cap"] == 4
    with mock.patch.dict(final_tb.BASE_PLAN_TABLE, {"R-TB-DK-1": {"method": "B2", "seed": 2, "release_order": 2}}):
        with pytest.raises(BindingError):
            e1.assert_plan_identities()


def test_02b_the_base_registration_keeps_three_plans_and_three_attempts(tmp_path):
    ledger = final_tb.Ledger(tmp_path / "base")
    state = ledger.init({p: {"status": "NOT_STARTED"} for p in final_tb.BASE_PLAN_TABLE})
    assert state["new_rl_attempts_cap"] == 3 and len(state["plans"]) == 3
    doc = final_tb.budget_reconciliation(final_tb.SMOKE_PRIOR_CHARGED)
    assert doc["rl_attempts"]["parent_cap"] == 3 and doc["rl_attempts"]["r2_not_additional"] == 3 and doc["rl_attempts"]["elastic"] == 0
    with pytest.raises(BindingError, match="elastic"):  # the base release path can never release the elastic plan
        final_tb.run_release(tmp_path / "base", "train", "a" * 40, "R-TB-E-1")


# 3 ---------------------------------------------------------------------------------------------------
def _base_facts(tmp_path):
    plans = {p: {"status": "COMPLETE", "attempt_id": p + "-x", "run_dir": str(tmp_path / p), "complete_updates": 14, "valid_transitions": 14500}
             for p in final_tb.BASE_PLAN_TABLE}
    return {"plans": plans, "old_files": {}}


def _plan_row(tmp_path):
    row = final_tb.PLAN_TABLE["R-TB-E-1"]
    return {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": "R-TB-E-1-t",
            "planned_run_dir": str(tmp_path / "run_e1"), "release_order": 4, "budget_slot": "ELASTIC-01"}


def test_03_total_attempt_cap_is_exactly_four_and_e1_is_the_fourth(tmp_path):
    assert final_tb.BASE_RL_ATTEMPTS == 3 and final_tb.MAX_NEW_RL_ATTEMPTS == 4
    state = e1.init_elastic_ledger(tmp_path / "e1", _base_facts(tmp_path), _plan_row(tmp_path), "0" * 64)
    assert state["new_rl_attempts_used"] == 3 and state["new_rl_attempts_cap"] == 4
    ledger = e1.ElasticLedger(tmp_path / "e1")
    ledger.reserve("R-TB-E-1", "R-TB-E-1-t", tmp_path / "run_e1", 123, 5)
    assert ledger.read()["new_rl_attempts_used"] == 4  # = cap: nothing else can ever be reserved
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-E-1", "again", tmp_path / "run_e1b", 124, 6)
    with pytest.raises(BindingError):
        e1.init_elastic_ledger(tmp_path / "e1", _base_facts(tmp_path), _plan_row(tmp_path), "0" * 64)  # no re-init


# 4 ---------------------------------------------------------------------------------------------------
def test_04_elastic01_can_only_be_taken_by_e1(tmp_path):
    e1.init_elastic_ledger(tmp_path / "e1", _base_facts(tmp_path), _plan_row(tmp_path), "0" * 64)
    ledger = e1.ElasticLedger(tmp_path / "e1")
    for other in ("R-TB-E-0", "R-TB-DK-1", "R-TB-K-1", "R-TB-E-2"):
        with pytest.raises(BindingError):
            ledger.reserve(other, "x", tmp_path / other, 1, 5)
    st = ledger.read()
    assert st["elastic_slots"]["ELASTIC-01"] == {**st["elastic_slots"]["ELASTIC-01"], "status": "ALLOCATED", "plan_id": "R-TB-E-1"}
    assert [st["elastic_slots"][s]["status"] for s in ("ELASTIC-02", "ELASTIC-03", "ELASTIC-04")] == ["UNALLOCATED"] * 3
    ledger.reserve("R-TB-E-1", "R-TB-E-1-t", tmp_path / "run_e1", 123, 5)
    assert ledger.read()["elastic_slots"]["ELASTIC-01"]["status"] == "IN_USE"
    ledger.finish("R-TB-E-1", "COMPLETE", stop_reason="Ncap")
    assert ledger.read()["elastic_slots"]["ELASTIC-01"]["status"] == "CONSUMED"  # an attempt, once started, is spent
    # tampered ledgers are refused: slot given to another plan, or another slot allocated
    for mutate in (lambda s: s["elastic_slots"]["ELASTIC-01"].update(plan_id="R-TB-E-0"),
                   lambda s: s["elastic_slots"]["ELASTIC-02"].update(status="ALLOCATED", plan_id="R-TB-E-1"),
                   lambda s: s.update(new_rl_attempts_used=2),
                   lambda s: s["plans"].update({"R-TB-E-0": {"status": "NOT_STARTED"}})):
        out = tmp_path / ("t%d" % id(mutate))
        e1.init_elastic_ledger(out, _base_facts(tmp_path), _plan_row(tmp_path), "0" * 64)
        led = e1.ElasticLedger(out)
        state = led.read()
        mutate(state)
        final_tb.write_json_atomic(led.state_path, state)
        with pytest.raises(BindingError):
            led.reserve("R-TB-E-1", "R-TB-E-1-t", tmp_path / "run_x", 1, 5)


@needs_base
def test_04b_e1_registration_is_single_use_and_does_not_start_a_run(tmp_path):
    out_a = tmp_path / "launch" / "a"
    plan = e1.register_e1(ROOT, out_a, FAKE_PREP, "authorization text", stamp="T0", git=FAKE_GIT)
    st = json.loads((out_a / "launch_state.json").read_text())
    assert st["new_rl_attempts_used"] == 3 and st["plans"]["R-TB-E-1"]["status"] == "NOT_STARTED"
    assert plan["plans"]["R-TB-E-1"]["planned_run_dir"].endswith("/T_B/B1-K+E/seed_1/R-TB-E-1-T0-eeeeeeee")
    assert not Path(plan["plans"]["R-TB-E-1"]["planned_run_dir"]).exists()
    with pytest.raises(BindingError, match="already exists"):  # ELASTIC-01 is allocated exactly once
        e1.register_e1(ROOT, tmp_path / "launch" / "b", FAKE_PREP, "authorization text", stamp="T1", git=FAKE_GIT)
    with pytest.raises(BindingError):
        e1.register_e1(ROOT, tmp_path / "launch2" / "c", e1.OLD_PREP, "x", stamp="T2", git=lambda r, *a: e1.OLD_PREP if a[0] == "rev-parse" else "")


# 5 ---------------------------------------------------------------------------------------------------
def test_05_seed1_reaches_policy_sampler_checkpoint_and_metadata(tmp_path):
    ctx = chk.make_context(tmp_path / "e1", "R-TB-E-1", ROOT)
    obs = chk.run_until_first_step(ROOT, ctx)
    assert obs["initial_seed_at_policy"] == 1 and obs["initial_seed_at_step"] == 1  # Policy init and collection stream
    assert obs["sampler_calls"] == 0  # no PriorSampler seed stream is built for +E
    again = chk.run_until_first_step(ROOT, chk.make_context(tmp_path / "e1b", "R-TB-E-1", ROOT))
    assert again["fingerprint"] == obs["fingerprint"]
    seed0 = chk.run_until_first_step(ROOT, chk.make_context(tmp_path / "e0", "R-TB-E-0", ROOT))
    assert seed0["initial_seed_at_step"] == 0 and seed0["fingerprint"] != obs["fingerprint"]  # not the seed-0 initialisation
    job = Path(ctx.output_directory)
    cfg = json.loads((job / "resolved_config.json").read_text())
    assert (cfg["plan_id"], cfg["method"], cfg["seed"], cfg["training_seed"]) == ("R-TB-E-1", "B1-K+E", 1, 1)
    man = json.loads((job / "manifest.json").read_text())
    assert (man["planned_id"], man["method"], man["seed"], man["training_seed"]) == ("R-TB-E-1", "B1-K+E", 1, 1)
    ck = json.loads((job / "checkpoints" / "n_000000.json").read_text())["manifest"]
    gen = json.loads((job / "persistence" / "generations" / "n_000000" / "manifest.json").read_text())
    for meta in (ck, gen):
        assert (meta["training_seed"], meta["plan_id"], meta["method"], meta["task"]) == (1, "R-TB-E-1", "B1-K+E", "T_B")


# 6 ---------------------------------------------------------------------------------------------------
def test_06_plus_e_is_still_empty_r_without_provider(tmp_path):
    assert chk.check_plus_e_empty_r(ROOT, tmp_path)
    ctx = chk.make_context(tmp_path / "e1", "R-TB-E-1", ROOT)
    assert ctx.prior_mode == "absent"
    chk.run_until_first_step(ROOT, ctx)
    cfg = json.loads((Path(ctx.output_directory) / "resolved_config.json").read_text())
    assert cfg["provider"] is None and cfg["sampler"] is None and cfg["prior_mode"] == "absent"
    sources = {"src/cp_disr/final_tb_e1.py": (ROOT / "src/cp_disr/final_tb_e1.py").read_text(encoding="utf-8"),
               "scripts/final_tb_e1_launch.py": (ROOT / "scripts/final_tb_e1_launch.py").read_text(encoding="utf-8")}
    assert final_tb.entry_forbidden_hits(sources) == {}  # no provider / sampler / dry-run / old-adapter identifiers in code
    assert e1.load_release_config(ROOT)["budget"]["provider"] == 0


# 7 ---------------------------------------------------------------------------------------------------
def test_07_e1_initialises_independently_and_never_loads_e0_weights(tmp_path):
    from cp_disr import torch_rl
    cfg = e1.load_release_config(ROOT)["plan"]
    assert cfg["init"] == "FROM_SCRATCH" and cfg["warm_start"] is False
    call = e1.production_call_fingerprint((ROOT / "src/cp_disr/final_tb.py").read_text(encoding="utf-8"))
    assert "resume" in call["keywords"] and "resume=False" in call["source"]
    ctx = chk.make_context(tmp_path / "e1", "R-TB-E-1", ROOT)
    assert not any("checkpoint" in k or "weights" in k for k in ctx.as_dict())  # no checkpoint-ish field on the context
    with mock.patch.object(torch_rl, "load_checkpoint", side_effect=AssertionError("E1 must not load any checkpoint")):
        obs = chk.run_until_first_step(ROOT, ctx)
    assert obs["reached_first_step"] and not (Path(ctx.output_directory) / "resume.json").exists()
    assert "R-TB-E-0" not in ctx.output_directory and "seed_0" not in ctx.output_directory
    assert obs["evals"][0]["checkpoint"].endswith("checkpoints/n_000000.pt")  # its own step-0 generation, not an E-0 file


# 8-9 -------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paths():
    """Real old-prep snapshot and the working tree (the E1 prep before it is committed)."""
    if not _have_old_prep():
        pytest.skip("old smoke prep commit not in this checkout")
    return e1.snapshot_paths(ROOT, e1.OLD_PREP), e1.snapshot_paths(ROOT, None)


@needs_old_prep
def test_08_real_old_prep_vs_new_prep_is_compatible_and_only_the_allowed_nodes_changed(paths):
    old, new = paths
    rep = e1.compare_execution_paths(old, new, e1.OLD_PREP, "working-tree")
    assert rep["verdict"] == "COMPATIBLE" and rep["execution_path_identical"] and rep["violations"] == []
    nodes = rep["final_tb_nodes"]
    assert set(nodes["changed"] + nodes["added"] + nodes["removed"]) <= set(e1.ALLOWED_FINAL_TB_CHANGES)
    assert all(row["identical"] for row in nodes["key_nodes"].values())
    assert nodes["key_nodes"]["run_smoke"]["identical"] and nodes["key_nodes"]["ScriptedSelector"]["identical"]
    assert rep["run_train_production_call"]["identical"] and "resume=False" in rep["run_train_production_call"]["new"]["source"]
    assert all(r["identical"] for r in rep["key_files"].values()) and len(rep["key_files"]) == len(e1.KEY_FILES)
    assert set(rep["source_sweep"]["changes"]) <= set(e1.ALLOWED_FILE_CHANGES)
    assert {"stage2a_v11.py", "collector.py", "neural.py", "torch_rl.py", "runtime_factory.py", "Verifier", "Evaluator", "SkillExecutor"} <= set(rep["key_files"])


def _mutate_node(src: str, name: str, how: str) -> str:
    """Change one node of the real source text: `ast` adds a statement/attribute, `comment` only adds a comment inside it."""
    import ast
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    target = None
    if "." in name:
        cls, meth = name.split(".")
        cdef = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        target = next(n for n in cdef.body if isinstance(n, ast.FunctionDef) and n.name == meth)
    else:
        target = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name), None)
    if target is None:  # module-level constant: always an AST change
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))
        seg = "".join(lines[node.lineno - 1:node.end_lineno])
        assert src.count(seg) == 1
        return src.replace(seg, seg.replace("=", "= (0) + ", 1), 1)
    pad = " " * (target.col_offset + 4)
    if how == "comment":
        at = target.lineno  # right after the def/class line: inside the node's own source segment
        return "".join(lines[:at]) + pad + "# mutated comment\n" + "".join(lines[at:])
    at = target.end_lineno
    return "".join(lines[:at]) + pad + "_mutation_marker = 1\n" + "".join(lines[at:])


MUTATED_NODES = ("run_smoke", "ScriptedSelector.__call__", "ScriptedSelector", "run_train", "configure_v11", "make_source_hashes",
                 "bind_worker_gpu", "first_update_gate", "Ledger.reserve", "verify_token", "verify_smoke_receipt", "CountingExecutor",
                 "RecordingSnapshotBuilder", "forward_probe", "validate_assignment")


@needs_old_prep
@pytest.mark.parametrize("name", MUTATED_NODES)
@pytest.mark.parametrize("how", ["ast", "comment"])
def test_09_any_change_of_a_key_function_fails_the_rebind(paths, name, how):
    old, new = paths
    ok = e1.compare_execution_paths(old, new, e1.OLD_PREP, "x")
    assert ok["verdict"] == "COMPATIBLE"  # control: the unmutated pair
    mutated = copy.deepcopy(new)
    mutated["final_tb_source"] = _mutate_node(mutated["final_tb_source"], name, how)
    assert mutated["final_tb_source"] != new["final_tb_source"]
    rep = e1.compare_execution_paths(old, mutated, e1.OLD_PREP, "x")
    assert rep["verdict"] == "INCOMPATIBLE" and not rep["execution_path_identical"]
    assert any(name in v or "production call" in v for v in rep["violations"]), rep["violations"]


@needs_old_prep
def test_09b_constants_files_sweep_and_production_call_changes_fail_the_rebind(paths):
    old, new = paths

    def verdict(mutated):
        return e1.compare_execution_paths(old, mutated, e1.OLD_PREP, "x")

    for const in ("N_CAP", "H_SECONDS", "D_REF_SECONDS", "EVAL_POINTS", "ROLLOUT_N", "TCAP_SECONDS"):
        m = copy.deepcopy(new)
        m["final_tb_source"] = _mutate_node(m["final_tb_source"], const, "ast")
        assert verdict(m)["verdict"] == "INCOMPATIBLE", const
    for name in ("stage2a_v11.py", "collector.py", "neural.py", "torch_rl.py", "runtime_factory.py", "Verifier", "Evaluator", "SkillExecutor",
                 "perception", "split (T_B dev10)", "task (T_B resolved)", "runtime manifest (profile)", "controller (controllers/__init__.py)"):
        m = copy.deepcopy(new)
        m["key_files"][name]["sha256"] = "f" * 64
        rep = verdict(m)
        assert rep["verdict"] == "INCOMPATIBLE" and any(name in v for v in rep["violations"]), name
    m = copy.deepcopy(new)
    m["final_tb_source"] = m["final_tb_source"].replace("resume=False", "resume=True")
    rep = verdict(m)
    assert rep["verdict"] == "INCOMPATIBLE" and not rep["run_train_production_call"]["identical"]
    m = copy.deepcopy(new)
    m["blob_map"]["src/cp_disr/stage2a_v11.py"] = "0" * 40  # any other source file touched
    assert verdict(m)["verdict"] == "INCOMPATIBLE"
    m = copy.deepcopy(new)
    m["blob_map"]["src/cp_disr/new_module.py"] = "1" * 40
    assert verdict(m)["verdict"] == "INCOMPATIBLE"
    m = copy.deepcopy(new)
    m["final_tb_source"] = _mutate_node(m["final_tb_source"], "storage_gate", "ast")  # an unlisted final_tb function
    rep = verdict(m)
    assert rep["verdict"] == "INCOMPATIBLE" and "storage_gate" in rep["final_tb_nodes"]["changed_outside_allowed"]
    with mock.patch.object(final_tb, "MAX_NEW_RL_ATTEMPTS", 5):  # the cap itself is guarded by assert_plan_identities, not by the rebind
        with pytest.raises(BindingError):
            e1.assert_plan_identities()


def test_09c_rebind_failure_writes_no_receipt_and_no_token(tmp_path):
    with pytest.raises(BindingError, match="INCOMPATIBLE"):
        e1.write_compat_receipt(tmp_path, {"old_files": {}}, {"verdict": "INCOMPATIBLE", "violations": ["x"]}, tmp_path / "r.json", FAKE_PREP)
    assert not (tmp_path / "smoke_compatibility_receipt.json").exists()
    with pytest.raises(BindingError, match="no smoke compatibility receipt"):
        e1.verify_compat_receipt(ROOT, tmp_path, FAKE_PREP)


# 10-11 + train refusals ------------------------------------------------------------------------------
@pytest.fixture
def reg(tmp_path):
    if not (BASE_OUT / "launch_state.json").is_file() or not (BASE_OUT / "smoke_receipt.json").is_file():
        pytest.skip("base launch evidence not in this checkout")
    out = tmp_path / "launch" / "e1"
    e1.register_e1(ROOT, out, FAKE_PREP, "authorization text", stamp="T0", git=FAKE_GIT)
    return out


def _prepare_release(out, monkeypatch, confirm=True, pool=True, receipt=True):
    facts = e1.base_launch_facts(BASE_OUT)
    rebind = {"kind": "smoke_compatibility_rebind", "verdict": "COMPATIBLE", "violations": [], "execution_path_identical": True,
              "comparison_digest_sha256": "d" * 64}
    monkeypatch.setattr(e1, "build_rebind", lambda root, new_ref, old_ref=e1.OLD_PREP, new_prep=None: dict(rebind))
    if receipt:
        path = out / "smoke_compatibility_rebind.json"
        final_tb.write_json_atomic(path, rebind)
        e1.write_compat_receipt(out, facts, rebind, path, FAKE_PREP)
    if confirm:
        final_tb.write_json_atomic(out / "nontrivial_learning_confirmation_R-TB-E-0.json", {"status": "CONFIRMED_STATIC"})
    if pool:
        final_tb.write_json_atomic(out / "elastic_pool_ledger_reconstructed.json",
                                   e1.build_elastic_pool_ledger(facts, {}, final_tb.authorization_digest(out)))
    return facts


def _release(out):
    return e1.release_e1(ROOT, out, FAKE_PREP, free_fn=lambda p: 6 * GIB, git=FAKE_GIT)


def test_10_old_smoke_and_train_tokens_cannot_release_the_new_prep(reg, monkeypatch):
    out = reg
    _prepare_release(out, monkeypatch)
    old_train = final_tb.issue_token(out, "train", e1.OLD_PREP, "R-TB-E-1", {"smoke_receipt_sha256": "x"})
    with pytest.raises(BindingError, match="prep commit mismatch"):
        e1.verify_release_token_e1(str(old_train), out, FAKE_PREP)
    with pytest.raises(BindingError, match="old prep"):
        e1.verify_release_token_e1(str(old_train), out, e1.OLD_PREP)
    old_smoke = final_tb.issue_token(out, "smoke", e1.OLD_PREP)
    with pytest.raises(BindingError):
        e1.verify_release_token_e1(str(old_smoke), out, FAKE_PREP)
    real = sorted((BASE_OUT / "release_tokens").glob("*.json")) if (BASE_OUT / "release_tokens").is_dir() else []
    for tok in real:  # the real tokens of the old registration: unusable for the new prep / registration
        with pytest.raises(BindingError):
            e1.verify_release_token_e1(str(tok), out, FAKE_PREP)
    # the new release is not blocked by the stray old-prep files above (different kind/name collision is not the case)
    assert "release_tokens" in str(old_train)


def test_11_new_release_token_is_bound_to_the_new_prep_and_the_compat_receipt(reg, monkeypatch):
    out = reg
    _prepare_release(out, monkeypatch)
    token = _release(out)
    doc = json.loads(Path(token).read_text())
    receipt_sha = final_tb.sha256_file(out / "smoke_compatibility_receipt.json")
    assert doc["kind"] == "train" and doc["plan_id"] == "R-TB-E-1" and doc["prep_commit"] == FAKE_PREP != e1.OLD_PREP
    ev = doc["evidence"]
    assert ev["compat_receipt_sha256"] == receipt_sha and ev["old_prep_commit"] == e1.OLD_PREP and ev["new_prep_commit"] == FAKE_PREP
    assert ev["old_token_reused"] is False and ev["elastic_slot"] == "ELASTIC-01" and ev["attempts"] == {"used_before": 3, "cap": 4}
    assert ev["storage_start_gate"]["passed"] and ev["storage_start_gate"]["free_bytes"] == 6 * GIB
    assert e1.verify_release_token_e1(str(token), out, FAKE_PREP)["plan_id"] == "R-TB-E-1"
    with pytest.raises(BindingError):
        e1.verify_release_token_e1(str(token), out, "f" * 40)  # another prep
    with pytest.raises(BindingError):
        _release(out)  # one token per plan
    receipt_path = out / "smoke_compatibility_receipt.json"
    original = receipt_path.read_text()
    receipt_path.write_text(original.replace('"issued"', '"issued_tampered"'))
    with pytest.raises(BindingError, match="not bound to the current compatibility receipt"):
        e1.verify_release_token_e1(str(token), out, FAKE_PREP)
    receipt_path.write_text(original)
    assert e1.verify_release_token_e1(str(token), out, FAKE_PREP)
    auth = out / "authorization.json"
    auth_text = auth.read_text()
    auth.write_text(auth_text.replace("authorization text", "another text"))
    with pytest.raises(BindingError):
        e1.verify_release_token_e1(str(token), out, FAKE_PREP)
    auth.write_text(auth_text)


def test_11b_release_requires_receipt_confirmation_pool_and_start_space(reg, monkeypatch):
    out = reg
    with pytest.raises(BindingError, match="no smoke compatibility receipt"):
        _release(out)
    _prepare_release(out, monkeypatch, confirm=False, pool=False)
    with pytest.raises(BindingError, match="NONTRIVIAL_LEARNING is not confirmed"):
        _release(out)
    final_tb.write_json_atomic(out / "nontrivial_learning_confirmation_R-TB-E-0.json", {"status": "CONFIRMED_STATIC"})
    with pytest.raises(BindingError, match="elastic_pool_ledger_reconstructed"):
        _release(out)
    facts = e1.base_launch_facts(BASE_OUT)
    pool = e1.build_elastic_pool_ledger(facts, {}, final_tb.authorization_digest(out))
    bad = copy.deepcopy(pool)
    bad["pool"]["slots"]["ELASTIC-02"] = {"status": "ALLOCATED", "plan_id": "R-TB-E-1"}
    final_tb.write_json_atomic(out / "elastic_pool_ledger_reconstructed.json", bad)
    with pytest.raises(BindingError, match="ELASTIC-01 to R-TB-E-1 alone"):
        _release(out)
    final_tb.write_json_atomic(out / "elastic_pool_ledger_reconstructed.json", pool)
    with pytest.raises(BindingError, match="STORAGE_START_GATE_REFUSED"):
        e1.release_e1(ROOT, out, FAKE_PREP, free_fn=lambda p: 6 * GIB - 1, git=FAKE_GIT)
    assert not (out / "release_tokens" / "train_R-TB-E-1.json").exists()
    # a receipt written for another prep never releases this one
    other = e1.verify_compat_receipt.__globals__["_json_file"](out / "smoke_compatibility_receipt.json")
    other["new_prep_commit"] = "9" * 40
    final_tb.write_json_atomic(out / "smoke_compatibility_receipt.json", other)
    with pytest.raises(BindingError, match="different prep"):
        _release(out)


def test_11c_pool_record_is_a_reconstruction_with_no_prior_allocations(reg, monkeypatch):
    out = reg
    facts = _prepare_release(out, monkeypatch)
    pool = json.loads((out / "elastic_pool_ledger_reconstructed.json").read_text())
    assert pool["status"].startswith("RECONSTRUCTED") and pool["external_master_ledger"] == "NOT_AVAILABLE"
    assert pool["prior_allocations_found"] == 0 and pool["attempt_accounting"]["base_attempts_used"] == 3
    assert pool["pool"]["slots"]["ELASTIC-01"]["plan_id"] == "R-TB-E-1"
    assert [pool["pool"]["slots"][s]["status"] for s in ("ELASTIC-02", "ELASTIC-03", "ELASTIC-04")] == ["UNALLOCATED"] * 3
    assert any("+E seed 2" in x for x in pool["not_authorized"])
    scan = e1.scan_elastic_markers([ROOT])
    assert str(ROOT) in scan and scan[str(ROOT)]["exists"]


def test_12_train_refuses_before_any_gpu_when_space_is_short_or_another_worker_runs(reg, monkeypatch):
    out = reg
    _prepare_release(out, monkeypatch)
    token = _release(out)
    boom = mock.Mock(side_effect=AssertionError("GPU/torch must not be touched before every check passed"))
    monkeypatch.setattr(final_tb, "bind_worker_gpu", boom)
    monkeypatch.setattr(final_tb, "query_gpu_uuid", boom)
    quiet_ps = "  PID ARGS\n 1 /sbin/init\n"
    with pytest.raises(BindingError, match="STORAGE_START_GATE_REFUSED"):  # start free < 6 GiB
        e1.run_e1_train(ROOT, out, 0, str(token), git=FAKE_GIT, free_fn=lambda p: 6 * GIB - 1, ps_text=quiet_ps)
    busy = quiet_ps + " 4242 python scripts/final_tb_e1_launch.py train --gpu 1 --token t\n"
    with pytest.raises(BindingError, match="another training worker"):
        e1.run_e1_train(ROOT, out, 0, str(token), git=FAKE_GIT, free_fn=lambda p: 7 * GIB, ps_text=busy)
    with pytest.raises(BindingError):
        e1.run_e1_train(ROOT, out, 0, str(out / "release_tokens" / "missing.json"), git=FAKE_GIT, free_fn=lambda p: 7 * GIB, ps_text=quiet_ps)
    boom.assert_not_called()
    # all checks pass -> exactly the unchanged production run_train is called with the E1 context and the elastic ledger
    monkeypatch.setattr(final_tb, "bind_worker_gpu", lambda g: {"physical_gpu_index": g})
    monkeypatch.setattr(final_tb, "query_gpu_uuid", lambda g: "GPU-TEST")
    seen = {}

    def stub(root, out_, ctx, gpu, ledger=None):
        seen.update(ctx=ctx, gpu=gpu, ledger=ledger)
        return {"stop_reason": "stub", "complete_updates": 0}

    summary = e1.run_e1_train(ROOT, out, 3, str(token), git=FAKE_GIT, free_fn=lambda p: 7 * GIB, ps_text=quiet_ps, runner=stub)
    ctx = seen["ctx"]
    assert summary["stop_reason"] == "stub" and seen["gpu"] == 3 and isinstance(seen["ledger"], e1.ElasticLedger)
    assert (ctx.plan_id, ctx.method, ctx.training_seed, ctx.physical_gpu_index, ctx.gpu_uuid) == ("R-TB-E-1", "B1-K+E", 1, 3, "GPU-TEST")
    assert ctx.source_commit == FAKE_PREP and ctx.study_envelope_ncap == 16384 and ctx.study_envelope_tcap == final_tb.TCAP_SECONDS
    evidence = json.loads((out / "launch_evidence.json").read_text())
    assert evidence["init"] == "FROM_SCRATCH" and evidence["warm_start"] is False and evidence["start_gate"]["passed"]


def test_12b_derived_inputs_are_the_ones_e0_trained_with(reg):
    doc = json.loads((reg / "derived_inputs_identity.json").read_text())
    assert doc["all_identical"], doc["rows"]
    prof = doc["frozen_profile"]
    assert prof["H_seconds"] == final_tb.H_SECONDS and prof["d_ref_seconds"] == final_tb.D_REF_SECONDS
    assert prof["study_envelope_ncap"] == 16384 and prof["rollout_n"] == 1024 and prof["max_complete_updates"] == 16


# 12 storage gate -------------------------------------------------------------------------------------
def test_13_storage_gate_constants_and_start_gate(tmp_path):
    assert e1.STORAGE_GATE == {"start_free_min_bytes": 6 * GIB, "hard_reserved_margin_bytes": 2 * GIB, "pause_trigger_bytes": 3 * GIB,
                               "resume_trigger_bytes": 5 * GIB, "sampling_interval_seconds": 5.0}
    assert e1.storage_gate_ok()
    for bad in ({"pause_trigger_bytes": 2 * GIB}, {"resume_trigger_bytes": 3 * GIB}, {"start_free_min_bytes": 4 * GIB},
                {"sampling_interval_seconds": 0}):
        with pytest.raises(BindingError):
            e1.storage_gate_ok({**e1.STORAGE_GATE, **bad})
    rec = e1.start_gate(tmp_path, free_fn=lambda p: 6 * GIB)
    assert rec["passed"] and rec["free_gib"] == 6.0 and rec["mount_point"]
    for free in (0, 3 * GIB, 6 * GIB - 1):
        with pytest.raises(BindingError, match="STORAGE_START_GATE_REFUSED"):
            e1.start_gate(tmp_path, free_fn=lambda p, f=free: f)
    assert e1.start_gate(tmp_path / "not" / "yet" / "created", free_fn=lambda p: 10 * GIB)["checked_path"] == str(tmp_path)  # nearest existing ancestor


def test_13b_pause_below_3_gib_resume_at_5_gib_with_hysteresis():
    d = e1.guard_decision
    assert d(False, 3 * GIB) == "NONE" and d(False, 3 * GIB - 1) == "PAUSE"  # strictly below 3 GiB
    assert d(False, 10 * GIB) == "NONE" and d(False, 5 * GIB) == "NONE"
    assert d(True, 5 * GIB - 1) == "NONE" and d(True, 5 * GIB) == "RESUME"  # only at/after 5 GiB
    assert d(True, 4 * GIB) == "NONE" and d(True, 1) == "NONE" and d(True, 3 * GIB) == "NONE"


class _FakeDisk:
    def __init__(self, series):
        self.series, self.i = list(series), 0

    def __call__(self, _path):
        v = self.series[min(self.i, len(self.series) - 1)]
        self.i += 1
        return v


def test_13c_guard_only_stops_and_continues_and_never_touches_files(tmp_path):
    watch = tmp_path / "run"
    watch.mkdir()
    (watch / "checkpoint.pt").write_bytes(b"x" * 100)
    before = {p.name: p.read_bytes() for p in watch.iterdir()}
    sent = []
    series = [10 * GIB, 4 * GIB, 3 * GIB, 3 * GIB - 1, 2 * GIB - 5, 4 * GIB, 5 * GIB - 1, 5 * GIB, 8 * GIB]
    g = e1.StorageGuard(watch, tmp_path / "guard.jsonl", lambda: [11, 12], free_fn=_FakeDisk(series),
                        signal_fn=lambda pid, sig: sent.append((pid, sig)), state_fn=lambda pid: "T")
    actions = [g.tick(now=float(i)) for i in range(len(series))]
    assert actions == ["NONE", "NONE", "NONE", "PAUSE", "NONE", "NONE", "NONE", "RESUME", "NONE"]
    assert sent == [(11, signal.SIGSTOP), (12, signal.SIGSTOP), (11, signal.SIGCONT), (12, signal.SIGCONT)]
    assert {s for _, s in sent} <= {signal.SIGSTOP, signal.SIGCONT} and signal.SIGKILL not in {s for _, s in sent}
    log = [json.loads(l) for l in (tmp_path / "guard.jsonl").read_text().splitlines()]
    events = [r["event"] for r in log]
    assert events.count("SIGSTOP") == 1 and events.count("SIGCONT") == 1 and events.count("HARD_MARGIN_BREACH") == 1
    stop = next(r for r in log if r["event"] == "SIGSTOP")
    assert stop["free_bytes"] == 3 * GIB - 1 and stop["pids"] == [11, 12] and stop["proc_states"] == {"11": "T", "12": "T"}
    summ = g.summary()
    assert summ["pauses"] == 1 and summ["resumes"] == 1 and summ["min_free_bytes_sampled"] == 2 * GIB - 5 and not summ["paused_now"]
    assert "between samples is not measured" in summ["note"]
    assert {p.name: p.read_bytes() for p in watch.iterdir()} == before  # nothing deleted, nothing rewritten
    with pytest.raises(BindingError):  # the only signals the guard may send
        g._send([11], signal.SIGKILL)
    with pytest.raises(BindingError):
        g._send([11], signal.SIGTERM)


def test_13d_guard_releases_a_stopped_worker_when_it_ends_and_survives_a_vanished_worker(tmp_path):
    sent = []
    g = e1.StorageGuard(tmp_path, tmp_path / "g.jsonl", lambda: [5], free_fn=_FakeDisk([GIB]),
                        signal_fn=lambda pid, sig: sent.append((pid, sig)))
    assert g.tick() == "PAUSE" and g.paused
    g.release_all("test")
    assert sent[-1] == (5, signal.SIGCONT) and not g.paused
    gone = e1.StorageGuard(tmp_path, tmp_path / "g2.jsonl", lambda: [6], free_fn=_FakeDisk([GIB]),
                           signal_fn=mock.Mock(side_effect=ProcessLookupError()))
    assert gone.tick() == "PAUSE"  # worker vanished between listing and signalling: recorded, not an exception
    none = e1.StorageGuard(tmp_path, tmp_path / "g3.jsonl", lambda: [], free_fn=_FakeDisk([GIB]), signal_fn=mock.Mock())
    assert none.tick() == "NONE" and not none.paused and none.signal_fn.call_count == 0  # nothing to stop: logged, no signal
    assert "PAUSE_TRIGGER_NO_WORKER" in (tmp_path / "g3.jsonl").read_text()
    stopped = []
    run = e1.StorageGuard(tmp_path, tmp_path / "g4.jsonl", lambda: [7], free_fn=_FakeDisk([GIB, GIB, GIB]),
                          signal_fn=lambda pid, sig: stopped.append(sig))
    summary = run.run(stop_fn=lambda: run.ticks >= 2, sleep_fn=lambda s: None)
    assert stopped == [signal.SIGSTOP, signal.SIGCONT] and not summary["paused_now"]  # run() always releases on exit


def test_13e_worker_discovery_uses_the_ledger_pid_the_command_line_and_the_process_tree(tmp_path):
    proc = tmp_path / "proc"
    for pid, ppid, cmd in ((123, 1, "python\0scripts/final_tb_e1_launch.py\0train"), (124, 123, "python\0child"), (125, 124, "python\0grandchild"),
                           (126, 1, "python\0unrelated")):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.encode())
        (d / "stat").write_text("%d (python) S %d 0 0\n" % (pid, ppid))
    out = tmp_path / "out"
    out.mkdir()
    state = {"plans": {"R-TB-E-1": {"status": "RUNNING", "pid": 123}}}
    final_tb.write_json_atomic(out / "launch_state.json", state)
    assert e1.find_worker_pids(out, proc_root=str(proc)) == [123, 124, 125]
    state["plans"]["R-TB-E-1"]["pid"] = 126  # a pid whose command line is not the E1 entry is never signalled
    final_tb.write_json_atomic(out / "launch_state.json", state)
    assert e1.find_worker_pids(out, proc_root=str(proc)) == []
    state["plans"]["R-TB-E-1"].update(pid=123, status="COMPLETE")
    final_tb.write_json_atomic(out / "launch_state.json", state)
    assert e1.find_worker_pids(out, proc_root=str(proc)) == []


def test_13f_release_config_must_equal_the_frozen_values(tmp_path):
    assert e1.load_release_config(ROOT)["storage_gate"]["pause_trigger_bytes"] == 3 * GIB
    root = tmp_path / "root"
    cfg = e1.write_release_config(root / e1.RELEASE_CONFIG_REL)
    assert e1.load_release_config(root) == cfg
    import yaml
    for mutate in (lambda c: c["storage_gate"].update(pause_trigger_bytes=2 * GIB), lambda c: c["frozen"].update(ncap=16385),
                   lambda c: c["plan"].update(warm_start=True), lambda c: c["plan"].update(seed=2),
                   lambda c: c["budget"]["elastic_pool"].update({"ELASTIC-02": "ALLOCATED:R-TB-E-1"}),
                   lambda c: c["frozen"].update(eval_points=[0, 4096, 8192, 12288]), lambda c: c["smoke"].update(reuse_old_release_token=True)):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        (root / e1.RELEASE_CONFIG_REL).write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(BindingError):
            e1.load_release_config(root)


# 13 success_seconds ----------------------------------------------------------------------------------
def test_14_success_seconds_is_not_presented_as_episode_total_time():
    v11 = (ROOT / e1.V11_REL).read_text(encoding="utf-8")
    rl = (ROOT / e1.RL_REL).read_text(encoding="utf-8")
    sem = e1.success_seconds_semantics(v11, rl)
    assert sem["guarded_by_hasattr"] and not sem["snapshot_has_elapsed_seconds"] and sem["recorded_value_is_final_skill_duration_only"]
    assert sem["label"] == "FINAL_SKILL_DURATION_ONLY" and sem["episode_total_success_time"] == "UNVERIFIED"
    desc = e1.describe_success_seconds(2.9)
    assert desc == {"recorded_success_seconds": 2.9, "label": "FINAL_SKILL_DURATION_ONLY", "episode_total_success_time": "UNVERIFIED"}
    assert not any("total" in k and k != "episode_total_success_time" for k in desc)
    # if the snapshot ever gained elapsed_seconds the claim "duration only" must stop being asserted
    patched = rl.replace("class Snapshot", "class Snapshot", 1)
    import re
    with_elapsed = re.sub(r"(class Snapshot[^\n]*\n)", r"\1    elapsed_seconds: float = 0.0\n", patched, count=1)
    assert with_elapsed != patched
    assert not e1.success_seconds_semantics(v11, with_elapsed)["recorded_value_is_final_skill_duration_only"]
    cfg = e1.load_release_config(ROOT)["reporting"]
    assert cfg == {"success_seconds_label": "FINAL_SKILL_DURATION_ONLY", "episode_total_success_time": "UNVERIFIED"}


# 14 evaluator eligibility ----------------------------------------------------------------------------
def test_15_evaluator_input_and_collector_call_sites_are_free_of_method_prior_logits_hidden_facts():
    ev = (ROOT / e1.EVALUATOR_REL).read_text(encoding="utf-8")
    reads = e1.evaluator_input_reads(ev)
    assert reads["ok"] and set(reads["attributes_read"]) <= e1.ALLOWED_VALUE_ATTRS and reads["bare_uses_of_input"] == 0
    assert reads["forbidden_identifiers_present"] == [] and reads["evaluator_version"] == "cp-disr-d0-task-evaluator-v1"
    assert e1.evaluation_input_fields((ROOT / e1.ADAPTERS_REL).read_text(encoding="utf-8")) == e1.EVAL_INPUT_FIELDS
    col = e1.collector_evaluation_calls((ROOT / e1.COLLECTOR_REL).read_text(encoding="utf-8"))
    assert col["ok"] and len(col["calls"]) == 2 and all(c["key_only_ids"] and not c["forbidden_hits"] for c in col["calls"])
    # mutations: the evaluator peeks at the policy / a prior, or the collector feeds logits/prior/facts into the input
    bad_eval = ev.replace("elapsed = float(value.elapsed_seconds)", "elapsed = float(value.elapsed_seconds)\n        _p = value.prior_edges", 1)
    assert not e1.evaluator_input_reads(bad_eval)["ok"]
    bad_eval2 = ev.replace("from cp_disr.adapters import TaskResult", "from cp_disr.adapters import TaskResult\nimport torch", 1)
    assert not e1.evaluator_input_reads(bad_eval2)["ok"]
    bad_eval3 = ev.replace("self.time_resolution_seconds = self._timestep()\n        success_now", "leak = value\n        success_now", 1)
    assert not e1.evaluator_input_reads(bad_eval3)["ok"]
    cs = (ROOT / e1.COLLECTOR_REL).read_text(encoding="utf-8")
    assert not e1.collector_evaluation_calls(cs.replace("EvaluationInput(self.bundle.task_id, *key, evidence, elapsed, start, end)",
                                                        "EvaluationInput(self.bundle.task_id, *key, evidence, elapsed, start, logits)", 1))["ok"]
    assert not e1.collector_evaluation_calls(cs.replace("key = (snapshot.env_id, snapshot.episode_id)", "key = (snapshot.prior_hash, snapshot.episode_id)", 1))["ok"]


def test_15b_task_success_has_a_single_producer_the_evaluator():
    sources = {}
    for p in (ROOT / "src" / "cp_disr").rglob("*.py"):
        sources[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    assert e1.task_success_producers(sources) == [e1.EVALUATOR_REL]
    sources["src/cp_disr/collector.py"] += '\n\ndef fake():\n    return TaskResult(success=True, terminated=True, truncated=False, reason="TASK_SUCCESS", reward_events=())\n'
    assert e1.task_success_producers(sources) == ["src/cp_disr/collector.py", e1.EVALUATOR_REL]


def _fake_trigger_run(tmp_path, root=ROOT, mutate=None):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "final_n_014512_u14.pt"
    ckpt.write_bytes(b"final-checkpoint-bytes")
    dev = final_tb.split_lists(root)["dev"]
    rows = [{"case_id": c, "success": True, "G": 0.49, "steps": 5, "reason": "TASK_SUCCESS", "success_seconds": 2.9, "source_n": 0, "prior_mode": "absent"} for c in dev]
    payload = {"task": "T_B", "method": "B1-K+E", "checkpoint": str(ckpt), "success_n": 10, "n": 10, "success_rate": 1.0, "mean_discounted_return": 0.49,
               "rows": rows, "hashes": {"git_commit": e1.OLD_PREP}, "eval_action": "deterministic_argmax", "label": "dev", "optimizer_steps": 0}
    if mutate:
        mutate(payload, run, ckpt)
    final_tb.write_json_atomic(run / "eval_final.json", payload)
    (run / "fresh_load_final_n_014512_u14.json").write_text(json.dumps({"adam": True, "fresh_process": True, "generation": "final_n_014512_u14", "model": True}))
    reg = {"trigger_evidence": {"post_update_dev10_file": {"sha256": final_tb.sha256_file(run / "eval_final.json")},
                                "bound_checkpoint": {"sha256": final_tb.sha256_file(ckpt)}}}
    return run, reg


def test_15c_trigger_eval_needs_10_of_10_independent_successes_bound_to_the_final_checkpoint(tmp_path):
    run, reg = _fake_trigger_run(tmp_path / "ok")
    ok = e1.check_trigger_eval(ROOT, run, reg)
    assert ok["ok"] and ok["problems"] == [] and ok["eval_final"]["reasons"] == ["TASK_SUCCESS"]
    assert ok["success_seconds"]["label"] == "FINAL_SKILL_DURATION_ONLY"

    def one_deadline(p, run_, ckpt):
        p["rows"][3].update(success=False, reason="DEADLINE")
        p["success_n"] = 9

    def other_ckpt(p, run_, ckpt):
        p["checkpoint"] = str(run_ / "checkpoints" / "update_complete_14.pt")

    def wrong_cases(p, run_, ckpt):
        p["rows"][0]["case_id"] = "T_B_dev_99"

    def stochastic(p, run_, ckpt):
        p["eval_action"] = "sample"

    for name, mut in (("deadline", one_deadline), ("checkpoint", other_ckpt), ("cases", wrong_cases), ("action", stochastic)):
        run2, reg2 = _fake_trigger_run(tmp_path / name, mutate=mut)
        res = e1.check_trigger_eval(ROOT, run2, reg2)
        assert not res["ok"], name
    run3, reg3 = _fake_trigger_run(tmp_path / "hash")
    reg3["trigger_evidence"]["bound_checkpoint"]["sha256"] = "0" * 64
    assert not e1.check_trigger_eval(ROOT, run3, reg3)["ok"]


# CLI -------------------------------------------------------------------------------------------------
def test_16_cli_refuses_without_registration_and_never_touches_the_gpu(tmp_path, monkeypatch):
    boom = mock.Mock(side_effect=AssertionError("GPU touched"))
    monkeypatch.setattr(final_tb, "bind_worker_gpu", boom)
    monkeypatch.setattr(final_tb, "query_gpu_uuid", boom)
    assert e1.main(["check-config", "--root", str(ROOT)]) == 0
    for argv in (["train", "--root", str(ROOT), "--out", str(tmp_path / "none"), "--gpu", "0", "--token", "/nope"],
                 ["release", "--root", str(ROOT), "--out", str(tmp_path / "none")],
                 ["rebind", "--root", str(ROOT), "--out", str(tmp_path / "none")]):
        assert e1.main(argv) == 2
    with pytest.raises(SystemExit) as exc:
        e1.main(["--help"])
    assert exc.value.code == 0
    boom.assert_not_called()
    assert "cp_disr.final_tb_e1" in (ROOT / "scripts" / "final_tb_e1_launch.py").read_text() or "final_tb_e1" in (ROOT / "scripts" / "final_tb_e1_launch.py").read_text()


def test_17_other_training_workers_detection_excludes_self(monkeypatch):
    ps = "PID ARGS\n 10 python scripts/final_tb_e1_launch.py train --gpu 2\n 11 python scripts/final_tb_launch.py train --plan R-TB-K-1\n" \
         " 12 python scripts/final_tb_e1_launch.py guard --out x\n 13 bash\n"
    rows = e1.other_training_workers(ps, self_pids={10})
    assert [r["pid"] for r in rows] == [11]  # guard/status processes are not training workers; own pid excluded
    assert e1.other_training_workers(ps, self_pids=set()) and len(e1.other_training_workers(ps, self_pids=set())) == 2
