"""Behavioural tests for the production T_B launch entry (CP-DISR-NEXT-ENGINEERING-FIRST-TB-1, A1).

CPU only. No simulator is constructed and no optimizer step is taken: the environment bundle,
dev evaluation and Collector.step are stubbed at the simulator boundary; everything else is the
production code path (final_tb, stage2a_v11.train_job, Policy/PPO, phase_a_v12 persistence).
The three mutation tests copy the source tree, re-introduce a known regression and require the
checks to fail on the mutant (and pass on the unmutated control copy).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cp_disr import final_tb
from cp_disr.common import BindingError
from tests.helpers import final_tb_checks as chk

ROOT = chk.ROOT
PLANS = [("R-TB-E-0", "B1-K+E", 0, "B1-K+E"), ("R-TB-DK-1", "B2", 1, "B2"), ("R-TB-K-1", "B1-K", 1, "B1")]


# 1 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("plan_id,method,seed,policy_method", PLANS)
def test_cli_binding_reaches_policy_dir_and_metadata(tmp_path, plan_id, method, seed, policy_method):
    ctx = chk.make_context(tmp_path, plan_id)
    obs = chk.run_until_first_step(ROOT, ctx)
    assert obs["reached_first_step"] and obs["policy_method"] == policy_method
    assert obs["task_id"] == "T_B" and obs["manifest_used"] == ctx.runtime_manifest
    job = Path(ctx.output_directory)
    assert obs["job_dir"] == ctx.output_directory and job.is_dir()
    cfg = json.loads((job / "resolved_config.json").read_text())
    assert (cfg["plan_id"], cfg["method"], cfg["seed"], cfg["training_seed"]) == (plan_id, method, seed, seed)
    assert cfg["attempt_id"] == ctx.attempt_id and cfg["prior_mode"] == "absent"
    assert cfg["sampler"] is None and cfg["provider"] is None
    # first dev evaluation (N=0) is bound to this run's own checkpoint generation
    assert obs["evals"] and Path(obs["evals"][0]["checkpoint"]).is_relative_to(job)
    meta = json.loads(Path(obs["evals"][0]["checkpoint"]).with_suffix(".json").read_text())
    blob = json.dumps(meta)
    assert method in blob and plan_id in blob


def test_illegal_plan_method_seed_combinations_are_rejected():
    for args in [("R-TB-E-0", "B2", 0), ("R-TB-E-0", "B1-K+E", 1), ("R-TB-DK-1", "B1-K", 1), ("R-TB-DK-1", "B2", 0),
                 ("R-TB-K-1", "B1-K", 0), ("R-TB-K-1", "B1-K", True), ("R-TB-K-1", "B1-K", "1"), ("R-TB-X-9", "B1-K", 1),
                 ("R-TB-K-1", "Full", 1), ("R-TB-K-1", "B0", 1)]:
        with pytest.raises(BindingError):
            final_tb.validate_assignment(*args)
    for plan_id, method, seed, _ in PLANS:
        assert final_tb.validate_assignment(plan_id, method, seed)["seed"] == seed


def test_run_config_rejects_placeholders_relative_paths_unknown_keys(tmp_path):
    good = chk.make_context(tmp_path, "R-TB-K-1").as_dict()
    keys = ["plan_id", "attempt_id", "method", "training_seed", "source_commit", "output_directory", "runtime_manifest",
            "train_split", "prior_mode", "study_envelope_ncap", "study_envelope_tcap", "evaluation_rule"]
    base = {k: good[k] for k in keys}
    final_tb.context_from_dict(base)
    for k, v in [("attempt_id", "ABSOLUTE_ATTEMPT_ID"), ("output_directory", "relative/dir"), ("source_commit", "abc"),
                 ("prior_mode", "provider"), ("study_envelope_ncap", 8192), ("study_envelope_tcap", 1.0),
                 ("evaluation_rule", "0/4096/final"), ("training_seed", 0), ("method", "B2")]:
        with pytest.raises(BindingError):
            final_tb.context_from_dict({**base, k: v})
    with pytest.raises(BindingError):
        final_tb.context_from_dict({**base, "surprise": 1})
    with pytest.raises(BindingError):
        final_tb.context_from_dict({k: v for k, v in base.items() if k != "training_seed"})


# 2 -------------------------------------------------------------------------------------------------
def test_training_seed_is_bound_and_reproducible(tmp_path):
    assert chk.check_seed_binding(ROOT, tmp_path)


# 3 -------------------------------------------------------------------------------------------------
def test_plus_e_and_siblings_use_empty_R_in_collection_and_evaluation(tmp_path):
    assert chk.check_plus_e_empty_r(ROOT, tmp_path)


@pytest.mark.parametrize("plan_id,method,seed,policy_method", PLANS)
def test_empty_R_reaches_the_collector_without_prior_sampler(tmp_path, plan_id, method, seed, policy_method):
    obs = chk.run_until_first_step(ROOT, chk.make_context(tmp_path, plan_id), prior_sampler_raises=True)
    assert obs["sampler_calls"] == 0
    assert obs["step_prior_edges"] == ()
    assert obs["bundle"].start_case_calls, "episode must have been started through the production path"


# 4 -------------------------------------------------------------------------------------------------
def test_effect_tokens_and_successor_semantics(snap, policy):
    import torch
    from dataclasses import replace
    from cp_disr.common import digest
    snap0 = replace(snap, prior_edges=(), prior_hash=digest(()))
    policy.eval()
    with torch.no_grad():
        policy.method = "B1-K+E"
        e = policy(snap0)
        assert e.diagnostics["effect_tokens"], "+E must read effect tokens from the contract"
        assert e.diagnostics["successor_used"] is False and e.diagnostics["differences"] == {}
        assert all(torch.count_nonzero(v) == 0 for v in e.diagnostics["prior_inputs"].values())
        policy.method = "B1"
        k = policy(snap0)
        assert k.diagnostics["effect_tokens"] == {} and k.diagnostics["successor_used"] is False
        policy.method = "B2"
        d = policy(snap)  # even with a non-empty source prior, B2 never feeds it
        assert d.diagnostics["successor_used"] is True and d.diagnostics["differences"], "B2 keeps DK/successor"
        assert all(torch.count_nonzero(v) == 0 for v in d.diagnostics["prior_inputs"].values())
        for out in (e, k, d):
            assert torch.isfinite(out.value) and torch.isfinite(out.logits[out.mask]).all()


# 5 -------------------------------------------------------------------------------------------------
def test_derived_manifest_and_split_bind_new_root_and_T_B(tmp_path):
    split = final_tb.derive_noprior_split(ROOT, tmp_path / "split.json")
    man = final_tb.derive_runtime_manifest(ROOT, tmp_path / "manifest.yaml", split["path"])
    text = Path(man["path"]).read_text()
    import re
    import yaml
    assert set(re.findall(r"graph_cp_disr_[A-Za-z0-9_]+", text)) <= {ROOT.name}
    doc = yaml.safe_load(text)
    assert doc["runtime"]["repository_path"] == str(ROOT) and doc["runtime"]["active_task_id"] == "T_B"
    assert doc["runtime"]["task_splits"]["T_B"] == split["path"]
    derived = json.loads(Path(split["path"]).read_text())
    original = json.loads((ROOT / final_tb.SPLIT_REL).read_text())
    assert [r["case_id"] for r in derived["train"]] == [r["case_id"] for r in original["train"]]
    assert [r["case_id"] for r in derived["dev"]] == [r["case_id"] for r in original["dev"]]
    assert not derived["test"] and all("cache_dir" not in r for r in derived["train"] + derived["dev"])
    # the frozen originals are not modified
    assert split["source_sha256"] == final_tb.sha256_file(ROOT / final_tb.SPLIT_REL)
    from cp_disr import stage2a_v11 as v11
    ctx = chk.make_context(tmp_path, "R-TB-K-1")
    with chk.configured(ROOT, ctx):
        assert str(v11.runtime_manifest_path(ROOT)) == ctx.runtime_manifest
        assert str(v11.ENABLED_SPLITS["T_B"]) == ctx.train_split
    assert v11.RUN_CONTEXT is None  # restored: default behaviour is unchanged for historical entries


# 6 -------------------------------------------------------------------------------------------------
def test_check_register_help_build_no_environment_and_smoke_train_refuse(tmp_path):
    assert chk.check_no_env_before_authorization(ROOT, tmp_path)


# 7 -------------------------------------------------------------------------------------------------
def test_two_workers_have_distinct_dirs_rng_and_reservations(tmp_path):
    ledger = final_tb.Ledger(tmp_path / "launch")
    ledger.init({p: {"status": "NOT_STARTED"} for p in final_tb.PLAN_TABLE})
    ledger.reserve("R-TB-E-0", "a0", tmp_path / "run_e", 100, 5)
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-E-0", "a0b", tmp_path / "run_e2", 101, 6)  # duplicate start
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-DK-1", "a1", tmp_path / "run_dk", 102, 5)  # same GPU already running
    ledger.reserve("R-TB-DK-1", "a1", tmp_path / "run_dk", 102, 6)
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-K-1", "a2", tmp_path / "run_k", 103, 7)  # third concurrent worker
    state = ledger.read()
    assert state["new_rl_attempts_used"] == 2 and state["new_rl_attempts_cap"] == 3
    assert {p["run_dir"] for p in state["plans"].values() if p.get("run_dir")} == {str(tmp_path / "run_e"), str(tmp_path / "run_dk")}
    ledger.finish("R-TB-E-0", "COMPLETED")
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-DK-1", "dup", tmp_path / "run_dk3", 104, 5)
    ledger.reserve("R-TB-K-1", "a2", tmp_path / "run_k", 103, 5)  # a free slot after one finished
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-K-1", "a3", tmp_path / "run_k2", 105, 5)  # attempt cap / duplicate
    assert ledger.read()["new_rl_attempts_used"] == 3
    # same directory cannot be reserved twice
    ledger2 = final_tb.Ledger(tmp_path / "launch2")
    ledger2.init({p: {"status": "NOT_STARTED"} for p in final_tb.PLAN_TABLE})
    ledger2.reserve("R-TB-E-0", "x", tmp_path / "same", 1, 5)
    with pytest.raises(BindingError):
        ledger2.reserve("R-TB-DK-1", "y", tmp_path / "same", 2, 6)


def test_workers_get_distinct_context_dirs_and_initialisation(tmp_path):
    e = chk.make_context(tmp_path / "e", "R-TB-E-0")
    d = chk.make_context(tmp_path / "d", "R-TB-DK-1")
    assert e.output_directory != d.output_directory and e.attempt_id != d.attempt_id
    oe, od = chk.run_until_first_step(ROOT, e), chk.run_until_first_step(ROOT, d)
    assert oe["initial_seed_at_step"] == 0 and od["initial_seed_at_step"] == 1
    assert oe["fingerprint"] != od["fingerprint"]


def test_gpu_parameters_reach_first_construction_and_env_factory(tmp_path):
    from cp_disr.platforms.libero import runtime_factory as rf
    from cp_disr import stage2a_v11 as v11
    seen = {}
    fake_env = mock.MagicMock()

    def fake_make_env(spec, **kw):
        seen.setdefault("calls", []).append(kw)
        return fake_env

    class Capture(Exception):
        pass

    def fake_bundle(**kw):
        seen["bundle"] = kw
        raise Capture()

    import dataclasses
    ctx = dataclasses.replace(chk.make_context(tmp_path, "R-TB-K-1"), render_gpu_device_id=6, physical_gpu_index=6)
    heavy = {n: mock.MagicMock() for n in ("DurationProvider", "SafetyManager", "SkillExecutor", "_Executor", "ObservationProvider",
                                           "PerceptionAdapter", "FactVerifier", "TaskEvaluator", "SnapshotBuilder")}
    with chk.configured(ROOT, ctx), mock.patch.object(rf, "make_env", fake_make_env), \
            mock.patch.object(rf, "RuntimeBundle", fake_bundle), mock.patch.multiple(rf, **heavy):
        with pytest.raises(Capture):
            v11.make_bundle(ROOT, "T_B")
    assert seen["calls"] == [{"gpu": 6}], "first construction must receive the explicit render device"
    assert seen["bundle"]["render_gpu_device_id"] == 6
    factory = seen["bundle"]["env_factory"]
    factory("spec")
    assert seen["calls"][-1] == {"gpu": 6}, "every reconstruction must use the same render device"
    assert final_tb.bind_worker_gpu(6, environ={})["render_gpu_device_id"] == 6
    env = {"MUJOCO_EGL_DEVICE_ID": "0", "DASHSCOPE_API_KEY": "k"}
    final_tb.bind_worker_gpu(5, environ=env)
    assert env["CUDA_VISIBLE_DEVICES"] == "5" and "MUJOCO_EGL_DEVICE_ID" not in env and "DASHSCOPE_API_KEY" not in env


# 8 -------------------------------------------------------------------------------------------------
def test_frozen_H_Tcap_invariant_and_rejections(tmp_path):
    from cp_disr import stage2a_v11 as v11
    ctx = chk.make_context(tmp_path, "R-TB-K-1")
    with chk.configured(ROOT, ctx) as v:
        prof = v.bind_H(ROOT)
        assert prof["H"] == final_tb.H_SECONDS == 23.09999999999752
        assert prof["d_ref"]["T_B"] == final_tb.D_REF_SECONDS == 4.199999999997672
        assert prof["Tcap"]["T_B"] == 16384 * 4.199999999997672 == final_tb.TCAP_SECONDS
        assert prof["task_deadlines"]["T_B"] == 60.0
    assert final_tb.envelope_stop(16383, 1.0) is None
    assert final_tb.envelope_stop(16384, 1.0) == "Ncap"
    assert final_tb.envelope_stop(10, final_tb.TCAP_SECONDS) == "Tcap"
    env = final_tb.envelope()
    assert env == {"ncap": 16384, "tcap": final_tb.TCAP_SECONDS, "d_ref": 4.199999999997672, "H": 23.09999999999752}
    assert (final_tb.ROLLOUT_N, final_tb.MAX_UPDATES, final_tb.EVAL_POINTS) == (1024, 16, (0, 4096, 8192, 16384))
    # source drift is refused
    ok = lambda r, *a: "c" * 40 if a[0] == "rev-parse" else ""
    assert final_tb.check_source_identity(ROOT, "c" * 40, git=ok)["tracked_tree_clean"]
    with pytest.raises(BindingError):
        final_tb.check_source_identity(ROOT, "d" * 40, git=ok)
    dirty = lambda r, *a: "c" * 40 if a[0] == "rev-parse" else " M src/x.py"
    with pytest.raises(BindingError):
        final_tb.check_source_identity(ROOT, "c" * 40, git=dirty)
    # a profile built from a tampered/other envelope is not accepted by the context
    for kw in ({"study_envelope_ncap": 32768}, {"study_envelope_tcap": 2 * final_tb.TCAP_SECONDS}):
        with pytest.raises(BindingError):
            chk.make_context(tmp_path / "bad", "R-TB-K-1", **kw)


# 9 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("plan_id,method,seed,policy_method", PLANS)
def test_checkpoint_save_load_consistency_and_tamper_detection(tmp_path, plan_id, method, seed, policy_method):
    import torch
    from cp_disr import stage2a_v11 as v11
    from cp_disr.persistence import GenerationStore, PersistenceError
    from cp_disr import torch_rl
    ctx = chk.make_context(tmp_path, plan_id)
    template = chk.make_template(ROOT, ctx.runtime_manifest)
    torch.manual_seed(seed)
    policy = v11.make_policy(template, method, torch.device("cpu"))
    fp = v11.param_fingerprint(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)  # constructed only: no optimizer.step anywhere
    path = tmp_path / "ck" / "model.pt"
    path.parent.mkdir()
    torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict()}, path)
    import hashlib
    path.with_suffix(".json").write_text(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "manifest": {}}))
    torch.manual_seed(seed + 100)
    twin = v11.make_policy(template, method, torch.device("cpu"))
    assert v11.param_fingerprint(twin) != fp
    torch_rl.load_checkpoint(path, twin)
    assert v11.param_fingerprint(twin) == fp
    # a state dict missing a parameter must fail strictly (no strict=False)
    state = torch.load(path, weights_only=False)
    victim = sorted(state["model"])[-1]
    del state["model"][victim]
    with pytest.raises(RuntimeError):
        twin.load_state_dict(state["model"])
    # published generations detect tampering
    store = GenerationStore(tmp_path / "persistence")
    store.publish({"model.pt": path, "manifest.json": {"N": 0}}, {"N": 0}, generation="initial")
    store.verify("initial")
    target = store.generations / "initial" / "model.pt"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(PersistenceError):
        store.verify("initial")


# 10 ------------------------------------------------------------------------------------------------
def test_smoke_classification_separates_controller_held_and_success():
    cs, ce = final_tb.classify_step, final_tb.classify_episode
    assert cs("PICK", "NORMAL_TERMINATION", "TRUE", False) == "STEP_OK"
    assert cs("PICK", "NORMAL_TERMINATION", "UNKNOWN", True) == "HELD_NOT_CONFIRMED"  # exit + success flag are not a grasp
    assert cs("PICK", "NORMAL_TERMINATION", "FALSE", False) == "HELD_NOT_CONFIRMED"
    assert cs("PICK", "TIMEOUT", "TRUE", False).startswith("CONTROLLER_EXIT:")
    assert cs("PLACE", "NORMAL_TERMINATION", "UNKNOWN", False) == "STEP_OK"
    assert cs("PICK", "NORMAL_TERMINATION", "TRUE", True, engineering_error="boom") == "ENGINEERING_FAILURE"
    assert ce([], True) == "FAIL:NO_STEPS"
    assert ce(["STEP_OK", "STEP_OK"], True) == "PASS"
    assert ce(["STEP_OK", "HELD_NOT_CONFIRMED"], True).startswith("FAIL:HELD")
    assert ce(["STEP_OK"], None) == "FAIL:TASK_NOT_SUCCESS" and ce(["STEP_OK"], False) == "FAIL:TASK_NOT_SUCCESS"


def test_smoke_budget_caps_and_case_selection():
    meter = final_tb.BudgetMeter(dict(final_tb.SMOKE_CAPS), {k: 0 for k in final_tb.SMOKE_CAPS})
    for _ in range(24):
        meter.take("skill_calls")
    with pytest.raises(final_tb.BudgetExceeded):
        meter.take("skill_calls")
    assert final_tb.SMOKE_CAPS == {"episode_attempts": 2, "construction_attempts": 4, "explicit_resets": 4, "skill_calls": 24}
    cases = final_tb.select_smoke_cases(ROOT, 2)
    assert [c["case_id"] for c in cases] == ["T_B_dev_00", "T_B_dev_04"]
    assert all(len(c["script"]) <= 12 and c["evidence_sha256"] for c in cases)
    dev = set(final_tb.split_lists(ROOT)["dev"])
    assert all(c["case_id"] in dev for c in cases)
    with pytest.raises(BindingError):
        final_tb.select_smoke_cases(ROOT, 2, episodes=[])


def test_entry_module_has_no_forbidden_calls():
    sources = {str(p): p.read_text() for p in (ROOT / "src/cp_disr/final_tb.py", ROOT / "scripts/final_tb_launch.py")}
    hits = final_tb.entry_forbidden_hits(sources)
    assert not any(hits.values()), hits


# mutation detection ---------------------------------------------------------------------------------
def _mutant_tree(tmp_path, name, edits):
    tree = tmp_path / name
    tree.mkdir()
    for item in ROOT.iterdir():
        if item.name in {"src", "tests"} or item.name == "__pycache__":
            continue
        os.symlink(item, tree / item.name)
    shutil.copytree(ROOT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tree / "tests").mkdir()
    shutil.copy(ROOT / "tests/__init__.py", tree / "tests/__init__.py")
    shutil.copytree(ROOT / "tests/helpers", tree / "tests/helpers", ignore=shutil.ignore_patterns("__pycache__"))
    os.symlink(ROOT / "tests/fixtures", tree / "tests/fixtures")
    for rel, old, new in edits:
        path = tree / rel
        text = path.read_text()
        assert text.count(old) == 1, (rel, old, text.count(old))
        path.write_text(text.replace(old, new))
    return tree


def _run_check(tree, tmp_path, check):
    code = ("import sys,tempfile;sys.path.insert(0,%r);from tests.helpers import final_tb_checks as c;"
            "from pathlib import Path\nwith tempfile.TemporaryDirectory() as t:\n    c.%s(Path(%r),t)\n") % (str(tree), check, str(tree))
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    return subprocess.run([sys.executable, "-c", code], cwd=str(tree), env=env, capture_output=True, text=True, timeout=280)


MUTATIONS = [
    ("check_seed_binding", "inner seed forced to 0",
     [("src/cp_disr/stage2a_v11.py", "s1.seed_all(seed)  # explicit training seed; never reset to 0 by the inner runner", "s1.seed_all(0)")]),
    ("check_plus_e_empty_r", "+E missing from the empty-R set",
     [("src/cp_disr/stage2a_v11.py", 'EMPTY_PRIOR_METHODS = {"B0", "B1", "B1-K", "B2", "B1-K+E"}', 'EMPTY_PRIOR_METHODS = {"B0", "B1", "B1-K", "B2"}')]),
    ("check_no_env_before_authorization", "GPU/environment binding before authorization",
     [("src/cp_disr/final_tb.py", "            # 1) authorization first:", "            bind_worker_gpu(args.gpu)\n            # 1) authorization first:")]),
]


@pytest.mark.parametrize("check,label,edits", MUTATIONS, ids=[m[1] for m in MUTATIONS])
def test_known_regressions_are_detected(tmp_path, check, label, edits):
    control = _run_check(_mutant_tree(tmp_path, "control", []), tmp_path, check)
    assert control.returncode == 0, "control copy must pass:\n" + control.stderr[-2000:]
    mutant = _run_check(_mutant_tree(tmp_path, "mutant", edits), tmp_path, check)
    assert mutant.returncode != 0, "regression NOT detected: %s" % label
    assert "AssertionError" in mutant.stderr or "Error" in mutant.stderr


# release gating -------------------------------------------------------------------------------------
def test_training_release_refuses_missing_or_unbound_smoke_receipts(tmp_path):
    """R2: the full pass/storage/predecessor chain is covered in tests/test_final_tb_smoke_r2.py (real smoke pair)."""
    head = "e" * 40
    git = lambda r, *a: head if a[0] == "rev-parse" else ""
    out = tmp_path / "launch"
    final_tb.run_register(ROOT, out, head, "auth text", stamp="T1", git=git)
    with pytest.raises(BindingError):  # no smoke receipt yet
        final_tb.run_release(out, "train", head, "R-TB-E-0")
    (out / "smoke_receipt.json").write_text(json.dumps({"label": "TB_LAUNCH_BLOCKED_RUNTIME"}))
    with pytest.raises(BindingError):
        final_tb.run_release(out, "train", head, "R-TB-E-0")
    (out / "smoke_receipt.json").write_text(json.dumps({"label": "TB_RUNTIME_SMOKE_PASS"}))
    with pytest.raises(BindingError):  # a bare PASS label is not a bound receipt (prep, authorization, case hashes)
        final_tb.run_release(out, "train", head, "R-TB-E-0")
