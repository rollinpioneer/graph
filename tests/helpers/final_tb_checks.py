"""Behavioural checks on the production T_B launch entry (CP-DISR-NEXT-ENGINEERING-FIRST-TB-1).

The checks use the real production functions (Policy, PPO, Collector, v11.train_job, v12 persistence,
final_tb.*).  Only the simulator / disk-heavy boundary is stubbed (environment bundle, dev evaluation,
Collector.step).  They are plain functions so that both pytest and the mutation driver can call them.
Synthetic fixtures are engineering-only and never enter training data or any reported performance.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures"


class Sentinel(Exception):
    """Raised from the stubbed Collector.step: collection (the simulator boundary) is never reached."""


def fixture_snapshot(prior=True):
    from cp_disr.fixtures import build_fixture, snapshot
    return snapshot(build_fixture(FIXTURE_DIR), prior=prior)


def make_template(root, manifest_path=None):
    """Same template construction as runtime_factory.create_task_runtime, without any environment."""
    from cp_disr.graph import build_template, Goal
    from cp_disr.platforms.libero import runtime_factory as rf
    manifest_path = manifest_path or Path(root) / "experiments/manifests/runtime_manifest_v211.yaml"
    runtime = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))["runtime"]
    timeouts = runtime["skill_timeouts"]["T_B"]
    contracts = rf._ground_contracts_for(Path(root) / runtime.get("stage_2a_contract_path", "configs/runtime/stage_2a_contract_registry.yaml"),
                                         timeouts, rf.TASK_OBJECTS["T_B"])
    for c in contracts:
        object.__setattr__(c, "timeout_seconds", float(timeouts[c.name]))
    goals = tuple(Goal(g, 1) for g in rf.TASK_GOALS["T_B"])
    return build_template(contracts, goals, rf.PREDICATES, rf.TASK_OBJECTS["T_B"])


class FakeBundle:
    """Stub of the simulator-side RuntimeBundle: just enough state for train_job / persistence."""

    def __init__(self, template, snap):
        self.template = template
        self._snap = snap
        self.environment = SimpleNamespace(
            sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3), ctrl=np.zeros(2), time=0.0),
                                forward=lambda: None),
            close=lambda: None)
        self.clock = SimpleNamespace(_origin=0.0, now_seconds=lambda: 0.0)
        self.episode_start_seconds = 0.0
        self._n = 0
        self.snapshot_builder = SimpleNamespace(episode_id="ep-0")
        self.evaluator = SimpleNamespace(_rewarded=False, _success_time=None, deadline=60.0)
        self.original_prior_edges = ()
        self.current_snapshot = None
        self.cases = {}
        self.caches = {}
        self.task_id = "T_B"
        self.start_case_calls = []

    def next_case(self, cases, index):
        return cases[0]

    def start_case(self, case_id):
        self.start_case_calls.append(case_id)
        self.original_prior_edges = self._snap.prior_edges  # a non-empty source prior must NOT reach the policy
        return self._snap


@contextlib.contextmanager
def configured(root, ctx):
    """Apply final_tb.configure_v11 and restore every module global afterwards (no test pollution)."""
    from cp_disr import final_tb, phase_a_v12 as v12, stage2a_v11 as v11
    v11_keys = ["N_CAP", "ROLLOUT_N", "MAX_UPDATES", "DEV_EPISODES", "TASKS", "METHODS", "ENABLED_SPLITS", "save_ckpt",
                "start_episode", "maybe_eval_at_n", "select_checkpoint", "RUN_CONTEXT"]
    v12_keys = ["EVAL_POINTS", "DEV10_N", "_source_hashes", "_fixed_mechanism_diagnostic"]
    saved11 = {k: getattr(v11, k) for k in v11_keys}
    saved12 = {k: getattr(v12, k) for k in v12_keys}
    try:
        final_tb.configure_v11(root, ctx)
        yield v11
    finally:
        for k, v in saved11.items():
            setattr(v11, k, v)
        for k, v in saved12.items():
            setattr(v12, k, v)


def make_context(tmp, plan_id, root=ROOT, **override):
    """Registered-style run config built from the real frozen sources (derived manifest/split in tmp)."""
    from cp_disr import final_tb
    tmp = Path(tmp)
    split = final_tb.derive_noprior_split(root, tmp / "split.json")
    manifest = final_tb.derive_runtime_manifest(root, tmp / "manifest.yaml", split["path"])
    row = final_tb.PLAN_TABLE[plan_id]
    cfg = {"plan_id": plan_id, "attempt_id": "%s-test-attempt" % plan_id, "method": row["method"],
           "training_seed": row["seed"], "source_commit": "a" * 40, "output_directory": str(tmp / "run" / plan_id),
           "runtime_manifest": manifest["path"], "train_split": split["path"], "prior_mode": "absent",
           "study_envelope_ncap": final_tb.N_CAP, "study_envelope_tcap": final_tb.TCAP_SECONDS,
           "evaluation_rule": final_tb.EVALUATION_RULE}
    cfg.update(override)
    return final_tb.context_from_dict(cfg)


def run_until_first_step(root, ctx, device="cpu", prior_sampler_raises=True):
    """Run the REAL v11.train_job (Policy, PPO, persistence, start_episode) up to the first collection step."""
    import torch
    from cp_disr import final_tb, phase_a_v12 as v12, prior as prior_mod, stage2a_v11 as v11
    from cp_disr.collector import Collector
    snap = fixture_snapshot(prior=True)
    assert snap.prior_edges, "fixture must carry a non-empty source prior"
    observed = {"sampler_calls": 0}
    holder = {}

    def fake_make_bundle(root_, task_id, split_rel=None):
        observed["manifest_used"] = str(v11.runtime_manifest_path(root_))
        observed["task_id"] = task_id
        bundle = FakeBundle(make_template(root_, v11.runtime_manifest_path(root_)), snap)
        holder["bundle"] = bundle
        return bundle

    real_make_policy = v11.make_policy

    def recording_make_policy(template, method, device_):
        policy = real_make_policy(template, method, device_)
        holder["policy"] = policy
        observed["policy_method"] = policy.method
        observed["fingerprint"] = v11.param_fingerprint(policy)
        observed["initial_seed_at_policy"] = torch.initial_seed()
        return policy

    def fake_eval(root_, task_id, method, ckpt_path, cases, split_index, device_, hashes_doc, out_eval, n_episodes=None, label="dev"):
        observed.setdefault("evals", []).append({"checkpoint": str(ckpt_path), "n_cases": len(cases), "label": label})
        payload = {"success_rate": 0.0, "success_n": 0, "mean_discounted_return": 0.0, "n": len(cases), "rows": [],
                   "optimizer_steps": 0}
        Path(out_eval).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_step(self, snapshot, deterministic=False):
        observed["initial_seed_at_step"] = torch.initial_seed()
        observed["step_prior_edges"] = tuple(snapshot.prior_edges)
        observed["step_prior_hash"] = snapshot.prior_hash
        raise Sentinel()

    class RaisingSampler:
        def __init__(self, *a, **k):
            observed["sampler_calls"] += 1
            raise AssertionError("PriorSampler must never be constructed for T_B runs")

    prof = None
    with configured(root, ctx) as v11_cfg:
        prof = v11_cfg.bind_H(root)
        prof["Ncap"], prof["max_updates"] = final_tb.N_CAP, final_tb.MAX_UPDATES
        hashes = final_tb.make_source_hashes(root, ctx)
        patches = [mock.patch.object(v11_cfg, "make_bundle", fake_make_bundle),
                   mock.patch.object(v11_cfg, "make_policy", recording_make_policy),
                   mock.patch.object(v11_cfg, "eval_episodes", fake_eval),
                   mock.patch.object(Collector, "step", fake_step)]
        if prior_sampler_raises:
            patches.append(mock.patch.object(prior_mod, "PriorSampler", RaisingSampler))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                v11_cfg.train_job(root, final_tb.TASK, ctx.method, torch.device(device), "cpu", prof, hashes,
                                  ctx.attempt_id, "deadbeef", max_updates=final_tb.MAX_UPDATES, resume=False)
            except Sentinel:
                observed["reached_first_step"] = True
    observed["job_dir"] = ctx.output_directory
    observed["bundle"] = holder.get("bundle")
    observed["prof"] = {"H": prof["H"], "Tcap": prof["Tcap"]["T_B"], "d_ref": prof["d_ref"]["T_B"]}
    return observed


# ---- the three regression checks (also run against mutated copies of the source) ----------------------
def check_seed_binding(root, tmp):
    """Seed 1 must reach the inner runner (never reset to 0) and be reproducible."""
    ctx_a = make_context(Path(tmp) / "a", "R-TB-K-1", root)
    ctx_b = make_context(Path(tmp) / "b", "R-TB-K-1", root)
    obs_a, obs_b = run_until_first_step(root, ctx_a), run_until_first_step(root, ctx_b)
    assert obs_a["initial_seed_at_step"] == 1, "inner runner reset the training seed (got %s)" % obs_a["initial_seed_at_step"]
    assert obs_a["fingerprint"] == obs_b["fingerprint"], "same seed must reproduce the initialisation"
    ctx_e = make_context(Path(tmp) / "e", "R-TB-E-0", root)
    ctx_e1 = make_context(Path(tmp) / "e1", "R-TB-E-0", root, training_seed=0)
    assert run_until_first_step(root, ctx_e)["initial_seed_at_step"] == 0
    cfg = json.loads((Path(ctx_a.output_directory) / "resolved_config.json").read_text())
    assert cfg["seed"] == 1 and cfg["training_seed"] == 1 and cfg["plan_id"] == "R-TB-K-1"
    return True


def check_plus_e_empty_r(root, tmp):
    """B1-K+E must use an empty effective R in collection, bootstrap and evaluation (no sampler)."""
    from cp_disr import stage2a_v11 as v11
    snap = fixture_snapshot(prior=True)
    assert snap.prior_edges
    def sampler_must_not_run(*a, **k):
        raise AssertionError("sampler touched")
    bomb = SimpleNamespace(start=sampler_must_not_run)
    for method in ("B1-K", "B1-K+E", "B2"):
        for eval_original in (False, True):  # collection and evaluation paths
            out, prior, source_n = v11.apply_episode_prior(method, SimpleNamespace(original_prior_edges=snap.prior_edges),
                                                           snap, bomb, eval_original=eval_original)
            assert out.prior_edges == () and prior.edges == () and prior.audit_mode == "absent", \
                "%s (eval_original=%s) did not get an empty R" % (method, eval_original)
            assert source_n == len(snap.prior_edges)
    return True


def check_no_env_before_authorization(root, tmp):
    """check/register/--help never construct an environment; smoke/train refuse before any GPU/env step."""
    from cp_disr import final_tb
    from cp_disr.platforms.libero import d0_env, runtime_factory as rf
    calls = []

    def boom(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError("forbidden call: %s" % name)
        return f

    out = Path(tmp) / "launch"
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(rf, "create_task_runtime", boom("create_task_runtime")))
        stack.enter_context(mock.patch.object(rf, "make_env", boom("rf.make_env")))
        stack.enter_context(mock.patch.object(d0_env, "make_env", boom("d0.make_env")))
        stack.enter_context(mock.patch.object(d0_env.D0ManipulationEnv, "__init__", boom("env.__init__")))
        stack.enter_context(mock.patch.object(d0_env.D0ManipulationEnv, "reset", boom("env.reset")))
        stack.enter_context(mock.patch.object(d0_env.D0ManipulationEnv, "step", boom("env.step")))
        stack.enter_context(mock.patch.object(final_tb, "bind_worker_gpu", boom("bind_worker_gpu")))
        stack.enter_context(mock.patch.object(final_tb, "query_gpu_uuid", boom("query_gpu_uuid")))
        auth_text = Path(tmp) / "authorization.txt"
        auth_text.write_text("test authorization text", encoding="utf-8")
        head = "b" * 40
        fake_git = lambda r, *a: head if a[0] == "rev-parse" else ""
        final_tb.run_register(root, out, head, "test authorization text", stamp="T0", git=fake_git)
        try:
            final_tb.main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        assert final_tb.main(["check", "--root", str(root), "--out", str(Path(tmp) / "check_out")]) == 0
        # no token -> refused first
        for argv in (["smoke", "--root", str(root), "--out", str(out), "--case-index", "1", "--gpu", "0"],
                     ["train", "--root", str(root), "--out", str(out), "--plan", "R-TB-E-0", "--gpu", "0"]):
            assert final_tb.main(argv) == 2, "must refuse without a release token: %s" % argv[0]
        # a smoke token must not release training; a valid smoke token still stops at source drift (fake prep)
        smoke_token = str(out / "release_tokens" / "smoke.json")
        assert final_tb.main(["train", "--root", str(root), "--out", str(out), "--plan", "R-TB-E-0", "--gpu", "0",
                              "--token", smoke_token]) == 2
        assert final_tb.main(["smoke", "--root", str(root), "--out", str(out), "--case-index", "1", "--gpu", "0",
                              "--token", smoke_token]) == 2  # HEAD != registered fake prep -> source drift
    assert calls == [], "environment/GPU code ran before authorization: %s" % calls
    return True
