"""CP-DISR-TB-REP-CONTROLS-01: representation-control semantics, legacy protection and launch binding (offline)."""
import ast
import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.torch_runtime


@pytest.fixture(scope="module")
def template():
    from cp_disr import tb_repctl_checks as C
    return C.tb_template(ROOT)


@pytest.fixture(scope="module")
def states(template):
    from cp_disr import tb_repctl_checks as C
    return C.state_suite(template, count=4)


# ----------------------------------------------------------------------------- Absolute Successor
def test_A1_f_after_identical_to_b2_field_by_field(template, states):
    from cp_disr import tb_repctl_checks as C
    assert C.check_A1_f_after_identical(template, states)["candidate_states_compared"] > 0


def test_A2_breaking_nominal_apply_fails_the_checks(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_A1_f_after_identical(template, states)
    with C.mutant_break_nominal_apply():
        with pytest.raises(AssertionError):
            C.check_A1_f_after_identical(template, states)
    C.check_nominal_apply_is_load_bearing(template, states)


def test_A3_restoring_latent_subtraction_is_detected(template, states):
    from cp_disr import tb_repctl_checks as C
    assert C.check_A3_rows_are_absolute(template, states)["candidate_rows_checked"] > 0
    with C.mutant_restore_latent_subtraction():
        with pytest.raises(AssertionError):
            C.check_A3_rows_are_absolute(template, states)


def test_A4_candidate_permutation_alignment(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_permutation(template, "B2-ABS", states)


def test_absolute_successor_has_b2_parameters_and_computation(template, states):
    from cp_disr import tb_repctl_checks as C
    snap = C.make_snapshot(template, states[0][1], states[0][0])
    b2, ab = C.make_policy(template, "B2", 0), C.make_policy(template, "B2-ABS", 0)
    assert list(b2.state_dict()) == list(ab.state_dict())
    assert C.effective_parameters(b2, snap)[0] == C.effective_parameters(ab, snap)[0]
    assert C.computation_counts(template, "B2", snap) == C.computation_counts(template, "B2-ABS", snap)


# ----------------------------------------------------------------------------- Matched Neural Composition
def test_N1_source_scan(template):
    from cp_disr import tb_repctl_checks as C
    assert C.check_N1_source_scan()["hits"] == []


def test_N2_runtime_trap_and_mutant(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_N2_runtime_trap(template, states)
    with C.mutant_nc_calls_nominal_apply():
        with pytest.raises(AssertionError):
            C.check_N2_runtime_trap(template, states)


def test_N1_mutant_with_four_views_in_source(template):
    from cp_disr import tb_repctl_checks as C
    with C.mutant_nc_source_calls_four_views():
        with pytest.raises(AssertionError):
            C.check_N1_source_scan()


def test_N3_effects_change_the_representation(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_N3_effects_change_representation(template, states)
    with C.mutant_nc_ignores_effects():
        with pytest.raises(AssertionError):
            C.check_N3_effects_change_representation(template, states)


def test_N4_current_state_changes_the_representation(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_N4_state_changes_representation(template, states)
    with C.mutant_nc_ignores_state():
        with pytest.raises(AssertionError):
            C.check_N4_state_changes_representation(template, states)


def test_N5_candidate_permutation_alignment(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_permutation(template, "B1-K+NC", states)


def test_N6_no_sidecar_no_file_read(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_N6_no_sidecar(template, states)
    with C.mutant_nc_reads_a_file():
        with pytest.raises(AssertionError):
            C.check_N6_no_sidecar(template, states)


def test_neural_composition_capacity_within_five_percent_of_b2(template, states):
    from cp_disr import tb_repctl_checks as C
    snap = C.make_snapshot(template, states[0][1], states[0][0])
    b2 = C.effective_parameters(C.make_policy(template, "B2", 0), snap)[0]
    nc = C.effective_parameters(C.make_policy(template, "B1-K+NC", 0), snap)[0]
    assert abs(nc - b2) / b2 <= 0.05


# ----------------------------------------------------------------------------- common
def test_common_boundaries(template, states):
    from cp_disr import tb_repctl_checks as C
    C.check_common_boundaries(template, states)


def test_registration_is_in_process_only_and_legacy_factory_is_delegated(monkeypatch):
    import types
    from cp_disr import neural, final_tb, final_tb_repctl as R, stage2a_v11 as v11, repctl_policy
    for m in ("B2-ABS", "B1-K+NC"):
        assert m not in neural.METHODS and m not in v11.EMPTY_PRIOR_METHODS  # neural.py / stage2a_v11.py are byte-identical to the base commit
    assert v11.METHODS == ("B0", "B1-K", "B2", "Full")  # the legacy sweep tuple is untouched
    fake = types.SimpleNamespace(EMPTY_PRIOR_METHODS={"B0", "B1", "B1-K", "B2", "B1-K+E"}, make_policy=lambda template, method, device: "production:" + method,
                                 OBS_DIM=48, CAND_DIM=8, B_PRIOR=0.5)
    monkeypatch.setattr(final_tb, "configure_v11", lambda root, ctx: fake)
    monkeypatch.setattr(neural, "METHODS", tuple(neural.METHODS))
    got = R.configure_v11(ROOT, None)
    assert got is fake and {"B2-ABS", "B1-K+NC", "B2", "B1-K+E"} <= fake.EMPTY_PRIOR_METHODS
    assert all(m in neural.METHODS and neural.canonical_method(m) == m for m in repctl_policy.CONTROL_METHODS)
    assert fake.make_policy(None, "B2", "cpu") == "production:B2" and fake.make_policy(None, "B1-K+E", "cpu") == "production:B1-K+E"
    assert R.configure_v11(ROOT, None).make_policy is fake.make_policy  # idempotent


def test_legacy_state_dicts_unchanged_by_the_new_modules(template):
    from cp_disr import tb_repctl_checks as C
    for m in ("B1-K", "B1-K+E", "B2"):
        assert not any(k.startswith("state_effect") for k in C.make_policy(template, m, 0).state_dict())
    nc = C.make_policy(template, "B1-K+NC", 0).state_dict()
    legacy = C.make_policy(template, "B2", 0).state_dict()
    assert set(legacy) < set(nc) and all(k.startswith("state_effect") for k in set(nc) - set(legacy))
    import torch
    assert all(torch.equal(legacy[k], nc[k]) for k in legacy), "shared parts must be initialised identically at the same seed"


def test_protected_production_files_are_unchanged_against_the_base_commit():
    from cp_disr import tb_repctl_checks as C
    protected = [
        "src/cp_disr/collector.py", "src/cp_disr/rl.py", "src/cp_disr/torch_rl.py", "src/cp_disr/adapters.py", "src/cp_disr/evaluation.py",
        "src/cp_disr/execution.py", "src/cp_disr/graph.py", "src/cp_disr/contracts.py", "src/cp_disr/facts.py", "src/cp_disr/phase_a_v12.py",
        "src/cp_disr/persistence.py", "src/cp_disr/final_tb.py", "src/cp_disr/final_tb_e1.py", "src/cp_disr/final_tb_eval_record.py", "src/cp_disr/stage2a_v11.py", "src/cp_disr/neural.py",
        "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/task_evaluator.py",
        "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/snapshot.py", "src/cp_disr/platforms/libero/safety.py",
        "configs/splits/T_B_phase_a_v13_r1_dev10.json", "configs/runtime/stage_2a_contract_registry.yaml", "configs/tasks/resolved/T_B.yaml",
        "experiments/manifests/runtime_manifest_v211.yaml",
    ]
    for rel in protected:
        base = subprocess.check_output(["git", "show", "%s:%s" % (C.BASE_COMMIT, rel)], cwd=str(ROOT))
        assert hashlib.sha256(base).hexdigest() == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), "%s changed" % rel


def test_no_pre_existing_tracked_file_is_modified():
    from cp_disr import tb_repctl_checks as C
    changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=MD", C.BASE_COMMIT, "--"], cwd=str(ROOT), text=True).split()
    assert changed == [], changed


def test_new_production_modules_import_no_provider_vlm_or_test_cache():
    forbidden = {"vlm_provider", "vlm_cache_pipeline", "vlm", "prior", "stage2a_runner", "phase_a_v13_r1", "phase_a_v13_r3"}
    for rel in ("src/cp_disr/final_tb_repctl.py", "src/cp_disr/tb_repctl_checks.py", "src/cp_disr/repctl_policy.py"):
        tree = ast.parse((ROOT / rel).read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[-1])
                mods.update(a.name for a in n.names)
            elif isinstance(n, ast.Import):
                mods.update(a.name.split(".")[-1] for a in n.names)
        assert not (mods & forbidden), (rel, mods & forbidden)


# ----------------------------------------------------------------------------- PPO path (synthetic transitions, CPU, no environment)
@pytest.mark.parametrize("method,changed_prefixes", [("B2-ABS", ("encoder.", "contract.", "v_head.", "q_head.")), ("B1-K+NC", ("state_effect.", "encoder.features.", "contract."))])
def test_ppo_update_runs_and_updates_the_expected_modules(template, states, method, changed_prefixes):
    import torch
    from dataclasses import replace
    from cp_disr import tb_repctl_checks as C
    from cp_disr.rl import Rollout, Transition
    from cp_disr.torch_rl import PPO, prefix_hidden
    policy = C.make_policy(template, method, 0)
    policy.train()
    rollout = Rollout()
    for ep, (seed, values) in enumerate(states[:3]):
        base = C.make_snapshot(template, values, seed)
        snaps = [replace(base, episode_id="ep%d" % ep, decision_id=d) for d in range(4)]
        for d in range(3):
            s, n, prefix = snaps[d], snaps[d + 1], tuple(snaps[:d])
            with torch.no_grad():
                out = policy(s, prefix_hidden(policy, prefix))
                v_next = float(policy(n, prefix_hidden(policy, prefix + (s,))).value)
            cid = next(c for c, ok in zip(s.candidate_ids, s.mask) if ok)
            logp = float(out.distribution.log_prob(torch.tensor(s.candidate_ids.index(cid))))
            last = d == 2
            rollout.append(Transition(s, n, cid, logp, float(out.value), 0.0 if last else v_next, 1.0 if last else 0.0, 1.0, 1.0, last, False, "synthetic", prefix))
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    logs = PPO(policy).update(rollout, epochs=2, minibatch=8, sequence_length=4)
    assert logs and all(torch.isfinite(torch.tensor(l["total"])) and torch.isfinite(torch.tensor(l["grad_norm"])) for l in logs)
    after = policy.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    for prefix in changed_prefixes:
        assert any(k.startswith(prefix) for k in moved), "no parameter under %s was updated" % prefix
    assert not any(k.startswith(("set_encoder.", "effect_readout.", "effect_fuse.", "b0_fuse.", "cat_proj.")) for k in moved), "an unused legacy module was updated"


# ----------------------------------------------------------------------------- launch binding
def test_plan_table_is_six_attempts_two_methods_three_seeds():
    from cp_disr import final_tb_repctl as R
    assert len(R.PLAN_TABLE) == 6 and R.MAX_ATTEMPTS == 6 and R.MAX_WORKERS == 3
    assert {(v["method"], v["seed"]) for v in R.PLAN_TABLE.values()} == {(m, s) for m in R.METHODS for s in (0, 1, 2)}
    assert sorted(v["queue_order"] for v in R.PLAN_TABLE.values()) == [1, 2, 3, 4, 5, 6]


def test_baseline_plan_table_untouched():
    from cp_disr import final_tb
    assert final_tb.METHODS == ("B1-K", "B1-K+E", "B2") and len(final_tb.PLAN_TABLE) == 4


def test_assignment_and_context_validation():
    from cp_disr import final_tb_repctl as R
    assert R.validate_assignment("R-TB-NC-2", "B1-K+NC", 2)["queue_order"] == 6
    for args in (("R-TB-NC-2", "B2-ABS", 2), ("R-TB-NC-2", "B1-K+NC", 1), ("R-TB-NC-9", "B1-K+NC", 2), ("R-TB-E-0", "B1-K+E", 0)):
        with pytest.raises(Exception):
            R.validate_assignment(*args)
    cfg = {"plan_id": "R-TB-ABS-1", "attempt_id": "R-TB-ABS-1-T-aaaaaaaa", "method": "B2-ABS", "training_seed": 1, "source_commit": "a" * 40,
           "output_directory": "/x/y", "runtime_manifest": "/x/m.yaml", "train_split": "/x/s.json", "prior_mode": "absent",
           "study_envelope_ncap": 16384, "study_envelope_tcap": 16384 * 4.199999999997672, "evaluation_rule": "0/4096/8192/final; frozen dev10"}
    ctx = R.context_from_dict(cfg)
    ctx.assert_worker("T_B", "B2-ABS")
    with pytest.raises(Exception):
        ctx.assert_worker("T_B", "B2")
    for bad in ({"prior_mode": "original"}, {"study_envelope_ncap": 8192}, {"training_seed": 5}, {"extra_key": 1}, {"evaluation_rule": "other"}):
        with pytest.raises(Exception):
            R.context_from_dict({**cfg, **bad})


def _ledger(tmp_path):
    from cp_disr import final_tb_repctl as R
    ledger = R.RepctlLedger(tmp_path)
    ledger.init({p: {"method": v["method"], "seed": v["seed"], "status": "NOT_STARTED", "queue_order": v["queue_order"]} for p, v in R.PLAN_TABLE.items()})
    return ledger


def test_ledger_allows_three_concurrent_workers_and_one_attempt_per_plan(tmp_path):
    ledger = _ledger(tmp_path)
    plans = ["R-TB-ABS-0", "R-TB-NC-0", "R-TB-ABS-1", "R-TB-NC-1"]
    for i, p in enumerate(plans[:3]):
        ledger.reserve(p, "att-" + p, "/runs/" + p, 100 + i, i)
    with pytest.raises(Exception, match="concurrent"):
        ledger.reserve(plans[3], "att", "/runs/x", 1, 5)
    with pytest.raises(Exception, match="duplicate"):
        ledger.reserve(plans[0], "att2", "/runs/y", 2, 6)
    ledger.finish("R-TB-ABS-0", "COMPLETE")
    ledger.reserve(plans[3], "att4", "/runs/p4", 9, 0)
    assert ledger.read()["new_rl_attempts_used"] == 4
    ledger.finish("R-TB-NC-0", "COMPLETE")
    with pytest.raises(Exception, match="GPU"):
        ledger.reserve("R-TB-ABS-2", "att5", "/runs/p5", 10, 2)  # GPU 2 still holds R-TB-ABS-1


def test_pick_gpu_skips_busy_and_taken():
    from cp_disr import final_tb_repctl as R
    table = {0: {"memory_used_mib": 20000, "utilization": 90}, 1: {"memory_used_mib": 10, "utilization": 0}, 2: {"memory_used_mib": 12, "utilization": 0},
             7: {"memory_used_mib": 5, "utilization": 0}}
    assert R.pick_gpu(table, set()) == 1
    assert R.pick_gpu(table, {1}) == 2
    assert R.pick_gpu(table, {1, 2}) is None  # GPU 7 is outside the allowed candidates


def test_supervisor_respects_queue_order_and_concurrency(tmp_path):
    from cp_disr import final_tb_repctl as R, final_tb as ftb
    ledger = _ledger(tmp_path)
    ftb.write_json_atomic(tmp_path / "launch_plan.json", {"queue": sorted(R.PLAN_TABLE, key=lambda p: R.PLAN_TABLE[p]["queue_order"])})
    spawned = []

    class P:
        pid = 4242
        def poll(self):
            return None

    def spawn(cmd, logfile):
        plan = cmd[cmd.index("--plan") + 1]
        spawned.append((plan, int(cmd[cmd.index("--gpu") + 1])))
        ledger.reserve(plan, "a-" + plan, "/r/" + plan, 4242, spawned[-1][1])
        return P()
    gpus = {i: {"memory_used_mib": 5, "utilization": 0} for i in range(7)}
    out = R.supervise(ROOT, tmp_path, python="py", poll_seconds=0, max_ticks=8, gpu_fn=lambda: gpus, spawn=spawn, sleep=lambda s: None, log=lambda *a: None)
    assert [p for p, _ in spawned] == ["R-TB-ABS-0", "R-TB-NC-0", "R-TB-ABS-1"]
    assert len({g for _, g in spawned}) == 3
    assert out["halted"] is None


def test_prep_receipt_gate_blocks_missing_receipts(tmp_path):
    from cp_disr import final_tb_repctl as R
    with pytest.raises(Exception, match="missing"):
        R.verify_prep_receipts(ROOT, tmp_path)
