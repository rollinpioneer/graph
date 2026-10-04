"""CP-DISR-TB-STRUCT-GEN-V1: launch binding, release gate and the PPO path over mixed goal templates (offline)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plan_table_is_twelve_attempts_four_methods_three_seeds_seed_major():
    from cp_disr import final_tb_structgen as S
    assert len(S.PLAN_TABLE) == 12 and S.MAX_ATTEMPTS == 12 and S.MAX_WORKERS == 6
    assert {(v["method"], v["seed"]) for v in S.PLAN_TABLE.values()} == {(m, s) for m in ("B1-K+E", "B2", "B2-ABS", "B1-K+NC") for s in (0, 1, 2)}
    order = sorted(S.PLAN_TABLE, key=lambda p: S.PLAN_TABLE[p]["queue_order"])
    assert [S.PLAN_TABLE[p]["seed"] for p in order] == [0] * 4 + [1] * 4 + [2] * 4 and sorted(S.PLAN_TABLE[p]["queue_order"] for p in order) == list(range(1, 13))


def test_assignment_and_context_validation():
    from cp_disr import final_tb_structgen as S
    assert S.validate_assignment("R-TB-SG-NC-2", "B1-K+NC", 2)["queue_order"] == 12
    for args in (("R-TB-SG-NC-2", "B2", 2), ("R-TB-SG-NC-2", "B1-K+NC", 1), ("R-TB-SG-NC-3", "B1-K+NC", 3), ("R-TB-NC-2", "B1-K+NC", 2)):
        with pytest.raises(Exception):
            S.validate_assignment(*args)
    cfg = {"plan_id": "R-TB-SG-ABS-1", "attempt_id": "R-TB-SG-ABS-1-T-aaaaaaaa", "method": "B2-ABS", "training_seed": 1, "source_commit": "a" * 40, "output_directory": "/x/y",
           "runtime_manifest": "/x/m.yaml", "train_split": "/x/s.json", "prior_mode": "absent", "study_envelope_ncap": 16384, "study_envelope_tcap": 16384 * 4.199999999997672,
           "evaluation_rule": "0/4096/8192/final; frozen dev10"}
    S.context_from_dict(cfg).assert_worker("T_B", "B2-ABS")
    for bad in ({"training_seed": 0}, {"prior_mode": "original"}, {"study_envelope_ncap": 8192}, {"extra": 1}, {"method": "B2"}):
        with pytest.raises(Exception):
            S.context_from_dict({**cfg, **bad})


def _ledger(tmp_path):
    from cp_disr import final_tb_structgen as S
    ledger = S.StructgenLedger(tmp_path)
    ledger.init({p: {"method": v["method"], "seed": v["seed"], "status": "NOT_STARTED", "queue_order": v["queue_order"]} for p, v in S.PLAN_TABLE.items()})
    return ledger


def test_ledger_six_workers_twelve_attempts_no_shared_gpu(tmp_path):
    from cp_disr import final_tb_structgen as S
    ledger = _ledger(tmp_path)
    order = sorted(S.PLAN_TABLE, key=lambda p: S.PLAN_TABLE[p]["queue_order"])
    for i, p in enumerate(order[:6]):
        ledger.reserve(p, "a-" + p, "/runs/" + p, 100 + i, i)
    with pytest.raises(Exception, match="concurrent"):
        ledger.reserve(order[6], "a", "/runs/x", 1, 6)
    ledger.finish(order[0], "COMPLETE")
    with pytest.raises(Exception, match="GPU"):
        ledger.reserve(order[6], "a", "/runs/x", 1, 1)
    ledger.reserve(order[6], "a", "/runs/x", 1, 0)
    with pytest.raises(Exception, match="duplicate"):
        ledger.reserve(order[1], "again", "/runs/y", 2, 5)


def test_pick_gpu_prefers_the_least_used_distinct_gpu():
    from cp_disr import final_tb_structgen as S
    table = {0: {"memory_used_mib": 9000, "utilization": 90}, 1: {"memory_used_mib": 100, "utilization": 0}, 2: {"memory_used_mib": 3000, "utilization": 50}, 7: {"memory_used_mib": 5, "utilization": 0},
             3: {"memory_used_mib": 30000, "utilization": 99}}
    assert S.pick_gpu(table, set()) == 1 and S.pick_gpu(table, {1}) == 2 and S.pick_gpu(table, {1, 2}) == 0 and S.pick_gpu(table, {0, 1, 2}) is None


def test_supervisor_starts_in_queue_order_with_at_most_six_workers(tmp_path):
    from cp_disr import final_tb_structgen as S, final_tb as ftb
    ledger = _ledger(tmp_path)
    ftb.write_json_atomic(tmp_path / "launch_plan.json", {"queue": sorted(S.PLAN_TABLE, key=lambda p: S.PLAN_TABLE[p]["queue_order"])})
    spawned = []

    class P:
        pid = 4242

        def poll(self):
            return None

    def spawn(cmd, logfile):
        plan = cmd[cmd.index("--plan") + 1]
        gpu = int(cmd[cmd.index("--gpu") + 1])
        spawned.append((plan, gpu))
        ledger.reserve(plan, "a-" + plan, "/r/" + plan, 4242, gpu)
        return P()
    gpus = {i: {"memory_used_mib": 100 + i, "utilization": 0} for i in range(7)}
    S.supervise(ROOT, tmp_path, python="py", poll_seconds=0, max_ticks=10, gpu_fn=lambda: gpus, spawn=spawn, sleep=lambda s: None, log=lambda *a: None)
    assert [p for p, _ in spawned] == sorted(S.PLAN_TABLE, key=lambda p: S.PLAN_TABLE[p]["queue_order"])[:6] and len({g for _, g in spawned}) == 6


def test_release_gate_refuses_missing_or_failed_release(tmp_path):
    from cp_disr import final_tb_structgen as S
    with pytest.raises(Exception, match="missing"):
        S.verify_release(ROOT, tmp_path)


def test_training_split_file_carries_no_test_rows_and_the_runtime_reads_only_it():
    from cp_disr import final_tb_structgen as S
    doc = json.loads((ROOT / S.TRAIN_DEV_REL).read_text())
    assert doc["test"] == [] and len(doc["train"]) == 60 and len(doc["dev"]) == 12
    test_ids = {r["case_id"] for r in json.loads((ROOT / S.TEST_REL).read_text())["test"]}
    assert not (test_ids & {r["case_id"] for r in doc["train"] + doc["dev"]})
    src = (ROOT / "src/cp_disr/final_tb_structgen.py").read_text()
    assert "TEST_REL" in src and "read_text" in src
    # the module only hashes the test file for the record; it never loads rows from it
    assert "_json(root / TEST_REL)" not in src and "_json(ROOT / TEST_REL)" not in src


def test_spawn_command_parses_with_the_real_cli():
    from cp_disr import final_tb_structgen as S
    args = S.build_parser().parse_args(["train", "--root", "/r", "--out", "/o", "--plan", "R-TB-SG-B2-2", "--gpu", "3", "--token", "/o/t.json"])
    assert (args.command, args.plan, args.gpu) == ("train", "R-TB-SG-B2-2", 3)


def test_in_process_configuration_sets_dev_size_and_routes_make_bundle(monkeypatch):
    import types
    from cp_disr import final_tb, final_tb_structgen as S, neural, phase_a_v12
    fake = types.SimpleNamespace(EMPTY_PRIOR_METHODS={"B1-K+E", "B2"}, make_policy=lambda *a: None, OBS_DIM=48, CAND_DIM=8, B_PRIOR=0.5, DEV_EPISODES=10,
                                 make_bundle=lambda root, task, split=None: None, ENABLED_SPLITS={"T_B": "x"})
    monkeypatch.setattr(final_tb, "configure_v11", lambda root, ctx: fake)
    monkeypatch.setattr(neural, "METHODS", tuple(neural.METHODS))
    monkeypatch.setattr(phase_a_v12, "DEV10_N", phase_a_v12.DEV10_N)
    got = S.configure_v11(ROOT, None)
    assert got.DEV_EPISODES == 12 and phase_a_v12.DEV10_N == 12 and getattr(got.make_bundle, "__structgen__", False)
    assert {"B2-ABS", "B1-K+NC"} <= got.EMPTY_PRIOR_METHODS


@pytest.mark.torch_runtime
@pytest.mark.parametrize("method", ["B1-K+E", "B2", "B2-ABS", "B1-K+NC"])
def test_ppo_update_over_transitions_with_different_goal_templates(method):
    import torch
    from dataclasses import replace
    from cp_disr import tb_repctl_checks as C, struct_gen as G
    from cp_disr.graph import Goal, build_template
    from cp_disr.platforms.libero.runtime_factory import PREDICATES
    from cp_disr.rl import Rollout, Transition
    from cp_disr.torch_rl import PPO, prefix_hidden
    base = C.tb_template(ROOT)
    templates = [build_template(base.contracts, tuple(Goal(a, 1) for a in G.GOAL_SETS[k]), PREDICATES, G.OBJECTS) for k in ("IN_T", "BUF_S", "IN_T+BUF_S")]
    policy = C.make_policy(base, method, 0)
    policy.train()
    rollout = Rollout()
    for ep, template in enumerate(templates):
        seed, values = C.state_suite(template, count=1, start=ep)[0]
        snaps = [replace(C.make_snapshot(template, values, seed), episode_id="ep%d" % ep, decision_id=d) for d in range(4)]
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
    logs = PPO(policy).update(rollout, epochs=1, minibatch=8, sequence_length=4)
    assert logs and all(torch.isfinite(torch.tensor(l["total"])) for l in logs)
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items())
