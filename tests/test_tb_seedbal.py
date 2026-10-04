"""CP-DISR-TB-SEED-BALANCE-01: registration, ledger, seed chain and frozen analysis rules (offline)."""
import ast
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plan_table_is_exactly_the_three_authorised_runs():
    from cp_disr import final_tb_seedbal as S
    assert {k: (v["method"], v["seed"]) for k, v in S.PLAN_TABLE.items()} == {
        "R-TB-B2-0-CURRENT": ("B2", 0), "R-TB-B2-2-CURRENT": ("B2", 2), "R-TB-E-2-CURRENT": ("B1-K+E", 2)}
    assert S.MAX_WORKERS == 3 and S.MAX_ATTEMPTS == 3 and S.RESULT_COMMIT == "4d0c426f67dfa1ce23247f89879c0b4952367d53"


def test_baseline_and_representation_control_tables_untouched():
    from cp_disr import final_tb, final_tb_repctl
    assert final_tb.METHODS == ("B1-K", "B1-K+E", "B2") and len(final_tb.PLAN_TABLE) == 4
    assert len(final_tb_repctl.PLAN_TABLE) == 6


def test_assignment_and_context_validation():
    from cp_disr import final_tb_seedbal as S
    assert S.validate_assignment("R-TB-E-2-CURRENT", "B1-K+E", 2)["queue_order"] == 3
    for args in (("R-TB-B2-0-CURRENT", "B2", 1), ("R-TB-B2-2-CURRENT", "B1-K+E", 2), ("R-TB-B2-3-CURRENT", "B2", 3), ("R-TB-E-1", "B1-K+E", 1)):
        with pytest.raises(Exception):
            S.validate_assignment(*args)
    cfg = {"plan_id": "R-TB-B2-2-CURRENT", "attempt_id": "R-TB-B2-2-CURRENT-T-aaaaaaaa", "method": "B2", "training_seed": 2, "source_commit": "a" * 40,
           "output_directory": "/x/y", "runtime_manifest": "/x/m.yaml", "train_split": "/x/s.json", "prior_mode": "absent", "study_envelope_ncap": 16384,
           "study_envelope_tcap": 16384 * 4.199999999997672, "evaluation_rule": "0/4096/8192/final; frozen dev10"}
    ctx = S.context_from_dict(cfg)
    ctx.assert_worker("T_B", "B2")
    for bad in ({"training_seed": 0}, {"prior_mode": "original"}, {"study_envelope_ncap": 8192}, {"extra": 1}, {"method": "B1-K+E"}):
        with pytest.raises(Exception):
            S.context_from_dict({**cfg, **bad})


def _ledger(tmp_path):
    from cp_disr import final_tb_seedbal as S
    ledger = S.SeedbalLedger(tmp_path)
    ledger.init({p: {"method": v["method"], "seed": v["seed"], "status": "NOT_STARTED", "queue_order": v["queue_order"]} for p, v in S.PLAN_TABLE.items()})
    return ledger


def test_ledger_three_workers_one_attempt_each(tmp_path):
    ledger = _ledger(tmp_path)
    plans = ["R-TB-B2-0-CURRENT", "R-TB-B2-2-CURRENT", "R-TB-E-2-CURRENT"]
    for i, p in enumerate(plans):
        ledger.reserve(p, "att-" + p, "/runs/" + p, 100 + i, i)
    with pytest.raises(Exception, match="duplicate"):
        ledger.reserve(plans[0], "again", "/runs/other", 9, 7)
    state = ledger.read()
    assert state["new_rl_attempts_used"] == 3 and state["new_rl_attempts_cap"] == 3


def test_ledger_refuses_shared_gpu_and_shared_directory(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.reserve("R-TB-B2-0-CURRENT", "a", "/runs/a", 1, 0)
    with pytest.raises(Exception, match="GPU"):
        ledger.reserve("R-TB-B2-2-CURRENT", "b", "/runs/b", 2, 0)
    with pytest.raises(Exception, match="directory"):
        ledger.reserve("R-TB-E-2-CURRENT", "c", "/runs/a", 3, 1)


def test_launch_is_one_shot_on_three_distinct_idle_gpus(tmp_path):
    from cp_disr import final_tb_seedbal as S
    _ledger(tmp_path)
    spawned = []

    class P:
        pid = 7

    def spawn(cmd, logfile):
        spawned.append((cmd[cmd.index("--plan") + 1], int(cmd[cmd.index("--gpu") + 1])))
        return P()
    idle = {i: {"memory_used_mib": 5, "utilization": 0} for i in range(7)}
    S.launch_all(ROOT, tmp_path, gpu_fn=lambda: idle, spawn=spawn, log=lambda *a: None)
    assert [p for p, _ in spawned] == ["R-TB-B2-0-CURRENT", "R-TB-B2-2-CURRENT", "R-TB-E-2-CURRENT"] and len({g for _, g in spawned}) == 3


def test_launch_refuses_when_fewer_than_three_gpus_are_idle(tmp_path):
    from cp_disr import final_tb_seedbal as S
    _ledger(tmp_path)
    busy = {0: {"memory_used_mib": 5, "utilization": 0}, 1: {"memory_used_mib": 5, "utilization": 0}, 2: {"memory_used_mib": 30000, "utilization": 90}}
    called = []
    with pytest.raises(Exception, match="three idle"):
        S.launch_all(ROOT, tmp_path, gpu_fn=lambda: busy, spawn=lambda c, l: called.append(c), log=lambda *a: None)
    assert called == []


def test_spawn_command_parses_with_the_real_cli():
    from cp_disr import final_tb_seedbal as S
    args = S.build_parser().parse_args(["train", "--root", "/r", "--out", "/o", "--plan", "R-TB-E-2-CURRENT", "--gpu", "2", "--token", "/o/t.json"])
    assert (args.command, args.plan, args.gpu) == ("train", "R-TB-E-2-CURRENT", 2)


def test_preflight_gate_refuses_a_missing_or_failed_preflight(tmp_path):
    from cp_disr import final_tb_seedbal as S
    with pytest.raises(Exception):
        S.verify_preflight(tmp_path, "a" * 40)
    (tmp_path / "seed_balance_preflight.json").write_text(json.dumps({"card": S.CARD, "verdict": "FAIL", "refs": {"prep_commit_head": "a" * 40}}))
    with pytest.raises(Exception, match="PASS"):
        S.verify_preflight(tmp_path, "a" * 40)
    (tmp_path / "seed_balance_preflight.json").write_text(json.dumps({"card": S.CARD, "verdict": "PASS", "refs": {"prep_commit_head": "b" * 40}}))
    with pytest.raises(Exception):
        S.verify_preflight(tmp_path, "a" * 40)


def test_no_pre_existing_tracked_file_is_modified_since_the_result_commit():
    from cp_disr import final_tb_seedbal as S
    changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=MD", S.RESULT_COMMIT, "HEAD"], cwd=str(ROOT), text=True).split()
    assert changed == [], changed


def test_new_modules_import_no_provider_vlm_or_test_cache():
    forbidden = {"vlm_provider", "vlm_cache_pipeline", "vlm", "prior", "stage2a_runner", "phase_a_v13_r1", "phase_a_v13_r3"}
    for rel in ("src/cp_disr/final_tb_seedbal.py", "src/cp_disr/tb_seedbal_analysis.py"):
        mods = set()
        for n in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[-1])
                mods.update(a.name for a in n.names)
            elif isinstance(n, ast.Import):
                mods.update(a.name.split(".")[-1] for a in n.names)
        assert not (mods & forbidden), (rel, mods & forbidden)


@pytest.mark.torch_runtime
def test_seed_reaches_the_whole_chain_and_policy_initialisation_is_independent_per_seed():
    from cp_disr import final_tb_seedbal as S
    ev = S.seed_chain_evidence(ROOT)
    assert ev["ok"], ev
    assert ev["train_job_seed_from_run_context"] and ev["train_job_seed_all_calls"] == ["seed"] and ev["seed_all_sets_python_numpy_torch"]
    assert ev["eval_rng_isolated_capture_restore"] and ev["policy_init_differs_per_seed_and_repeats"]


# ----------------------------------------------------------------------------- frozen analysis rules
def test_early_labels():
    from cp_disr import tb_seedbal_analysis as A
    assert A.classify_seed({"0": 0, "4096": 10, "8192": 10, "final": 10}) == "4096"
    assert A.classify_seed({"0": 0, "4096": 0, "8192": 10, "final": 10}) == "8192"
    assert A.classify_seed({"0": 0, "4096": 0, "8192": 0, "final": 10}) == "FINAL_ONLY"
    assert A.classify_seed({"0": 0, "4096": 0, "8192": 0, "final": 0}) == "NEVER"
    assert A.classify_seed({"0": 0, "4096": 9, "8192": 9, "final": 9}) == "NEVER"


def _matrix(**cells):
    m = {}
    base = {("+E", 0): "FINAL_ONLY", ("+E", 1): "FINAL_ONLY", ("ABS", 0): "FINAL_ONLY", ("ABS", 1): "4096", ("ABS", 2): "8192",
            ("NC", 0): "NEVER", ("NC", 1): "FINAL_ONLY", ("NC", 2): "NEVER", ("B2", 1): "4096"}
    m.update(base)
    for k, v in cells.items():
        fam, seed = k.rsplit("_s", 1)
        m[({"E": "+E"}.get(fam, fam), int(seed))] = v
    return m


@pytest.mark.parametrize("cells,expected", [
    ({"B2_s0": "4096", "B2_s2": "8192", "E_s2": "FINAL_ONLY"}, "MECHANISM_PATTERN_REPLICATED"),
    ({"B2_s0": "FINAL_ONLY", "B2_s2": "NEVER", "E_s2": "FINAL_ONLY"}, "SEED_SENSITIVITY_HIGH"),
    ({"B2_s0": "FINAL_ONLY", "B2_s2": "FINAL_ONLY", "E_s2": "8192"}, "SEED_SENSITIVITY_HIGH"),
    ({"B2_s0": "4096", "B2_s2": "NEVER", "E_s2": "FINAL_ONLY"}, "MECHANISM_PATTERN_REPLICATED"),
    ({"B2_s0": "4096", "B2_s2": "4096", "E_s2": "4096"}, "MIXED_REPRESENTATION_PATTERN"),
])
def test_conclusion_rules(cells, expected):
    from cp_disr import tb_seedbal_analysis as A
    assert A.conclude(A.family_counts(_matrix(**cells)))["conclusion"] == expected


def test_conclusion_inconclusive_when_ABS_is_not_replicated_and_when_cells_missing():
    from cp_disr import tb_seedbal_analysis as A
    m = _matrix(B2_s0="4096", B2_s2="4096", E_s2="FINAL_ONLY")
    m[("ABS", 2)] = "FINAL_ONLY"
    assert A.conclude(A.family_counts(m))["conclusion"] == "INCONCLUSIVE"
    del m[("B2", 0)]
    assert A.conclude(A.family_counts(m))["conclusion"] == "INCONCLUSIVE"


def test_frozen_existing_cells_are_reproduced_from_the_committed_tables():
    from cp_disr import tb_seedbal_analysis as A
    by_run, _ = A._csv_matrix(ROOT)
    got = {}
    for meta in by_run.values():
        fam = A.METHOD_FAMILY.get(meta["method"])
        if fam:
            got[(fam, meta["seed"])] = A.classify_seed(meta["points"])
    for key, want in A.FROZEN_EXISTING.items():
        assert got[key] == want, (key, got.get(key), want)
