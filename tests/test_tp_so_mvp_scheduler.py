"""S01-S09: budget caps, the two-branch technical wave, the remaining-branch hard block, seeds and fault stopping."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_master/s4_family_a_soft_ordering_mvp.yaml"
CFG = m.load_config(CONFIG)
SRC = Path(m.__file__).read_text(encoding="utf-8")


def _ident():
    return {m.RUNTIME_REL: "0" * 64, "src/cp_disr/platforms/libero/tp_sr_instrumentation.py": "1" * 64}


def test_S01_branch_cap_is_twelve():
    assert int(CFG["budgets"]["physical_branch_attempts"]) == 12
    assert int(CFG["budgets"]["total_branch_attempts"]) == 12
    assert int(CFG["budgets"]["environment_resets"]) == 12
    assert m.PHYS_CAP == 12
    assert int(CFG["budgets"]["remaining_branch_attempts"]) == 10
    assert int(CFG["budgets"]["technical_wave_branch_attempts"]) == 2
    # 3 configs x 2 routes x 2 repeats
    assert 3 * 2 * 2 == 12


def test_S02_technical_wave_is_exactly_two_branches(tmp_path):
    assert int(CFG["authorization"]["technical_wave_first"]) == 2
    assert int(CFG["technical_wave"]["workers"]) == 2
    out = tmp_path
    phys = out / "physical"
    (phys / "witnesses").mkdir(parents=True)
    reg = {"branches": [{"branch_id": f"b{i}", "wave": "technical" if i < 3 else "remaining"} for i in range(5)]}
    (phys / "witnesses/e4_branch_registration.json").write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(m.StopRun) as e:
        m.technical_wave(ROOT, CONFIG, out, [0, 1], 2)
    assert e.value.code == "STOPPED_EVIDENCE_INTEGRITY"
    assert "exactly two branches" in e.value.detail


def test_S03_remaining_branches_blocked_before_technical_pass(tmp_path):
    out = tmp_path
    phys = out / "physical"
    phys.mkdir(parents=True)
    # no gate file at all
    with pytest.raises(m.StopRun) as e:
        m.run_remaining(ROOT, CONFIG, out, [0, 1], 4)
    assert e.value.code == "STOPPED_TECHNICAL_WAVE"
    # gate present but FAIL
    (phys / "technical_wave_check.json").write_text(json.dumps({"technical_wave": "FAIL", "remaining_branches_released": False}), encoding="utf-8")
    with pytest.raises(m.StopRun) as e2:
        m.run_remaining(ROOT, CONFIG, out, [0, 1], 4)
    assert e2.value.code == "STOPPED_TECHNICAL_WAVE"
    # gate PASS but not released
    (phys / "technical_wave_check.json").write_text(json.dumps({"technical_wave": "PASS", "remaining_branches_released": False}), encoding="utf-8")
    with pytest.raises(m.StopRun) as e3:
        m.run_remaining(ROOT, CONFIG, out, [0, 1], 4)
    assert e3.value.code == "STOPPED_TECHNICAL_WAVE"


def test_S04_paired_routes_share_one_restore_seed():
    for cid in m.CONFIG_ORDER:
        for repeat in (0, 1):
            seed = m.restore_seed(cid, repeat)
            tf = m.make_branch(cid, m.ROUTE_TARGET_FIRST, repeat, "/tmp/m.yaml", "case", _ident(), "remaining")
            sf = m.make_branch(cid, m.ROUTE_SECOND_FIRST, repeat, "/tmp/m.yaml", "case", _ident(), "remaining")
            assert tf["restore_seed"] == sf["restore_seed"] == seed
            assert tf["branch_id"] != sf["branch_id"]
    # different repeats and different configs must not collide
    seeds = {m.restore_seed(c, r) for c in m.CONFIG_ORDER for r in (0, 1)}
    assert len(seeds) == 6


def _minimal_physical(tmp_path, branch):
    out = tmp_path
    phys = out / "physical"
    (phys / "witnesses/branch_receipts").mkdir(parents=True)
    (phys / "budget_ledger.json").write_text(json.dumps({"physical_witness_episodes": {"cap": 12, "used": 0}}), encoding="utf-8")
    (phys / "attempt_registry.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("runtime: {}\n", encoding="utf-8")
    branch = dict(branch, manifest_path=str(manifest))
    (phys / "witnesses/e4_branch_registration.json").write_text(json.dumps({"branches": [branch]}), encoding="utf-8")
    return phys, branch


def test_S05_duplicate_branch_attempt_rejected(tmp_path):
    from cp_disr.analysis.s1_integration import IntegrationError, reserve_branch_attempt
    b = m.make_branch("SO_TARGET_FIRST", m.ROUTE_TARGET_FIRST, 0, "/tmp/m.yaml", "case", _ident(), "technical")
    phys, b = _minimal_physical(tmp_path, b)
    reserve_branch_attempt(phys, b["branch_id"])
    with pytest.raises(IntegrationError) as e:
        reserve_branch_attempt(phys, b["branch_id"])
    assert "DUPLICATE_ATTEMPT" in str(e.value)


def test_S06_max_workers_is_four():
    assert int(CFG["remaining"]["workers_max"]) == 4
    assert int(CFG["technical_wave"]["workers"]) == 2
    # dispatch never runs more workers than GPUs, and clamps the requested maximum
    assert "slots = list(gpus)[:max(1, min(int(max_workers), len(gpus)))]" in SRC
    assert 'min(int(max_workers), int(cfg["remaining"]["workers_max"]))' in SRC
    assert 'min(int(max_workers), int(cfg["technical_wave"]["workers"]))' in SRC


def test_S07_no_arbitrary_gpu_memory_threshold():
    lowered = SRC.lower()
    for token in ("free_memory", "mem_get_info", "memory_threshold", "nvidia-smi --query-gpu=memory",
                  "min_free", "mem_free"):
        assert token not in lowered, token
    # GPU selection is a plain explicit list; one worker per GPU
    assert "def _spawn_env(root, gpu)" in SRC
    assert 'CUDA_VISIBLE_DEVICES' in SRC


def test_S08_coordinator_is_sole_merger():
    # workers write only their own branch directory; the dispatcher owns the merged reports
    worker = SRC.split("def physical_worker", 1)[1].split("# ------------------------------------------------------------------ dispatch", 1)[0]
    assert 'phys / "branch_results" / f"{branch_id}.json"' in worker
    for merged in ("throughput_report", "worker_events.jsonl", "paired_restore.json", "per_config_summary.json"):
        assert merged not in worker, merged
    dispatch = SRC.split("def _dispatch", 1)[1].split("# ------------------------------------------------------------------ technical wave", 1)[0]
    assert 'throughput_report_{phase}.json' in dispatch
    assert 'worker_events.jsonl' in dispatch
    # the budget ledger is charged only inside the coordinator
    assert worker.count("charge_many") == 0
    assert dispatch.count("charge_many") == 1


def test_S09_two_consecutive_engineering_faults_stop_dispatch():
    assert "consecutive += 1" in SRC
    assert "if consecutive >= 2:" in SRC
    assert '"CONSECUTIVE_EXCEPTIONS:"' in SRC
    assert "while running or (todo and not faults):" in SRC
    # a worker exit failure is a fault too, and closes the attempt UNKNOWN
    assert "WORKER_EXIT:" in SRC
    assert 'finish_branch_attempt(phys, bid, "UNKNOWN"' in SRC
    # a reserve fault stops the loop as well
    assert "RESERVE:" in SRC