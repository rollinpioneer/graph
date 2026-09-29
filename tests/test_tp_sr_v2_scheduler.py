"""S01-S08: Wave scheduling, budget charging and stop rules (no environment, no provider)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"


class FakeProc:
    def __init__(self, rc):
        self.rc, self.pid = rc, 4242

    def poll(self):
        return self.rc


def _setup(tmp_path, monkeypatch, behaviours=None, dry_ok=True):
    out = tmp_path / "o"
    for sub in ("inventory", "spec", "diagnostics", "decision"):
        (out / sub).mkdir(parents=True)
    (out / "stage_manifest.json").write_text(json.dumps({"phases_done": []}))
    m.init_ledger(out, m.load_config(CFG))
    m.prepare_diagnostics(ROOT, CFG, out)
    phys = out / "diagnostics" / "wave_d_branch_attempts" if False else out / m.WAVES["D"]["dir"]
    state = {"active": {}, "max_active": 0, "gpus_seen": [], "launched": []}
    behaviours = list(behaviours or [])

    def dry(root, o, wave, bid):
        if not dry_ok:
            raise RuntimeError("binding broken")

    def spawn(root, o, wave, bid, gpu):
        assert gpu not in state["active"].values(), "two workers on one GPU"     # S01
        state["active"][bid] = gpu
        state["max_active"] = max(state["max_active"], len(state["active"]))
        state["gpus_seen"].append(gpu)
        state["launched"].append(bid)
        beh = behaviours.pop(0) if behaviours else "ok"
        rc = 0
        if beh == "crash":
            rc = 3
        else:
            (phys / "branch_results").mkdir(parents=True, exist_ok=True)
            status = "EXCEPTION" if beh == "exception" else "TERMINATED"
            (phys / "branch_results" / f"{bid}.json").write_text(json.dumps({"branch_id": bid, "execution_status": status, "termination_reason": "x"}))
        p = FakeProc(rc)
        orig = p.poll

        def poll():
            state["active"].pop(bid, None)
            return orig()
        p.poll = poll
        return p

    monkeypatch.setattr(m, "worker_dry_run", dry)
    monkeypatch.setattr(m, "_spawn_worker", spawn)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    return out, phys, state


def _ledger(out):
    return m.rd(out / "budget_ledger.json")


def test_S01_S02_one_worker_per_gpu_and_one_charge_per_dispatch(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch)
    rep = m.run_wave(ROOT, CFG, out, "D", [1, 3], 2)
    assert rep["workers"] == 2 and st["max_active"] <= 2 and set(st["gpus_seen"]) <= {1, 3}
    led = _ledger(out)
    assert led["wave_d_branch_attempts"]["used"] == led["environment_resets"]["used"] == led["total_branch_attempts"]["used"] == 4


def test_S03_refill_on_completion_with_single_gpu(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch)
    m.run_wave(ROOT, CFG, out, "D", [1], 4)
    assert len(st["launched"]) == 4 and st["max_active"] == 1


def test_S04_wave_p_not_released_without_gate(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch)
    pphys = out / m.WAVES["P"]["dir"] / "witnesses"
    pphys.mkdir(parents=True)
    (pphys / "e4_branch_registration.json").write_text(json.dumps({"branches": [{"branch_id": "px"}]}))
    (out / "diagnostics/wave_d_gate.json").write_text(json.dumps({"wave_d_status": "FAIL"}))
    with pytest.raises(m.StopRun):
        m.run_wave(ROOT, CFG, out, "P", [0, 1, 2, 3], 4)
    assert _ledger(out)["wave_p_branch_attempts"]["used"] == 0


def test_S05_two_consecutive_exceptions_stop_dispatch(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch, behaviours=["exception", "exception", "ok", "ok"])
    rep = m.run_wave(ROOT, CFG, out, "D", [1], 1)
    assert len(st["launched"]) == 2 and rep["faults"]
    assert _ledger(out)["wave_d_branch_attempts"]["used"] == 2                      # no refund, nothing dispatched afterwards


def test_S06_worker_crash_counts_and_stops(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch, behaviours=["crash", "ok", "ok", "ok"])
    rep = m.run_wave(ROOT, CFG, out, "D", [1], 1)
    assert rep["faults"][0]["fault"].startswith("WORKER_EXIT") and len(st["launched"]) == 1
    from cp_disr.analysis.s1_integration import _load_attempts
    assert _load_attempts(phys)[st["launched"][0]] == "UNKNOWN"


def test_S07_no_speedup_claim_without_baseline(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch)
    rep = m.run_wave(ROOT, CFG, out, "D", [1, 3], 2)
    assert rep["single_worker_baseline"] == "NOT_MEASURED" and rep["speedup_claim"].startswith("NONE")


def test_S08_dry_run_failure_spends_nothing(tmp_path, monkeypatch):
    out, phys, st = _setup(tmp_path, monkeypatch, dry_ok=False)
    with pytest.raises(m.StopRun):
        m.run_wave(ROOT, CFG, out, "D", [1, 3], 2)
    assert _ledger(out)["total_branch_attempts"]["used"] == 0 and not st["launched"]
