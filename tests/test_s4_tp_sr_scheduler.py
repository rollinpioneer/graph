import json
from pathlib import Path

import pytest

from cp_disr.analysis import s4_tp_sr_design as d


def st(agg, med, mn, faults=()):
    return {"aggregate": agg, "median_wall": med, "min_wall": mn, "faults": list(faults)}


def test_S01_two_workers_no_gain_drops_to_one():
    assert d.decide_workers(st(0.104, 20, 10), None)[0] == 1


def test_S02_four_kept_when_faster_and_no_contention():
    assert d.decide_workers(st(0.15, 10, 8), st(0.2, 12, 8))[0] == 4


def test_S03_four_rejected_on_small_gain():
    assert d.decide_workers(st(0.15, 10, 8), st(0.16, 12, 8))[0] == 2


def test_S04_four_rejected_on_median_wall_blowup():
    assert d.decide_workers(st(0.15, 10, 8), st(0.3, 16, 8))[0] == 2


def test_S05_four_rejected_on_fault():
    assert d.decide_workers(st(0.15, 10, 8), st(0.3, 11, 8, [{"fault": "x"}]))[0] == 2


def test_S06_four_not_tried_keeps_two():
    assert d.decide_workers(st(0.15, 10, 8), None)[0] == 2


def test_S07_charge_overflow_consumes_nothing(tmp_path):
    cfg = {"budgets": {"static_capture_resets": 2}}
    d.init_ledger(tmp_path, cfg)
    d.charge(tmp_path, "static_capture_resets", 2)
    with pytest.raises(d.StopRun):
        d.charge(tmp_path, "static_capture_resets", 1)
    assert d.ledger(tmp_path)["static_capture_resets"]["used"] == 2
    assert d.ledger(tmp_path)["rl_transitions"]["cap"] == 0
    with pytest.raises(d.StopRun):
        d.charge(tmp_path, "optimizer_steps", 1)


def test_S08_selection_deterministic_and_outcome_blind():
    rows = [{"case_id": f"c{i}", "occlusion_bin": i % 4, "status": "PASS"} for i in range(64)]
    a, _ = d.select_physical(rows)
    b, _ = d.select_physical([dict(r, junk=i) for i, r in enumerate(rows)])
    assert a == b and len(a) == 24 and len(set(a)) == 24
    assert sum(1 for c in a if rows[int(c[1:])]["occlusion_bin"] in (3, 2)) == 12
