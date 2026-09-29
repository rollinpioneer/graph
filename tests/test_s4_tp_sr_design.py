from cp_disr.analysis import s4_tp_sr_design as d


def row(ok, t=10.0, eng=False, cf=0, route="direct"):
    return {"success": ok, "engineering_failure": eng, "controller_failures": cf, "sim_time": t, "route": route}


def cls(dr, rr):
    return d.classify_scene([row(x, t, route="direct") for x, t in dr], [row(x, t, route="relocation") for x, t in rr])[0]


def test_O01_strong_helpful():
    assert cls([(True, 10), (False, 20)], [(True, 15), (True, 15)]) == "STRONG_HELPFUL"
    assert cls([(False, 20), (False, 20)], [(True, 15), (True, 15)]) == "STRONG_HELPFUL"


def test_O02_cost_helpful():
    assert cls([(True, 20), (True, 20)], [(True, 16), (True, 16)]) == "COST_HELPFUL"


def test_O03_neutral_and_harmful():
    assert cls([(True, 10), (True, 10)], [(True, 11), (True, 11)]) == "NEUTRAL"
    assert cls([(True, 10), (True, 10)], [(True, 16), (True, 16)]) == "HARMFUL"
    assert cls([(True, 10), (True, 10)], [(True, 10), (False, 30)]) == "HARMFUL"


def test_O04_engineering_failure_is_unknown_not_dropped():
    r = d.classify_scene([row(True), row(True)], [row(True, eng=True), row(True)])
    assert r[0] == "UNKNOWN"


def test_O05_unmatched_is_unknown():
    assert cls([(True, 10), (True, 10)], [(True, 13), (True, 13)]) == "UNKNOWN"   # between 20% and 50%
    assert cls([(False, 0), (False, 0)], [(False, 0), (False, 0)]) == "UNKNOWN"


def scenes(labels, bins=None):
    return [{"case_id": f"c{i}", "opportunity": l, "occlusion_bin": (bins or [3] * len(labels))[i]} for i, l in enumerate(labels)]


def brs(n, eng=0):
    return [{"engineering_failure": i < eng} for i in range(n)]


STAT = [{"case_id": f"c{i}", "direct_reachable": True, "relocation_reachable": True} for i in range(24)]


def test_P01_gate_pass():
    s = scenes(["STRONG_HELPFUL"] * 8 + ["NEUTRAL"] * 4 + ["UNKNOWN"] * 12, [3] * 4 + [2] * 4 + [1] * 16)
    assert d.physical_gate(s, brs(96), STAT)["passed"]


def test_P02_gate_fails_on_helpful_count():
    s = scenes(["STRONG_HELPFUL"] * 7 + ["NEUTRAL"] * 5 + ["UNKNOWN"] * 12, [3] * 4 + [2] * 4 + [1] * 16)
    assert not d.physical_gate(s, brs(96), STAT)["passed"]


def test_P03_gate_fails_on_single_bin():
    s = scenes(["STRONG_HELPFUL"] * 8 + ["NEUTRAL"] * 4 + ["UNKNOWN"] * 12)
    assert not d.physical_gate(s, brs(96), STAT)["checks"]["helpful_bins_ge_2"]


def test_P04_gate_fails_on_engineering_rate():
    s = scenes(["STRONG_HELPFUL"] * 8 + ["NEUTRAL"] * 4 + ["UNKNOWN"] * 12, [3] * 4 + [2] * 4 + [1] * 16)
    assert not d.physical_gate(s, brs(96, eng=20), STAT)["passed"]     # 20/96 > 20%


def test_P05_gate_needs_neutral_or_harmful():
    s = scenes(["STRONG_HELPFUL"] * 12 + ["NEUTRAL"] * 3 + ["UNKNOWN"] * 9, [3] * 6 + [2] * 6 + [1] * 12)
    assert not d.physical_gate(s, brs(96), STAT)["passed"]


def test_P06_reachability_counts_static_rows():
    s = scenes(["STRONG_HELPFUL"] * 8 + ["NEUTRAL"] * 4 + ["UNKNOWN"] * 12, [3] * 4 + [2] * 4 + [1] * 16)
    bad = [dict(r, relocation_reachable=False) for r in STAT[:13]] + STAT[13:]
    assert not d.physical_gate(s, brs(96), bad)["checks"]["both_reachable_ge_12"]


# ---- provider / representation gating (R01-R04)
import json

import pytest

from cp_disr.analysis import s4_tp_sr_provider as p


def _out(tmp_path, passed):
    (tmp_path / "opportunity").mkdir(parents=True)
    (tmp_path / "provider").mkdir()
    (tmp_path / "opportunity/physical_gate.json").write_text(json.dumps({"passed": passed}))
    d.init_ledger(tmp_path, {"budgets": {"provider_first_calls": 12, "provider_retries": 12, "representation_forward_cases": 12}})
    return tmp_path


def test_R01_provider_sample_refused_when_gate_failed(tmp_path):
    out = _out(tmp_path, False)
    with pytest.raises(d.StopRun) as e:
        p.freeze_provider_sample(".", "cfg", out)
    assert e.value.code == "PHYSICAL_GATE_NOT_PASSED"
    assert (out / "provider/freeze-provider-sample_NOT_RUN.json").is_file()


def test_R02_provider_call_refused_and_budget_untouched(tmp_path):
    out = _out(tmp_path, False)
    with pytest.raises(d.StopRun):
        p.call_provider(".", "cfg", out)
    led = d.ledger(out)
    assert led["provider_first_calls"]["used"] == 0 and led["provider_retries"]["used"] == 0


def test_R03_representation_refused_and_budget_untouched(tmp_path):
    out = _out(tmp_path, False)
    with pytest.raises(d.StopRun):
        p.probe_representation(".", "cfg", out, [0])
    assert d.ledger(out)["representation_forward_cases"]["used"] == 0


def test_R04_gate_passed_reads_gate_file(tmp_path):
    assert not p.gate_passed(tmp_path)
    assert p.gate_passed(_out(tmp_path, True)) and not p.gate_passed(_out(tmp_path / "x", False))
