"""A03 evaluation-entry tests on artificial fixtures only (train-split cases and untrained policies); no test-split case is evaluated here."""
import json
from pathlib import Path

import pytest

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld import train as T
from cp_disr.blocksworld.environment import Case
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / E.TRAIN_DEV["path"])
    set_suite_half_life(H)
    return cs


def _root(tmp_path, ids):
    rr = tmp_path / "rr"
    for sub in ("registration", "receipts", "eval", "results"):
        (rr / sub).mkdir(parents=True)
    (rr / "receipts" / "case_ledger.jsonl").write_text("")
    manifest = {s: [] for s in E.ORDER}
    manifest["a0"] = ids
    (rr / "registration" / "a03_spec.json").write_text(json.dumps({"case_manifest": manifest}))
    return rr


def _fake_eval(calls):
    def f(policy, case, solver, hops, record):
        calls.append(case.case_id)
        return {"case_id": case.case_id, "split": "x", "n_blocks": case.problem.n, "success": True, "decision_perfect": True, "steps": 1, "optimal_length": 1, "step_cap": 4, "excess_steps": 0,
                "first_divergence": None, "cycle": False, "destroyed_satisfied_goal_count": 0, "decisions": [{"probs": {"a": 1.0}, "selected": "a", "selected_is_optimal": True}]}
    return f


def test_every_case_is_evaluated_once_and_a_started_case_is_never_rerun(tmp_path, cases):
    mini = cases[:3]
    rr = _root(tmp_path, [c.case_id for c in mini])
    calls = []
    split_cases = {s: [] for s in E.ORDER}
    split_cases["a0"] = mini
    E.ledger_append(rr, "STARTED", "B2", "a0", mini[1].case_id)                      # interrupted earlier: started, no record
    c1 = E.evaluate_method(rr, "B2", None, split_cases, None, None, _fake_eval(calls))
    assert calls == [mini[0].case_id, mini[2].case_id] and c1["technical_incomplete"] == 1
    c2 = E.evaluate_method(rr, "B2", None, split_cases, None, None, _fake_eval(calls))   # second invocation: nothing runs again
    assert calls == [mini[0].case_id, mini[2].case_id] and c2["completed"] == 0 and c2["skipped_completed"] == 2
    st = E.ledger_state(rr)
    assert st[("B2", "a0", mini[1].case_id)] == "TECHNICAL_INCOMPLETE" and st[("B2", "a0", mini[0].case_id)] == "COMPLETED"


def test_an_evaluator_exception_is_a_technical_gap_not_a_retry(tmp_path, cases):
    mini = cases[:2]
    rr = _root(tmp_path, [c.case_id for c in mini])
    split_cases = {s: [] for s in E.ORDER}
    split_cases["a0"] = mini

    def boom(*a, **k):
        raise RuntimeError("x")
    c = E.evaluate_method(rr, "QMARK", None, split_cases, None, None, boom)
    assert c["technical_incomplete"] == 2 and c["completed"] == 0
    assert all(v == "TECHNICAL_INCOMPLETE" for v in E.ledger_state(rr).values())


def test_nonfinite_probabilities_are_not_recorded_as_results(tmp_path, cases):
    mini = cases[:1]
    rr = _root(tmp_path, [mini[0].case_id])
    split_cases = {s: [] for s in E.ORDER}
    split_cases["a0"] = mini

    def bad(policy, case, solver, hops, record):
        ep = _fake_eval([])(policy, case, solver, hops, record)
        ep["decisions"][0]["probs"] = {"a": float("nan")}
        return ep
    assert E.evaluate_method(rr, "ASNET", None, split_cases, None, None, bad)["technical_incomplete"] == 1


def test_real_models_run_on_a_train_fixture_without_optimizer_or_grad(tmp_path, cases):
    import torch
    from cp_disr.blocksworld import planner as P
    policy = I.make_imitation_policy("ASNET-READOUT", "cpu", 0).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    ep = T.evaluate_case(policy, cases[0], P.Solver(), T.Hops(), True)
    assert ep["steps"] >= 1 and all(p.grad is None for p in policy.parameters())
    assert torch.is_grad_enabled()


def test_source_has_no_training_path():
    src = (ROOT / "src/cp_disr/blocksworld/eval_a03.py").read_text() + (ROOT / "scripts/c1_bw_eval_a03.py").read_text()
    for token in ("backward(", "torch.optim", "Adam(", "a0_gate", "load_checkpoint(" + "str(old"):
        assert token not in src
    assert "CL.a0_gate" not in src and "id_gate(" not in src


def test_frozen_constants_and_state_labels():
    assert sum(m["bytes"] > 0 for m in E.MODELS.values()) == 3 and set(E.MODELS) == {"B2", "QMARK", "ASNET"}
    assert sum(s["n"] for s in E.SPLITS.values()) == 136 and 3 * 136 == 408
    assert E.STATE_LABELS["original_a02_state"] == "IMITATION_ID_GATE_FAIL"
    assert all("final.pt" in m["path"] and "/imitation_a02/" in m["path"] for m in E.MODELS.values())


def test_a0_pairing_finds_the_relabelling_and_compares_the_mapped_sequence():
    names_dev, names_a0 = ("b0", "b1", "b2"), ("k0", "k1", "k2")
    dev = Case("d", "dev", S.Problem(names_dev, (0, 0, 1), (1, -1, -1), (2, 0, -1)), 4, 12)
    a0 = Case("a", "a0", S.Problem(names_a0, (0, 1, 0), (2, -1, -1), (1, -1, 0)), 4, 12)   # dev b0->k0, b1->k2, b2->k1
    maps = E._relabelings(dev, a0)
    assert maps, "the fixture must be a genuine relabelling"
    perm = maps[0]
    sel_dev = ["a:PICK_UP:b0:v1", "a:STACK:b0:b1:v1"]
    mapped = [E._map_action(x, names_dev, names_a0, perm) for x in sel_dev]
    ep_dev = {"success": True, "decision_perfect": True, "steps": 2, "decisions": [{"selected": s} for s in sel_dev]}
    ep_a0 = {"success": True, "decision_perfect": True, "steps": 2, "decisions": [{"selected": s} for s in mapped]}
    r = E.a0_pair(dev, a0, ep_dev, ep_a0)
    assert r["status"] == "PAIRED" and r["same_selected_sequence"] and r["same_success"]
    assert E.a0_pair(dev, a0, None, ep_a0)["status"] == "PAIR_DETAIL_UNAVAILABLE"


def test_paired_counts_use_the_full_denominator():
    a = [{"case_id": str(i), "decision_perfect": i < 3} for i in range(5)]
    b = [{"case_id": str(i), "decision_perfect": i in (1, 4)} for i in range(5)]
    assert E.paired_counts(a, b, "decision_perfect") == {"n": 5, "both": 1, "only_first": 2, "only_second": 1, "neither": 1}
