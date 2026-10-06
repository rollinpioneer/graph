"""Fixtures of runbook v2 section 12: only the switches of this card are tested; train-split cases and artificial examples only."""
import json
import shutil
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import a04p_controls as AC
from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import gp_attribution as A
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld.environment import BwEpisode, Case
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / E.MODELS["B2"]["path"]).is_file() and (ROOT / A.GP_ROOT / "datasets" / "D_train.json").is_file()
need_assets = pytest.mark.skipif(not HAVE, reason="A02 B2 checkpoint / D_train not on this machine")


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


@pytest.fixture(scope="module")
def data():
    d = json.loads((ROOT / A.GP_ROOT / "datasets" / "D_train.json").read_text())
    lab = json.loads((ROOT / A.GP_ROOT / "datasets" / "rank_labels.json").read_text())
    return d, lab


@need_assets
def test_b2_rank_with_zero_weight_reproduces_the_old_m0_step(cases, data):
    d, lab = data
    r = A.check_rank_zero_equivalence(ROOT, cases, d, lab, torch.device("cpu"), n_traj=4)
    assert r["pass"], r
    assert r["gru_grad_abs_sum"] > 0 and r["rank_weight_one_changes_gradient"]


@need_assets
def test_m1_goal_switch_off_is_the_old_m1_and_on_uses_the_marks(cases, data):
    d, lab = data
    r = A.check_goal_switch_equivalence(ROOT, cases, d, lab, torch.device("cpu"), n_traj=4)
    assert r["pass"], r
    assert r["head_parameters"] == 82688 and r["trainable_parameters"] == 513472 and r["head_sha_matches_M1"]


def test_colour_twins_change_only_the_colours(cases):
    p = cases[-1].problem
    q = A.colour_swapped(p)
    assert q.names == p.names and q.init == p.init and q.goal == p.goal and all(a + b == 1 for a, b in zip(p.colors, q.colors))
    e1, e2 = BwEpisode(Case("a", "t", p, 4, 20)), BwEpisode(Case("b", "t", q, 4, 20))
    s1, s2 = e1.snapshot(), e2.snapshot()
    assert s1.candidate_ids == s2.candidate_ids and s1.mask == s2.mask
    for a in e1.legal_ids():
        assert S.apply(e1.state, e1._action_of[a]) == S.apply(e2.state, e2._action_of[a])
    solver = P.Solver()
    assert solver.cost_to_go(p.init, p.goal) == solver.cost_to_go(q.init, q.goal)
    assert A.destruction_flag(p, solver) == A.destruction_flag(q, solver)


def test_unknown_destruction_label_is_not_coerced(monkeypatch, cases):
    def boom(problem, solver):
        raise P.PlannerLimit("limit")
    monkeypatch.setattr("cp_disr.blocksworld.planner.destruction_labels", boom)
    assert A.destruction_flag(cases[0].problem, P.Solver()) is None


def _fixture(goal, state, names=("b0", "b1", "b2")):
    p = S.Problem(names, (0, 1, 0), state, goal)
    ep = BwEpisode(Case("x", "t", p, 2, 8))
    return ep, ep.snapshot()


def _logits(snap, fav):
    lg = torch.full((len(snap.candidate_ids),), -1.0)
    lg[snap.candidate_ids.index(fav)] = 5.0
    return lg


def test_g1_c3_combination_order_and_cases():
    names = ("b0", "b1", "b2")
    stack, put = S.action_id(names, ("STACK", 1, 0)), S.action_id(names, ("PUT_DOWN", 1))
    # (a) unvisited successors exist and one adds a goal atom: G1 picks among them even against the raw preference
    ep, snap = _fixture((S.TABLE, 0, S.TABLE), (S.TABLE, S.HELD, S.TABLE))
    mem = AC.Memory(ep.state)
    sel, info = A.combo_choose("G1C3", ep, snap, _logits(snap, put), mem)
    assert snap.candidate_ids[sel] == stack and info["intervened"] and info["trigger"] == "GOAL_PROGRESS_SET"
    # (b) the progress move leads to a visited state -> it is excluded first; the remainder has no progress move -> raw choice among the remainder
    mem2 = AC.Memory(ep.state)
    mem2.visited.add(S.apply(ep.state, ep._action_of[stack]))
    sel, info = A.combo_choose("G1C3", ep, snap, _logits(snap, put), mem2)
    assert snap.candidate_ids[sel] == put and info["trigger"] is None
    # (c) every successor visited -> stop
    mem3 = AC.Memory(ep.state)
    for a in ep.legal_ids():
        mem3.visited.add(S.apply(ep.state, ep._action_of[a]))
    sel, info = A.combo_choose("G1C3", ep, snap, _logits(snap, put), mem3)
    assert sel is None and info["trigger"] == "NO_UNVISITED_SUCCESSOR"


def test_combination_does_not_read_planner_labels():
    import inspect
    src = inspect.getsource(A.combo_choose)
    assert "solver" not in src and "optimal" not in src
    assert list(inspect.signature(A.combo_choose).parameters) == ["controller", "ep", "snap", "logits", "mem"]


@need_assets
def test_runner_matches_the_registered_controller_loop(cases):
    pol = E.load_model(ROOT, "B2", torch.device("cpu"))
    for c in cases[:2]:
        for cond, ctrl in (("C0", "C0"), ("G1", "G1"), ("C3", "C3")):
            a = A.run_episode_with(pol, c, P.Solver(), A.chooser_for(cond))
            b = AC.run_episode(pol, c, P.Solver(), ctrl)
            assert [d["selected"] for d in a["decisions"]] == [d["selected"] for d in b["decisions"]]
            assert (a["success"], a["decision_perfect"], a["steps"]) == (b["success"], b["decision_perfect"], b["steps"])


def test_direction_thresholds():
    assert A.net_threshold(32) == 5 and A.net_threshold(16) == 3 and A.net_threshold(24) == 4 and A.net_threshold(15) is None
    assert A.direction(5, 32) == "A_AHEAD" and A.direction(4, 32) == "NO_CLEAR_DIFFERENCE" and A.direction(-3, 16) == "B_AHEAD" and A.direction(3, 12) == "NO_LABEL_N_LT_16"


def test_artifact_whitelist_and_source_drift(tmp_path):
    """A registered source whose hash differs makes the identity check fail; files inside the run root (ledger, logs) never do."""
    rr = ROOT / A.GPA_REL / "_unit_test_run_root"
    try:
        (rr / "prep").mkdir(parents=True)
        ident = A.source_identity(ROOT, ("tests/test_c1_bw_gp_attribution.py",))
        (rr / "prep" / "source_identity.json").write_text(json.dumps(ident))
        if E.git(ROOT, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % str(rr.relative_to(ROOT))):
            pytest.skip("tracked tree dirty during development")
        (rr / "receipts").mkdir()
        (rr / "receipts" / "case_ledger.jsonl").write_text("growing\n")
        A.check_identity(ROOT, rr)                                                   # run-root writes do not count as source drift
        bad = json.loads((rr / "prep" / "source_identity.json").read_text())
        first = next(iter(bad["frozen_sources"]))
        bad["frozen_sources"][first]["sha256"] = "0" * 64
        (rr / "prep" / "source_identity.json").write_text(json.dumps(bad))
        with pytest.raises(E.A03Error):
            A.check_identity(ROOT, rr)
    finally:
        shutil.rmtree(rr, ignore_errors=True)
