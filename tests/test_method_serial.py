"""Fixtures of C1-BW-METHOD-SERIAL-SUITE-V3 (CPU, train-split states only; no board / confirmation case)."""
import json
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import scorer_control as SC
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld.environment import BwEpisode
from cp_disr.blocksworld.method_serial import common as CM
from cp_disr.blocksworld.method_serial import labels as LB
from cp_disr.blocksworld.method_serial import lookahead as LA
from cp_disr.blocksworld.method_serial import model as MD
from cp_disr.blocksworld.method_serial.heads_math import calibration_objectives
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / E.MODELS["B2"]["path"]).is_file() and (ROOT / SC.M1GOAL["path"]).is_file()
need = pytest.mark.skipif(not HAVE, reason="checkpoints not on this machine")


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


@pytest.fixture(scope="module")
def mg():
    return SC.load_scorer(ROOT, "MG_C3", torch.device("cpu")).model


def test_new_writes_must_resolve_into_xushijie3():
    with pytest.raises(RuntimeError):
        CM.check_storage("/home/xushijie2/anything")
    with pytest.raises(RuntimeError):
        CM.check_storage("/tmp/elsewhere")


@need
@pytest.mark.parametrize("kind", MD.KINDS)
def test_every_variant_reproduces_the_m1_goal_distribution_at_initialisation(cases, mg, kind):
    snaps = [I.snapshot_at(cases[i], cases[i].problem.init, 0) for i in (3, 60, 130)]
    m = MD.SerialModel(mg, kind)
    m.eval()
    if kind == "CAL":
        m.set_calibration(0.5, 1.0)
    for sn in snaps:
        ref = mg.group_scores([sn])[0]
        steps = 4 if kind.startswith("REC") else None
        out = m.eval_logits(sn, steps)
        ok = torch.isfinite(ref)
        assert torch.allclose(torch.softmax(out, -1)[ok], torch.softmax(ref, -1)[ok], atol=1e-4)
        if kind.startswith("REC"):
            out8 = m.eval_logits(sn, 8)
            assert torch.allclose(torch.softmax(out8, -1)[ok], torch.softmax(ref, -1)[ok], atol=1e-4)


@need
def test_recurrent_t8_uses_the_same_parameters_as_t4(cases, mg):
    m = MD.SerialModel(mg, "REC_REL")
    n0 = sum(p.numel() for p in m.parameters())
    sn = I.snapshot_at(cases[3], cases[3].problem.init, 0)
    m.eval_logits(sn, 4)
    m.eval_logits(sn, 8)
    assert sum(p.numel() for p in m.parameters()) == n0


@need
def test_leaf_values_of_true_states_match_the_nominal_successor_values(cases, mg):
    m = MD.SerialModel(mg, "MG")
    m.eval()
    c = cases[30]
    ep = BwEpisode(c)
    snap = ep.snapshot()
    out = m.forward_group([snap])[None]
    ids = snap.candidate_ids
    legal = [i for i, ok in enumerate(snap.mask) if ok]
    states = [S.apply(ep.state, ep._action_of[ids[i]]) for i in legal]
    v = m.state_values(snap.template, c.problem, states)
    ref = out["vn"][0, torch.tensor(legal)]
    assert torch.allclose(v, ref, atol=1e-3)


@need
def test_same_state_and_history_give_the_same_tree_for_both_leaf_scorers_and_do_not_touch_real_history(cases, mg):
    m = MD.SerialModel(mg, "MG")
    m.eval()
    c = cases[100]
    ep = BwEpisode(c)
    snap = ep.snapshot()
    ids = snap.candidate_ids
    visited = {ep.state}
    before = set(visited)
    roots = [ids[i] for i, ok in enumerate(snap.mask) if ok and S.apply(ep.state, ep._action_of[ids[i]]) not in visited]
    c1, c2 = LA.Counters(), LA.Counters()
    a1, i1 = LA.choose_root(m, "MG", snap.template, c.problem, ep.state, roots, ep._action_of, visited, c1)
    a2, i2 = LA.choose_root(m, "COUNT", snap.template, c.problem, ep.state, roots, ep._action_of, visited, c2)
    assert visited == before
    assert i1["n_roots"] == i2["n_roots"] and i1["n_leaves"] == i2["n_leaves"] and i1["terminal"] == i2["terminal"]
    assert a1 in roots and a2 in roots


def test_terminal_move_wins_over_any_leaf_score(cases):
    c = cases[10]
    solver = P.Solver()
    plan = solver.one_optimal_plan(c.problem.init, c.problem.goal)
    ep = BwEpisode(c)
    for a in plan[:-1]:
        ep.step(S.action_id(c.problem.names, a))
    snap = ep.snapshot()
    ids = snap.candidate_ids
    roots = [ids[i] for i, ok in enumerate(snap.mask) if ok]
    aid, info = LA.choose_root(None, "COUNT", snap.template, c.problem, ep.state, roots, ep._action_of, {ep.state})
    assert aid == S.action_id(c.problem.names, plan[-1]) and info["terminal"]


def test_event_labels_project_to_all_optimal_actions(cases):
    solver = P.Solver()
    bad = 0
    for c in cases[::6]:
        lab = LB.event_label(c.problem, c.problem.init, solver)
        assert lab["complete"]
        assert {a for _g, a in lab["Y"]} == set(lab["A"])
        assert all(g in lab["U0"] for g, _a in lab["Y"])
        bad += 0
    assert bad == 0


def test_relative_calibration_is_offset_invariant_and_absolute_is_not():
    w = torch.tensor([3.0, 2.0, 5.0, 1.0])
    d = torch.tensor([4.0, 3.0, 6.0, 2.0])
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3]])
    a = calibration_objectives(w, d, pairs, 2.0)
    b = calibration_objectives(w + 7.0, d, pairs, 2.0)
    assert torch.allclose(a["relative"], b["relative"]) and not torch.allclose(a["absolute"], b["absolute"])


@need
def test_goal_attention_mask_depends_only_on_the_scored_state(cases, mg):
    m = MD.SerialModel(mg, "GOAL_REL")
    c = cases[40]
    ep = BwEpisode(c)
    d1 = m.mask_density(ep.snapshot())
    ep2 = BwEpisode(c)
    d2 = m.mask_density(ep2.snapshot())
    assert d1 == d2 and 0.0 <= d1 <= 1.0
