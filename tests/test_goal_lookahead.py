"""Fixtures of the goal-lookahead integration card: the sparsity-matched random-mask control and the shared look-ahead path (CPU; GPU equality when a GPU is visible)."""
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import scorer_control as SC
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld.environment import BwEpisode
from cp_disr.blocksworld.method_serial import model as MD
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


def _masks(model, case_state):
    snap, = [case_state]
    st, gprop = model.mg.goal_free_static(snap.template)
    gs = model.attn._goal_static(st, gprop, snap.template)
    codes = model.mg.base._codes(st, snap.facts.values).unsqueeze(0)
    return model.attn.pair_features(gs, codes, gprop)[1][0], gs


@need
def test_random_control_keeps_row_counts_self_and_is_a_pure_function_of_the_state(cases, mg):
    rel, rnd = MD.SerialModel(mg, "GOAL_REL"), MD.SerialModel(mg, "GOAL_RAND")
    rnd2 = MD.SerialModel(mg, "GOAL_RAND")
    differs = 0
    for i in range(0, len(cases), 7):
        c = cases[i]
        ep = BwEpisode(c)
        for _ in range(3):
            snap = ep.snapshot()
            m_rel, gs = _masks(rel, snap)
            m_r, _ = _masks(rnd, snap)
            m_r2, _ = _masks(rnd2, snap)
            assert torch.equal(m_r, m_r2)                                              # same state -> same mask, across instances and calls
            assert bool(m_r.diagonal().all())
            assert torch.equal(m_r.sum(-1), m_rel.sum(-1))                             # sparsity matched row by row
            differs += int(not torch.equal(m_r, m_rel))
            legal = [j for j, ok in enumerate(snap.mask) if ok]
            ep.step(snap.candidate_ids[legal[0]])
    assert differs > 5                                                                 # the control is not the relation mask


@need
def test_random_control_depends_on_the_state_not_on_the_history(cases, mg):
    rnd = MD.SerialModel(mg, "GOAL_RAND")
    c = cases[60]
    e1, e2 = BwEpisode(c), BwEpisode(c)
    a = [x for x in S.legal_actions(e1.state)][0]
    e1.step(S.action_id(c.problem.names, a))
    e2.state = e1.state                                                                  # same state reached without the move history
    assert torch.equal(_masks(rnd, e1.snapshot())[0], _masks(rnd, e2.snapshot())[0])


@need
@pytest.mark.parametrize("kind", ["GOAL_DENSE", "GOAL_REL", "GOAL_RAND"])
def test_leaf_values_use_the_same_state_function_as_nominal_successors_even_with_active_attention(cases, mg, kind):
    m = MD.SerialModel(mg, kind)
    with torch.no_grad():
        m.attn.o.weight.normal_(0, 0.2)
        m.attn.o.bias.normal_(0, 0.2)
    m.eval()
    c = cases[90]
    ep = BwEpisode(c)
    snap = ep.snapshot()
    out = m.forward_group([snap])[None]
    ids = snap.candidate_ids
    legal = [i for i, ok in enumerate(snap.mask) if ok]
    states = [S.apply(ep.state, ep._action_of[ids[i]]) for i in legal]
    v = m.state_values(snap.template, c.problem, states)
    assert torch.allclose(v, out["vn"][0, torch.tensor(legal)], atol=1e-3)


@need
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible")
def test_random_control_mask_is_identical_on_cpu_and_gpu(cases, mg):
    cpu = MD.SerialModel(mg, "GOAL_RAND")
    gpu_mg = SC.load_scorer(ROOT, "MG_C3", torch.device("cuda", 0)).model
    gpu = MD.SerialModel(gpu_mg, "GOAL_RAND")
    for i in range(0, len(cases), 11):
        snap_c = I.snapshot_at(cases[i], cases[i].problem.init, 0)
        m_c, _ = _masks(cpu, snap_c)
        m_g, _ = _masks(gpu, snap_c)
        assert torch.equal(m_c, m_g.cpu())
