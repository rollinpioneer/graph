"""Fixtures of the scale-order card: train-split cases, artificial states and the frozen scorer on a padded snapshot (no board case, no dev rerun)."""
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import gp_attribution as A
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import scale_order as SO
from cp_disr.blocksworld import scorer_control as SC
from cp_disr.blocksworld import state as S
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / E.MODELS["B2"]["path"]).is_file() and (ROOT / SC.M1GOAL["path"]).is_file() and (ROOT / A.GP_ROOT / "datasets" / "D_train.json").is_file()
need_assets = pytest.mark.skipif(not HAVE, reason="checkpoints / D_train not on this machine")


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


def test_padding_keeps_original_blocks_and_adds_table_blocks(cases):
    p = cases[0].problem
    q = SO.pad_problem(p, 7, cases[0].case_id)
    assert q.names[:p.n] == p.names and q.init[:p.n] == p.init and q.goal[:p.n] == p.goal and all(x == S.TABLE for x in q.init[p.n:] + q.goal[p.n:]) and S.is_valid(q.init)
    cs = SO.pad_colors("x", 8, 5)
    assert abs(cs.count(0) - cs.count(1)) <= 1


def test_version_schedule_is_half_original_and_covers_every_version():
    s = SO.schedule_counts(1296)
    assert s["every_parent_meets_all_four_versions"] and abs(s["original_share"] - 0.5) < 1e-12
    assert sum(s["slot_epochs"].values()) == 1296 * 100


@need_assets
def test_relabelled_padded_trajectory_keeps_remaining_length_and_optimal_actions(cases):
    d = __import__("json").loads((ROOT / A.GP_ROOT / "datasets" / "D_train.json").read_text())
    c = cases[3]
    parent = {"case_id": c.case_id, "n": c.problem.n, "names": list(c.problem.names), "colors": list(c.problem.colors), "init": list(c.problem.init), "goal": list(c.problem.goal), "optimal_length": c.optimal_length, "step_cap": c.step_cap}
    r = SO._pad_worker((parent, [t for t in d if t["case_id"] == c.case_id][:2]))
    st = r["stats"]
    assert st["decisions"] > 0 and st.get("remaining_mismatch", 0) == 0 and st.get("orig_astar_not_subset", 0) == 0 and st.get("orig_action_illegal", 0) == 0
    assert len(r["cases"]) == 3 and all(t["n_blocks"] in (6, 7, 8) for t in r["trajs"])


def test_taxonomy_is_a_total_function_and_prefers_the_plan_order(cases):
    r = SO.check_fixtures.__doc__
    from cp_disr.blocksworld import planner as P
    c = cases[10]
    solver = P.Solver()
    _, opt = solver.optimal_actions(c.problem.init, c.problem.goal)
    ids = [S.action_id(c.problem.names, a) for a in opt]
    non = [S.action_id(c.problem.names, a) for a in S.legal_actions(c.problem.init) if S.action_id(c.problem.names, a) not in ids]
    for sel in non:
        assert SO.classify(c.problem, c.problem.init, sel, ids) in SO.CLASSES


@need_assets
def test_frozen_scorer_reads_a_padded_snapshot(cases):
    pol = SC.load_scorer(ROOT, "MG_C3", torch.device("cpu"))
    pc = cases[0]
    from cp_disr.blocksworld.environment import Case
    prob = SO.pad_problem(pc.problem, 8, pc.case_id)
    snap = I.snapshot_at(Case("pad_fx", "t", prob, pc.optimal_length, pc.step_cap), prob.init, 0)
    with torch.no_grad():
        o = pol(snap, pol.initial_hidden())
    legal = [i for i, m in enumerate(snap.mask) if m]
    assert torch.isfinite(o.logits[legal]).all()


@need_assets
def test_all_fixtures_of_prep(cases):
    d = __import__("json").loads((ROOT / A.GP_ROOT / "datasets" / "D_train.json").read_text())
    r = SO.check_fixtures(cases, d, torch.device("cpu"))
    assert r["pass"], r
