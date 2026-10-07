"""Fixtures of the scorer-control plan: train-split cases and artificial logits only (no fresh case, no dev rerun)."""
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import gp_attribution as A
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import scorer_control as C
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / E.MODELS["B2"]["path"]).is_file() and (ROOT / C.M1GOAL["path"]).is_file()


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


def test_rules_match_the_plan_section_4_1_restatement(cases):
    r = C.check_rule_fixtures(cases)
    assert r["pass"], r
    assert r["decisions_compared"] > 100 and r["ties_present"]


def test_four_conditions_swap_only_the_scorer():
    assert {C.SCORER[k] for k in C.CONDITIONS} == {"B2", "M1GOAL"} and {C.RULE[k] for k in C.CONDITIONS} == {"C3", "G1C3"}
    assert C.chooser_for("B_G1C3") is A.combo_choose and C.chooser_for("MG_G1C3") is A.combo_choose
    assert C.chooser_for("B_C3").__code__ is C.chooser_for("MG_C3").__code__


@pytest.mark.skipif(not HAVE, reason="checkpoints not on this machine")
def test_scorers_load_and_forward_leaves_weights_unchanged(cases):
    r = C.check_scorer_loading(ROOT, cases, torch.device("cpu"))
    assert r["pass"], r


def test_colour_twin_helper_flips_only_colours(cases):
    p = cases[-1].problem
    q = A.colour_swapped(p)
    assert q.names == p.names and q.init == p.init and q.goal == p.goal and all(a + b == 1 for a, b in zip(p.colors, q.colors))
