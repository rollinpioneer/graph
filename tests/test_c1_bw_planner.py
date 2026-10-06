"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: exact planner correctness (runbook 17.2)."""
import random

import pytest

from cp_disr.blocksworld import generator as G
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import state as S


def _instances(count, seed, sizes=(3, 4, 5)):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.choice(sizes)
        colors = [rng.randint(0, 1) for _ in range(n)]
        out.append((G.random_initial(n, colors, rng), G.random_initial(n, colors, rng)))
    return out


def test_astar_length_equals_bfs_on_random_small_instances():
    sol = P.Solver()
    for init, goal in _instances(120, 1):
        assert sol.cost_to_go(init, goal) == P.bfs_cost(init, goal)


def test_heuristic_is_admissible():
    sol = P.Solver()
    for init, goal in _instances(120, 2, sizes=(3, 4)):
        assert P.heuristic(init, goal) <= sol.cost_to_go(init, goal)


def test_optimal_action_set_is_complete_and_exact():
    sol = P.Solver()
    for init, goal in _instances(60, 3, sizes=(3, 4)):
        L, acts = sol.optimal_actions(init, goal)
        if L == 0:
            continue
        brute = sorted(a for a, t in S.successors(init) if P.bfs_cost(t, goal) == L - 1)
        assert acts == brute and acts


def test_tie_break_does_not_change_the_set_and_cache_state_is_irrelevant():
    warm = P.Solver()
    for init, goal in _instances(40, 4, sizes=(3, 4)):
        warm.cost_to_go(init, goal)
    for init, goal in _instances(40, 4, sizes=(3, 4)):
        cold = P.Solver()
        assert cold.optimal_actions(init, goal) == warm.optimal_actions(init, goal)


def test_plan_reaches_the_goal_and_has_optimal_length():
    sol = P.Solver()
    for init, goal in _instances(40, 5):
        plan = sol.one_optimal_plan(init, goal)
        s = init
        for a in plan:
            assert a in S.legal_actions(s)
            s = S.apply(s, a)
        assert s == goal and len(plan) == sol.cost_to_go(init, goal)


def test_every_frozen_split_case_is_solvable_with_the_recorded_length():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in ("c1_bw_train_dev_v1", "c1_bw_a0_iso_v1", "c1_bw_a1_color_reverse_v1", "c1_bw_a2_noniso_v1", "c1_bw_b_scale_v1"):
        path = root / ("configs/splits/%s.json" % rel)
        if not path.is_file():
            pytest.skip("splits not built")
        doc = json.loads(path.read_text())
        cases = doc["train"] + doc["dev"] if "train" in doc else doc["cases"]
        sol = P.Solver()
        for c in cases[::7]:
            assert sol.cost_to_go(tuple(c["init"]), tuple(c["goal"])) == c["optimal_length"]
            assert c["step_cap"] == 2 * c["optimal_length"] + 4


def test_destruction_labels_are_complementary_on_a_known_case():
    # a goal atom satisfied at the start that every optimal plan must undo: b2 sits on the table under nothing, goal needs it on b0 while b0 is under b1
    prob = S.Problem(S.default_names(3), (0, 1, 0), (S.TABLE, 0, S.TABLE), (1, S.TABLE, 0))
    sol = P.Solver()
    requires, monotone, L = P.destruction_labels(prob, sol)
    assert requires != monotone and L == sol.cost_to_go(prob.init, prob.goal)
