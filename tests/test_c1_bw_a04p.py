"""Phase-1 (A04P) tests on artificial fixtures and train-split cases only; no pilot / A03 outcome is used."""
import inspect
import json
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import a04p_controls as AC
from cp_disr.blocksworld import a04p_registry as R
from cp_disr.blocksworld import goal_probe as GP
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld import train as T
from cp_disr.blocksworld.environment import BwEpisode, Case
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


def _logits(snap, favourite):
    lg = torch.full((len(snap.candidate_ids),), -1.0)
    lg[snap.candidate_ids.index(favourite)] = 5.0
    return lg


def test_c0_reproduces_the_frozen_evaluation(cases):
    ck = ROOT / R.B2_CKPT["path"]
    if not ck.is_file():
        pytest.skip("checkpoint not on this machine")
    from cp_disr.blocksworld import eval_a03 as E
    pol = E.load_model(ROOT, "B2", torch.device("cpu"))
    for c in cases[:3]:
        a = AC.run_episode(pol, c, P.Solver(), "C0")
        b = T.evaluate_case(pol, c, P.Solver(), T.Hops(), True)
        assert [d["selected"] for d in a["decisions"]] == [d["selected"] for d in b["decisions"]]
        assert (a["success"], a["decision_perfect"], a["steps"], a["cycle"]) == (b["success"], b["decision_perfect"], b["steps"], b["cycle"])


def test_c1_counts_attempts_per_state_action_and_resets_per_episode(cases):
    ep = BwEpisode(cases[0])
    snap = ep.snapshot()
    legal = [i for i, m in zip(snap.candidate_ids, snap.mask) if m]
    assert len(legal) >= 2
    fav = legal[0]
    mem = AC.Memory(ep.state)
    sel, info = AC.choose("C1", ep, snap, _logits(snap, fav), mem)
    assert snap.candidate_ids[sel] == fav and not info["intervened"]            # untried: the raw choice is kept
    mem.visits[(ep.state, fav)] += 1
    sel, info = AC.choose("C1", ep, snap, _logits(snap, fav), mem)
    assert snap.candidate_ids[sel] != fav and info["intervened"] and info["trigger"] == "RAW_ALREADY_TRIED"
    assert AC.Memory(ep.state).visits[(ep.state, fav)] == 0                      # a new episode starts from an empty table


def test_c3_excludes_visited_successors_and_can_run_out(cases):
    ep = BwEpisode(cases[0])
    snap = ep.snapshot()
    ids = snap.candidate_ids
    legal = [i for i, m in zip(ids, snap.mask) if m]
    mem = AC.Memory(ep.state)
    succ = {a: S.apply(ep.state, ep._action_of[a]) for a in legal}
    mem.visited |= {succ[legal[0]]}
    sel, info = AC.choose("C3", ep, snap, _logits(snap, legal[0]), mem)
    assert ids[sel] != legal[0] and info["intervened"]
    mem.visited |= set(succ.values())
    sel, info = AC.choose("C3", ep, snap, _logits(snap, legal[0]), mem)
    assert sel is None and info["trigger"] == "NO_UNVISITED_SUCCESSOR"


def test_g1_requires_a_strict_superset_of_satisfied_goal_atoms():
    names = ("b0", "b1", "b2")
    # goal: b1 on b0, b0 and b2 on the table. State: b0 on table, b1 held, b2 on table -> STACK(b1,b0) adds On(b1,b0) and destroys nothing
    p = S.Problem(names, (0, 1, 0), (S.TABLE, S.HELD, S.TABLE), (S.TABLE, 0, S.TABLE))
    case = Case("g1", "t", p, 1, 6)
    ep = BwEpisode(case)
    snap = ep.snapshot()
    ids = snap.candidate_ids
    stack = S.action_id(names, ("STACK", 1, 0))
    put = S.action_id(names, ("PUT_DOWN", 1))
    assert stack in ids and put in ids
    sel, info = AC.choose("G1", ep, snap, _logits(snap, put), AC.Memory(ep.state))
    assert ids[sel] == stack and info["intervened"]
    # all-on-table goal: only PUT_DOWN(b1) adds an atom (OnTable b1); STACK(b1,b0) adds none, so a raw STACK preference is overridden
    p2 = S.Problem(names, (0, 1, 0), (S.TABLE, S.HELD, S.TABLE), (S.TABLE, S.TABLE, S.TABLE))
    ep2 = BwEpisode(Case("g1b", "t", p2, 1, 6))
    snap2 = ep2.snapshot()
    sel2, info2 = AC.choose("G1", ep2, snap2, _logits(snap2, stack), AC.Memory(ep2.state))
    assert snap2.candidate_ids[sel2] == put and info2["intervened"]
    # no strict-superset move exists (hand empty, everything already on the table in the goal and in the state): the raw choice is kept
    p3 = S.Problem(names, (0, 1, 0), (S.TABLE, S.TABLE, S.TABLE), (S.TABLE, 0, S.TABLE))
    ep3 = BwEpisode(Case("g1c", "t", p3, 1, 6))
    snap3 = ep3.snapshot()
    raw = S.action_id(names, ("PICK_UP", 2))
    sel3, info3 = AC.choose("G1", ep3, snap3, _logits(snap3, raw), AC.Memory(ep3.state))
    assert snap3.candidate_ids[sel3] == raw and not info3["intervened"]


def test_controllers_do_not_read_planner_labels():
    src = inspect.getsource(AC.choose)
    assert "solver" not in src and "optimal" not in src and "A*" not in src
    assert list(inspect.signature(AC.choose).parameters) == ["controller", "ep", "snap", "logits", "mem"]


def test_first_error_fields_and_denominators():
    names = ("b0", "b1", "b2")
    p = S.Problem(names, (0, 1, 0), (S.TABLE, S.HELD, S.TABLE), (S.TABLE, 0, S.TABLE))
    stack, put = S.action_id(names, ("STACK", 1, 0)), S.action_id(names, ("PUT_DOWN", 1))
    decs = [{"state": [-1, -2, -1], "selected": put, "optimal_actions": [stack], "probs": {put: 0.7, stack: 0.3}},
            {"state": [-1, -2, -1], "selected": put, "optimal_actions": [stack], "probs": {put: 0.7, stack: 0.3}}]
    f = AC.first_error_fields(decs, p)
    assert f["has_first_error"] and f["first_error_is_put_down_instead_of_goal_stack"] and f["best_optimal_rank"] == 2
    assert f["optimal_action_types"] == "STACK" and f["first_repeated_action_is_optimal"] is False
    clean = AC.first_error_fields([{"state": [0], "selected": stack, "optimal_actions": [stack], "probs": {stack: 1.0}}], p)
    assert not clean["has_first_error"] and clean["best_optimal_rank"] is None


def test_goal_flattening_keeps_the_target_tower_and_the_atom_count():
    goal = (S.TABLE, 0, S.TABLE, 2, S.TABLE)               # towers: [0,1], [2,3], [4]
    comps = GP.goal_components(goal)
    assert sorted(comps) == [(0, 1), (2, 3), (4,)]
    flat = GP.flatten_other_towers(goal, 1)
    assert flat == (S.TABLE, 0, S.TABLE, S.TABLE, S.TABLE) and len(flat) == len(goal)


def test_time_intervention_changes_only_the_two_time_dimensions(cases):
    snap = BwEpisode(cases[0]).snapshot()
    s2 = GP.with_time(snap, 0.25)
    assert s2.base_input[0] == 0.25 and s2.base_input[1] == 0.75 and s2.base_input[2:] == snap.base_input[2:]
    assert s2.candidate_ids == snap.candidate_ids and s2.facts == snap.facts


def test_probe_decision_rules_are_the_registered_ones():
    recs = [{"case_id": "c%d" % i, "status": "VALID"} for i in range(12)]

    def rows(g10_delta, g01_delta):
        out = []
        for i in range(12):
            out += [{"case_id": "c%d" % i, "condition": "G00", "margin": -2.0, "argmax_is_goal_stack": False, "optimal_action_probability_mass": 0.1},
                    {"case_id": "c%d" % i, "condition": "G10", "margin": -2.0 + g10_delta, "argmax_is_goal_stack": g10_delta > 2.5, "optimal_action_probability_mass": 0.2},
                    {"case_id": "c%d" % i, "condition": "G01", "margin": -2.0 + g01_delta, "argmax_is_goal_stack": g01_delta > 2.5, "optimal_action_probability_mass": 0.2},
                    {"case_id": "c%d" % i, "condition": "G11", "margin": -2.0, "argmax_is_goal_stack": False, "optimal_action_probability_mass": 0.1}]
        return out
    assert GP.summarize_probe(recs, rows(3.0, 0.0))["state"] == "GOAL_PROGRESS_READY"
    assert GP.summarize_probe(recs, rows(0.0, 3.0))["state"] == "TIME_INPUT_FIRST"
    assert GP.summarize_probe(recs, rows(0.2, 0.2))["state"] == "PROBE_INCONCLUSIVE"
    assert GP.summarize_probe(recs, rows(1.5, 0.0))["state"] == "GOAL_PROGRESS_READY"          # margin improved in 100% with median 1.5 >= 1.0
    assert GP.summarize_probe(recs[:11], rows(3.0, 0.0))["state"] == "INSUFFICIENT_PROBE"


def test_pilot_split_is_new_and_disjoint_if_built():
    p = ROOT / R.PILOT_REL
    if not p.is_file():
        pytest.skip("pilot not built yet")
    pilot = json.loads(p.read_text())
    assert pilot["counts"] == 80 and pilot["namespace"] == "C1-BW-A04P-PILOT-v1"
    old = set()
    td = json.loads((ROOT / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    for c in td["train"] + td["dev"]:
        old.add(c["problem_iso_hash"])
    for f in ("a0_iso", "a1_color_reverse", "a2_noniso", "b_scale"):
        old |= {c["problem_iso_hash"] for c in json.loads((ROOT / ("configs/splits/c1_bw_%s_v1.json" % f)).read_text())["cases"]}
    mine = [c["problem_iso_hash"] for c in pilot["cases"]]
    assert not (set(mine) & old) and len(set(mine)) == len(mine)
    assert all(c["step_cap"] == 2 * c["optimal_length"] + 4 for c in pilot["cases"])
