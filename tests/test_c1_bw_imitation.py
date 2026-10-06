"""CP-DISR-C1-BW-IMITATION-A02 unit tests (A02 section 13): planner labels, loss, sequence handling, shared aggregation, model boundaries and the negative (must-refuse) tests."""
import inspect
import json
import math
import re
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import imitation_launch as IL
from cp_disr.blocksworld import launch as L
from cp_disr.blocksworld import planner as P
from cp_disr.blocksworld import state as S
from cp_disr.rl import Snapshot, set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / L.TRAIN_SPLIT_REL
METHODS = ("B2-CACHED", "B1-K+QMARK-BW", "ASNET-READOUT")


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(SPLIT)
    set_suite_half_life(H)
    return cs


@pytest.fixture(scope="module")
def mini(cases):
    by_n = {}
    for c in cases:
        by_n.setdefault(c.problem.n, []).append(c)
    out = []
    for n in sorted(by_n):
        out += by_n[n][:2]
    return out


@pytest.fixture(scope="module")
def d0_mini(mini):
    return I.build_d0(mini)


def _sources(cases):
    return {c.case_id: c for c in cases}


# ------------------------------------------------------------------ planner labels
def test_astar_is_exactly_the_distance_minus_one_set(cases, d0_mini):
    by_id = _sources(cases)
    seen_multi = 0
    for t in d0_mini:
        case = by_id[t["case_id"]]
        goal = case.problem.goal
        for s, ids in zip(t["states"][:-1], t["astar"]):
            s = tuple(s)
            d = P.bfs_cost(s, goal)
            truth = sorted(S.action_id(case.problem.names, a) for a, nxt in S.successors(s) if P.bfs_cost(nxt, goal) == d - 1)
            assert sorted(ids) == truth and truth
            legal = {S.action_id(case.problem.names, a) for a in S.legal_actions(s)}
            assert set(ids) <= legal
            for a, nxt in S.successors(s):
                aid = S.action_id(case.problem.names, a)
                assert (aid in ids) == (P.bfs_cost(nxt, goal) == d - 1)
            seen_multi += len(ids) > 1
    assert seen_multi >= 0


def test_multi_solution_decisions_keep_the_full_set(cases):
    d0 = I.build_d0(cases)
    multi = [(t, i) for t in d0 for i, a in enumerate(t["astar"]) if len(a) > 1]
    assert multi, "D0 must contain decisions with several optimal actions"
    for t, i in multi:
        assert t["actions"][i] == t["astar"][i][0]          # the expert's single action is the lexicographically first, the label keeps all
        assert len(t["astar"][i]) >= 2
    assert I.dataset_identity(d0)["decisions_with_multiple_optimal_actions"] == len(multi)


def test_terminal_states_carry_no_decision(d0_mini):
    for t in d0_mini:
        assert len(t["states"]) == len(t["actions"]) + 1 == len(t["astar"]) + 1


# ------------------------------------------------------------------ loss
def test_loss_counts_every_optimal_action_as_correct_mass():
    logits = torch.tensor([[2.0, 1.0, 0.5, -torch.inf]])
    p = torch.softmax(logits, -1)[0]
    two = torch.tensor([[True, True, False, False]])
    one = torch.tensor([[True, False, False, False]])
    assert math.isclose(float(I.imitation_nll(logits, two)), -math.log(float(p[0] + p[1])), rel_tol=1e-6)
    assert math.isclose(float(I.imitation_nll(logits, one)), -math.log(float(p[0])), rel_tol=1e-6)
    assert float(I.imitation_nll(logits, two)) < float(I.imitation_nll(logits, one))
    # an illegal action is masked (-inf) and never receives mass
    assert float(torch.softmax(logits, -1)[0, 3]) == 0.0


def test_loss_gradient_is_zero_when_all_mass_is_on_the_optimal_set():
    logits = torch.tensor([[30.0, 30.0, -30.0, -30.0]], requires_grad=True)
    I.imitation_nll(logits, torch.tensor([[True, True, False, False]])).sum().backward()
    assert float(logits.grad.abs().max()) < 1e-6


# ------------------------------------------------------------------ sequence handling
@pytest.mark.parametrize("method", METHODS)
def test_batched_nll_matches_the_per_snapshot_forward(method, mini, d0_mini):
    tag = I.TAG_OF[method]
    policy = I.make_imitation_policy(method, "cpu", 0)
    tr = I.ImitationTrainer(policy, I.ROW_KIND[tag], mini, torch.device("cpu"))
    trajs = [d0_mini[0], d0_mini[-1], d0_mini[len(d0_mini) // 2]]                 # different lengths in one batch (padding)
    per = [tr.snapshots(t) for t in trajs]
    zo = I.trajectory_hidden(policy, per, torch.device("cpu"))
    snaps = [s for p in per for s in p]
    astar = [a for t in trajs for a in t["astar"]]
    nll, _hit = I.batch_losses(policy, tr.kind, snaps, torch.cat(zo, 0), astar)
    ref = []
    with torch.no_grad():
        for t, p in zip(trajs, per):
            hidden = policy.initial_hidden()                                          # zero ONLY at the trajectory start, carried along it
            for sn, ids in zip(p, t["astar"]):
                out = policy(sn, hidden)
                hidden = out.hidden
                probs = out.distribution.probs
                ref.append(-math.log(sum(float(probs[sn.candidate_ids.index(a)]) for a in ids)))
    assert torch.allclose(nll.detach(), torch.tensor(ref), atol=2e-4, rtol=1e-4)


def test_padding_is_excluded_and_hidden_does_not_leak_between_trajectories(mini, d0_mini):
    policy = I.make_imitation_policy("ASNET-READOUT", "cpu", 0)
    tr = I.ImitationTrainer(policy, "asnet", mini, torch.device("cpu"))
    short = min(d0_mini, key=lambda t: len(t["actions"]))
    long_ = max(d0_mini, key=lambda t: len(t["actions"]))
    assert len(short["actions"]) < len(long_["actions"])
    alone = I.trajectory_hidden(policy, [tr.snapshots(short)], torch.device("cpu"))[0]
    both = I.trajectory_hidden(policy, [tr.snapshots(long_), tr.snapshots(short)], torch.device("cpu"))
    assert both[1].shape[0] == len(short["actions"]) and both[0].shape[0] == len(long_["actions"])
    assert torch.allclose(alone, both[1], atol=1e-6)
    first = policy.advance_hidden(tr.snapshots(short)[0].base_input, policy.initial_hidden())
    assert torch.allclose(both[1][0], first, atol=1e-6)


def test_every_trajectory_has_the_same_total_weight_loops_do_not_weigh_more(d0_mini):
    base = d0_mini[0]
    loop = dict(base, states=base["states"][:1] * 21, actions=base["actions"][:1] * 20, astar=base["astar"][:1] * 20, remaining=base["remaining"][:1] * 20)
    batch = [base, loop, d0_mini[-1]]
    ws = I.decision_weights(batch)
    totals = [sum(w) for w in ws]
    assert all(math.isclose(x, 1.0 / len(batch), rel_tol=1e-9) for x in totals)
    assert len(ws[1]) == 20 and ws[1][0] < ws[0][0]
    cyc = I.dataset_identity([loop])
    assert cyc["cycle_containing_trajectories"] == 1


# ------------------------------------------------------------------ shared aggregation
def test_aggregation_is_identical_for_all_methods_and_one_trajectory_per_source_case(mini, d0_mini):
    solver = P.Solver()
    rolls = {}
    for method in METHODS:
        tag = I.TAG_OF[method]
        pol = I.make_imitation_policy(method, "cpu", 0)
        pol.eval()
        rolls[tag] = [I.rollout_trajectory(pol, c, solver, "%s_r1" % tag) for c in mini]
        assert len(rolls[tag]) == len(mini)
        for c, t in zip(mini, rolls[tag]):
            assert len(t["actions"]) <= c.step_cap                                    # the frozen step cap
            assert t["tid"] == "%s_r1:%s" % (tag, c.case_id)
    d1_a = I.aggregate(d0_mini, rolls)
    d1_b = I.aggregate(list(d0_mini), {k: list(v) for k, v in rolls.items()})
    assert I.dataset_identity(d1_a)["sha256"] == I.dataset_identity(d1_b)["sha256"]
    assert [t["source"] for t in d1_a][len(d0_mini):len(d0_mini) + len(mini)] == ["B2_r1"] * len(mini)
    assert len(d1_a) == len(d0_mini) + 3 * len(mini)
    I.verify_labels(d1_a, mini)                                                       # every visited state is planner-labelled with the full set
    ident = I.dataset_identity(d1_a)
    assert ident["source_counts"] == {"expert": len(d0_mini), "B2_r1": len(mini), "QMARK_r1": len(mini), "ASNET_r1": len(mini)}


def test_evaluation_splits_and_the_dev_rows_cannot_be_opened_by_training_code():
    for name in L.EVAL_SPLIT_NAMES:
        with pytest.raises(L.LaunchError):
            I.load_train_cases(ROOT / "configs" / "splits" / name)
    src = (ROOT / "src/cp_disr/blocksworld/imitation.py").read_text() + (ROOT / "src/cp_disr/blocksworld/imitation_launch.py").read_text()
    assert re.search(r"\[\"dev\"\]|'dev'|load_cases|a0_iso|a1_color|a2_noniso|b_scale", src) is None
    assert "load_checkpoint" not in src


# ------------------------------------------------------------------ model boundaries
def test_shared_modules_share_the_init_seed_and_architecture_specific_parameters_are_derived():
    pols = {m: I.make_imitation_policy(m, "cpu", 0) for m in METHODS}
    ref = pols["B2-CACHED"]
    for m in ("B1-K+QMARK-BW", "ASNET-READOUT"):
        for name in ("encoder", "observation", "gru", "candidate", "contract", "base", "v_head", "q_head"):
            a, b = getattr(ref, name).state_dict(), getattr(pols[m], name).state_dict()
            assert all(torch.equal(a[k], b[k]) for k in a), (m, name)
    again = I.make_imitation_policy("B1-K+QMARK-BW", "cpu", 0)
    assert torch.equal(again.query_projection.weight, pols["B1-K+QMARK-BW"].query_projection.weight)
    assert I.derived_seed("query_projection", 0) == I.derived_seed("query_projection", 0) != I.derived_seed("query_projection", 1)


@pytest.mark.parametrize("method", METHODS)
def test_one_step_trains_policy_parameters_only(method, mini, d0_mini):
    tag = I.TAG_OF[method]
    policy = I.make_imitation_policy(method, "cpu", 0)
    tr = I.ImitationTrainer(policy, I.ROW_KIND[tag], mini, torch.device("cpu"))
    frozen = I.module_digest(policy, I.FROZEN_MODULES)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    row = tr.step(d0_mini[:8])
    assert math.isfinite(row["loss"]) and row["decisions"] > 0
    assert I.module_digest(policy, I.FROZEN_MODULES) == frozen
    for name in ("v_head", "q_head"):
        assert all(p.grad is None and not p.requires_grad for p in getattr(policy, name).parameters())
    changed = [k for k, v in policy.state_dict().items() if not torch.equal(before[k], v)]
    assert changed and not any(k.startswith(("v_head", "q_head")) for k in changed)
    assert hasattr(policy, "query_projection") == (method == "B1-K+QMARK-BW")
    assert not hasattr(policy, "nominal_apply_calls") or policy.nominal_apply_calls == 0
    if tag == "QMARK":
        assert not hasattr(policy, "nominal_apply_calls")


def test_planner_labels_are_not_a_model_input():
    assert list(inspect.signature(I._group_logits).parameters) == ["policy", "kind", "snaps", "zos"]
    assert "astar" not in {f for f in Snapshot.__dataclass_fields__}
    assert not any("astar" in f or "optimal" in f for f in Snapshot.__dataclass_fields__)


def test_every_child_module_is_classified():
    assert not set(I.TRAINABLE_MODULES) & set(I.FROZEN_MODULES)
    for m in METHODS:
        pol = I.make_imitation_policy(m, "cpu", 0)
        names = {n for n, _ in pol.named_children()}
        assert names <= set(I.TRAINABLE_MODULES) | set(I.FROZEN_MODULES)
        assert {"v_head", "q_head"} <= set(I.FROZEN_MODULES)


# ------------------------------------------------------------------ negative tests (each must fail the corresponding illegal variant)
def test_ppo_checkpoint_and_mid_checkpoint_loading_is_refused():
    with pytest.raises(I.ImitationError):
        I.refuse_checkpoint_init({"init": "/x/blocksworld_main_v1/20261006T015313Z_490a3b11/runs/R-C1-BW-ASNET-0/checkpoints/final.pt"})
    with pytest.raises(I.ImitationError):
        I.refuse_checkpoint_init({"init": "/x/checkpoints/n_0032768.pt"})            # the ASNET N=32768 checkpoint may not be chosen
    with pytest.raises(I.ImitationError):
        I.refuse_checkpoint_init({"init_checkpoint": None})
    I.refuse_checkpoint_init({"init": "FROM_SCRATCH", "seed": 0})


def _cfg(method, **over):
    c = {"method": method, "d0_sha256": "d", "loss": I.LOSS_NAME, "optimizer": dict(I.OPT), "stages": [dict(s) for s in I.STAGES], "seed": 0, "train_dev_split_sha256": "t", "aggregation_rounds": 2}
    c.update(over)
    return c


def test_planner_supervision_for_a_subset_of_methods_is_refused():
    cfgs = [_cfg(m) for m in METHODS]
    I.validate_identical_supervision(cfgs)
    for key, val in (("d0_sha256", "other"), ("loss", "PPO"), ("seed", 1), ("aggregation_rounds", 0)):
        bad = [_cfg(METHODS[0]), _cfg(METHODS[1]), _cfg(METHODS[2], **{key: val})]
        with pytest.raises(I.ImitationError):
            I.validate_identical_supervision(bad)
    with pytest.raises(I.ImitationError):
        I.validate_identical_supervision([_cfg(METHODS[0]), _cfg(METHODS[1])])


def test_different_aggregation_data_across_methods_is_refused(tmp_path):
    I.check_shared_hash(tmp_path, "D1", "abc")
    I.check_shared_hash(tmp_path, "D1", "abc")
    with pytest.raises(I.ImitationError):
        I.check_shared_hash(tmp_path, "D1", "abd")


def test_a_single_tie_break_action_replacing_the_full_set_is_refused(cases):
    d0 = I.build_d0(cases)
    t = next(t for t in d0 if any(len(a) > 1 for a in t["astar"]))
    i = next(i for i, a in enumerate(t["astar"]) if len(a) > 1)
    broken = dict(t, astar=[list(a) for a in t["astar"]])
    broken["astar"][i] = [t["actions"][i]]
    with pytest.raises(I.ImitationError):
        I.verify_labels([broken], cases)
    I.verify_labels([t], cases)


def test_tuning_epochs_rounds_or_optimizer_is_refused():
    I.validate_schedule([dict(s) for s in I.STAGES], dict(I.OPT))
    longer = [dict(s) for s in I.STAGES]
    longer[0]["epochs"] = 120
    with pytest.raises(I.ImitationError):
        I.validate_schedule(longer, dict(I.OPT))
    with pytest.raises(I.ImitationError):
        I.validate_schedule([dict(s) for s in I.STAGES] + [{"name": "round3", "epochs": 50, "shuffle_seed": 3, "dataset": "D3"}], dict(I.OPT))
    with pytest.raises(I.ImitationError):
        I.validate_schedule([dict(s) for s in I.STAGES], dict(I.OPT, lr=1e-3))
    assert [s["epochs"] for s in I.STAGES] == [100, 50, 50]


def test_frozen_config_matches_the_code_constants():
    doc = json.loads((ROOT / IL.CONFIG_REL).read_text())
    assert doc["optimizer"] == I.OPT and doc["stages"] == [dict(s) for s in I.STAGES]
    assert doc["authorized_training_runs"] == 3 and doc["ppo_checkpoints_loaded"] is False
    assert doc["id_gate"] == {"success_min": 33, "decision_perfect_min": 33, "nan": 0, "hard_fail": 0, "n_dev": 36}
    assert set(doc["runs"]) == set(IL.RUN_IDS)
    assert all(v["unchanged_since_freeze"] for v in doc["reused_files"].values())
