"""Phase-2 tests (runbook section 15, items 4-6): decomposition constraint, matched M1/M2 parameters and inputs, data isolation, label rules. Train-split fixtures only."""
import inspect
import itertools
import json
from pathlib import Path

import pytest
import torch

from cp_disr.blocksworld import eval_a03 as E
from cp_disr.blocksworld import goal_progress as GP
from cp_disr.blocksworld import imitation as I
from cp_disr.blocksworld import launch as L
from cp_disr.blocksworld import state as S
from cp_disr.blocksworld.environment import BwEpisode, Case
from cp_disr.rl import set_suite_half_life

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / E.MODELS["B2"]["path"]
pytestmark = pytest.mark.skipif(not CK.is_file(), reason="A02 B2 checkpoint not on this machine")


@pytest.fixture(scope="module")
def cases():
    cs, H = I.load_train_cases(ROOT / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(H)
    return cs


@pytest.fixture(scope="module")
def mini(cases):
    by = {}
    for c in cases:
        by.setdefault(c.problem.n, []).append(c)
    return [x for n in sorted(by) for x in by[n][:2]]


def model(mode):
    return GP.GoalProgressModel(GP.load_base(ROOT, torch.device("cpu")), mode)


def test_new_heads_have_the_registered_size_and_identical_initial_tensors():
    m1, m2 = model("global"), model("additive")
    n1 = sum(p.numel() for p in m1.heads.parameters())
    assert n1 == sum(p.numel() for p in m2.heads.parameters()) == 82688
    for a, b in zip(m1.heads.state_dict().values(), m2.heads.state_dict().values()):
        assert torch.equal(a, b)
    for k, v in m1.base.encoder.state_dict().items():
        assert torch.equal(v, m2.base.encoder.state_dict()[k])
    assert m1.mode != m2.mode


def test_only_the_encoder_and_the_new_heads_are_trainable():
    m = model("additive")
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith("base.encoder.") or n.startswith("heads.") for n in names)
    assert not any(n.startswith(("base.v_head", "base.q_head", "base.gru", "base.contract", "base.observation", "base.base")) for n in names)


def _snap(case):
    return BwEpisode(case).snapshot()


def test_additive_scores_decompose_over_goal_sets_and_ignore_order(mini):
    torch.manual_seed(1)
    m2 = model("additive")
    snap = _snap(mini[-1])
    ng = len(snap.template.goals)
    assert ng >= 4
    idx = list(range(ng))
    G = torch.zeros(ng, dtype=torch.bool)
    H = torch.zeros(ng, dtype=torch.bool)
    G[idx[: ng // 2]] = True
    H[idx[ng // 2:]] = True
    with torch.no_grad():
        sG, sH, sGH = (m2.group_scores([snap], goal_mask=g)[0] for g in (G, H, G | H))
        legal = torch.tensor(snap.mask)
        assert torch.allclose(sG[legal] + sH[legal], sGH[legal], atol=1e-4)
        perm = torch.randperm(ng)
        full = m2.group_scores([snap])[0]
        # a permutation of the goal set (same members, other order) leaves the score unchanged
        st0, gprop = m2.goal_free_static(snap.template)
        base = m2.base
        st = base._static(snap.template)
        codes0 = base._codes(st, snap.facts.values).unsqueeze(0)
        x = m2.features(st0, gprop, codes0)
        v_perm = m2.values(x[:, perm])
        assert torch.allclose(v_perm, m2.values(x), atol=1e-5)
        empty = m2.group_scores([snap], goal_mask=torch.zeros(ng, dtype=torch.bool))[0]
        assert torch.allclose(empty[legal], torch.zeros(int(legal.sum())), atol=1e-6)
        assert torch.isfinite(full[legal]).all()
    m1 = model("global")
    with torch.no_grad():
        a, b, ab = (m1.group_scores([snap], goal_mask=g)[0][legal] for g in (G, H, G | H))
    assert not torch.allclose(a + b, ab, atol=1e-4)                                # the global head is not additive (control for the test itself)


def test_per_goal_inputs_do_not_depend_on_the_other_goals(cases):
    m = model("additive")
    p = cases[-1].problem                                                           # n=5, goal has a tower
    comps = [c for c in __import__("cp_disr.blocksworld.goal_probe", fromlist=["x"]).goal_components(p.goal) if len(c) >= 2]
    from cp_disr.blocksworld import goal_probe as GPB
    x_block = comps[0][-1]
    flat = GPB.flatten_other_towers(p.goal, x_block)
    p2 = S.Problem(p.names, p.colors, p.init, flat)
    s1, s2 = _snap(Case("a", "t", p, cases[-1].optimal_length, 30)), _snap(Case("b", "t", p2, cases[-1].optimal_length, 30))
    with torch.no_grad():
        def feats(snap):
            st0, gprop = m.goal_free_static(snap.template)
            st = m.base._static(snap.template)
            x = m.features(st0, gprop, m.base._codes(st, snap.facts.values).unsqueeze(0))[0]
            return {g.fact_id: x[i] for i, g in enumerate(snap.template.goals)}
        f1, f2 = feats(s1), feats(s2)
    common = set(f1) & set(f2)
    assert len(common) >= 3
    for k in common:
        assert torch.allclose(f1[k], f2[k], atol=1e-6)


def test_labels_are_not_forward_inputs():
    assert list(inspect.signature(GP.GoalProgressModel.group_scores).parameters) == ["self", "snaps", "goal_mask"]
    assert list(inspect.signature(GP.ScorePolicy.__call__).parameters) == ["self", "snap", "hidden"]


def test_rank_loss_and_il_loss_on_fixtures():
    scores = torch.tensor([[3.0, 1.0, -torch.inf], [0.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, False], [True, True, True]])
    dist = torch.tensor([[1.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
    astar = torch.tensor([[True, False, False], [True, True, True]])
    nll, rank, cnt = GP.nll_and_rank(scores, astar, dist, legal)
    assert cnt.tolist() == [1, 0] and float(rank[1]) == 0.0                         # equal distances: no pair, no rank loss
    assert abs(float(rank[0]) - float(torch.nn.functional.softplus(torch.tensor(-2.0)))) < 1e-6
    assert abs(float(nll[1])) < 1e-6                                                # every legal action optimal -> probability mass 1


@pytest.mark.parametrize("mode", ["global", "additive"])
def test_one_matched_step_updates_encoder_and_heads_only(mode, mini):
    d0 = I.build_d0(mini)
    labels = GP.build_rank_labels(d0, mini)
    m = model(mode)
    tr = GP.GPTrainer(m, mini, labels, torch.device("cpu"))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    row = tr.step(d0[:6])
    assert row["decisions"] > 0 and torch.isfinite(torch.tensor(row["loss"]))
    changed = [k for k, v in m.state_dict().items() if not torch.equal(before[k], v)]
    assert any(k.startswith("heads.") for k in changed) and any(k.startswith("base.encoder.") for k in changed)
    assert all(k.startswith(("heads.", "base.encoder.")) for k in changed)


def test_training_loader_cannot_open_pilot_confirm_or_probe_files():
    for rel in ("configs/splits/c1_bw_a04p_pilot80_v1.json", "configs/splits/c1_bw_gp_confirm112_v1.json", "configs/splits/c1_bw_b_scale_v1.json"):
        with pytest.raises(L.LaunchError):
            I.load_train_cases(ROOT / rel)
    src = (ROOT / "src/cp_disr/blocksworld/goal_progress.py").read_text()
    for token in ("pilot", "confirm", "probe", "a04p"):
        assert token not in src.lower().replace("goal_probe", "")


def test_direction_labels_follow_the_registered_rules():
    def tab(d):
        t = {}
        for c, (perf, succ) in d.items():
            for s in GP.CORE:
                t[(c, s)] = {"n": 8, "success": succ, "perfect": perf}
            t[(c, "A1-4")] = {"n": 8, "success": 8 if c != "M2" else 8, "perfect": 8}
            t[(c, "A1-5")] = {"n": 8, "success": 8, "perfect": 8}
        return t
    good = tab({"M0": (1, 1), "M1": (2, 2), "M2": (5, 5), "C0": (1, 1), "C3": (2, 8), "G1": (3, 3)})
    lab = GP.direction_labels(good)["labels"]
    assert "DECOMPOSITION_PROMISING" in lab and "NO_CLEAR_ADDITIVE_ADVANTAGE" not in lab
    flat = tab({"M0": (1, 1), "M1": (4, 4), "M2": (4, 4), "C0": (1, 1), "C3": (1, 1), "G1": (1, 1)})
    lab2 = GP.direction_labels(flat)["labels"]
    assert "NO_CLEAR_ADDITIVE_ADVANTAGE" in lab2 and "DECOMPOSITION_PROMISING" not in lab2
    bad = tab({"M0": (3, 3), "M1": (3, 3), "M2": (2, 2), "C0": (3, 3), "C3": (1, 1), "G1": (1, 1)})
    assert "NEW_MODELS_NOT_SUPPORTED" in GP.direction_labels(bad)["labels"]
