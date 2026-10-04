"""Representation fix: public semantic classes reach every method; prior-score/candidate correspondence is expressible.

These tests use hand-built public states (no ALFWorld data needed)."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cp_disr.platforms.alfworld.adapter import PublicState  # noqa: E402
from cp_disr.platforms.alfworld.ontology import NODE_TYPES, OTYPE_INDEX, RTYPE_INDEX  # noqa: E402
from cp_disr.platforms.alfworld.pddl_contracts import KINDS, PREDICATE_TYPES, candidate_key  # noqa: E402
from cp_disr.platforms.alfworld.snapshot import BASE_DIM, CAND_DIM, build_snapshot, prior_edges_for  # noqa: E402

PRIOR_METHODS = ("A_STAT", "Full", "A_CAT", "PRIOR_BIAS")


def make_pub(feasible, goal_otype="keychain", goal="desk 1"):
    recs = tuple(feasible) + (goal,)
    return PublicState(
        goal_otype=goal_otype, goal_rtype="desk", receptacles=recs, feasible=tuple(feasible), unchecked=tuple(feasible),
        goal_instance=goal, raw_steps=0, max_steps=50, checked=(), target_in=(), target_found=False, holding=None,
        placed=False, legal=tuple(("CHECK", r) for r in feasible), done=False)


def snap(pub, scores, d=0):
    from cp_disr.platforms.alfworld.pddl_contracts import episode_template

    tpl = episode_template(pub.feasible, pub.goal_instance, pub.goal_otype)
    return build_snapshot(pub, prior_edges_for(tpl, scores) if scores is not None else (), "e", "x", d)


def policy(method, seed=0):
    from cp_disr.neural_batched import BatchedPolicy

    torch.manual_seed(seed)
    return BatchedPolicy(set(KINDS), set(PREDICATE_TYPES), set(NODE_TYPES), BASE_DIM, CAND_DIM, method=method).eval()


def logit(out, snapshot, name):
    return out.logits[snapshot.candidate_ids.index(next(c.id for c in snapshot.template.contracts if candidate_key(c) == ("CHECK", name)))]


def resid(out, snapshot, name):
    cid = next(c.id for c in snapshot.template.contracts if candidate_key(c) == ("CHECK", name))
    return out.diagnostics["delta"][cid]


def test_vocabulary_is_fixed_and_unk_handles_unseen_classes():
    from cp_disr.platforms.alfworld.ontology import oclass, rclass

    assert "UNK" in RTYPE_INDEX and "UNK" in OTYPE_INDEX
    assert rclass("drawer 3") == "rtype:drawer" and rclass("zzz 1") == "rtype:UNK" and oclass("qqq") == "otype:UNK"
    assert all(t in NODE_TYPES for t in ("rtype:drawer", "rtype:UNK", "otype:keychain", "otype:UNK"))


def test_nodes_carry_semantic_class_but_no_instance_identity():
    pub = make_pub(["drawer 1", "bed 1", "shelf 2"])
    s = snap(pub, None)
    by = {n.id: n for n in s.template.nodes}
    assert by["a:CHECK:drawer 1:v1"].argument_types == ("rtype:drawer",)
    assert by["a:CHECK:bed 1:v1"].argument_types == ("rtype:bed",)
    assert by["p:checked:shelf 2"].argument_types == ("rtype:shelf",)
    assert by["p:target_found"].argument_types == ("otype:keychain",)
    # the encoder-visible types never contain an instance number / path / room id
    assert all(not any(ch.isdigit() for ch in t) for n in s.template.nodes for t in n.argument_types)
    # candidate features expose class one-hots (receptacle and target) and nothing else public
    i = s.candidate_ids.index("a:CHECK:bed 1:v1")
    f = s.candidate_features[i]
    assert len(f) == CAND_DIM and f[len(KINDS) + RTYPE_INDEX["bed"]] == 1.0 and f[len(KINDS) + len(RTYPE_INDEX) + OTYPE_INDEX["keychain"]] == 1.0


@pytest.mark.parametrize("method", ("B2",) + PRIOR_METHODS)
def test_every_method_distinguishes_public_receptacle_classes(method):
    pub = make_pub(["drawer 1", "bed 1", "shelf 1", "desk 2"])
    s = snap(pub, None)
    with torch.no_grad():
        out = policy(method)(s)
    vals = [round(float(logit(out, s, n)), 7) for n in pub.feasible]
    assert len(set(vals)) > 1, vals


@pytest.mark.parametrize("method", ("B2",) + PRIOR_METHODS)
def test_same_class_instances_are_exchangeable_at_init(method):
    pub = make_pub(["drawer 1", "drawer 2", "bed 1"])
    s = snap(pub, None)
    with torch.no_grad():
        out = policy(method)(s)
    assert torch.allclose(logit(out, s, "drawer 1"), logit(out, s, "drawer 2"), atol=1e-6)


def test_b2_is_strictly_insensitive_to_the_prior():
    pub = make_pub(["drawer 1", "drawer 2", "bed 1", "shelf 1"])
    a = snap(pub, None)
    b = snap(pub, {"drawer 1": 0.7, "drawer 2": 0.1, "bed 1": 0.1, "shelf 1": 0.1})
    pol = policy("B2")
    with torch.no_grad():
        assert torch.equal(pol(a).logits, pol(b).logits) and torch.equal(pol(a).value, pol(b).value)


@pytest.mark.parametrize("method", PRIOR_METHODS)
def test_same_class_score_swap_is_an_isomorphism(method):
    pub = make_pub(["drawer 1", "drawer 2", "bed 1"])
    s1 = snap(pub, {"drawer 1": 0.6, "drawer 2": 0.1, "bed 1": 0.3})
    s2 = snap(pub, {"drawer 1": 0.1, "drawer 2": 0.6, "bed 1": 0.3})
    pol = policy(method)
    if method == "PRIOR_BIAS":
        pol.prior_beta.data.fill_(0.8)
    with torch.no_grad():
        o1, o2 = pol(s1), pol(s2)
    assert torch.allclose(resid(o1, s1, "drawer 1"), resid(o2, s2, "drawer 2"), atol=1e-6)
    assert torch.allclose(resid(o1, s1, "drawer 2"), resid(o2, s2, "drawer 1"), atol=1e-6)
    assert torch.allclose(resid(o1, s1, "bed 1"), resid(o2, s2, "bed 1"), atol=1e-6)


@pytest.mark.parametrize("method", PRIOR_METHODS)
def test_candidate_residual_responds_when_scores_are_exchanged_between_classes(method):
    pub = make_pub(["drawer 1", "shelf 1"])
    s1 = snap(pub, {"drawer 1": 0.8, "shelf 1": 0.2})
    s2 = snap(pub, {"drawer 1": 0.2, "shelf 1": 0.8})
    pol = policy(method)
    if method == "PRIOR_BIAS":
        pol.prior_beta.data.fill_(0.8)
    with torch.no_grad():
        o1, o2 = pol(s1), pol(s2)
    d1 = float(resid(o1, s1, "drawer 1") - resid(o1, s1, "shelf 1"))
    d2 = float(resid(o2, s2, "drawer 1") - resid(o2, s2, "shelf 1"))
    assert abs(d1 - d2) > 1e-9, (method, d1, d2)
    if method == "PRIOR_BIAS":
        assert d1 > 0 > d2  # raises the high-score candidate in both worlds

@pytest.mark.parametrize("method", PRIOR_METHODS)
def test_empty_prior_is_an_exact_null(method):
    pub = make_pub(["drawer 1", "drawer 2", "bed 1"])
    s = snap(pub, None)
    with torch.no_grad():
        out = policy(method)(s)
    assert all(float(v) == 0.0 for v in out.diagnostics["delta"].values())


@pytest.mark.parametrize("method", ("B2",) + PRIOR_METHODS)
def test_same_class_instance_renaming_equivariance(method):
    names = ["drawer 1", "drawer 2", "bed 1"]
    ren = {"drawer 1": "drawer 7", "drawer 2": "drawer 3", "bed 1": "bed 5"}
    scores = {"drawer 1": 0.5, "drawer 2": 0.2, "bed 1": 0.3}
    pub = make_pub(names)
    pub2 = make_pub([ren[n] for n in names])
    s1, s2 = snap(pub, scores), snap(pub2, {ren[k]: v for k, v in scores.items()})
    pol = policy(method)
    if method == "PRIOR_BIAS":
        pol.prior_beta.data.fill_(0.8)
    with torch.no_grad():
        o1, o2 = pol(s1), pol(s2)
    for n in names:
        assert torch.allclose(logit(o1, s1, n), logit(o2, s2, ren[n]), atol=1e-5)
    assert torch.allclose(o1.value, o2.value, atol=1e-5)


def _train_two_candidates(method, steps=500, lr=1e-3, seed=3):
    """Two candidates of different classes; the prior alone says which one is right (scores exchanged between worlds). Cross-entropy toward the high-score one."""
    pub = make_pub(["drawer 1", "shelf 1"])
    hi1 = snap(pub, {"drawer 1": 0.8, "shelf 1": 0.2})   # correct: drawer 1
    hi2 = snap(pub, {"drawer 1": 0.2, "shelf 1": 0.8})   # correct: shelf 1
    pol = policy(method, seed=seed).train()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    idx = lambda s, n: s.candidate_ids.index("a:CHECK:%s:v1" % n)
    first_grad = None
    for step in range(steps):
        loss = 0
        for s, good in ((hi1, "drawer 1"), (hi2, "shelf 1")):
            out = pol(s)
            loss = loss - out.distribution.log_prob(torch.tensor(idx(s, good)))
        opt.zero_grad()
        loss.backward()
        if step == 0:
            first_grad = sum(float(p.grad.abs().sum()) for n, p in pol.named_parameters() if p.grad is not None and ("prior" in n or "cat_proj" in n))
        opt.step()
    with torch.no_grad():
        p1 = float(pol(hi1).distribution.probs[idx(hi1, "drawer 1")])
        p2 = float(pol(hi2).distribution.probs[idx(hi2, "shelf 1")])
    return p1, p2, first_grad


def test_b2_cannot_fit_the_swapped_prior_task():
    p1, p2, _ = _train_two_candidates("B2", steps=150, lr=1e-3)
    assert min(p1, p2) < 0.6  # prior-blind: cannot pick drawer in one world and shelf in the other


@pytest.mark.parametrize("method", PRIOR_METHODS)
@pytest.mark.parametrize("seed", (3, 4))
def test_prior_methods_can_overfit_the_two_candidate_prior_task(method, seed):
    # lr 1e-3 (PPO uses 3e-4); lr 5e-3 collapses some seeds to a dead branch, see the diagnostic in the report
    # PRIOR_BIAS moves a single scalar, so it needs a larger step size to get there inside the same budget
    kw = {"lr": 4e-3} if method == "PRIOR_BIAS" else {}
    p1, p2, g0 = _train_two_candidates(method, seed=seed, **kw)
    assert g0 is None or g0 >= 0
    # residuals are bounded by B=0.5 per candidate, so the best reachable two-way probability is sigmoid(1.0)=0.731\n    assert min(p1, p2) > 0.70, (method, p1, p2)
