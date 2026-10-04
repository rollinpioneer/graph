"""E0: weighted prior edges, batched encoder equivalence, PRIOR_BIAS, shadow policy, permutation."""
from dataclasses import replace

import pytest

from .helpers.alfworld_fixture import FakePrior, gamepath, load_all, require_data

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

PRIOR_METHODS = ("A_STAT", "Full", "A_CAT", "PRIOR_BIAS")
FAST = ("B2",) + PRIOR_METHODS


def _policy(method, batched=False, seed=0):
    from cp_disr.neural import Policy
    from cp_disr.neural_batched import BatchedPolicy
    from cp_disr.platforms.alfworld.pddl_contracts import KINDS, PREDICATE_TYPES
    from cp_disr.platforms.alfworld.snapshot import BASE_DIM, CAND_DIM

    torch.manual_seed(seed)
    cls = BatchedPolicy if batched else Policy
    return cls(set(KINDS), set(PREDICATE_TYPES), {"receptacle"}, BASE_DIM, CAND_DIM, method=method).eval()


def _snapshots(n_games=6, with_prior=True):
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.snapshot import build_snapshot, prior_edges_for

    rows, splits, tables = load_all()
    prior = FakePrior()
    out = []
    for rel in splits["dev"][:n_games]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            d = 0
            while not ep.done and d < 4:
                pub = ep.public()
                tpl = build_snapshot(pub, (), "e", "x", d).template
                edges = prior_edges_for(tpl, prior.scores(pub, tables)) if with_prior else ()
                out.append(build_snapshot(pub, edges, "e", "x", d))
                checks = sorted(a for a in pub.legal if a[0] == "CHECK")
                ep.execute(*(checks[0] if checks else sorted(pub.legal)[0]))
                d += 1
        finally:
            ep.close()
    return out


def test_weighted_rgcn_equals_rgcn_for_unit_weights_and_scales_messages():
    from torch_geometric.nn import RGCNConv
    from cp_disr.neural import WeightedRGCNConv

    torch.manual_seed(1)
    ref = RGCNConv(16, 16, 12, num_bases=4, aggr="mean", root_weight=True)
    mine = WeightedRGCNConv(16, 16, 12, num_bases=4, aggr="mean", root_weight=True)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(9, 16)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 1], [1, 2, 3, 4, 5, 6, 7, 8, 0]])
    et = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0])
    assert torch.allclose(ref(x, ei, et), mine(x, ei, et), atol=1e-6)
    assert torch.allclose(ref(x, ei, et), mine(x, ei, et, torch.ones(9)), atol=1e-6)
    w = torch.ones(9)
    w[0] = 0.0
    assert not torch.allclose(mine(x, ei, et, w), ref(x, ei, et), atol=1e-6)  # weight really multiplies the message


@pytest.mark.parametrize("method", FAST)
@pytest.mark.parametrize("with_prior", (True, False))
def test_batched_policy_matches_reference_policy(method, with_prior):
    require_data()
    ref, fast = _policy(method), _policy(method, batched=True)
    fast.load_state_dict(ref.state_dict())
    with torch.no_grad():
        for s in _snapshots(5, with_prior):
            a, b = ref(s), fast(s)
            assert a.candidate_ids == b.candidate_ids and torch.equal(a.mask, b.mask)
            va, vb = a.logits[a.mask], b.logits[b.mask]
            assert torch.allclose(va, vb, atol=2e-5, rtol=1e-4), (method, va, vb)
            assert torch.allclose(a.value, b.value, atol=2e-5, rtol=1e-4)
            assert torch.allclose(a.q[a.mask], b.q[b.mask], atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("method", PRIOR_METHODS)
def test_empty_prior_gives_exactly_zero_residual(method):
    require_data()
    pol = _policy(method, batched=True)
    for s in _snapshots(4, with_prior=False):
        out = pol(s)
        assert all(float(abs(v)) == 0.0 for v in out.diagnostics["delta"].values())


@pytest.mark.parametrize("method", ("A_STAT", "Full", "A_CAT"))
def test_prior_present_gives_nonzero_state_dependent_residual(method):
    require_data()
    pol = _policy(method, batched=True)
    vals = []
    for s in _snapshots(6, with_prior=True):
        out = pol(s)
        vals += [float(v) for v in out.diagnostics["delta"].values()]
    assert any(abs(v) > 1e-7 for v in vals) and len({round(v, 6) for v in vals}) > 3


@pytest.mark.parametrize("method", FAST)
def test_candidate_permutation_equivariance(method):
    require_data()
    pol = _policy(method, batched=True)
    with torch.no_grad():
        for s in _snapshots(3, with_prior=True):
            n = len(s.candidate_ids)
            perm = list(reversed(range(n)))
            sp = replace(s, candidate_ids=tuple(s.candidate_ids[i] for i in perm), candidate_features=tuple(s.candidate_features[i] for i in perm), mask=tuple(s.mask[i] for i in perm))
            a, b = pol(s), pol(sp)
            assert torch.allclose(a.logits[perm][b.mask], b.logits[b.mask], atol=1e-5, rtol=1e-4)


def test_prior_methods_share_one_weighted_prior_and_prior_bias_is_scalar_controlled():
    require_data()
    from cp_disr.platforms.alfworld.pddl_contracts import candidate_key

    pol = _policy("PRIOR_BIAS", batched=True)
    assert [n for n, _ in pol.named_parameters() if n == "prior_beta"] == ["prior_beta"]
    for m in ("B2", "Full", "A_STAT", "A_CAT"):
        assert "prior_beta" not in dict(_policy(m).named_parameters())
    snaps = _snapshots(6, with_prior=True)
    for s in snaps:
        sc = pol.prior_bias_scores(s, s.mask)
        weights = {e[0]: e[3] for e in s.prior_edges}
        valid = [i for i, ok in enumerate(s.mask) if ok]
        assert all(-1 - 1e-9 <= sc[i] <= 1 + 1e-9 for i in valid)
        if weights and len(valid) > 1 and len({weights.get(s.candidate_ids[i], 0.0) for i in valid}) > 1:
            order_w = sorted(valid, key=lambda i: weights.get(s.candidate_ids[i], 0.0))
            order_s = sorted(valid, key=lambda i: sc[i])
            assert [weights.get(s.candidate_ids[i], 0.0) for i in order_w] == sorted(weights.get(s.candidate_ids[i], 0.0) for i in order_s)
    # beta = 0 => PRIOR_BIAS logits equal B2 logits built from the same parameters
    ref = _policy("B2", batched=True)
    ref_state = {k: v for k, v in ref.state_dict().items()}
    pol.load_state_dict({**{k: v for k, v in pol.state_dict().items()}, **ref_state})
    with torch.no_grad():
        for s in snaps[:4]:
            assert torch.allclose(pol(s).logits[pol(s).mask], ref(s).logits[ref(s).mask], atol=1e-5)
    # the scalar receives gradient and moves logits in prior order
    s = next(x for x in snaps if sum(x.mask) > 2)
    out = pol(s)
    out.distribution.log_prob(torch.tensor(int(out.mask.nonzero()[0]))).backward()
    assert pol.prior_beta.grad is not None


def test_shadow_policy_zeroes_only_the_prior_residual():
    require_data()
    for method in PRIOR_METHODS:
        pol = _policy(method, batched=True)
        with torch.no_grad():
            if method == "PRIOR_BIAS":
                pol.prior_beta.fill_(0.7)
            for s in _snapshots(3, with_prior=True):
                normal = pol(s)
                pol.shadow_zero_prior = True
                shadow = pol(s)
                pol.shadow_zero_prior = False
                assert normal.candidate_ids == shadow.candidate_ids and torch.equal(normal.mask, shadow.mask)
                resid = torch.stack([normal.diagnostics["delta"][c] for c, ok in zip(normal.candidate_ids, normal.mask) if ok])
                assert torch.allclose((normal.logits[normal.mask] - resid.reshape(-1)), shadow.logits[shadow.mask], atol=1e-5)
                assert all(float(v) == 0.0 for v in shadow.diagnostics["delta"].values())
                assert torch.allclose(normal.value, shadow.value)  # only the final residual changes
