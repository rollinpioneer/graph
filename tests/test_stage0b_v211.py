"""Stage 0B v2.1.1 production invariants: B1-K, A_CAT, Set permutation, unweighted actor."""
from dataclasses import replace
import math
import pytest
from cp_disr.common import digest
from cp_disr.rl import Transition, Rollout, gamma
from cp_disr.neural import Policy, canonical_method

@pytest.mark.torch_runtime
def test_b1k_does_not_compute_successor(snap, policy):
    policy.method = "B1"
    before = policy.encoder.forward_calls
    out = policy(snap)
    assert canonical_method(policy.method) == "B1"
    assert out.diagnostics["successor_used"] is False
    assert out.diagnostics["differences"] == {}
    # current G_K only (one encoder call), no nominal successor / GH
    assert policy.encoder.forward_calls == before + 1
    assert all(int(v.abs().sum().item()) == 0 for v in out.diagnostics["prior_inputs"].values())

@pytest.mark.torch_runtime
def test_acat_null_prior_is_exact_zero(snap, fixture):
    import torch
    t = fixture["template"]
    cat = Policy({n.schema for n in t.nodes if n.kind == "ACTION"}, {n.schema for n in t.nodes if n.kind == "PROPOSITION"}, {x for n in t.nodes for x in n.argument_types}, 4, 3, method="A_CAT")
    empty = replace(snap, prior_edges=(), prior_hash=digest(()))
    out = cat(empty)
    for cid in out.diagnostics["delta"]:
        assert torch.count_nonzero(out.diagnostics["delta"][cid]) == 0
        assert torch.count_nonzero(out.diagnostics["up"][cid]) == 0

@pytest.mark.torch_runtime
def test_b0_candidate_permutation(snap, fixture):
    import torch
    t = fixture["template"]
    b0 = Policy({n.schema for n in t.nodes if n.kind == "ACTION"}, {n.schema for n in t.nodes if n.kind == "PROPOSITION"}, {x for n in t.nodes for x in n.argument_types}, 4, 3, method="B0")
    a = b0(snap)
    perm = tuple(reversed(range(len(snap.candidate_ids))))
    b = b0(replace(snap, candidate_ids=tuple(snap.candidate_ids[i] for i in perm), candidate_features=tuple(snap.candidate_features[i] for i in perm), mask=tuple(snap.mask[i] for i in perm)))
    assert torch.allclose(a.logits[list(perm)], b.logits, atol=1e-5, rtol=1e-4)

@pytest.mark.torch_runtime
def test_actor_ignores_episode_prefix_weights(snap, policy):
    import torch
    from cp_disr.torch_rl import ppo_losses
    with torch.no_grad():
        out = policy(snap)
    idx = 0
    lp = out.distribution.log_prob(torch.tensor(idx))
    adv = torch.tensor([0.5])
    v = out.value.unsqueeze(0)
    q = out.q[idx].unsqueeze(0)
    ent = out.distribution.entropy().unsqueeze(0)
    a1 = ppo_losses(lp.unsqueeze(0), lp.detach().unsqueeze(0), adv, torch.tensor([1.0]), v, v.detach(), q, q.detach(), ent, 0.1, False)
    a2 = ppo_losses(lp.unsqueeze(0), lp.detach().unsqueeze(0), adv, torch.tensor([0.01]), v, v.detach(), q, q.detach(), ent, 0.1, False)
    assert torch.allclose(a1["actor"], a2["actor"], atol=1e-8, rtol=1e-7)
    w1 = ppo_losses(lp.unsqueeze(0), lp.detach().unsqueeze(0), adv, torch.tensor([1.0]), v, v.detach(), q, q.detach(), ent, 0.1, True)
    w2 = ppo_losses(lp.unsqueeze(0), lp.detach().unsqueeze(0), adv, torch.tensor([0.01]), v, v.detach(), q, q.detach(), ent, 0.1, True)
    # old weighted path would still be identical for a single sample; two-sample check:
    lp2 = torch.stack([lp, lp])
    adv2 = torch.tensor([0.5, -0.25])
    v2 = torch.stack([out.value, out.value])
    q2 = torch.stack([out.q[idx], out.q[idx]])
    ent2 = torch.stack([out.distribution.entropy(), out.distribution.entropy()])
    u = ppo_losses(lp2, lp2.detach(), adv2, torch.tensor([1.0, 0.01]), v2, v2.detach(), q2, q2.detach(), ent2, 0.1, False)
    w = ppo_losses(lp2, lp2.detach(), adv2, torch.tensor([1.0, 0.01]), v2, v2.detach(), q2, q2.detach(), ent2, 0.1, True)
    assert not torch.allclose(u["actor"], w["actor"], atol=1e-6, rtol=1e-5)

@pytest.mark.pure
def test_gamma_units_are_seconds():
    assert gamma(1, H=1) == pytest.approx(0.5)
    # 1 tick is not 1 second; callers must convert ticks to seconds before gamma.
    assert gamma(0.002, H=1) == pytest.approx(2 ** (-0.002))
