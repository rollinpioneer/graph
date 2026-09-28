from dataclasses import replace

import torch

from cp_disr.common import digest, seed32
from cp_disr.contracts import Atom, Effects
from cp_disr.graph import build_template
from cp_disr.neural import Policy, canonical_method
from cp_disr.rl import Rollout, Transition
from cp_disr.torch_rl import PPO


def make_policy(fixture, method):
    template = fixture["template"]
    torch.manual_seed(seed32("s0_final_variants", method, 0, 0))
    torch.set_num_threads(2)
    return Policy(
        {n.schema for n in template.nodes if n.kind == "ACTION"},
        {n.schema for n in template.nodes if n.kind == "PROPOSITION"},
        {x for n in template.nodes for x in n.argument_types},
        4,
        3,
        method=method,
        B=0.5,
    )


def test_s0_methods_are_registered():
    assert canonical_method("B1-K+E") == "B1-K+E"
    assert canonical_method("A_STAT") == "A_STAT"


def test_b1_k_plus_e_reads_effect_tokens_without_successor(snap, fixture):
    out = make_policy(fixture, "B1-K+E")(snap)
    assert out.diagnostics["method"] == "B1-K+E"
    assert out.diagnostics["successor_used"] is False
    assert all(count > 0 for count in out.diagnostics["effect_tokens"].values())
    assert all(torch.count_nonzero(value).item() == 0 for value in out.diagnostics["prior_inputs"].values())


def test_a_stat_exact_r_null_matches_full_with_shared_weights(snap, fixture):
    empty = replace(snap, prior_edges=(), prior_hash=digest(()))
    full = make_policy(fixture, "Full")
    stat = make_policy(fixture, "A_STAT")
    stat.load_state_dict(full.state_dict())
    full_out = full(empty)
    stat_out = stat(empty)
    assert torch.equal(full_out.logits, stat_out.logits)
    assert torch.equal(full_out.value, stat_out.value)
    assert torch.equal(full_out.q, stat_out.q)
    assert all(torch.count_nonzero(value).item() == 0 for value in stat_out.diagnostics["stat_relation"].values())


def test_a_stat_uses_only_static_relation_difference_and_respects_bound(snap, fixture):
    assert snap.prior_edges
    out = make_policy(fixture, "A_STAT")(snap)
    for cid, delta in out.diagnostics["differences"].items():
        expected = delta.zh.float() - delta.zk.float()
        actual = out.diagnostics["prior_inputs"][cid]
        assert torch.allclose(actual, expected)
        assert float(out.diagnostics["delta"][cid].abs().max()) <= 0.5 + 1e-6


def test_a_stat_candidate_reorder_equivariance(snap, fixture):
    policy = make_policy(fixture, "A_STAT")
    original = policy(snap)
    order = tuple(reversed(range(len(snap.candidate_ids))))
    reordered = replace(
        snap,
        candidate_ids=tuple(snap.candidate_ids[i] for i in order),
        candidate_features=tuple(snap.candidate_features[i] for i in order),
        mask=tuple(snap.mask[i] for i in order),
    )
    permuted = policy(reordered)
    by_id = {cid: i for i, cid in enumerate(permuted.candidate_ids)}
    for i, cid in enumerate(original.candidate_ids):
        assert torch.allclose(original.logits[i], permuted.logits[by_id[cid]])
        assert torch.allclose(original.q[i], permuted.q[by_id[cid]])
    assert torch.allclose(original.value, permuted.value)


def test_production_clip_logging_records_pre_and_post_norm(snap, fixture):
    policy = make_policy(fixture, "Full")
    initial = policy(snap)
    next_snapshot = replace(snap, decision_id=snap.decision_id + 1)
    transition = Transition(
        snap,
        next_snapshot,
        snap.candidate_ids[0],
        float(initial.distribution.log_prob(torch.tensor(0))),
        float(initial.value),
        0.0,
        0.3,
        1.0,
        1.0,
        False,
        False,
        "synthetic_unit_fixture",
        (),
    )
    rollout = Rollout()
    rollout.append(transition)
    logs = PPO(policy).update(rollout, epochs=1, minibatch=1)
    row = logs[0]
    assert row["optimizer_step_id"] == 1
    assert row["clip_threshold"] == 0.5
    assert row["pre_clip_global_grad_norm"] >= 0.0
    assert row["post_clip_global_grad_norm"] >= 0.0
    assert row["post_clip_global_grad_norm"] <= row["pre_clip_global_grad_norm"] + 1e-6
    assert row["clip_triggered"] == (row["pre_clip_global_grad_norm"] > 0.5)


def test_full_empty_patch_null_uses_production_four_views(snap, fixture):
    old = next(c for c in snap.template.contracts if c.name == "MOVE")
    empty_contract = replace(old, pre_pos=(), pre_neg=(), effects=Effects())
    contracts = tuple(empty_contract if c.id == old.id else c for c in snap.template.contracts)
    template = build_template(
        contracts,
        snap.template.goals,
        fixture["registry"].predicate_types,
        fixture["objects"],
        extra_atoms=tuple(
            Atom(parts[1], tuple(parts[2:]))
            for record in snap.facts.records
            for parts in [record.fact_id.split(":")]
        ),
    )
    empty = replace(snap, template=template)
    out = make_policy(fixture, "Full")(empty)
    delta = out.diagnostics["differences"][old.id]
    assert torch.count_nonzero(delta.dk).item() == 0
    assert torch.count_nonzero(delta.dp).item() == 0
    assert torch.count_nonzero(out.diagnostics["up"][old.id]).item() == 0
    assert torch.count_nonzero(out.diagnostics["delta"][old.id]).item() == 0


def test_a_stat_and_a_cat_shared_null_bound_and_permission_isolation(snap, fixture):
    import torch
    facts_before = snap.facts.values
    mask_before = snap.mask
    for method in ("A_STAT", "A_CAT"):
        policy = make_policy(fixture, method)
        empty = replace(snap, prior_edges=(), prior_hash=digest(()))
        null_out = policy(empty)
        assert all(torch.count_nonzero(value).item() == 0 for value in null_out.diagnostics["up"].values())
        assert all(torch.count_nonzero(value).item() == 0 for value in null_out.diagnostics["delta"].values())
        out = policy(snap)
        assert all(float(value.abs().max()) <= 0.5 + 1e-6 for value in out.diagnostics["delta"].values())
        assert snap.facts.values == facts_before and snap.mask == mask_before

def test_full_delta_is_bounded_on_nonempty_prior(snap, fixture):
    out = make_policy(fixture, "Full")(snap)
    assert all(
        float(value.abs().max()) <= 0.5 + 1e-6
        for value in out.diagnostics["delta"].values()
    )
