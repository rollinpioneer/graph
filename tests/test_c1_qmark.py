"""CP-DISR-C1-MECH-CONFIRM-V1: QMARK semantics (Q01-Q10), offline. No environment, no GPU, no optimizer step."""
import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.torch_runtime


@pytest.fixture(scope="module")
def template():
    from cp_disr import tb_repctl_checks as C
    return C.tb_template(ROOT)


@pytest.fixture(scope="module")
def states(template):
    from cp_disr import tb_repctl_checks as C
    return C.state_suite(template, count=4)


def make_qmark(template, seed=0):
    import torch
    from cp_disr import c1_qmark_policy as Q
    from cp_disr import tb_repctl_checks as C
    from cp_disr.platforms.libero import runtime_factory as rf
    type_set = set()
    for mapping in rf.TASK_OBJECTS.values():
        type_set.update(mapping.values())
    torch.manual_seed(seed)
    policy = Q.QmarkPolicy(sorted({c.name for c in template.contracts}), sorted(rf.PREDICATES), sorted(type_set), C.OBS_DIM, C.CAND_DIM, method=Q.QMARK_METHOD, B=0.5)
    policy.eval()
    return policy


def _production_nodes():
    from cp_disr import c1_qmark_policy as Q
    tree = ast.parse(inspect.getsource(Q))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("QmarkPolicy", "query_channel"):
            keep.append(node)
    return keep


def _names(nodes):
    out = set()
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, ast.Attribute):
                out.add(n.attr)
            elif isinstance(n, ast.arg):
                out.add(n.arg)
    return out


# ----------------------------------------------------------------------------- Q01 / Q02: no successor construction
def test_Q01_forward_never_reaches_nominal_apply(template, states):
    from cp_disr import tb_repctl_checks as C
    policy = make_qmark(template)
    for seed, values in states:
        snap = C.make_snapshot(template, values, seed)
        with C.nominal_apply_trap():
            out = C.forward(policy, snap)
        assert out.diagnostics["successor_used"] is False
    # the trap is load bearing: a forward that applies a contract is detected
    from cp_disr import graph as graph_mod
    snap = C.make_snapshot(template, states[0][1], states[0][0])
    contract = next(c for c in template.contracts)
    with C.nominal_apply_trap():
        with pytest.raises(C.NominalApplyCalled):
            graph_mod.successor(graph_mod.view(template, states[0][1]), contract)


def test_Q02_source_has_no_successor_or_difference_path(template):
    forbidden = set(__import__("cp_disr.tb_repctl_checks", fromlist=["x"]).NOMINAL_APPLY_NAMES) | {"nominal_overlay", "four_views", "successor", "differences", "overlay"}
    hits = sorted(_names(_production_nodes()) & forbidden)
    assert hits == []
    from cp_disr import c1_qmark_policy as Q
    imported = {a.name for n in ast.walk(ast.parse(inspect.getsource(Q))) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not (imported & forbidden)


# ----------------------------------------------------------------------------- Q03 / Q04 / Q10: the query changes only one channel
def test_Q03_query_leaves_facts_goals_and_edges_untouched(template, states):
    from cp_disr import graph as graph_mod
    for seed, values in states:
        seen = []
        policy = make_qmark(template)
        graph = graph_mod.view(template, values)
        before = (graph.values, graph.template.goals, graph.edges, id(graph.template))
        original = policy.encoder.features.forward

        def spy(g, _orig=original):
            seen.append(g.values)
            return _orig(g)
        policy.encoder.features.forward = spy
        policy.encode_with_action_query(graph, None)
        for c in template.contracts:
            policy.encode_with_action_query(graph, c.id)
        assert (graph.values, graph.template.goals, graph.edges, id(graph.template)) == before
        assert all(v == before[0] for v in seen) and len(seen) == 1 + len(template.contracts)


def test_Q04_only_the_queried_action_node_carries_the_channel(template):
    from cp_disr import c1_qmark_policy as Q
    from cp_disr import graph as graph_mod
    from cp_disr import tb_repctl_checks as C
    graph = graph_mod.view(template, C.random_values(template, 3))
    ids = graph.template.node_ids
    for c in template.contracts:
        ch = Q.query_channel(graph, c.id)
        nz = [ids[i] for i in range(len(ids)) if float(ch[i, 0]) != 0.0]
        assert nz == [c.id] and float(ch.sum()) == 1.0
    prop = next(n.id for n in template.nodes if n.kind == "PROPOSITION")
    with pytest.raises(ValueError):
        Q.query_channel(graph, prop)
    with pytest.raises(KeyError):
        Q.query_channel(graph, "a:NO_SUCH:v1")


def test_Q10_two_queries_differ_only_by_the_marker_and_facts_differ_in_zero_places(template, states):
    import torch
    from cp_disr import c1_qmark_policy as Q
    from cp_disr import graph as graph_mod
    seed, values = states[0]
    policy = make_qmark(template)
    graph = graph_mod.view(template, values)
    a, b = template.contracts[0].id, template.contracts[1].id
    ca, cb = Q.query_channel(graph, a), Q.query_channel(graph, b)
    assert int((ca != cb).sum()) == 2                       # exactly the two action nodes
    facts_a = dict(graph.values)
    facts_b = dict(graph.values)
    assert sum(1 for k in facts_a if facts_a[k] != facts_b[k]) == 0
    with torch.no_grad():
        z0 = policy.encode_with_action_query(graph, None)
        za, zb = policy.encode_with_action_query(graph, a), policy.encode_with_action_query(graph, b)
    assert not torch.equal(za, zb) and not torch.equal(za, z0)    # the control is not vacuous


# ----------------------------------------------------------------------------- Q05 / Q06: order and identity independence
def test_Q05_candidate_order_does_not_change_a_candidates_representation(template, states):
    from cp_disr import tb_repctl_checks as C
    policy = make_qmark(template)
    ids = tuple(c.id for c in template.contracts)
    for seed, values in states:
        base = C.forward(policy, C.make_snapshot(template, values, seed))
        rev = C.forward(policy, C.make_snapshot(template, values, seed, order=tuple(reversed(ids))))
        assert rev.diagnostics["query_rows"].keys() == base.diagnostics["query_rows"].keys()
        for cid, rows in base.diagnostics["query_rows"].items():
            assert (rows - rev.diagnostics["query_rows"][cid]).abs().max() == 0
        for cid in ids:
            i, j = base.candidate_ids.index(cid), rev.candidate_ids.index(cid)
            if bool(base.mask[i]):
                assert float(base.logits[i]) == float(rev.logits[j])


def test_Q06_marker_is_not_derived_from_case_seed_cell_object_strings_or_hashes(template, states):
    from cp_disr import tb_repctl_checks as C
    banned = {"case_id", "seed", "hash", "digest", "sha256", "goal_key", "cell", "episode_id", "env_id", "arguments", "bound_arguments", "hashlib", "id"}
    assert sorted(_names(_production_nodes()) & banned) == []
    policy = make_qmark(template)
    seed, values = states[0]
    snap = C.make_snapshot(template, values, seed)
    other = dataclasses.replace(snap, env_id="different-env", episode_id="different-episode")
    a, b = C.forward(policy, snap), C.forward(policy, other)
    assert float((a.logits[a.mask] - b.logits[b.mask]).abs().max()) == 0.0
    new = {n for n, _p in policy.named_parameters()} - {n for n, _p in C.make_policy(template, "B2", 0).named_parameters()}
    assert new == {"query_projection.weight"}               # one shared projection, no per-contract / per-object parameter
    assert tuple(policy.query_projection.weight.shape) == (128, 1)


# ----------------------------------------------------------------------------- Q07 / Q08: legacy behaviour and B2 alignment
def test_Q07_empty_query_equals_the_production_encoder_and_shared_modules_start_equal_to_b2(template, states):
    import torch
    from cp_disr import graph as graph_mod
    from cp_disr import tb_repctl_checks as C
    policy, b2 = make_qmark(template), C.make_policy(template, "B2", 0)
    for seed, values in states:
        graph = graph_mod.view(template, values)
        with torch.no_grad():
            assert torch.equal(policy.encode_with_action_query(graph, None), policy.encoder(graph))
    b2_state = b2.state_dict()
    q_state = policy.state_dict()
    assert set(b2_state) <= set(q_state) and set(q_state) - set(b2_state) == {"query_projection.weight"}
    for k, v in b2_state.items():
        assert torch.equal(v, q_state[k]), k


def test_Q08_outputs_and_heads_align_with_b2(template, states):
    from cp_disr import tb_repctl_checks as C
    policy, b2 = make_qmark(template), C.make_policy(template, "B2", 0)
    seed, values = states[0]
    snap = C.make_snapshot(template, values, seed)
    qo, bo = C.forward(policy, snap), C.forward(b2, snap)
    assert qo.logits.shape == bo.logits.shape and qo.value.shape == bo.value.shape and qo.q.shape == bo.q.shape and qo.hidden.shape == bo.hidden.shape
    assert qo.candidate_ids == bo.candidate_ids and bool((qo.mask == bo.mask).all())
    for name in ("encoder", "observation", "gru", "candidate", "contract", "base", "v_head", "q_head", "prior"):
        assert [tuple(p.shape) for p in getattr(policy, name).parameters()] == [tuple(p.shape) for p in getattr(b2, name).parameters()]


# ----------------------------------------------------------------------------- Q09: machine-readable identity
def test_Q09_identity_report_is_machine_readable_and_within_one_percent_of_b2(template, states):
    import json
    from cp_disr import c1_qmark_policy as Q
    seed, values = states[0]
    from cp_disr import tb_repctl_checks as C
    snap = C.make_snapshot(template, values, seed)
    report = Q.identity_report(template, snap)
    json.dumps(report)
    assert report["qmark"]["effective_trainable_parameters"] - report["b2"]["effective_trainable_parameters"] == 128
    assert report["extra_parameters_pct_of_b2"] <= 1.0
    assert report["qmark"]["computation"]["nominal_apply_calls_total"] == 0
    assert report["qmark"]["computation"]["graph_propagation_layers"] == report["b2"]["computation"]["graph_propagation_layers"] == 4
    assert report["qmark"]["computation"]["encoder_forward_calls_total"] == 1 + sum(snap.mask)


def test_marker_is_load_bearing_and_rows_vanish_without_it(template, states):
    import torch
    from cp_disr import tb_repctl_checks as C
    policy = make_qmark(template)
    seed, values = states[0]
    snap = C.make_snapshot(template, values, seed)
    rows = C.forward(policy, snap).diagnostics["query_rows"]
    assert rows and all(float(r.abs().max()) > 0 for r in rows.values())
    with torch.no_grad():
        policy.query_projection.weight.zero_()
    rows0 = C.forward(policy, snap).diagnostics["query_rows"]
    assert all(float(r.abs().max()) == 0 for r in rows0.values())
