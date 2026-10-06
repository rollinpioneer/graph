"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: B2-CACHED / QMARK / ASNET-READOUT semantics, equivalence and leakage tests (runbook 17.4 - 17.7). Offline, CPU."""
import random

import pytest
import torch

pytestmark = pytest.mark.torch_runtime


@pytest.fixture(scope="module")
def snaps():
    from cp_disr.blocksworld import generator as G
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld.environment import BwEpisode, Case
    from cp_disr.rl import set_suite_half_life
    set_suite_half_life(6.0)
    rng = random.Random(4)
    out = []
    for n, heights in ((3, (2, 1)), (4, (3, 1)), (5, (3, 1, 1)), (6, (3, 2, 1))):
        for k in range(3):
            p = G.make_problem("t", n, k, heights, S.RED, 0)
            if p is None:
                continue
            ep = BwEpisode(Case("t%d" % k, "train", p, 6, 16))
            for _ in range(rng.randint(0, 3)):
                ep.step(rng.choice(sorted(ep.legal_ids())))
            out.append((p, ep.snapshot()))
    return out


def _models():
    from cp_disr import c1_blocksworld_policies as B
    ms = {m: B.make_policy(m, "cpu", 0) for m in B.METHODS}
    for m in ms.values():
        m.eval()
    return B, ms


def _ref_b2(b2):
    from cp_disr import neural
    from cp_disr.blocksworld import contracts as C
    ref = neural.Policy(sorted(C.SCHEMA_NAMES), list(C.PREDICATES), list(C.TYPES), 48, 8, method="B2", B=0.5)
    ref.load_state_dict(b2.state_dict())
    return ref.eval()


def test_b2_cached_forward_and_gradients_equal_the_original_b2(snaps):
    B, ms = _models()
    ref, mine = _ref_b2(ms[B.B2_CACHED]), ms[B.B2_CACHED]
    for _p, s in snaps[:5]:
        with torch.no_grad():
            o1, o2 = ref(s, None), mine(s, None)
        mk = o1.mask
        assert float((o1.logits[mk] - o2.logits[mk]).abs().max()) < 1e-5 and float((o1.value - o2.value).abs()) < 1e-5 and float((o1.q - o2.q).abs().max()) < 1e-5
    s = snaps[2][1]
    for m in (ref, mine):
        m.zero_grad()
        o = m(s, None)
        (o.logits[o.mask].sum() + o.value + o.q[o.mask].sum()).backward()
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), mine.named_parameters()):
        g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
        g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
        assert float((g1 - g2).abs().max()) < 1e-5, n1


def test_b2_cached_counts_encodings_and_nominal_applies(snaps):
    B, ms = _models()
    m = ms[B.B2_CACHED]
    for _p, s in snaps[:4]:
        with torch.no_grad():
            out = m(s, None)
        K = int(sum(s.mask))
        assert out.diagnostics["encoder_graph_encodings"] == 1 + K and out.diagnostics["nominal_apply_calls"] == K


def test_qmark_matches_the_frozen_unbatched_qmark_and_never_applies_a_contract(snaps):
    from cp_disr import tb_repctl_checks as T
    B, ms = _models()
    m = ms[B.QMARK_BW]
    for _p, s in snaps[:5]:
        with torch.no_grad(), T.nominal_apply_trap():
            o2 = m(s, None)
        with torch.no_grad():
            o1 = m.reference_forward(s, None)
        mk = o1.mask
        assert float((o1.logits[mk] - o2.logits[mk]).abs().max()) < 1e-5 and float((o1.value - o2.value).abs()) < 1e-5
        assert o2.diagnostics["encoder_graph_encodings"] == 1 + int(sum(s.mask)) and o2.diagnostics["nominal_apply_calls"] == 0 and o2.diagnostics["successor_used"] is False


def test_qmark_query_changes_only_the_queried_action_node_and_shares_one_projection(snaps):
    from cp_disr import c1_qmark_policy as Q
    from cp_disr import graph as graph_mod
    B, ms = _models()
    m = ms[B.QMARK_BW]
    p, s = snaps[2]
    graph = graph_mod.view(s.template, s.facts.values)
    before = graph.values
    ids = s.template.node_ids
    legal = [c for c, ok in zip(s.candidate_ids, s.mask) if ok]
    for cid in legal[:3]:
        ch = Q.query_channel(graph, cid)
        assert [ids[i] for i in range(len(ids)) if float(ch[i, 0]) != 0.0] == [cid]
    assert graph.values == before
    new = {n for n, _p in m.named_parameters()} - {n for n, _p in ms[B.B2_CACHED].named_parameters()}
    assert new == {"query_projection.weight"} and tuple(m.query_projection.weight.shape) == (128, 1)


def test_asnet_readout_has_no_marker_no_successor_one_encoding_and_action_node_rows(snaps):
    from cp_disr import tb_repctl_checks as T
    B, ms = _models()
    m = ms[B.ASNET]
    assert not hasattr(m, "query_projection")
    assert {n for n, _p in m.named_parameters()} == {n for n, _p in ms[B.B2_CACHED].named_parameters()}
    for _p, s in snaps[:4]:
        with torch.no_grad(), T.nominal_apply_trap():
            out = m(s, None)
        assert out.diagnostics["encoder_graph_encodings"] == 1 and out.diagnostics["nominal_apply_calls"] == 0
    # the candidate row IS the ACTION node's final hidden state: recompute it with the production encoder pieces
    p, s = snaps[1]
    st = m._static(s.template)
    codes = m._codes(st, s.facts.values).unsqueeze(0)
    with torch.no_grad():
        _z, H = m._encode(st, codes)
    legal = [i for i, ok in enumerate(s.mask) if ok]
    assert H.shape[1] == st.N and len(legal) > 0 and all(int(st.action_pos[st.action_row[s.candidate_ids[i]]]) < st.N for i in legal)


def test_policies_do_not_read_planner_labels_or_hidden_state():
    import ast
    import inspect
    from cp_disr import c1_blocksworld_policies as B
    tree = ast.parse(inspect.getsource(B))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    forbidden = {"Solver", "cost_to_go", "optimal_actions", "one_optimal_plan", "hidden_truth", "goal_iso_hash", "problem_iso_hash", "case_id", "split", "nominal_overlay", "four_views", "successor"}
    assert not (names & forbidden), names & forbidden


def test_candidate_features_carry_no_block_identity():
    from cp_disr import c1_blocksworld_policies as B
    assert B.candidate_feature("PICK_UP") == (1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    assert B.candidate_feature("STACK") == (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert len({B.candidate_feature(s) for s in ("PICK_UP", "PUT_DOWN", "UNSTACK", "STACK")}) == 4


@pytest.mark.parametrize("method", ["B2-CACHED", "B1-K+QMARK-BW", "ASNET-READOUT"])
def test_relabelling_blocks_permutes_the_policy_output_and_a_name_leak_breaks_it(snaps, method):
    """Colour-preserving renaming + reindexing is a pure permutation of the problem: every method's logits must follow it (names / ids / order carry no information)."""
    from cp_disr import c1_blocksworld_policies as B
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld.environment import BwEpisode, Case
    _B, ms = _models()
    m = ms[method]
    p, s = snaps[2]
    rng = random.Random(9)
    # reds and blues may swap among themselves only
    label = list(range(p.n))
    for color in (S.RED, S.BLUE):
        idx = [i for i in range(p.n) if p.colors[i] == color]
        shuf = idx[:]
        rng.shuffle(shuf)
        for a, b in zip(idx, shuf):
            label[a] = b
    names = tuple("z%d" % (7 - i) for i in range(p.n))
    q = K.rename(p, names, label)
    case = Case("q", "train", q, 6, 16)
    ep = BwEpisode(case)
    # bring q to the image of the state of s (same step index, so the recurrent input is identical)
    ep.state = K._relabel(_state_of(p, s), label)
    ep.step_index, ep.previous_ok = s.decision_id, s.decision_id > 0
    sq = ep.snapshot()
    with torch.no_grad():
        o1, o2 = m(s, None), m(sq, None)
    id1 = {cid: float(o1.logits[i]) for i, cid in enumerate(s.candidate_ids) if s.mask[i]}
    id2 = {cid: float(o2.logits[i]) for i, cid in enumerate(sq.candidate_ids) if sq.mask[i]}
    mapped = {}
    for cid, v in id1.items():
        parts = cid.split(":")                                   # a:SCHEMA:b0[:b1]:v1
        blocks = [p.names.index(x) for x in parts[2:-1]]
        new = "a:%s:%s:v1" % (parts[1], ":".join(q.names[label[b]] for b in blocks))
        mapped[new] = v
    assert set(mapped) == set(id2)
    assert max(abs(mapped[k] - id2[k]) for k in mapped) < 1e-4


def _state_of(problem, snap):
    """Recover the block-state tuple of a snapshot from its facts."""
    from cp_disr.blocksworld import state as S
    f = {k: str(v.value) if hasattr(v, "value") else str(v) for k, v in snap.facts.values.items()}
    st = []
    for x in range(problem.n):
        if f[S.atom_id(problem.names, "OnTable", x)] == "TRUE":
            st.append(S.TABLE)
        elif f[S.atom_id(problem.names, "Holding", x)] == "TRUE":
            st.append(S.HELD)
        else:
            st.append(next(y for y in range(problem.n) if y != x and f[S.atom_id(problem.names, "On", x, y)] == "TRUE"))
    return tuple(st)


def test_a_policy_that_reads_a_name_hash_would_fail_the_relabelling_test(snaps):
    """Mutant: add a hash of the grounded action id to the candidate context; the permutation property must then break."""
    import hashlib
    from cp_disr import c1_blocksworld_policies as B
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld.environment import BwEpisode, Case
    _B, ms = _models()
    m = ms[B.B2_CACHED]
    original = B.BwMixin._contexts

    def leaky(self, snapshot, zo, legal, zk_mean):
        ctx = original(self, snapshot, zo, legal, zk_mean)
        bump = torch.tensor([int(hashlib.sha256(snapshot.candidate_ids[i].encode()).hexdigest()[:6], 16) / 16777216.0 for i in legal], dtype=ctx.dtype)
        return ctx + bump.unsqueeze(-1)
    p, s = snaps[2]
    names = tuple("z%d" % (7 - i) for i in range(p.n))
    q = K.rename(p, names)
    ep = BwEpisode(Case("q", "train", q, 6, 16))
    ep.state = _state_of(p, s)
    ep.step_index, ep.previous_ok = s.decision_id, s.decision_id > 0
    sq = ep.snapshot()
    B.BwMixin._contexts = leaky
    try:
        with torch.no_grad():
            a, b = m(s, None), m(sq, None)
    finally:
        B.BwMixin._contexts = original
    la = sorted(float(x) for x in a.logits[a.mask])
    lb = sorted(float(x) for x in b.logits[b.mask])
    assert max(abs(x - y) for x, y in zip(la, lb)) > 1e-4


def test_batched_ppo_outputs_equal_the_per_snapshot_forward(snaps):
    from cp_disr.c1_blocksworld_policies import batched_outputs
    B, ms = _models()
    for method, m in ms.items():
        group = [s for _p, s in snaps[:6]]
        selected = [next(c for c, ok in zip(s.candidate_ids, s.mask) if ok) for s in group]
        zos = torch.stack([m.advance_hidden(s.base_input, m.initial_hidden()) for s in group])
        lp, vs, qs, ent = batched_outputs(m, m.row_kind, group, zos, selected)
        for i, s in enumerate(group):
            o = m(s, m.initial_hidden())
            sel = s.candidate_ids.index(selected[i])
            assert abs(float(o.distribution.log_prob(torch.tensor(sel))) - float(lp[i])) < 1e-5, method
            assert abs(float(o.value) - float(vs[i])) < 1e-5 and abs(float(o.q[sel]) - float(qs[i])) < 1e-5 and abs(float(o.distribution.entropy()) - float(ent[i])) < 1e-5
