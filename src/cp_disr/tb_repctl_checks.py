"""CP-DISR-TB-REP-CONTROLS-01 semantic checks, mutants and capacity instrumentation.

Used by tests/test_tb_repctl_semantics.py (pytest) and scripts/tb_repctl_receipts.py (receipts). Everything here is
offline: no environment, no GPU, no provider, no optimizer step, no test split. The real T_B template (7 grounded
candidates, 17 nodes) is built from the registered contract registry; states are synthetic three-valued assignments.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import inspect
import pathlib
import pickle
import random
import textwrap
from dataclasses import replace
from pathlib import Path

import torch

from . import contracts as contracts_mod
from . import graph as graph_mod
from . import neural
from . import repctl_policy as rp
from .common import ContractError, digest
from .facts import FactRecord, FactStore, Truth
from .rl import Snapshot

BASE_COMMIT = "9422b837cf1dfc43afc056a77bdb59d63512b3b6"
OBS_DIM = 48
CAND_DIM = 8
CONTROL_METHODS = ("B2-ABS", "B1-K+NC")
REFERENCE_METHODS = ("B1-K", "B1-K+E", "B2")
ALL_METHODS = REFERENCE_METHODS + CONTROL_METHODS
CONTRACT_REGISTRY_REL = Path("configs/runtime/stage_2a_contract_registry.yaml")
RUNTIME_REL = Path("experiments/manifests/runtime_manifest_v211.yaml")
# Names that may never appear in the Matched Neural Composition production path (N1).
NOMINAL_APPLY_NAMES = frozenset({
    "nominal_overlay", "nominal_apply", "successor", "four_views", "differences", "ReadOnlyOverlay", "derive",
    "precondition_value",
})
SIDECAR_TOKENS = ("sidecar", "successor_cache", "f_after", "after_cache", "torch.load", "pickle", "open(", "read_text", "read_bytes")


class NominalApplyCalled(AssertionError):
    pass


class FileReadAttempt(AssertionError):
    pass


# ----------------------------------------------------------------------------------------------- fixtures
def tb_template(root):
    import yaml
    from .platforms.libero import runtime_factory as rf
    root = Path(root)
    runtime = yaml.safe_load((root / RUNTIME_REL).read_text(encoding="utf-8"))["runtime"]
    objects = rf.TASK_OBJECTS["T_B"]
    contracts = rf._ground_contracts_for(root / CONTRACT_REGISTRY_REL, runtime["skill_timeouts"]["T_B"], objects)
    goals = tuple(graph_mod.Goal(g, 1) for g in rf.TASK_GOALS["T_B"])
    return graph_mod.build_template(contracts, goals, rf.PREDICATES, objects)


def make_policy(template, method, seed=0):
    from .platforms.libero import runtime_factory as rf
    type_set = set()
    for mapping in rf.TASK_OBJECTS.values():
        type_set.update(mapping.values())
    torch.manual_seed(seed)
    policy = rp.policy_class(method)(sorted({c.name for c in template.contracts}), sorted(rf.PREDICATES), sorted(type_set),
                                     OBS_DIM, CAND_DIM, method=method, B=0.5)
    policy.eval()
    return policy


def _allowed(contract, template, values):
    if contracts_mod.precondition_value(contract, values) != Truth.TRUE:
        return False
    try:
        contracts_mod.nominal_overlay(contract, values, template.derived_rules, template.exclusive_groups)
    except ContractError:
        return False
    return True


def proposition_ids(template):
    return tuple(n.id for n in template.nodes if n.kind == "PROPOSITION")


def random_values(template, seed):
    rng = random.Random(seed)
    return {fid: rng.choices((Truth.TRUE, Truth.FALSE, Truth.UNKNOWN), weights=(5, 3.5, 1.5))[0] for fid in proposition_ids(template)}


def make_snapshot(template, values, seed=0, order=None):
    values = dict(values)
    ids = tuple(c.id for c in template.contracts)
    contracts = {c.id: c for c in template.contracts}
    rng = random.Random(10_000 + seed)
    base = tuple(rng.uniform(-1, 1) for _ in range(OBS_DIM))
    feats = {cid: tuple(int.from_bytes(hashlib.sha256(("%s|%d" % (cid, j)).encode()).digest()[:4], "big") / 2 ** 32 for j in range(CAND_DIM)) for cid in ids}
    mask = {cid: _allowed(contracts[cid], template, values) for cid in ids}
    order = tuple(ids if order is None else order)
    facts = FactStore(tuple(FactRecord(k, Truth(v), 0.0, 0.0, ("synthetic",), Truth.TRUE, 0.0, "SYNTHETIC", 0.9) for k, v in values.items()))
    return Snapshot("env0", "ep0", 0, template, facts, order, tuple(mask[c] for c in order), (), digest(()), base,
                    tuple(feats[c] for c in order), "synthetic", 0.0, synthetic_unit_fixture=True)


def state_suite(template, count=6, min_allowed=3, start=0):
    """Deterministic synthetic states with at least `min_allowed` legal candidates."""
    out, seed = [], start
    contracts = list(template.contracts)
    while len(out) < count:
        values = random_values(template, seed)
        if sum(_allowed(c, template, values) for c in contracts) >= min_allowed:
            out.append((seed, values))
        seed += 1
        if seed > start + 5000:
            raise RuntimeError("could not build the synthetic state suite")
    return out


def reference_successor(contract, template, values):
    """Independent F_after: the unpatched contracts.nominal_overlay."""
    overlay = contracts_mod.nominal_overlay(contract, dict(values), template.derived_rules, template.exclusive_groups)
    return dict(overlay.items())


# ----------------------------------------------------------------------------------------------- patching
@contextlib.contextmanager
def patched(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextlib.contextmanager
def nominal_apply_trap():
    def trap(*_a, **_k):
        raise NominalApplyCalled("nominal_apply reached")
    with contextlib.ExitStack() as stack:
        for target, name in ((neural, "four_views"), (neural, "differences"), (graph_mod, "successor"),
                             (graph_mod, "four_views"), (graph_mod, "nominal_overlay"), (contracts_mod, "nominal_overlay")):
            stack.enter_context(patched(target, name, trap))
        yield


@contextlib.contextmanager
def file_read_trap():
    def trap(*_a, **_k):
        raise FileReadAttempt("file or cache read during forward")
    with contextlib.ExitStack() as stack:
        stack.enter_context(patched(builtins, "open", trap))
        for name in ("open", "read_text", "read_bytes"):
            stack.enter_context(patched(pathlib.Path, name, trap))
        stack.enter_context(patched(torch, "load", trap))
        stack.enter_context(patched(pickle, "load", trap))
        yield


@contextlib.contextmanager
def count_nominal_apply():
    counter = {"calls": 0}
    original = graph_mod.nominal_overlay

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)
    with patched(graph_mod, "nominal_overlay", counted):
        yield counter


def forward(policy, snap):
    with torch.no_grad():
        return policy(snap, policy.initial_hidden())


# ----------------------------------------------------------------------------------------------- A checks
def _record_differences():
    record = []
    original = neural.differences

    def recording(encoder, graphs):
        out = original(encoder, graphs)
        record.append((graphs, out))
        return out
    return record, recording


def check_A1_f_after_identical(template, states):
    """F_after of Absolute Successor equals B2's, field by field, and equals the unpatched nominal overlay."""
    compared = 0
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        records = {}
        for method in ("B2", "B2-ABS"):
            record, recording = _record_differences()
            with patched(neural, "differences", recording):
                forward(make_policy(template, method, 0), snap)
            records[method] = record
        assert len(records["B2"]) == len(records["B2-ABS"]) > 0, "no successor views recorded"
        contracts = {c.id: c for c in template.contracts}
        legal = [cid for cid, ok in zip(snap.candidate_ids, snap.mask) if ok]
        assert len(legal) == len(records["B2"])
        for cid, (g_b2, _), (g_abs, _) in zip(legal, records["B2"], records["B2-ABS"]):
            k_b2, _h, ki_b2, _hi = g_b2
            k_abs, _h2, ki_abs, _hi2 = g_abs
            reference = reference_successor(contracts[cid], template, values)
            assert ki_b2.values == ki_abs.values, "F_after differs between B2 and Absolute Successor for %s" % cid
            assert k_b2.values == k_abs.values
            assert dict(ki_abs.values) == reference, "F_after differs from the unpatched nominal_apply for %s" % cid
            assert set(dict(ki_abs.values)) == set(values), "F_after fields differ from F fields"
            compared += 1
    return {"candidate_states_compared": compared, "states": len(states)}


def check_A3_rows_are_absolute(template, states):
    """The rows handed to the candidate readout are E(F_after_i) - never E(F_after_i) - E(F)."""
    checked = 0
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        policy = make_policy(template, "B2-ABS", 0)
        seen = []
        original = policy.contract.forward

        def recording(context, rows, mask=None, _orig=original):
            seen.append(rows.detach().clone())
            return _orig(context, rows, mask)
        contracts = {c.id: c for c in template.contracts}
        legal = [cid for cid, ok in zip(snap.candidate_ids, snap.mask) if ok]
        with patched(policy.contract, "forward", recording):
            forward(policy, snap)
        assert len(seen) == len(legal)
        with torch.no_grad():
            k = graph_mod.view(template, snap.facts.values)
            zk = policy.encoder(k)
            for cid, rows in zip(legal, seen):
                ki = graph_mod.successor(k, contracts[cid])
                zki = policy.encoder(ki)
                assert torch.allclose(rows, zki.float(), atol=1e-6, rtol=1e-5), "rows are not E(F_after) for %s" % cid
                if float((zki - zk).abs().max()) > 1e-6:  # successor differs from F: absolute and difference must be distinguishable
                    assert not torch.allclose(rows, (zki - zk).float(), atol=1e-6), "rows equal the latent difference for %s" % cid
                    assert float(zk.abs().max()) > 1e-3
                checked += 1
    return {"candidate_rows_checked": checked}


def check_nominal_apply_is_load_bearing(template, states):
    """Breaking nominal_apply must change Absolute Successor logits (so the control really applies the contract)."""
    changed = 0
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        base = forward(make_policy(template, "B2-ABS", 0), snap).logits
        noop = lambda contract, facts, derived=(), exclusive_groups=(): contracts_mod.ReadOnlyOverlay(facts, {})  # noqa: E731
        with patched(graph_mod, "nominal_overlay", noop):
            broken = forward(make_policy(template, "B2-ABS", 0), snap).logits
        legal = torch.tensor(snap.mask)
        if not torch.allclose(base[legal], broken[legal], atol=1e-7):
            changed += 1
    assert changed > 0, "nominal_apply is not load-bearing for Absolute Successor"
    return {"states_where_logits_changed": changed, "states": len(states)}


def check_permutation(template, method, states):
    checked = 0
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        ids = list(snap.candidate_ids)
        perm = list(reversed(range(len(ids))))
        permuted = make_snapshot(template, values, seed, order=[ids[i] for i in perm])
        a = forward(make_policy(template, method, 0), snap)
        b = forward(make_policy(template, method, 0), permuted)
        assert b.candidate_ids == tuple(ids[i] for i in perm)
        assert tuple(b.mask.tolist()) == tuple(snap.mask[i] for i in perm)
        legal_a = a.logits[perm]
        assert torch.allclose(legal_a[b.mask], b.logits[b.mask], atol=1e-6, rtol=1e-5), "permutation misaligns logits for %s" % method
        assert torch.equal(a.logits[perm] == -torch.inf, b.logits == -torch.inf)
        checked += 1
    return {"states": checked}


# ----------------------------------------------------------------------------------------------- N checks
def _forbidden_names(node):
    hits = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in NOMINAL_APPLY_NAMES:
            hits.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in NOMINAL_APPLY_NAMES:
            hits.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in NOMINAL_APPLY_NAMES:
            hits.add(sub.value)
    return hits


def nc_production_nodes():
    """AST nodes of the Matched Neural Composition production path: the interaction class, the node-feature prelude
    and the candidate branch of RepctlPolicy.forward."""
    cls = ast.parse(textwrap.dedent(inspect.getsource(rp.StateEffectInteraction)))
    tree = ast.parse(textwrap.dedent(inspect.getsource(rp.RepctlPolicy.forward)))

    def is_method_test(test, name):
        return isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "method" \
            and any(isinstance(c, ast.Constant) and c.value == name for c in test.comparators)
    prelude, branch = [], []
    for sub in ast.walk(tree):
        if isinstance(sub, ast.If) and is_method_test(sub.test, "B1-K+NC"):
            prelude.append(ast.Module(body=sub.body, type_ignores=[]))
        if isinstance(sub, ast.If) and is_method_test(sub.test, "B2-ABS"):
            branch.append(ast.Module(body=sub.orelse, type_ignores=[]))
    assert len(prelude) == 1 and len(branch) == 1, "expected one node-feature prelude and one NC candidate branch"
    return cls, prelude + branch


def _code_without_docstrings(node):
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.unparse(node)


def check_N1_source_scan():
    cls, branches = nc_production_nodes()
    hits = _forbidden_names(cls)
    for branch in branches:
        hits |= _forbidden_names(branch)
    assert not hits, "Matched Neural Composition production path references %s" % sorted(hits)
    return {"forbidden_names": sorted(NOMINAL_APPLY_NAMES), "hits": []}


def check_N2_runtime_trap(template, states):
    """Runtime trap: NC forward must not reach nominal_apply; the same trap must fire for B2/Absolute Successor."""
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        with nominal_apply_trap():
            forward(make_policy(template, "B1-K+NC", 0), snap)
    controls_fire = {}
    seed, values = states[0]
    snap = make_snapshot(template, values, seed)
    for method in ("B2", "B2-ABS"):
        try:
            with nominal_apply_trap():
                forward(make_policy(template, method, 0), snap)
        except NominalApplyCalled:
            controls_fire[method] = True
        else:
            controls_fire[method] = False
    assert all(controls_fire.values()), "the runtime trap does not fire on the successor methods: %s" % controls_fire
    with count_nominal_apply() as counter:
        forward(make_policy(template, "B1-K+NC", 0), snap)
    assert counter["calls"] == 0
    return {"nc_nominal_apply_calls": 0, "trap_fires_on": sorted(controls_fire)}


def _nc_rows(policy, snap, cid, values_override=None, contract_override=None):
    """Candidate rows and uk of the NC module for one candidate (no policy-level randomness)."""
    template = snap.template
    values = dict(snap.facts.values) if values_override is None else values_override
    k = graph_mod.view(template, values)
    with torch.no_grad():
        node_h = policy.encoder.features(k)
        node_index = {v: i for i, v in enumerate(template.node_ids)}
        contract = {c.id: c for c in template.contracts}[cid] if contract_override is None else contract_override
        rows, _ = policy.state_effect(node_h, node_index, contract)
    return rows


def check_N3_effects_change_representation(template, states):
    """Changing the grounded effects (ADD <-> DEL swap, dropped effect) changes the candidate rows."""
    from .contracts import Effects
    changed = 0
    policy = make_policy(template, "B1-K+NC", 0)
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        for c in template.contracts:
            if not (c.effects.add or c.effects.delete):
                continue
            base = _nc_rows(policy, snap, c.id)
            swapped = replace(c, effects=Effects(add=c.effects.delete, delete=c.effects.add, unknown=c.effects.unknown))
            moved = _nc_rows(policy, snap, c.id, contract_override=swapped)
            dropped = replace(c, effects=Effects(add=c.effects.add[:-1] if c.effects.add else (), delete=c.effects.delete, unknown=c.effects.unknown))
            fewer = _nc_rows(policy, snap, c.id, contract_override=dropped)
            if base.shape != moved.shape or not torch.allclose(base, moved):
                changed += 1
            if base.shape != fewer.shape or not torch.allclose(base, fewer):
                changed += 1
    assert changed > 0, "candidate representation does not respond to the grounded effects"
    return {"changes_observed": changed}


def check_N4_state_changes_representation(template, states):
    """Same contract, different current F -> different candidate rows."""
    changed = 0
    policy = make_policy(template, "B1-K+NC", 0)
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        for c in template.contracts:
            atoms = list(c.pre_pos) + list(c.effects.add) + list(c.effects.delete)
            base = _nc_rows(policy, snap, c.id)
            for atom in atoms:
                flipped = dict(values)
                flipped[atom.id] = Truth.FALSE if values[atom.id] == Truth.TRUE else Truth.TRUE
                if not torch.allclose(base, _nc_rows(policy, snap, c.id, values_override=flipped)):
                    changed += 1
                    break
    assert changed > 0, "candidate representation does not respond to the current state F"
    return {"changes_observed": changed}


def check_N6_no_sidecar(template, states):
    cls, branches = nc_production_nodes()
    source = (_code_without_docstrings(cls) + "\n" + "\n".join(_code_without_docstrings(b) for b in branches)).lower()
    hits = [t for t in SIDECAR_TOKENS if t in source]
    assert not hits, "sidecar-like tokens in the NC production path: %s" % hits
    fields = [f for f in Snapshot.__dataclass_fields__]
    bad = [f for f in fields if any(w in f.lower() for w in ("after", "successor", "sidecar"))]
    assert not bad, "Snapshot carries successor-like fields: %s" % bad
    for seed, values in states:
        with file_read_trap():
            forward(make_policy(template, "B1-K+NC", 0), make_snapshot(template, values, seed))
    return {"snapshot_fields": fields, "tokens_scanned": list(SIDECAR_TOKENS), "forward_under_file_read_trap": True}


# ----------------------------------------------------------------------------------------------- common checks
def check_common_boundaries(template, states):
    """Hard mask, candidate IDs, finite outputs and no-provider equivalence between reference and control methods."""
    for seed, values in states:
        snap = make_snapshot(template, values, seed)
        reference = None
        for method in ALL_METHODS:
            out = forward(make_policy(template, method, 0), snap)
            assert out.candidate_ids == snap.candidate_ids
            assert tuple(out.mask.tolist()) == snap.mask, "hard mask differs for %s" % method
            assert torch.isfinite(out.logits[out.mask]).all() and torch.isfinite(out.value) and torch.isfinite(out.q[out.mask]).all()
            assert bool((out.logits[~out.mask] == -torch.inf).all())
            assert out.diagnostics["prior_inputs"] and all(int(torch.count_nonzero(v)) == 0 for v in out.diagnostics["prior_inputs"].values())
            assert all(int(torch.count_nonzero(v)) == 0 for v in out.diagnostics["delta"].values())
            reference = reference or (out.candidate_ids, tuple(out.mask.tolist()))
            assert (out.candidate_ids, tuple(out.mask.tolist())) == reference
    return {"methods": list(ALL_METHODS), "states": len(states)}


def all_semantic_checks(template, states):
    return {
        "A1_f_after_identical_to_b2": check_A1_f_after_identical(template, states),
        "A1b_nominal_apply_load_bearing": check_nominal_apply_is_load_bearing(template, states),
        "A3_rows_are_absolute_successor": check_A3_rows_are_absolute(template, states),
        "A4_permutation_absolute_successor": check_permutation(template, "B2-ABS", states),
        "N1_source_scan": check_N1_source_scan(),
        "N2_runtime_trap": check_N2_runtime_trap(template, states),
        "N3_effects_change_representation": check_N3_effects_change_representation(template, states),
        "N4_state_changes_representation": check_N4_state_changes_representation(template, states),
        "N5_permutation_neural_composition": check_permutation(template, "B1-K+NC", states),
        "N6_no_sidecar_or_file_read": check_N6_no_sidecar(template, states),
        "common_boundaries": check_common_boundaries(template, states),
    }


# ----------------------------------------------------------------------------------------------- mutants
@contextlib.contextmanager
def mutant_break_nominal_apply():
    noop = lambda contract, facts, derived=(), exclusive_groups=(): contracts_mod.ReadOnlyOverlay(facts, {})  # noqa: E731
    with patched(graph_mod, "nominal_overlay", noop):
        yield


@contextlib.contextmanager
def mutant_restore_latent_subtraction():
    original = neural.differences

    def subtracting(encoder, graphs):
        d = original(encoder, graphs)
        zk, zh, zki, zhi = d.encodings
        return neural.Differences(d.zk, d.zh, d.dk, d.dh, d.dp, (zk, zh, zki.float() - zk.float(), zhi.float() - zh.float()))
    with patched(neural, "differences", subtracting):
        yield


@contextlib.contextmanager
def mutant_nc_calls_nominal_apply():
    original = rp.StateEffectInteraction.forward

    def calling(self, node_h, node_index, contract):
        facts = {a.id: Truth.UNKNOWN for a in contract.atoms()}
        facts.update({a.id: Truth.TRUE for a in contract.pre_pos})
        facts.update({a.id: Truth.FALSE for a in contract.pre_neg})
        contracts_mod.nominal_overlay(contract, facts)
        return original(self, node_h, node_index, contract)
    with patched(rp.StateEffectInteraction, "forward", calling):
        yield


@contextlib.contextmanager
def mutant_nc_reads_a_file():
    original = rp.StateEffectInteraction.forward

    def reading(self, node_h, node_index, contract):
        with open("/etc/hostname", "r") as handle:
            handle.read()
        return original(self, node_h, node_index, contract)
    with patched(rp.StateEffectInteraction, "forward", reading):
        yield


@contextlib.contextmanager
def mutant_nc_ignores_state():
    original = rp.StateEffectInteraction.forward

    def blind(self, node_h, node_index, contract):
        return original(self, torch.zeros_like(node_h), node_index, contract)
    with patched(rp.StateEffectInteraction, "forward", blind):
        yield


@contextlib.contextmanager
def mutant_nc_ignores_effects():
    original = rp.StateEffectInteraction.forward

    def blind(self, node_h, node_index, contract):
        from .contracts import Effects
        stripped = replace(contract, pre_pos=(), pre_neg=(), effects=Effects(), conditional=())
        return original(self, node_h, node_index, stripped)
    with patched(rp.StateEffectInteraction, "forward", blind):
        yield


@contextlib.contextmanager
def mutant_nc_source_calls_four_views():
    """Source-scan mutant: a copy of the NC class whose forward mentions four_views."""
    original = inspect.getsource

    def faked(obj):
        text = original(obj)
        if obj is rp.StateEffectInteraction:
            text = text.replace("self.forward_calls += 1", "self.forward_calls += 1\n        _ = four_views", 1)
        return text
    with patched(inspect, "getsource", faked):
        yield


MUTANTS = (
    # (id, covers, context manager, check callable name)
    ("M-A2-break-nominal-apply", "A2", mutant_break_nominal_apply, lambda t, s: check_A1_f_after_identical(t, s)),
    ("M-A2b-break-nominal-apply-load-bearing", "A2", mutant_break_nominal_apply, lambda t, s: _expect_unchanged_detects(t, s)),
    ("M-A3-restore-latent-subtraction", "A3", mutant_restore_latent_subtraction, lambda t, s: check_A3_rows_are_absolute(t, s)),
    ("M-N1-nc-source-mentions-four-views", "N1", mutant_nc_source_calls_four_views, lambda t, s: check_N1_source_scan()),
    ("M-N2-nc-calls-nominal-apply", "N2", mutant_nc_calls_nominal_apply, lambda t, s: check_N2_runtime_trap(t, s)),
    ("M-N3-nc-ignores-effects", "N3", mutant_nc_ignores_effects, lambda t, s: check_N3_effects_change_representation(t, s)),
    ("M-N4-nc-ignores-state", "N4", mutant_nc_ignores_state, lambda t, s: check_N4_state_changes_representation(t, s)),
    ("M-N6-nc-reads-a-file", "N6", mutant_nc_reads_a_file, lambda t, s: check_N6_no_sidecar(t, s)),
)


def _expect_unchanged_detects(template, states):
    # Under the broken-nominal-apply mutant the load-bearing check must fail (logits would not change vs itself).
    return check_nominal_apply_is_load_bearing(template, states)


def run_mutants(template, states):
    """Every mutant must make its check fail; the unmutated check must pass."""
    rows = []
    for mutant_id, covers, mutant, check in MUTANTS:
        check(template, states)  # unmutated control: must pass
        detected, error = False, None
        try:
            with mutant():
                check(template, states)
        except (AssertionError, NominalApplyCalled, FileReadAttempt) as exc:
            detected, error = True, "%s: %s" % (type(exc).__name__, str(exc)[:200])
        rows.append({"mutant": mutant_id, "covers": covers, "unmutated_check_passed": True, "mutant_detected": detected, "detection": error})
    return rows


# ----------------------------------------------------------------------------------------------- capacity audit
def module_parameter_table(policy):
    return {name: sum(p.numel() for p in module.parameters()) for name, module in policy.named_children()}


def effective_parameters(policy, snap):
    """Parameters that actually receive a gradient from one forward/backward (no optimizer step is taken)."""
    policy.zero_grad(set_to_none=True)
    out = policy(snap, policy.initial_hidden())
    mask = out.mask
    loss = out.logits[mask].sum() + out.value + out.q[mask].sum()
    loss.backward()
    per_module, total = {}, 0
    for name, module in policy.named_children():
        count = sum(p.numel() for p in module.parameters() if p.grad is not None)
        if count:
            per_module[name] = count
            total += count
    policy.zero_grad(set_to_none=True)
    return total, per_module


def computation_counts(template, method, snap):
    """Encoder / nominal_apply / interaction-module calls in one policy forward (K = legal candidates)."""
    policy = make_policy(template, method, 0)
    legal = sum(snap.mask)
    calls = {"effect_readout": 0, "state_effect": 0}
    hooks = []
    for name in calls:
        module = getattr(policy, name, None)
        if module is not None:
            hooks.append(module.register_forward_hook(lambda _m, _i, _o, _n=name: calls.__setitem__(_n, calls[_n] + 1)))
    before = policy.encoder.forward_calls
    with count_nominal_apply() as counter:
        forward(policy, snap)
    for hook in hooks:
        hook.remove()
    encoder_calls = policy.encoder.forward_calls - before
    return {
        "legal_candidates": legal,
        "encoder_forward_calls_total": encoder_calls,
        "encoder_forward_calls_current_state": 1,
        "encoder_forward_calls_per_candidate": (encoder_calls - 1) / legal if legal else None,
        "nominal_apply_calls_total": counter["calls"],
        "nominal_apply_calls_per_candidate": counter["calls"] / legal if legal else None,
        "interaction_module_calls_total": {k: v for k, v in calls.items() if getattr(policy, k, None) is not None and v},
        "interaction_module_calls_per_candidate": {k: v / legal for k, v in calls.items() if getattr(policy, k, None) is not None and v},
        "node_feature_embedding_calls_outside_encoder": 1 if method == "B1-K+NC" else 0,
    }
