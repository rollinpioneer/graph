"""Fixtures of the search-scope diagnostic (plan section 18): logic only, no GPU, no model. Real-asset checks (LM-cut admissibility, SAS mapping, observer with the real scorers) run in the card's ``fixtures`` command.
The three synthetic searches (plain, observed, observed with another scorer) run once per module and are booked in the resource ledger of the card."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cp_disr.pddl import task as T
from cp_disr.pddl.scope_diag import analysis as AN
from cp_disr.pddl.scope_diag import labels as LB
from cp_disr.pddl.scope_diag.core import hx, ix, local_positions, local_structure
from cp_disr.pddl.scope_diag.observed import gbfs_observed, snapshot_schedule
from cp_disr.pddl.scope_diag.references import path_states
from cp_disr.pddl.search_match import evaluators as EV
from cp_disr.pddl.search_match.budget import Budget
from cp_disr.pddl.search_match.engine import SuccessorIndex, gbfs
from cp_disr.pddl.search_match.runner import OldRun

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e"


class FakeEval(EV.Evaluator):
    """Deterministic scorer: unmet goal atoms plus a state-hash tie breaker in {0,..,6}/10 (many near equal values, deterministic)."""
    name = "FAKE"

    def __init__(self, scale=1.0, salt=""):
        self.scale, self.salt = scale, salt

    def prepare(self, task, ctx, budget):
        self.task = task

    def evaluate(self, states, budget):
        return [self.scale * (self.task.unmet_goals(s) + (int(hashlib.md5((self.salt + str(s)).encode()).hexdigest(), 16) % 7) / 10.0) for s in states]


@pytest.fixture(scope="module")
def train_task():
    old = OldRun(OLD)
    c = next(x for x in old.cases("train") if x["case_id"] == "train_n3_000")
    from cp_disr.pddl import depots as DP
    return T.depots_task(DP.DOMAIN_TYPED, c["file"])


FIX_EXP = 16                                                                          # plan 13.2: fixture prefixes are limited to 16 expansions


@pytest.fixture(scope="module")
def runs(train_task):
    """Three synthetic fixture prefixes (<= 16 expansions each) of the card's fixture ledger; the reference plan is the exact optimal plan of the small Train problem."""
    task = train_task
    index = SuccessorIndex(task)
    oracle = T.ExactOracle(task)
    try:
        ids = oracle.plan()
    finally:
        oracle.close()
    seq, ref = path_states(task, ids)
    snaps_at = tuple(range(1, FIX_EXP + 1))
    o1, o2, o3 = [], [], []
    b1 = Budget(60, FIX_EXP, 8 * 2 ** 30, 20.0)
    e1 = FakeEval()
    e1.prepare(task, {}, b1)
    r1 = gbfs(task, index, e1, b1, milestones=(), observer=lambda s, n: o1.append(s))                         # (1) plain engine
    b2 = Budget(60, FIX_EXP, 8 * 2 ** 30, 20.0)
    e2 = FakeEval()
    e2.prepare(task, {}, b2)
    r2, s2, x2 = gbfs_observed(task, index, e2, b2, snapshot_at=snaps_at, ref=ref, case_id="train_n3_000", n_comp=3, observer=lambda s, n: o2.append(s))   # (2) observed, same scorer
    b3 = Budget(60, FIX_EXP, 8 * 2 ** 30, 20.0)
    e3 = FakeEval(scale=3.0, salt="other")
    e3.prepare(task, {}, b3)
    r3, s3, x3 = gbfs_observed(task, index, e3, b3, snapshot_at=snaps_at, ref=ref, case_id="train_n3_000", n_comp=3, observer=lambda s, n: o3.append(s))   # (3) observed, another scorer
    return dict(task=task, index=index, r1=r1, o1=o1, r2=r2, o2=o2, s2=s2, r3=r3, o3=o3, s3=s3, ids=ids, seq=seq, ref=ref)


# ---- 1. observer has no side effects
def test_observer_does_not_change_the_search(runs):
    assert runs["r1"].status in ("NODE_LIMIT", "SOLVED")
    assert runs["o1"] == runs["o2"] and len(runs["o1"]) >= 5
    for k in ("status", "plan", "expanded", "generated", "duplicates", "path_updates", "solved_at_expansion", "best_h", "open_size", "closed_size"):
        assert getattr(runs["r1"], k) == getattr(runs["r2"], k), k
    assert len(runs["s2"]) >= 5


# ---- 2. OPEN anchor lifecycle
def test_anchor_lifecycle(runs):
    ref, order, snaps = runs["ref"], runs["o3"], runs["s3"]
    assert snaps
    anchors = 0
    for sn in snaps:
        e = sn["event"]
        closed_before = set(order[:e - 1])
        assert sn["popped"]["state"] not in closed_before
        a = sn["anchor"]
        if a is not None:
            anchors += 1
            assert a["state"] in ref and a["state"] not in closed_before             # a popped / closed reference state is never an anchor
            assert a["ref_U"] == ref[a["state"]][1]
            assert a["ref_k"] <= sn["ref_deepest_generated"]
        else:
            assert sn["ref_deepest_generated"] == sn["ref_deepest_closed"]            # nothing generated-but-unclosed on the path => NO_ACTIVE_REFERENCE_ANCHOR
        for c in sn["competitors"]:
            assert c["state"] not in closed_before and c["state"] != sn["popped"]["state"]
        assert len(sn["competitors"]) <= 3
    assert anchors > 0


def test_run_trace_file_roundtrip(runs, tmp_path):
    """The stage function ``run_trace`` (4th synthetic prefix): file layout, hex round trip, same pops / competitors as the direct observed run of the same scorer."""
    from cp_disr.pddl.scope_diag.core import gz_read, run_trace, trace_states
    old = OldRun(OLD)
    case = dict(next(x for x in old.cases("train") if x["case_id"] == "train_n3_000"), set="train", group="C")
    cfg = {"traces": {"control": {"max_expansions": FIX_EXP, "wall_seconds": 15, "memory_gib": 8}, "ipc": {"max_expansions": FIX_EXP, "wall_seconds": 15, "memory_gib": 8}, "snapshots": 32, "competitors": 3, "grace_seconds": 20}}
    out = tmp_path / "t.jsonl.gz"
    rec = run_trace(case, "fake", FakeEval(scale=3.0, salt="other"), {"ids": runs["ids"], "sha256": "x"}, cfg, out)
    lines = gz_read(out)
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "summary" and lines[-1]["status"] == rec["status"]
    snaps = [d for d in lines if d["type"] == "snapshot"]
    assert snaps and all(isinstance(d["popped"]["state"], str) for d in snaps)
    by_event = {s["event"]: s for s in runs["s3"]}
    for d in snaps:
        direct = by_event[d["event"]]
        assert ix(d["popped"]["state"]) == direct["popped"]["state"]
        assert [ix(c["state"]) for c in d["competitors"]] == [c["state"] for c in direct["competitors"]]        # fixed-hash competitors are deterministic
        assert (d["anchor"] is None) == (direct["anchor"] is None)
    assert trace_states(lines)


def test_snapshot_schedule():
    s = snapshot_schedule(4096, 32)
    assert s[0] == 1 and s[-1] == 4096 and len(s) == len(set(s)) <= 32 and s == sorted(s)
    s2 = snapshot_schedule(128, 32)
    assert s2[0] == 1 and s2[-1] == 128


# ---- 3. generation groups and path updates
def test_groups_and_sibling(runs):
    index, order, snaps = runs["index"], runs["o3"], runs["s3"]
    succ_of = {}

    def succ(p):
        if p not in succ_of:
            succ_of[p] = {(p & ~a.del_mask) | a.add_mask for a in index.applicable(p)}
        return succ_of[p]
    multi = 0
    for sn in snaps:
        before = order[:sn["event"] - 1]
        for rec in [sn["popped"]] + sn["competitors"]:
            expected = {p for p in before if rec["state"] in succ(p)}               # an OPEN state is never closed, so every earlier parent that generates it has been recorded
            assert set(rec["parents"]) == expected
            multi += len(rec["parents"]) > 1
        if sn["event"] > 1:
            assert sn["popped"]["parents"]
    print("states with several recorded parents in the fixture prefix:", multi)


def test_shared_parent_classification():
    a = {"parents": ["p1", "p2"]}
    b = {"parents": ["p2"]}
    c = {"parents": ["p3"]}
    assert AN.shared_parent(a, b) is True
    assert AN.shared_parent(a, c) is False
    assert AN.shared_parent({"parents": []}, c) is None


# ---- 4. serialisation
def test_hex_roundtrip_high_bits():
    for s in (0, 1, (1 << 63), (1 << 64) + 1, (1 << 1621) | 5, (1 << 2000) - 1):
        assert ix(json.loads(json.dumps(hx(s)))) == s
    assert isinstance(json.loads(json.dumps(hx(1 << 70))), str)


# ---- 5. certificates
def test_certificate_direction_and_unknown():
    assert AN.cert_remaining(3, 4) == "CERT"
    assert AN.cert_remaining(4, 4) == "NO"                                           # equality is not a strict order
    assert AN.cert_remaining(5, 4) == "NO"                                           # overlapping intervals are not equal either
    assert AN.cert_remaining(None, 4) == "UNKNOWN" and AN.cert_remaining(3, None) == "UNKNOWN"
    assert AN.cert_remaining(AN.INF, 4) == "NO"                                      # infinite upper bound: no information
    assert AN.cert_remaining(3, AN.INF) == "CERT"                                    # worse state proven dead by the relaxation
    assert AN.num(None) is None and AN.num("INF") == AN.INF and AN.num(0) == 0.0     # null is never 0
    assert AN.order_of(1.0, 2.0) == "CORRECT" and AN.order_of(2.0, 1.0) == "INVERTED" and AN.order_of(1.0, 1.0 + 1e-9) == "TIE"


# ---- 6. near ties / alternative optimal actions are not errors (R2 logic on constructed data)
class FakeInputs:
    def __init__(self, succ, values, lower, upper, parent="P", child="B"):
        self.local = {"c": [{"k": 0, "parent": parent, "child": child, "goal_successor": False, "succ": [[h, i, ["a%d" % i]] for i, h in enumerate(succ)]}]}
        self._v, self._l, self._u = values, lower, upper
        self.ref_u = {"c": {}}

    def value(self, cid, sc, h):
        return self._v[sc].get(h)

    def lower(self, cid, h):
        return self._l.get(h)

    def upper(self, cid, h):
        return self._u.get(h)


def local_for(values_dense, lower, upper, succ=("B", "Y", "Z")):
    inp = FakeInputs(list(succ), {"dense": dict(values_dense), "wl": {h: float(i) for i, h in enumerate(succ)}}, lower, upper)
    inp._v["dense"]["P"] = 5.0
    inp._v["wl"]["P"] = 5.0
    return AN.local_rows(inp, "c")[0]


def test_r2_decision_error_vs_tie_vs_alternative():
    lower = {"P": 3, "B": 1, "Y": 5, "Z": 2}
    upper = {"B": 3, "P": 4}                                                        # B provably better than Y (3 < 5), not than Z (3 >= 2)
    r = local_for({"B": 2.0, "Y": 1.0, "Z": 3.0}, lower, upper)                      # model prefers Y by a clear margin: decision error
    assert r["R2_dense_error"] == "DECISION_ERROR"
    r = local_for({"B": 1.0 + 1e-9, "Y": 1.0, "Z": 3.0}, lower, upper)               # preference inside the numerical tolerance: tie, not an error
    assert r["R2_dense_error"] == "TIE_CHOICE"
    r = local_for({"B": 2.0, "Y": 4.0, "Z": 1.0}, lower, upper)                      # model chooses Z: B is not provably better than Z: no error (alternative / unknown relation)
    assert r["R2_dense_error"] == "OK"
    r = local_for({"B": 0.5, "Y": 4.0, "Z": 1.0}, lower, upper)                      # model chooses the plan child
    assert r["R2_dense_error"] == "OK" and r["R2_dense_choice_is_plan_child"]
    r = local_for({"B": 2.0, "Y": 1.0, "Z": 3.0}, {"P": 3, "B": 1, "Z": 2}, upper)   # unknown lower bound of the choice: unknown, never a silent OK
    assert r["R2_dense_error"] == "UNKNOWN_L"


def test_r1_requires_certificate():
    inp = FakeInputs(["B", "Y"], {"dense": {"P": 5.0, "B": 6.0, "Y": 1.0}, "wl": {"P": 5.0, "B": 4.0, "Y": 1.0}}, {"P": 4, "B": 3, "Y": 1}, {"B": 3, "P": 4})
    r = AN.local_rows(inp, "c")[0]
    assert r["R1_cert"] == "CERT" and r["R1_dense_order"] == "INVERTED" and r["R1_wl_order"] == "CORRECT"
    inp2 = FakeInputs(["B", "Y"], {"dense": {"P": 5.0, "B": 6.0}, "wl": {"P": 5.0, "B": 4.0}}, {"P": 3, "B": 3}, {"B": 3, "P": 4})
    assert AN.local_rows(inp2, "c")[0]["R1_cert"] == "NO"


# ---- 7. decision rule on constructed summaries
def test_decision_rule_precedence():
    cfg = {"decision": {"signal": {"min_problems": 2, "min_cross_events_per_problem": 8, "min_delayed_states_per_problem": 4}, "local": {"min_problems": 2, "min_error_parents_per_problem": 6},
                        "coverage": {"min_active_anchor_events": 8, "min_bounded_events": 8, "majority": 3}, "tie_dominated_fraction": 0.5}}
    F = ["a", "b", "c", "d"]
    good = {"cross_inverted_events": 9, "cross_inverted_delayed_states": 5, "anchor_active_events": 20, "bounded_events": 20, "resolved_events": 20, "certified_cross_group": 9, "cross_tie_under_driver": 0, "cross_inverted_under_driver": 9,
            "certified_remaining": 9}
    bad = dict(good, cross_inverted_events=0, cross_inverted_delayed_states=0, certified_cross_group=0, cross_inverted_under_driver=0, certified_remaining=0)
    loc0 = {"R2_dense_error_parents": 0}
    r = AN.decide(cfg, F, {"a": good, "b": good, "c": bad, "d": bad}, {c: loc0 for c in F}, False)
    assert r["scientific_label"] == "CROSS_SCOPE_SIGNAL" and not r["mixed_local_and_cross"]
    r = AN.decide(cfg, F, {c: bad for c in F}, {"a": {"R2_dense_error_parents": 7}, "b": {"R2_dense_error_parents": 6}, "c": loc0, "d": loc0}, False)
    assert r["scientific_label"] == "LOCAL_WITNESS_PRESENT_NO_CROSS_LOCALIZATION"
    r = AN.decide(cfg, F, {c: dict(bad, anchor_active_events=2) for c in F}, {c: loc0 for c in F}, False)
    assert r["scientific_label"] == "REFERENCE_ANCHOR_MISSING"
    r = AN.decide(cfg, F, {c: dict(bad, resolved_events=1) for c in F}, {c: loc0 for c in F}, False)
    assert r["scientific_label"] == "INCONCLUSIVE_CERTIFICATES"
    r = AN.decide(cfg, F, {c: dict(bad, resolved_events=1) for c in F}, {c: loc0 for c in F}, False, coverage_key="bounded_events")   # the registered quantity (bounds exist) does not see the overlap
    assert r["scientific_label"] == "NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX"
    r = AN.decide(cfg, F, {c: dict(bad, cross_tie_under_driver=9, certified_cross_group=9) for c in F}, {c: loc0 for c in F}, False)
    assert r["scientific_label"] == "NUMERICAL_TIE_DOMINATED"
    r = AN.decide(cfg, F, {c: bad for c in F}, {c: loc0 for c in F}, False)
    assert r["scientific_label"] == "NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX"
    r = AN.decide(cfg, F, {"a": good, "b": good, "c": good, "d": good}, {"a": {"R2_dense_error_parents": 7}, "b": {"R2_dense_error_parents": 7}, "c": loc0, "d": loc0}, False)
    assert r["scientific_label"] == "CROSS_SCOPE_SIGNAL" and r["mixed_local_and_cross"]


# ---- 7b. analysis end to end on a constructed run root (no search, no model)
DEC = {"signal": {"min_problems": 2, "min_cross_events_per_problem": 8, "min_delayed_states_per_problem": 4}, "local": {"min_problems": 2, "min_error_parents_per_problem": 6},
       "coverage": {"min_active_anchor_events": 8, "min_bounded_events": 8, "majority": 3}, "tie_dominated_fraction": 0.5}


def build_root(tmp, anchor_h=5.5, shared_events=(), nF=2, popped_lower=6, self_event=None):
    from cp_disr.pddl.scope_diag.core import gz_write, jdump
    rr = Path(tmp) / "rr"
    cases = [{"case_id": "f%d" % i, "group": "F", "set": "ipc"} for i in range(1, nF + 1)] + [{"case_id": "c1", "group": "C", "set": "train"}]
    jdump(rr / "registration" / "case_manifest.json", {"cases": cases})
    jdump(rr / "registration" / "config.json", {"decision": DEC})
    jdump(rr / "references" / "plans.json", {c["case_id"]: {"length": 10, "sha256": "x", "optimality": "UNKNOWN", "path_states_hex": ["r%d" % k for k in range(11)], "ids": []} for c in cases})
    bounds, scores = [], {}
    for cid in [c["case_id"] for c in cases if c["group"] == "F"]:
        snaps = []
        for e in range(1, 11):
            k = 7 + e % 4
            parents_a = ["x%d" % e] if e in shared_events else ["y"]
            anchor = {"role": "ANCHOR", "state": "r%d" % k, "h": anchor_h, "serial": 100 + e, "g": 3, "parents": parents_a, "first_event": 0, "ref_k": k, "ref_U": 10 - k}
            popped = {"role": "POPPED", "state": "s%d" % e, "h": 5.0, "serial": e, "g": 5, "parents": ["x%d" % e], "first_event": e - 1, "ref_k": None, "ref_U": None}
            if e == self_event:                                                                  # the popped state is itself the deepest OPEN reference state
                anchor = dict(popped, role="ANCHOR", ref_k=k, ref_U=10 - k)
            snaps.append({"type": "snapshot", "event": e, "open": 50, "closed": e - 1, "expanded": e - 1, "elapsed": 0.1, "popped": popped, "anchor": anchor, "competitors": [], "ref_deepest_generated": 10, "ref_deepest_closed": 2})
            bounds.append({"case_id": cid, "state_hex": "s%d" % e, "lower": popped_lower, "upper": None, "lower_source": "LM_CUT", "upper_source": "UNAVAILABLE", "status": "OK", "category": "R3_POPPED"})
        gz_write(rr / "traces" / cid / "dense.jsonl.gz", [{"type": "header", "case_id": cid}] + snaps + [{"type": "summary", "status": "NODE_LIMIT", "expanded": 10, "wall_total": 1.0, "schedule": list(range(1, 11))}])
        scores[cid] = {"s%d" % e: 5.0 for e in range(1, 11)}
        scores[cid].update({"r%d" % k: anchor_h for k in range(7, 11)})
    (rr / "labels").mkdir(parents=True, exist_ok=True)
    (rr / "labels" / "bounds.jsonl").write_text("\n".join(json.dumps(b) for b in bounds) + "\n")
    for cid, sc in scores.items():
        jdump(rr / "states" / ("scores_dense_%s.json" % cid), sc)
        jdump(rr / "states" / ("scores_wl_%s.json" % cid), {h: float(i) for i, h in enumerate(sc)})
    return rr


def test_analysis_end_to_end_cross_signal(tmp_path):
    out = AN.analyze(build_root(tmp_path))
    r = out["r3"]["f1|dense"]
    assert out["decision"]["scientific_label"] == "CROSS_SCOPE_SIGNAL"
    assert r["cross_inverted_events"] == 10 and r["cross_inverted_delayed_states"] == 4 and r["certified_cross_group"] == 10 and r["certified_sibling"] == 0
    assert r["anchor_active_events"] == 10 and r["bounded_events"] == 10 and r["resolved_events"] == 10 and r["pairs_unknown_bound"] == 0
    assert out["totals"]["certified_pairs"] == 20
    import csv
    rows = list(csv.DictReader(open(Path(tmp_path) / "rr" / "labels" / "certified_pairs.csv")))
    assert len(rows) == 20 and all(x["certificate_remaining"] == "CERT" and x["dense_order"] == "INVERTED" for x in rows)


def test_analysis_sibling_is_not_cross_and_ties_are_not_witnesses(tmp_path):
    out = AN.analyze(build_root(tmp_path / "a", shared_events=(3, 4, 5)))
    r = out["r3"]["f1|dense"]
    assert r["certified_sibling"] == 3 and r["certified_cross_group"] == 7 and r["cross_inverted_events"] == 7               # shared recorded parent: not a cross-group witness
    assert out["decision"]["scientific_label"] != "CROSS_SCOPE_SIGNAL"
    out = AN.analyze(build_root(tmp_path / "b", anchor_h=5.0 + 1e-9))
    r = out["r3"]["f1|dense"]
    assert r["cross_inverted_events"] == 0 and r["cross_tie_under_driver"] == 10                                              # numerical ties never count as inverted witnesses
    assert out["decision"]["scientific_label"] != "CROSS_SCOPE_SIGNAL"


def test_overlapping_bounds_are_unresolved_and_self_pairs_are_skipped(tmp_path):
    # L(popped) = 0 <= U(anchor) in every event: bounds exist but overlap -> UNKNOWN, never counted as resolved; the registered quantity (bounds exist) would still see all 10 events
    out = AN.analyze(build_root(tmp_path / "o", nF=4, popped_lower=0))
    r = out["r3"]["f1|dense"]
    assert r["bounded_events"] == 10 and r["resolved_events"] == 0 and r["certified_remaining"] == 0 and r["pairs_resolved"] == 0
    assert out["decision"]["scientific_label"] == "INCONCLUSIVE_CERTIFICATES"
    assert out["decision"]["as_registered"]["scientific_label"] == "NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX"
    # an anchor that is the popped state itself forms no pair
    out = AN.analyze(build_root(tmp_path / "s", self_event=4))
    r = out["r3"]["f1|dense"]
    assert r["anchor_is_popped_events"] == 1 and r["candidate_pairs"] == 9 and r["cross_inverted_events"] == 9


# ---- 8. formula invariance, budget allocation, no optimiser
def test_action_scores_ignore_the_parent_value():
    rng = np.random.default_rng(0)
    child = rng.normal(size=7)
    base = 5.0 - child
    q = np.exp(base - base.max())
    q /= q.sum()
    for c in (0.0, 3.7, -11.2):
        logits = (5.0 + c) - child                                                    # l(s, a) = V(s) - V(T(s, a))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        assert np.allclose(p, q)
        assert np.allclose(logits[:, None] - logits[None, :], base[:, None] - base[None, :])


def test_allocation_is_fair_and_bounded():
    cases = ["a", "b", "c"]
    demand = {"a": [(0, "x%d" % i) for i in range(10)], "b": [(0, "y%d" % i) for i in range(2)], "c": [(1, "z%d" % i) for i in range(9)]}
    taken, rest = LB.allocate(demand, 12, cases)
    assert sum(len(v) for v in taken.values()) == 12
    assert len(taken["b"]) == 2 and len(taken["a"]) + len(taken["c"]) == 10
    assert min(len(taken["a"]), len(taken["c"])) >= 4


def test_no_optimizer_or_training_code_in_the_package():
    pkg = ROOT / "src/cp_disr/pddl/scope_diag"
    for p in pkg.glob("*.py"):
        text = p.read_text()
        for bad in ("torch.optim", ".backward(", "optimizer.step", "zero_grad", "nn.Module"):
            assert bad not in text, (p.name, bad)


def test_local_positions_and_structure(runs):
    assert local_positions(1, 16) == [] and local_positions(2, 16) == [0] and local_positions(5, 16) == [0, 1, 2, 3]
    pos = local_positions(60, 16)
    assert len(pos) == 16 and pos[0] == 0 and pos[-1] == 58 and pos == sorted(set(pos))
    task, index, seq = runs["task"], runs["index"], runs["seq"]
    loc = local_structure(task, index, seq, local_positions(len(runs["ids"]), 16))
    assert loc
    for p in loc:
        states = [t for t, _fa, _aids in p["succ"]]
        assert len(states) == len(set(states))
        assert p["child"] in states or p["goal_successor"]
        assert [fa for _t, fa, _a in p["succ"]] == sorted(fa for _t, fa, _a in p["succ"])           # canonical action order
