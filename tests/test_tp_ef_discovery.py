"""Offline tests for the T_P-EF discovery card pure logic (no simulator, no provider)."""
import math

from cp_disr.analysis import tp_ef_discovery as m

A, B = m.PAIR


def trace(*ends, success_at=None, extra=()):
    rows = []
    for i, e in enumerate(ends):
        rows.append({"action": f"s{i}", "elapsed_end": e, "task_success": success_at == i})
    return rows


def test_branch_outcome_success_discounted():
    out = m.branch_outcome(trace(4, 8, 12, 16, 20, success_at=4), nominal_depth=5)
    assert out["success"] and out["skill_count"] == 5 and out["rework_count"] == 0
    assert math.isclose(out["g_start_discounted"], 2 ** (-20 / m.H_SECONDS))


def test_branch_outcome_failure_is_zero_and_counted():
    out = m.branch_outcome(trace(4, 8), nominal_depth=5)
    assert not out["success"] and out["g_start_discounted"] == 0.0 and out["rework_count"] is None


def test_branch_outcome_late_success_not_credited():
    out = m.branch_outcome(trace(30, 61, success_at=1), nominal_depth=5)
    assert out["g_start_discounted"] == 0.0


def row(case, cand, g, t, success=True, valid=True, skills=5, term="TASK_SUCCESS"):
    return {"case_id": case, "candidate_id": cand, "valid": valid, "g_start_discounted": g, "completion_seconds": t,
            "success": success, "skill_count": skills, "termination": term}


def test_epsilon_uses_floor_and_repeat_range_only():
    rows = [row("c", A, 0.5, 20.0), row("c", A, 0.5, 20.0), row("c", B, 0.4, 25.0), row("c", B, 0.4, 25.0)]
    eps = m.epsilon_from_repeats(rows)
    assert eps["epsilon_q"] == m.EPS_G_FLOOR and eps["epsilon_t"] == m.EPS_T_FLOOR and eps["repeat_groups_complete"]
    noisy = rows[:1] + [row("c", A, 0.6, 22.0)] + rows[2:]
    assert math.isclose(m.epsilon_from_repeats(noisy)["epsilon_q"], 0.1)


def test_compare_pair_reliable_direction():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    rows = [row("c", A, 0.55, 20.0), row("c", B, 0.40, 30.0, skills=7)]
    out = m.compare_pair("c", rows, eps)
    assert out["reliable"] and out["direction"] == A and "return" in out["reasons"]


def test_compare_pair_within_tolerance_is_low_value():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.50, 20.0), row("c", B, 0.51, 20.5)], eps)
    assert not out["reliable"] and out["status"] == "LOW_DIAGNOSTIC_VALUE"


def test_compare_pair_planner_artefact_excluded():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.5, 20.0), row("c", B, 0.0, 60.0, success=False, term="NO_PLAN")], eps)
    assert not out["reliable"] and out["status"] == "ARTEFACT_EXCLUDED"


def test_compare_pair_invalid_branch_not_established():
    eps = {"epsilon_q": 0.02, "epsilon_t": 2.1}
    out = m.compare_pair("c", [row("c", A, 0.5, 20.0), row("c", B, 0.0, 0.0, valid=False)], eps)
    assert out["status"] == "NOT_ESTABLISHED_INVALID_BRANCH"


def test_state_rank_is_deterministic_and_outcome_free():
    keys = [m.state_rank_key(f"T_B_dev_{i:02d}") for i in range(20)]
    assert keys == [m.state_rank_key(f"T_B_dev_{i:02d}") for i in range(20)] and len(set(keys)) == 20


def test_branch_ids_unique_and_budget_fits():
    ids = {m.branch_id_of(c, cand, r) for c in ("a", "b", "c") for cand in m.PAIR for r in (0, 1)}
    assert len(ids) == 12
    assert 4 + 2 + 2 <= m.PHYSICAL_EPISODE_CAP


def test_classify_e6_mapping():
    assert m.classify_e6(0, True, True, True) == "PROVIDER_SCHEMA_LIMITATION"
    assert m.classify_e6(0, False, False, False) == "CONTRACT_SUFFICIENT"
    assert m.classify_e6(0, False, False, True) == "HEURISTIC_INCONCLUSIVE"
    assert m.classify_e6(3, True, True, True) == "BENEFICIAL_IMPERFECT"
    assert m.classify_e6(3, False, True, True) == "HARMFUL_BUT_INFORMATIVE"


# ---------------------------------------------------------------- worker control loop with a mock runtime
import json
from types import SimpleNamespace as NS


class _T:  # Truth-like
    def __init__(self, v):
        self.value = v

    def __eq__(self, o):
        return getattr(o, "value", o) == self.value

    def __hash__(self):
        return hash(self.value)


class MockWorld:
    """Five-skill world: OPEN, PICK(second), PLACE_BUFFER, PICK(target), PLACE; success after the fifth executed skill."""

    def __init__(self, fail_first=False):
        self.t = 0.0
        self.done = 0
        self.fail_first = fail_first
        self.held = None
        self.open = False
        self.snapshot_identity = "ep-1-identity"
        self.restore_receipt = {"requested_case_id": "c", "state_identity_sha256": "EP-DEPENDENT", "public_facts_sha256": "f"}
        self.restore_verified = True
        self.episode_start_seconds = 0.0
        self.task_id = "T_B"
        self.environment = NS(close=lambda: None)
        self.clock = NS(now_seconds=lambda: self.t)
        ids = ("a:OPEN:container:v1", "a:PICK:second_object:v1", "a:PICK:target:v1")
        self.template = NS(contracts=tuple(NS(id=i, timeout_seconds=9.0) for i in ids), goals=(NS(fact_id="g", sign=1),))
        self.observations = NS(observe=lambda: NS(proprioception=(0.0, 1.0), frame_id="f"))
        self.perception = NS(infer=lambda o: o)
        self.executor = NS(execute=self._exec)
        self.verifier = NS(verify=lambda m, e: ())
        self.snapshot_builder = NS(build=lambda snap, rec, obs, ex, end: self._snap())
        self.evaluator = NS(evaluate=self._eval)

    def _snap(self):
        facts = NS(values={"p:Open:container": _T("TRUE" if self.open else "FALSE"),
                           "p:Held:second_object": _T("TRUE" if self.held == "second_object" else "FALSE")})
        return NS(facts=facts, template=self.template, candidate_ids=tuple(c.id for c in self.template.contracts),
                  mask=(True, True, True), env_id="e", episode_id="ep-1")

    def start_case(self, case_id, restore_seed=None):
        return self._snap()

    def _exec(self, cid, timeout):
        start = self.t
        self.t += 4.0
        self.done += 1
        if ":OPEN:" in cid:
            self.open = not self.fail_first or self.done > 1
        if ":PICK:second_object" in cid:
            self.held = "second_object" if not self.fail_first else None
        return NS(execution_id=f"x{self.done}", controller_exit="NORMAL_TERMINATION", start_seconds=start, end_seconds=self.t, evidence_ids=())

    def _eval(self, v):
        ok = self.done >= 5
        return NS(success=ok, terminated=ok, truncated=False, reason="TASK_SUCCESS" if ok else "CONTINUE")


class ScriptedPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, facts, template, remaining):
        self.calls += 1
        return NS(status="PLAN_FOUND", plan=("a:PICK:target:v1",), expanded_nodes=1)


def _registry(tmp_path, world, cand):
    snap = world._snap()
    ident = m.pre_state_identity(world, snap)
    b = {"branch_id": "bid", "case_id": "c", "candidate_id": cand, "repeat": 0, "wave": 1, "snapshot_hash": ident["snapshot_hash"],
         "facts_hash": ident["facts_hash"], "candidate_hash": ident["candidate_hash"], "qpos_hash": ident["qpos_hash"],
         "nominal_depth": 5, "restore_seed": 1}
    (tmp_path / "branch_registration.json").write_text(json.dumps({"branches": [b]}))


def test_worker_success_path_and_outcome(tmp_path):
    w = MockWorld()
    _registry(tmp_path, w, A)
    res = m.worker(tmp_path, tmp_path, "bid", bundle_factory=lambda: w, planner_factory=ScriptedPlanner)
    assert res["valid"] and res["success"] and res["skill_count"] == 5 and res["rework_count"] == 0
    assert res["trace"][0]["postcondition_fact"] == "p:Open:container" and res["trace"][0]["postcondition_true"]
    assert math.isclose(res["g_start_discounted"], 2 ** (-20.0 / m.H_SECONDS))
    assert (tmp_path / "branches" / "bid.json").exists()


def test_worker_candidate_postcondition_failure_is_invalid_not_retried(tmp_path):
    w = MockWorld(fail_first=True)
    _registry(tmp_path, w, B)
    res = m.worker(tmp_path, tmp_path, "bid", bundle_factory=lambda: w, planner_factory=ScriptedPlanner)
    assert not res["valid"] and res["termination"] == "CANDIDATE_UNSTABLE" and not res["g3_candidate_stable"]
    try:
        m.worker(tmp_path, tmp_path, "bid", bundle_factory=lambda: MockWorld(), planner_factory=ScriptedPlanner)
        raise AssertionError("second claim must be refused")
    except RuntimeError as exc:
        assert "STOPPED_BUDGET" in str(exc)


def test_worker_restore_identity_is_episode_counter_independent(tmp_path):
    w = MockWorld()
    _registry(tmp_path, w, A)
    w.snapshot_identity = "ep-99-identity"
    w.restore_receipt = dict(w.restore_receipt, state_identity_sha256="DIFFERENT-EPISODE")
    res = m.worker(tmp_path, tmp_path, "bid", bundle_factory=lambda: w, planner_factory=ScriptedPlanner)
    assert res["restore_matches_registry"] and res["valid"]
