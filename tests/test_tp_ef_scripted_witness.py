"""Offline tests for the scripted-witness card: registration, analysis rules, route execution with a mock runtime."""
import math
from types import SimpleNamespace as NS

from cp_disr.analysis import tp_ef_protocol_review as pr
from cp_disr.analysis import tp_ef_scripted_witness as w


def test_routes_orders_and_ids():
    assert w.ROUTES == {"T": ["pt", "plt", "ps", "pbs"], "S": ["ps", "pbs", "pt", "plt"]}
    assert [o[1] for o in w.ORDER] == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    assert [o[3:] for o in w.ORDER[:4]] == [("T", 0), ("S", 0), ("T", 1), ("S", 1)]
    ids = {w.branch_id_of(c, r, p) for _, _, c, r, p in w.ORDER}
    assert len(ids) == 8 and pr.IDS["open"] not in {a for r in w.ROUTE_IDS.values() for a in r}
    assert all(len(v) == 4 for v in w.ROUTE_IDS.values())


def row(case, route, rep, t, valid=True, success=True, skills=4, anomalies=()):
    q = 2.0 ** (-t / w.H)
    return {"case_id": case, "route": route, "repeat": rep, "valid": valid, "task_success": success, "skill_count": skills, "anomalies": list(anomalies),
            "decision_relative_completion_time": t, "Q_ref_decision": q, "G_episode": q}


def full(case, tt, ts, tt2=None, ts2=None):
    return [row(case, "T", 0, tt), row(case, "S", 0, ts), row(case, "T", 1, tt if tt2 is None else tt2), row(case, "S", 1, ts if ts2 is None else ts2)]


def test_epsilon_floor_and_repeat_ranges():
    e = w.epsilons(full("a", 10.0, 15.0))
    assert e["epsilon_G"] == w.EPS_G_FLOOR and e["epsilon_T"] == w.EPS_T_FLOOR
    e2 = w.epsilons(full("a", 10.0, 15.0, tt2=14.0))
    assert math.isclose(e2["epsilon_T"], 4.0) and e2["epsilon_G"] > w.EPS_G_FLOOR


def test_reliable_witness_requires_all_conditions():
    rows = full("a", 12.0, 18.0)
    j = w.judge_state("a", rows, w.epsilons(rows))
    assert j["reliable"] and j["winner"] == "T" and j["status"] == "RELIABLE_PHYSICAL_WITNESS"
    close = full("a", 12.0, 13.0)
    assert not w.judge_state("a", close, w.epsilons(close))["reliable"]
    flip = [row("a", "T", 0, 12.0), row("a", "S", 0, 18.0), row("a", "T", 1, 20.0), row("a", "S", 1, 14.0)]
    jf = w.judge_state("a", flip, {"epsilon_G": 0.0, "epsilon_T": 0.0})
    assert not jf["reliable"] and not jf["conditions"]["4_repeat_winner_direction_consistent"]


def test_failed_or_wrong_skill_count_never_forms_a_winner():
    rows = full("a", 12.0, 18.0)
    rows[1] = row("a", "S", 0, 18.0, valid=False, success=False)
    assert w.judge_state("a", rows, w.epsilons(rows))["status"] == "NOT_RELIABLE_EXECUTION_CONDITIONS"
    rows = full("a", 12.0, 18.0)
    rows[0]["skill_count"] = 5
    assert not w.judge_state("a", rows, w.epsilons(rows))["reliable"]


def test_mechanism_gate_classes():
    r03, r05 = full("T_B_dev_03", 12.0, 18.0), full("T_B_dev_05", 18.0, 12.0)
    eps = w.epsilons(r03 + r05)
    j3, j5 = w.judge_state("T_B_dev_03", r03 + r05, eps), w.judge_state("T_B_dev_05", r03 + r05, eps)
    assert w.mechanism_gate(j3, j5, r03 + r05, False, False)["status"] == "PASS"
    same = full("T_B_dev_05", 12.0, 18.0)
    j5s = w.judge_state("T_B_dev_05", r03 + same, eps)
    assert w.mechanism_gate(j3, j5s, r03 + same, False, False)["status"] == "FIXED_ACTION_BIAS"
    tie = full("T_B_dev_05", 12.0, 12.5)
    j5t = w.judge_state("T_B_dev_05", r03 + tie, w.epsilons(r03 + tie))
    assert w.mechanism_gate(j3, j5t, r03 + tie, False, False)["status"] == "INSUFFICIENT_TWO_WITNESSES"
    assert w.mechanism_gate(j3, j5, r03 + r05, True, False)["status"] == "ENGINEERING_OR_EXECUTION_FAILURE"
    weak = full("T_B_dev_03", 12.0, 12.5)
    jw = w.judge_state("T_B_dev_03", weak, w.epsilons(weak))
    assert w.mechanism_gate(jw, None, weak, False, True)["status"] == "INSUFFICIENT_FIRST_WITNESS"


def test_time_and_return_definitions():
    q = w.q_values(10.0, 17.6)
    assert math.isclose(q["Q_ref_decision"], 2 ** (-10.0 / 23.1)) and math.isclose(q["G_episode"], 2 ** (-17.6 / 23.1))
    assert w.H == 23.1 and w.DEADLINE == 60.0


# ------------------------------------------------------------------------ route execution with a mock runtime
class Truth:
    def __init__(self, v):
        self.value = v


class Atom:
    def __init__(self, i):
        self.id = i


EFFECTS = {pr.IDS["pt"]: ({"p:Held:target": "TRUE", "p:GripperEmpty": "FALSE"}, ["p:Held:target"]), pr.IDS["plt"]: ({"p:Inside:target:container": "TRUE", "p:Held:target": "FALSE", "p:GripperEmpty": "TRUE"}, ["p:Inside:target:container"]),
           pr.IDS["ps"]: ({"p:Held:second_object": "TRUE", "p:GripperEmpty": "FALSE"}, ["p:Held:second_object"]), pr.IDS["pbs"]: ({"p:AtBuffer:second_object:buffer": "TRUE", "p:Held:second_object": "FALSE", "p:GripperEmpty": "TRUE"}, ["p:AtBuffer:second_object:buffer"])}


class Mock:
    def __init__(self, break_at=None, skip_effect_at=None, early_success=False, controller_fail_at=None):
        self.t = 7.6
        self.n = 0
        self.break_at, self.skip_effect_at, self.early_success, self.controller_fail_at = break_at, skip_effect_at, early_success, controller_fail_at
        self.facts = {"p:GripperEmpty": "TRUE"}
        self.task_id = "T_B"
        self.contracts = {i: NS(id=i, timeout_seconds=9.0, effects=NS(add=[Atom(a) for a in add])) for i, (_, add) in EFFECTS.items()}
        self.clock = NS(now_seconds=lambda: self.t)
        self.executor = NS(execute=self._exec, last={})
        self.observations = NS(observe=lambda: NS(frame_id="f"))
        self.perception = NS(infer=lambda o: o)
        self.verifier = NS(verify=lambda m, e: tuple(NS(fact_id=k, value=Truth(v), reason="r", last_confirmed_value=Truth(v), evidence_ids=("rgb",)) for k, v in sorted(self.facts.items())))
        self.snapshot_builder = NS(build=lambda snap, rec, obs, ex, end: self._snap())
        self.evaluator = NS(evaluate=self._eval)

    def _snap(self):
        ids = tuple(self.contracts)
        mask = tuple(not (self.break_at is not None and self.n == self.break_at and i == list(self.contracts)[0]) for i in ids)
        return NS(facts=NS(values={k: Truth(v) for k, v in self.facts.items()}), candidate_ids=ids, mask=mask, env_id="e", episode_id="ep")

    def _exec(self, cid, timeout):
        self.n += 1
        start = self.t
        self.t += 4.0
        eff = EFFECTS[cid][0]
        if self.skip_effect_at != self.n:
            self.facts.update(eff)
        exit_ = "NORMAL_TERMINATION" if self.controller_fail_at != self.n else "TIMEOUT"
        self.executor.last = {"sim_duration": 4.0, "steps": 80, "states": []}
        return NS(execution_id=f"x{self.n}", controller_exit=exit_, start_seconds=start, end_seconds=self.t, evidence_ids=())

    def _eval(self, v):
        ok = self.n >= (2 if self.early_success else 4)
        return NS(success=ok, terminated=ok, truncated=False, reason="TASK_SUCCESS" if ok else "CONTINUE", reward_events=((0.0, 1.0),) if ok else ())


def run_mock(**kw):
    m = Mock(**kw)
    snap = m._snap()
    return w.execute_route(m, snap, w.ROUTE_IDS["T"], "T", lambda: "h", 7.6, 0.0, m.contracts), m


def test_execute_route_success_path():
    (skills, anomalies, ok), m = run_mock()
    assert ok and not anomalies and len(skills) == 4 and m.n == 4
    assert skills[-1]["post"]["evaluator"]["success"] and not any(s["post"]["evaluator"]["success"] for s in skills[:3])
    assert skills[0]["pre"]["mask_for_action"] and skills[0]["post"]["expected_add_facts"] == ["p:Held:target"]
    assert math.isclose(skills[-1]["post"]["clock_end"], 7.6 + 16.0)


def test_execute_route_flags_each_gate():
    (s, a, ok), _ = run_mock(early_success=True)
    assert not ok and any(x.startswith("EARLY_SUCCESS") for x in a) and len(s) == 2
    (s, a, ok), _ = run_mock(skip_effect_at=2)
    assert not ok and any("POSTCONDITION_NOT_CONFIRMED_SKILL_1" in x for x in a)
    (s, a, ok), _ = run_mock(controller_fail_at=1)
    assert not ok and any("CONTROLLER_EXIT_SKILL_0" in x for x in a) and len(s) == 1
    (s, a, ok), _ = run_mock(break_at=0)
    assert not ok and any(x.startswith("MASK_FALSE_BEFORE_SKILL_0") for x in a)


def test_budget_design():
    assert w.CAPS["physical_branch_attempts"] == 8 and w.CAPS["environment_constructions"] == 8 and w.CAPS["scripted_skill_calls"] == 32
    assert w.CAPS["skill_retries"] == 0 and w.CAPS["open_skill_calls"] == 0 and w.CAPS["provider_requests"] == 0
