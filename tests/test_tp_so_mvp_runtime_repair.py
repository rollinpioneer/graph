from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cp_disr.facts import FactStore, Truth
from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
from cp_disr.platforms.libero.d0_env import CaseSpec
from cp_disr.platforms.libero.verifier import FactVerifier
from cp_disr.platforms.libero import tp_so_mvp_runtime as r
from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m


def _case(lid_closed=False):
    return CaseSpec(
        case_id="repair-case", split="train", seed=17,
        target_xy=(-0.1, 0.0), second_xy=(0.1, 0.0),
        container_xy=(0.0, 0.15), buffer_xy=(0.0, -0.15),
        lid_closed=lid_closed, task_id=r.TASK_ID,
        second_role=r.SECOND_ROLE, deadline=60.0,
    )


class PublicEnv:
    def __init__(self, lid_closed=False, writers=()):
        self.case = _case(lid_closed)
        self.task_manifest = self.case
        self.reset_identity = {
            "case_id": self.case.case_id, "seed": self.case.seed,
            "lid_closed": self.case.lid_closed,
        }
        self.task_id = r.TASK_ID
        self.task_contracts = tuple(writers) or (SimpleNamespace(id="a:PICK:target:v1", effects=SimpleNamespace(add=(), delete=(), unknown=()), conditional=()),)
        self.second_role = r.SECOND_ROLE
        self.hidden_called = False

    def public_layout(self):
        return {"table_top_z": 0.8}

    def hidden_truth(self):
        self.hidden_called = True
        raise AssertionError("hidden_truth must not be read by task verifier")


def measurement(blobs=None, grip=(0.04, -0.04)):
    return SimpleNamespace(measurements={
        "blobs": blobs or {},
        "eef_pos": [0.4, 0.4, 1.0],
        "gripper_qpos": list(grip),
    })


def facts_by_id(records):
    return {x.fact_id: x for x in records}


def test_B01_B02_generic_defaults_are_preserved():
    from cp_disr.platforms.libero.runtime_factory import RuntimeBundle
    assert RuntimeBundle.__dataclass_fields__["verifier_factory"].default is None
    assert RuntimeBundle.__dataclass_fields__["evaluator_factory"].default is None
    assert r.TPSOMVPFactVerifier is not FactVerifier


def test_B03_B05_task_fact_and_evaluator_bindings():
    env = PublicEnv()
    verifier = r.TPSOMVPFactVerifier(env)
    evaluator = r.make_tp_so_evaluator(env, 60.0, r.TASK_ID)
    assert isinstance(verifier, r.TPSOMVPFactVerifier)
    assert isinstance(evaluator, r.TPSOMVPTaskEvaluator)
    assert isinstance(evaluator, r.TaskEvaluator)


def test_B06_B08_evaluator_requires_both_goals():
    class E:
        def __init__(self, t, s):
            self.t, self.s = t, s
        def hidden_truth(self):
            def pos(inside):
                return [0.0, 0.0, 0.82] if inside else [0.2, 0.2, 0.82]
            return {
                "target": pos(self.t), "second_object": pos(self.s),
                "container": [0.0, 0.0, 0.8], "lid": [1.0, 0.0, 0.8],
                "table_top_z": 0.8,
            }
    ev = r.TPSOMVPTaskEvaluator(E(True, False), 60.0, task_id=r.TASK_ID)
    assert ev.goal_true() is False
    ev = r.TPSOMVPTaskEvaluator(E(False, True), 60.0, task_id=r.TASK_ID)
    assert ev.goal_true() is False
    ev = r.TPSOMVPTaskEvaluator(E(True, True), 60.0, task_id=r.TASK_ID)
    assert ev.goal_true() is True


def test_F01_F02_unknown_open_is_publicly_repaired():
    env = PublicEnv(False)
    verifier = r.TPSOMVPFactVerifier(env)
    recs = facts_by_id(verifier.verify(measurement()))
    assert recs["p:Open:container"].value is Truth.TRUE
    assert "tp-so-mvp-static-public-invariant" in recs["p:Open:container"].reason
    assert set(recs["p:Open:container"].evidence_ids) >= {"task_manifest", "reset_config"}
    assert env.hidden_called is False


def test_F01_measured_open_true_is_retained():
    env = PublicEnv(False)
    verifier = r.TPSOMVPFactVerifier(env)
    blobs = {"container": {"xyz": [0.0, 0.0, 0.8]},
             "lid": {"xyz": [0.2, 0.0, 0.8]}}
    rec = facts_by_id(verifier.verify(measurement(blobs)))["p:Open:container"]
    assert rec.value is Truth.TRUE
    assert rec.reason.endswith("lid_offset_from_container")


def test_F03_F04_false_or_closed_fail_closed():
    with pytest.raises(r.StaticInvariantContradiction, match="STOPPED_STATIC_INVARIANT_CONTRADICTION"):
        r.TPSOMVPFactVerifier(PublicEnv(False)).verify(
            measurement({"container": {"xyz": [0.0, 0.0, 0.8]},
                         "lid": {"xyz": [0.0, 0.0, 0.8]}}))
    with pytest.raises(r.StaticInvariantContradiction, match="lid_closed=true"):
        r.TPSOMVPFactVerifier(PublicEnv(True)).verify(measurement())


def test_F05_registered_open_writer_is_rejected():
    class C:
        id = "a:OPEN:container:v1"
        effects = SimpleNamespace(add=(SimpleNamespace(id="p:Open:container"),),
                                  delete=(), unknown=())
        conditional = ()
    with pytest.raises(r.StaticInvariantContradiction, match="registered action"):
        r.TPSOMVPFactVerifier(PublicEnv(False, writers=(C(),))).verify(measurement())


def test_F06_F07_unknown_facts_are_not_filled():
    recs = facts_by_id(r.TPSOMVPFactVerifier(PublicEnv(False)).verify(measurement()))
    assert recs["p:Held:target"].value is Truth.UNKNOWN
    assert recs["p:OnTable:target"].value is Truth.UNKNOWN


def test_F08_non_open_records_match_production_verifier():
    env = PublicEnv(False)
    base = FactVerifier(env).verify(measurement())
    task = r.TPSOMVPFactVerifier(env).verify(measurement())
    b, t = facts_by_id(base), facts_by_id(task)
    for key in b:
        if key != "p:Open:container":
            assert t[key].fact_id == b[key].fact_id
            assert t[key].value == b[key].value
            assert t[key].evidence_ids == b[key].evidence_ids
            assert t[key].reason == b[key].reason
            assert t[key].last_confirmed_value == b[key].last_confirmed_value


def test_P01_P08_real_contract_template_and_b_plan():
    contracts, template, _ = m._build_offline_template(__import__("pathlib").Path(__file__).resolve().parents[1])
    facts = m._initial_fact_store()
    result = BPlanPlanner(SearchConfig(
        depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0,
        reference_skill_seconds=m.D_REF,
    )).plan(facts, template, 60.0)
    assert result.status == "PLAN_FOUND"
    assert result.depth == 4
    assert result.plan == (
        "a:PICK:second_object:v1", "a:PLACE:second_object:container:v1",
        "a:PICK:target:v1", "a:PLACE:target:container:v1",
    )
