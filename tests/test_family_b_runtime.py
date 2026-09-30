"""Family B factory and public binding checks; never constructs MuJoCo."""
from pathlib import Path
from types import SimpleNamespace

from cp_disr.analysis import family_b_pilot as p
from cp_disr.platforms.libero import family_b_runtime as r
from cp_disr.platforms.libero.family_b_adapters import (
    FamilyBEvaluator, FamilyBPerception, FamilyBVerifier)
from cp_disr.platforms.libero.family_b_env import FamilyBEnv

ROOT = Path(__file__).resolve().parents[1]
CFG = p.config(ROOT, "configs/final_master/s4_family_b_staging.yaml")


def test_factory_has_three_objects_two_destinations_and_specialists(tmp_path):
    p.inspect_bindings(ROOT, CFG, tmp_path)
    import yaml
    manifest = yaml.safe_load((tmp_path / "spec/runtime_manifest_T_P_FB.yaml").read_text())
    bundle = r.create_family_b_runtime(manifest)
    assert bundle.environment.__class__.__name__ == "_NullEnv"
    assert set(bundle.cases) == {"layout_0", "layout_1"}
    assert bundle.env_factory.__name__ == "make_family_b_env"
    assert bundle.perception_cls is FamilyBPerception
    assert bundle.verifier_factory is FamilyBVerifier
    assert bundle.evaluator_factory is FamilyBEvaluator
    assert len(bundle.template.contracts) == 7
    assert len(r.VERIFIER_FACT_IDS) == 17
    assert {g.fact_id for g in bundle.template.goals} == {
        "p:Inside:obj_b:receiver", "p:Inside:obj_c:receiver", "p:GripperEmpty"}


def test_start_case_guards_specialized_runtime(monkeypatch):
    from cp_disr.platforms.libero.runtime_factory import RuntimeBundle
    bundle = r.FamilyBBundle(
        environment=None, executor=None, observations=None, perception=None,
        verifier=None, evaluator=None, safety=None, clock=None,
        snapshot_builder=None, task_id=r.TASK_ID)
    env = FamilyBEnv.__new__(FamilyBEnv)
    env.resolve_skill_destination = lambda *a: [0, 0, 0]
    bundle.environment = env
    bundle.perception = FamilyBPerception.__new__(FamilyBPerception)
    bundle.verifier = FamilyBVerifier.__new__(FamilyBVerifier)
    bundle.evaluator = FamilyBEvaluator.__new__(FamilyBEvaluator)
    sentinel = SimpleNamespace()
    monkeypatch.setattr(RuntimeBundle, "start_case", lambda *a, **kw: sentinel)
    assert bundle.start_case("layout_0", restore_seed=1) is sentinel
    bundle.verifier = object()
    import pytest
    with pytest.raises(Exception, match="verifier binding lost"):
        bundle.start_case("layout_0", restore_seed=1)
