import json
from pathlib import Path
from types import SimpleNamespace
from cp_disr.analysis import s1_revision_resume as resume
from cp_disr.analysis.s1_revision import execute_registered_branch
from cp_disr.platforms.libero.runtime_factory import CaseSpec, RuntimeBundle
from tests.helpers.s1_runner_doubles import FakeBundle

def fixture(tmp_path):
    (tmp_path/"input_binding").mkdir(); (tmp_path/"witnesses").mkdir()
    (tmp_path/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").write_text(json.dumps({
        "runtime":{"active_task_id":"T_A","reference_skill_seconds_by_task":{"T_A":3.7},
                   "task_deadlines":{"T_A":60.0}}}))
    branch={"branch_id":"TEST-branch","case_id":"TEST-case","candidate_id":"a:A:x:v1",
            "restore_seed":42,"authorized":True,"execute_now":True}
    (tmp_path/"witnesses/e4_branch_registration.json").write_text(json.dumps({"branches":[branch]}))
    (tmp_path/"budget_ledger.json").write_text(json.dumps({"physical_witness_episodes":{"used":0,"cap":1}}))
    (tmp_path/"budget_events.jsonl").write_text("")
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    reserve_branch_attempt(tmp_path,"TEST-branch")

def test_production_runtime_method_body_creates_restore_receipt(monkeypatch):
    import cp_disr.platforms.libero.runtime_factory as rf
    events=[]
    class Env:
        def close(self): events.append("close")
        def reset(self): events.append(("reset",self.case.seed))
        def _apply_case_poses(self): events.append("poses")
        def _open_gripper_reset(self): events.append("gripper")
    class Clock:
        def now_seconds(self): return 0.0
    class Obs:
        def __init__(self,*a): pass
        def observe(self): return object()
    class Perc:
        def __init__(self,*a): pass
        def infer(self,x): return object()
    class Ver:
        def __init__(self,*a): pass
        def verify(self,*a): return ()
    class Eval:
        def __init__(self,*a,**k): pass
        def reset_episode(self): pass
    class ExecInner:
        def __init__(self,*a,**k): pass
    class Builder:
        def __init__(self,*a,**k): pass
        def initial(self,facts,obs,episode): return SimpleNamespace(
            episode_id=episode, candidate_ids=("a:A:x:v1",), mask=(True,),
            facts=facts, template=template)
    monkeypatch.setattr(rf,"make_env",lambda spec: (events.append(("new",spec.seed)) or Env()))
    monkeypatch.setattr(rf,"DurationProvider",lambda env: Clock())
    monkeypatch.setattr(rf,"SafetyManager",lambda env: object())
    monkeypatch.setattr(rf,"ObservationProvider",Obs)
    monkeypatch.setattr(rf,"PerceptionAdapter",Perc)
    monkeypatch.setattr(rf,"FactVerifier",Ver)
    monkeypatch.setattr(rf,"TaskEvaluator",Eval)
    monkeypatch.setattr(rf,"SkillExecutor",ExecInner)
    monkeypatch.setattr(rf,"SnapshotBuilder",Builder)
    monkeypatch.setattr(rf,"load_cache_edges",lambda cache: ())
    contract=SimpleNamespace(id="a:A:x:v1",version="v1")
    template=SimpleNamespace(contracts=(contract,))
    case=CaseSpec("case","TEST",3,(0,0),(0,0),(0,0),(0,0))
    old=Env()
    bundle=RuntimeBundle(old,object(),object(),object(),object(),object(),object(),Clock(),object(),
                         task_id="T_A",template=template,cases={"case":case},caches={})
    snap=bundle.start_case("case",restore_seed=7)
    assert events[0]=="close" and ("new",7) in events
    assert bundle.restore_verified is True
    assert bundle.restore_receipt["applied_restore_seed"]==7
    assert bundle.snapshot_identity==bundle.restore_receipt["state_identity_sha256"]
    assert snap.candidate_ids==("a:A:x:v1",)

def test_missing_restore_measurements_are_rejected_before_action(tmp_path):
    fixture(tmp_path)
    bundle=FakeBundle([{"success":True,"terminated":True}],[])
    original=bundle.start_case
    def no_receipt(case_id,restore_seed):
        snap=original(case_id,restore_seed)
        bundle.restore_receipt=None
        bundle.restore_verified=True
        return snap
    bundle.start_case=no_receipt
    result=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b:bundle)
    assert result["termination_reason"]=="RESTORE_RECEIPT_MISSING"
    assert bundle.executor.executed_candidate_ids==[]

def test_restore_measurement_seed_mismatch_is_rejected(tmp_path):
    fixture(tmp_path)
    bundle=FakeBundle([{"success":True,"terminated":True}],[])
    original=bundle.start_case
    def wrong_seed(case_id,restore_seed):
        snap=original(case_id,restore_seed)
        bundle.restore_receipt["applied_restore_seed"]=41
        return snap
    bundle.start_case=wrong_seed
    result=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b:bundle)
    assert result["termination_reason"]=="RESTORE_SEED_APPLIED_MISMATCH"
    assert bundle.executor.executed_candidate_ids==[]
