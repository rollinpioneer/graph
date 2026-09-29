from pathlib import Path
import json
import pytest
from cp_disr.analysis.s1_revision import execute_registered_branch
from cp_disr.baselines.b_plan import PlanResult
from tests.helpers.s1_runner_doubles import FakeBundle, FakePlanner, write_fixture

def call(tmp_path,bundle,plans=None,manifest=None,branch=None):
    write_fixture(tmp_path,manifest,branch)
    return execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b: bundle,planner_factory=(lambda: bundle.planner))

def test_B01_nested_reference_is_used_and_default_config_can_construct(tmp_path,monkeypatch):
    bundle=FakeBundle([{"success":False,"terminated":False},{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}],[{"status":"PLAN_FOUND","plan":["a:B:x:v1"]}])
    seen=[]
    class SpyPlanner:
        def __init__(self,config): seen.append(config)
        def plan(self,*args): return bundle.planner.plan(*args)
    import cp_disr.baselines.b_plan as bp
    monkeypatch.setattr(bp,"BPlanPlanner",SpyPlanner)
    write_fixture(tmp_path)
    result=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b: bundle)
    assert result["task_success"] is True and seen[0].reference_skill_seconds == 3.7
    assert result["protocol_complete"] is True

@pytest.mark.parametrize("value", [None,0,-1,float("nan"),float("inf")])
def test_B02_invalid_reference_rejected_before_bundle(tmp_path,value):
    calls=[]
    manifest={"runtime":{"active_task_id":"T_A","reference_skill_seconds_by_task":{"T_A":value},"task_deadlines":{"T_A":60.0}}}
    write_fixture(tmp_path,manifest)
    from cp_disr.analysis.s1_revision import execute_registered_branch
    r=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b:calls.append(1))
    assert r["execution_status"]=="EXCEPTION" and calls==[]

def test_B03_plan_found_is_executed_and_success_is_evaluator_only(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False,"reason":"CONTINUE"},{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}],[{"status":"PLAN_FOUND","plan":["a:B:x:v1"]}])
    r=call(tmp_path,bundle)
    assert bundle.executor.executed_candidate_ids==["a:A:x:v1","a:B:x:v1"]
    assert len(bundle.evaluator.inputs)==2 and r["task_success"] is True

def test_B04_normal_controller_does_not_equal_task_success(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False,"reason":"CONTINUE"},{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}],[{"status":"PLAN_FOUND","plan":["a:B:x:v1"]}])
    r=call(tmp_path,bundle)
    assert r["task_success"] is True and r["trace"][0]["controller_exit"]=="NORMAL_TERMINATION"

def test_B05_updated_mask_blocks_next_action(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False},{"success":True,"terminated":True}],[{"status":"PLAN_FOUND","plan":["a:B:x:v1"]}],masks=[[True]*9,[True,False,True,True,True,True,True,True,True]])
    r=call(tmp_path,bundle)
    assert bundle.executor.executed_candidate_ids==["a:A:x:v1"]
    assert r["termination_reason"]=="CURRENT_MASK_REJECTED"

def test_B06_deadline_before_action_does_not_execute(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False}],[],deadline=0.5,start_offset=0.5)
    manifest={"runtime":{"active_task_id":"T_A","reference_skill_seconds_by_task":{"T_A":3.7},"task_deadlines":{"T_A":0.5}}}
    r=call(tmp_path,bundle,manifest=manifest)
    assert bundle.executor.executed_candidate_ids==[] and r["termination_reason"]=="DEADLINE"

def test_B07_elapsed_is_not_reset_between_replans(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False},{"success":False,"terminated":False},{"success":True,"terminated":True}],[{"status":"PLAN_FOUND","plan":["a:B:x:v1"]},{"status":"PLAN_FOUND","plan":["a:C:x:v1"]}],durations=[2,3,1])
    r=call(tmp_path,bundle)
    assert [round(x["elapsed_seconds"]) for x in r["trace"]]==[2,5,6]
    events=[json.loads(line) for line in (tmp_path/"witnesses/branch_TEST-branch.jsonl").read_text().splitlines()]
    assert [round(x["relative_time"]) for x in events if x["phase"]=="decision_start"] == [0,2,5]

def test_B08_truncated_stops_without_replan(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False,"truncated":True,"reason":"TRUNCATED"}],[])
    r=call(tmp_path,bundle)
    assert r["execution_status"]=="TRUNCATED" and len(bundle.planner.inputs)==0

def test_B09_more_than_seven_actions_are_allowed(tmp_path):
    outcomes=[{"success":False,"terminated":False} for _ in range(7)]+[{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}]
    plans=[{"status":"PLAN_FOUND","plan":[f"a:{x}:x:v1"]} for x in "BCDEFGH"]
    bundle=FakeBundle(outcomes,plans,durations=[1]*16)
    r=call(tmp_path,bundle)
    assert len(bundle.executor.executed_candidate_ids)==8 and r["task_success"] is True

def test_B10_search_timeout_is_not_valid_E4(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False}],[{"status":"SEARCH_TIMEOUT","plan":[]}])
    r=call(tmp_path,bundle)
    assert r["execution_status"]=="SEARCH_TIMEOUT" and r["eligible_for_e4"] is False

def test_B11_symbolic_goal_does_not_override_evaluator(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False}],[{"status":"GOAL_ALREADY_SATISFIED","plan":[]}])
    r=call(tmp_path,bundle)
    assert r["termination_reason"]=="SYMBOLIC_EVALUATOR_MISMATCH" and r["task_success"] is False

def test_B13_unknown_controller_is_not_success(tmp_path):
    bundle=FakeBundle([{"success":False,"terminated":False}],[])
    class Unknown(bundle.executor.__class__):
        def execute(self,*a): 
            out=super().execute(*a); return out.__class__(out.execution_id,"UNKNOWN_CODE",out.start_seconds,out.end_seconds,out.evidence_ids)
    bundle.executor=Unknown(bundle)
    r=call(tmp_path,bundle)
    assert r["execution_status"]=="UNKNOWN_CONTROLLER_EXIT" and r["task_success"] is False

def test_B14_initialization_failure_is_traced(tmp_path):
    write_fixture(tmp_path)
    r=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b: (_ for _ in ()).throw(RuntimeError("boom")))
    assert r["error_phase"]=="bundle_create" and (tmp_path/"witnesses/branch_TEST-branch.jsonl").read_text()

def test_B15_restore_seed_is_paired_and_new_env_receives_it(tmp_path):
    events=[]
    b1=FakeBundle([{"success":True,"terminated":True}],[],events=events)
    r=call(tmp_path,b1)
    assert any(item[:2] == ("restore_seed_new_env",42) for item in events) and r["protocol_complete"]
