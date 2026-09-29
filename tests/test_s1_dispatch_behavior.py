import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from cp_disr.analysis.s1_revision import reserve_budget, BudgetExceeded, execute_registered_branch
from tests.helpers.s1_runner_doubles import FakeBundle, write_fixture

def ledger(p,used=0,cap=1):
    (p/"budget_ledger.json").write_text(json.dumps({"physical_witness_episodes":{"used":used,"cap":cap}}))
    (p/"budget_events.jsonl").write_text("")

def test_B18_budget_exhausted_before_direct_runtime(tmp_path):
    ledger(tmp_path,8,8); called=[]
    try: reserve_budget(tmp_path,"physical_witness_episodes",1,"x")
    except BudgetExceeded: pass
    else: assert False
    write_fixture(tmp_path)
    b=FakeBundle([{"success":True,"terminated":True}],[])
    r=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,x: called.append(x))
    assert r["error_type"]=="BudgetExceeded" and called==[]

def test_B19_concurrent_reserve_has_one_winner(tmp_path):
    ledger(tmp_path,0,1)
    def one():
        try: reserve_budget(tmp_path,"physical_witness_episodes",1,"id-"+str(object())); return True
        except BudgetExceeded: return False
    with ThreadPoolExecutor(max_workers=2) as ex: got=list(ex.map(lambda _:one(),range(2)))
    assert sum(got)==1

def test_B20_duplicate_attempt_is_rejected_before_runtime(tmp_path):
    ledger(tmp_path,0,1)
    write_fixture(tmp_path)
    (tmp_path/"attempt_registry.json").write_text(json.dumps({"TEST-branch":"STARTED"}))
    called=[]
    r=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,x: called.append(x))
    assert r["error_type"]=="RevisionError" and called==[]
