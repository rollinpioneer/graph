import json, multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import pytest

from cp_disr.analysis import s1_revision_resume as resume
from cp_disr.analysis.s1_integration import reserve_branch_attempt
from cp_disr.analysis.s1_revision import run_witnesses, execute_registered_branch
from tests.helpers.s1_runner_doubles import FakeBundle

def setup_fixture(path, branches, cap):
    path=Path(path); (path/"input_binding").mkdir(parents=True); (path/"witnesses").mkdir()
    (path/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").write_text(json.dumps({"runtime":{"active_task_id":"T_A","reference_skill_seconds_by_task":{"T_A":3.7},"task_deadlines":{"T_A":60.0}}}))
    (path/"witnesses/e4_branch_registration.json").write_text(json.dumps({"branches":branches}))
    (path/"budget_ledger.json").write_text(json.dumps({"physical_witness_episodes":{"used":0,"cap":cap}}))
    (path/"budget_events.jsonl").write_text("")

def branch(i="TEST-branch", case="TEST-case"):
    return {"branch_id":i,"case_id":case,"candidate_id":"a:A:x:v1","restore_seed":42,
            "authorized":True,"execute_now":True}

def _reserve_worker(path, queue):
    try:
        reserve_branch_attempt(path,"race")
        queue.put("PASS")
    except Exception as exc:
        queue.put(type(exc).__name__)

def test_last_reserved_slot_runs_through_batch_entry(tmp_path, monkeypatch):
    setup_fixture(tmp_path,[branch()],1)
    fake=FakeBundle([{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}],[])
    constructed=[]
    monkeypatch.setattr(resume,"load_runtime",lambda manifest: constructed.append(True) or fake)
    result=run_witnesses(tmp_path,tmp_path)
    assert len(constructed)==1
    assert fake.executor.executed_candidate_ids==["a:A:x:v1"]
    assert result["episodes"]==1
    result2=run_witnesses(tmp_path,tmp_path)
    assert len(constructed)==1
    assert result2["episodes"]==0

def test_eight_slots_and_ninth_duplicate_are_enforced(tmp_path, monkeypatch):
    branches=[branch(f"b{i}",f"case{i}") for i in range(8)]
    setup_fixture(tmp_path,branches,8)
    bundles=[FakeBundle([{"success":True,"terminated":True,"reason":"TASK_SUCCESS"}],[]) for _ in branches]
    monkeypatch.setattr(resume,"load_runtime",lambda manifest: bundles.pop(0))
    result=run_witnesses(tmp_path,tmp_path)
    assert result["episodes"]==8
    assert json.loads((tmp_path/"budget_ledger.json").read_text())["physical_witness_episodes"]["used"]==8
    assert run_witnesses(tmp_path,tmp_path)["episodes"]==0

def test_two_processes_compete_for_last_slot(tmp_path):
    setup_fixture(tmp_path,[branch("race","case")],1)
    queue=mp.get_context("fork").Queue()
    ps=[mp.get_context("fork").Process(target=_reserve_worker,args=(tmp_path,queue)) for _ in range(2)]
    [p.start() for p in ps]; [p.join(20) for p in ps]
    results=[queue.get(timeout=2) for _ in ps]
    assert results.count("PASS")==1
    assert json.loads((tmp_path/"budget_ledger.json").read_text())["physical_witness_episodes"]["used"]==1

def test_direct_entry_rejects_missing_ledger_before_factory(tmp_path):
    setup_fixture(tmp_path,[branch()],1)
    (tmp_path/"budget_ledger.json").unlink()
    called=[]
    result=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b: called.append(1))
    assert result["execution_status"]=="EXCEPTION" and called==[]

def test_forged_registration_hash_rejects_before_factory(tmp_path):
    setup_fixture(tmp_path,[branch()],1)
    reserve_branch_attempt(tmp_path,"TEST-branch")
    reg=json.loads((tmp_path/"witnesses/e4_branch_registration.json").read_text())
    reg["branches"][0]["candidate_id"]="a:B:x:v1"
    (tmp_path/"witnesses/e4_branch_registration.json").write_text(json.dumps(reg))
    called=[]
    result=execute_registered_branch(tmp_path,"TEST-branch",tmp_path,bundle_factory=lambda m,b: called.append(1))
    assert result["execution_status"]=="EXCEPTION" and called==[]


def test_reserved_receipt_claims_when_cap_is_full(tmp_path, monkeypatch):
    setup_fixture(tmp_path, [branch()], 1)
    reserve_branch_attempt(tmp_path, "TEST-branch")
    fake = FakeBundle([{"success": True, "terminated": True, "reason": "TASK_SUCCESS"}], [])
    called = []
    monkeypatch.setattr(resume, "load_runtime", lambda manifest: called.append(True) or fake)
    result = execute_registered_branch(tmp_path, "TEST-branch", tmp_path)
    assert result["execution_status"] == "TERMINATED" and called == [True]
    assert json.loads((tmp_path / "witnesses/branch_receipts/TEST-branch.json").read_text())["state"] == "COMPLETED"


def test_claim_event_is_written_before_execution(tmp_path, monkeypatch):
    setup_fixture(tmp_path, [branch()], 1)
    reserve_branch_attempt(tmp_path, "TEST-branch")
    fake = FakeBundle([{"success": True, "terminated": True, "reason": "TASK_SUCCESS"}], [])
    monkeypatch.setattr(resume, "load_runtime", lambda manifest: fake)
    execute_registered_branch(tmp_path, "TEST-branch", tmp_path)
    events = (tmp_path / "budget_events.jsonl").read_text().splitlines()
    assert any('"status": "STARTED"' in line for line in events)


def test_branch_authorization_is_required_before_factory(tmp_path):
    item = branch(); item["authorized"] = False
    setup_fixture(tmp_path, [item], 1)
    from cp_disr.analysis.s1_integration import IntegrationError
    try:
        reserve_branch_attempt(tmp_path, "TEST-branch")
    except IntegrationError as exc:
        assert str(exc) == "EXPLICIT_BRANCH_AUTHORIZATION_REQUIRED"
    else:
        assert False, "unauthorized branch was reserved"
