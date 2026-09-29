from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from cp_disr.adapters import ExecutionResult, EvaluationInput, TaskResult, Observation, Perception
from cp_disr.common import digest
from cp_disr.analysis.s1_integration import canonical_hash, reserve_branch_attempt
from cp_disr.contracts import Atom, Effects, SkillContract, TypedArgument
from cp_disr.facts import FactRecord, FactStore, Truth
from cp_disr.graph import GraphTemplate, Goal, Node
from cp_disr.rl import Snapshot
from cp_disr.baselines.b_plan import PlanResult

@dataclass
class FakeClock:
    value: float = 100.0
    def now_seconds(self): return self.value
    def advance(self, seconds): self.value += float(seconds)

class FakeEnvironment:
    def __init__(self, events):
        self.events=events; self.closed=False; self.generation_id=len([x for x in events if x[0]=="env_create"])+1
    def set_seed(self, seed):
        self.events.append(("old_or_current_env_set_seed", int(seed)))
    def close(self):
        self.closed=True; self.events.append(("env_close", self.generation_id))

class FakeObservations:
    def __init__(self, bundle): self.bundle=bundle
    def observe(self):
        self.bundle.events.append(("observe", self.bundle.env.generation_id))
        return object()

class FakePerception:
    def infer(self, observation): return object()

class FakeVerifier:
    def __init__(self,bundle): self.bundle=bundle
    def verify(self, measurement, execution=None):
        self.bundle.events.append(("verify", execution.execution_id if execution else None))
        return self.bundle.records_for_step()

class FakeSnapshotBuilder:
    def __init__(self,bundle,deadline): self.bundle=bundle; self.deadline=deadline
    def build(self, previous, records, observation, execution, clock_seconds):
        self.bundle.step += 1
        masks=self.bundle.masks[min(self.bundle.step,len(self.bundle.masks)-1)]
        return self.bundle.snapshot(masks, self.bundle.step, records, clock_seconds)

class FakeExecutor:
    def __init__(self,bundle): self.bundle=bundle; self.executed_candidate_ids=[]
    def execute(self,candidate_id,timeout_seconds):
        self.executed_candidate_ids.append(candidate_id)
        self.bundle.events.append(("execute",candidate_id))
        start=self.bundle.clock.now_seconds(); self.bundle.clock.advance(self.bundle.durations[self.bundle.step])
        end=self.bundle.clock.now_seconds()
        return ExecutionResult(f"exec-{len(self.executed_candidate_ids)}","NORMAL_TERMINATION",start,end,(f"e{len(self.executed_candidate_ids)}",))

class FakeEvaluator:
    def __init__(self,bundle,outcomes): self.bundle=bundle; self.outcomes=list(outcomes); self.inputs=[]
    def evaluate(self,value):
        self.inputs.append(value); self.bundle.events.append(("evaluate",len(self.inputs)))
        item=self.outcomes.pop(0) if self.outcomes else {"success":False,"terminated":False,"truncated":False,"reason":"CONTINUE"}
        return TaskResult(bool(item.get("success")),bool(item.get("terminated")),bool(item.get("truncated")),item.get("reason","CONTINUE"),())

class FakePlanner:
    def __init__(self, plans): self.plans=list(plans); self.inputs=[]
    def plan(self,facts,template,deadline):
        self.inputs.append((facts,template,deadline))
        item=self.plans.pop(0) if self.plans else {"status":"NO_PLAN","plan":[]}
        return PlanResult(item["status"],tuple(item.get("plan",())),0.0,1,1,0.001,0,1,1)

class FakeBundle:
    task_id="T_A"
    def __init__(self, outcomes, plans, masks=None, durations=None, deadline=60.0, events=None, start_offset=0.0):
        self.events=events if events is not None else []
        self.clock=FakeClock()
        self.env=FakeEnvironment(self.events)
        self.environment=self.env
        self.outcomes=outcomes
        self.plans=plans
        self.masks=masks or [[True]*9,[True]*9]
        self.durations=durations or [1.0]*16
        self.deadline=deadline
        self.start_offset=float(start_offset)
        self.step=0
        self.snapshot_builder=FakeSnapshotBuilder(self,deadline)
        self.observations=FakeObservations(self)
        self.perception=FakePerception()
        self.verifier=FakeVerifier(self)
        self.evaluator=FakeEvaluator(self,outcomes)
        self.executor=FakeExecutor(self)
        self.restore_verified=False
        self.restore_receipt=None
        self.snapshot_identity=None
        self.episode_start_seconds=self.clock.now_seconds()
        self.template=self._template()
        self.initial_records=self._records()
        self.planner=FakePlanner(plans)
    def _template(self):
        ready=Atom("Ready",("x",)); goal=Atom("Goal",("x",))
        cs=[]
        for name in ("A","B","C","D","E","F","G","H","I"):
            cs.append(SkillContract(name,(TypedArgument("x","object"),),(ready,),(),Effects(add=(goal,)),"v1","TEST",timeout_seconds=9.0,bound_arguments=("x",)))
        nodes=[Node("p:Ready:x","PROPOSITION","Ready",("x",),("object",)),Node("p:Goal:x","PROPOSITION","Goal",("x",),("object",))]
        nodes += [Node(c.id,"ACTION",c.name,c.bound_arguments,("object",)) for c in cs]
        return GraphTemplate(tuple(sorted(nodes,key=lambda n:n.id)),(),(Goal("p:Goal:x",1),),tuple(cs))
    def _records(self):
        return (FactRecord("p:Ready:x",Truth.TRUE,0,0,last_confirmed_value=Truth.TRUE,last_confirmed_time=0), FactRecord("p:Goal:x",Truth.FALSE,0,0,last_confirmed_value=Truth.FALSE,last_confirmed_time=0))
    def records_for_step(self):
        return self._records()
    def snapshot(self,mask,step,records,clock):
        return Snapshot("fake-env",f"episode-{self.env.generation_id}",step,self.template,FactStore(tuple(records)),tuple(c.id for c in self.template.contracts),tuple(mask),(),digest(()),(0.0,),(tuple(0.0 for _ in range(8)),)*len(self.template.contracts),f"frame-{step}",clock,synthetic_unit_fixture=True)
    def start_case(self,case_id,restore_seed):
        old=self.env; old.close()
        self.events.append(("env_create",case_id,int(restore_seed)))
        self.env=FakeEnvironment(self.events); self.environment=self.env
        self.events.append(("restore_seed_new_env",int(restore_seed),self.env.generation_id))
        self.episode_start_seconds=self.clock.now_seconds() - self.start_offset
        snap = self.snapshot(self.masks[0],0,self.initial_records,self.episode_start_seconds)
        state_hash = canonical_hash({"case_id": case_id, "seed": int(restore_seed),
                                     "facts": [r.fact_id for r in self.initial_records],
                                     "mask": list(snap.mask)})
        self.snapshot_identity = state_hash
        self.restore_receipt = {
            "requested_case_id": case_id, "applied_case_id": case_id,
            "requested_restore_seed": int(restore_seed), "applied_restore_seed": int(restore_seed),
            "normalized_reset_config_sha256": canonical_hash({"case_id": case_id}),
            "runtime_source_sha256": "fixture-runtime-source",
            "task_contract_identity": canonical_hash([c.id for c in self.template.contracts]),
            "state_identity_kind": "ENGINEERING_FIXTURE_ONLY",
            "state_identity_sha256": state_hash,
            "public_facts_sha256": canonical_hash([r.fact_id for r in self.initial_records]),
            "candidate_ids_sha256": canonical_hash(list(snap.candidate_ids)),
            "candidate_mask_sha256": canonical_hash(list(snap.mask)),
            "restore_checks": {"case_id": True, "seed": True, "facts": True, "candidate_ids": True, "mask": True},
        }
        self.restore_verified = True
        return snap

def write_fixture(output_dir, manifest=None, branch=None):
    output_dir=Path(output_dir); (output_dir/"input_binding").mkdir(parents=True,exist_ok=True); (output_dir/"witnesses").mkdir(parents=True,exist_ok=True)
    manifest=manifest or {"runtime":{"active_task_id":"T_A","reference_skill_seconds_by_task":{"T_A":3.7},"task_deadlines":{"T_A":60.0}}}
    (output_dir/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").write_text(json.dumps(manifest),encoding="utf-8")
    branch=branch or {"branch_id":"TEST-branch","case_id":"TEST-case","candidate_id":"a:A:x:v1","restore_seed":42,
                      "authorized": True, "execute_now": True}
    (output_dir/"witnesses/e4_branch_registration.json").write_text(json.dumps({
        "branches":[branch]}),encoding="utf-8")
    ledger = output_dir/"budget_ledger.json"
    if not ledger.exists():
        ledger.write_text(json.dumps({"physical_witness_episodes":{"used":0,"cap":1}}), encoding="utf-8")
    events = output_dir/"budget_events.jsonl"
    if not events.exists():
        events.write_text("", encoding="utf-8")
    if not (output_dir/"witnesses/branch_receipts/TEST-branch.json").exists():
        try:
            reserve_branch_attempt(output_dir, "TEST-branch")
        except Exception:
            pass
    return branch
