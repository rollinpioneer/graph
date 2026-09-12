from __future__ import annotations

import concurrent.futures, json
from pathlib import Path

import upgrade_v2.l2r_logical_clock_confirmation.collector as r20
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json
from .clock import PHYSICS_STEP_NS
from .registry import CASES,FAMILIES


class CanonicalTimeCapture(r20.R20Capture):
    def capture_frame(self,*,action_end:bool)->None:
        super().capture_frame(action_end=action_end)
        ns=int(self.physics_step_index)*PHYSICS_STEP_NS
        for rows in (self.frame_rows,self.contact_rows,self.lifecycle_rows,self.command_rows,self.request_rows,self.logical_rows):
            rows[-1]["physical_time_ns"]=ns


def _task(args):
    family,family_seed,rollout_seed,case,output=args
    original=r20.R20Capture; r20.R20Capture=CanonicalTimeCapture
    try: return r20.collect_rollout(Path(output),family,family_seed,rollout_seed,case)
    finally: r20.R20Capture=original


def collect(output:Path,workers:int=2)->dict:
    if workers not in (1,2): raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output.mkdir(parents=True,exist_ok=True); tasks=[]
    for family,fseed,base in FAMILIES:
        for index,case in enumerate(CASES):
            root=output/f"{family}__{case.case_id}"
            if (root/"termination.json").is_file() and json.loads((root/"termination.json").read_text())["status"]=="COMPLETE": continue
            if root.exists(): raise RuntimeError(f"INCOMPLETE_ROLLOUT_PRESENT:{root}")
            tasks.append((family,fseed,base+index,case,str(root)))
    completed=[]; failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending={}; iterator=iter(tasks); stop=False
        for _ in range(min(workers,len(tasks))):
            task=next(iterator,None)
            if task: pending[pool.submit(_task,task)]=task
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future)
                try: completed.append(future.result())
                except Exception as exc: failures.append({"rollout":Path(task[-1]).name,"error":f"{type(exc).__name__}: {exc}"}); stop=True
                if not stop:
                    nxt=next(iterator,None)
                    if nxt: pending[pool.submit(_task,nxt)]=nxt
    terms=list(output.glob("*/termination.json")); passed=len(terms)==24 and all(json.loads(p.read_text())["status"]=="COMPLETE" for p in terms)
    result={"schema":"l2rar2_r22_collection_v1","status":"PASS" if passed else "FAIL","all_24_complete":passed,"newly_completed":len(completed),"failures":failures}
    write_json(output/"collection_status.json",result); return result
