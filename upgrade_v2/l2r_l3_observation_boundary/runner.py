from __future__ import annotations
import concurrent.futures,json
from pathlib import Path
from typing import Any
import numpy as np

from upgrade_v2.l2r_canonical_time_confirmation.collector import CanonicalTimeCapture
from upgrade_v2.l2r_canonical_time_confirmation.guard import apply_guard
from upgrade_v2.l2r_l3_closed_loop.runner import L3Capture,L3ClosedLoopTabletop,_controlled_move,_execute_recovery,_finish_task
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json,write_jsonl
from upgrade_v2.l2r_logical_clock_confirmation.registry import ConfirmationCase
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import validate_online_row
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o
from upgrade_v2.l2r_rgb_temporal_confirmation.rgb_capture import _prehold
from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec
from upgrade_v2.visual_refine_l2.vision import detect_frame
from . import CLP3_ID,METHODS
from .registry import CASES,FAMILIES,Case

class R24Sim(L3ClosedLoopTabletop):
 def __init__(self,*a,fault=None,**kw): self.r24_fault=fault; self.disturbed=False; self.secondary=False; super().__init__(*a,**kw)
 def _advance(self,target=None,controls=4):
  capture=getattr(getattr(self,"after_physics_step",None),"__self__",None); phase=getattr(capture,"phase","")
  if self.r24_fault=="secondary_loss" and phase=="transport_to_target" and self.recovered and not self.secondary:
   self.secondary=True
   if self.attached:self._detach("contact_lost")
  if self.r24_fault=="relocation_error" and phase.endswith("_relocate") and target is not None: target=np.asarray(target)+np.array([.25,0,0])
  return super()._advance(target,controls)
 def physics_step(self):
  capture=getattr(getattr(self,"after_physics_step",None),"__self__",None); phase=getattr(capture,"phase","")
  if self.r24_fault=="regrasp_disturbance" and phase=="recovery_hold_verification" and not self.disturbed:
   self.disturbed=True
   if self.attached:self._detach("contact_lost")
  if self.r24_fault=="secondary_loss" and phase=="transport_to_target" and self.recovered and not self.secondary:
   self.secondary=True
   if self.attached:self._detach("contact_lost")
  return super().physics_step()

class R24Capture(L3Capture):
 def __init__(self,*a,r24_case:Case,**kw): self.r24_case=r24_case; self.physical_loss_started=False; self.loss_start_order=None; self.false_physical_captures=0; self.post_loss_captures=0; self.raw_trigger_observed=False; self.action_edges=[]; self._last_selected=None; super().__init__(*a,l3_case=type("C",(),{"contact_dropout":False})(),**kw)
 def capture_frame(self,*,action_end:bool):
  CanonicalTimeCapture.capture_frame(self,action_end=action_end); frame=self.frame_rows[-1]
  result=detect_frame(self.root/str(frame["jpeg_path"])) if not frame["frame_missing"] else {}
  det={"time":float(frame["time"]),"capture_order":int(frame["capture_order"]),"object_centroid":result.get("object_centroid"),"object_confidence":result.get("object_confidence",0.0),"gripper_centroid":result.get("gripper_centroid"),"gripper_confidence":result.get("gripper_confidence",0.0),"width":result.get("width",192),"height":result.get("height",144)}
  contact=dict(self.contact_rows[-1]); actual=bool(contact["contact_present"])
  if self.physical_loss_started:self.post_loss_captures+=1
  if self.physical_loss_started and not actual:self.false_physical_captures+=1
  mode=self.r24_case.signal_mode
  if self.physical_loss_started and mode=="delayed" and self.false_physical_captures<=2:contact["contact_present"]=True
  elif self.physical_loss_started and mode=="missing":contact["contact_present"]=True
  elif self.physical_loss_started and mode=="partial" and self.post_loss_captures==1:contact["contact_present"]=False
  elif self.physical_loss_started and mode=="multisource" and self.false_physical_captures<=2:
   contact["contact_present"]=None; det.update({"object_centroid":None,"gripper_centroid":None,"object_confidence":0.0,"gripper_confidence":0.0})
  merged={**det,**contact,**self.lifecycle_rows[-1],**self.command_rows[-1],**self.request_rows[-1]}; row=validate_online_row(merged); row["physical_time_ns"]=int(self.contact_rows[-1]["physical_time_ns"]); self.online_rows.append(row)
  base=[{k:v for k,v in x.items() if k!="physical_time_ns"} for x in self.online_rows]; raw=_run_o(base,"O_C3"); self.raw_trigger_observed=self.raw_trigger_observed or raw[-1].get("selected_action")=="recover_object"; records=apply_guard(self.online_rows,raw) if self.method in {CLP3_ID,"RECOVERY_DISABLED"} else raw; latest=records[-1]
  decision={"time":row["time"],"physical_time_ns":row["physical_time_ns"],"capture_order":row["capture_order"],"method":self.method,"selected_action":latest.get("selected_action"),"reason_code":latest.get("reason_code"),"guard_state":latest.get("guard_state")}; self.decision_rows.append(decision)
  action=decision["selected_action"]
  if action in {"retry_grasp","recover_object"} and action!=self._last_selected:
   self.action_edges.append(decision)
   if not self.action_execution_active and self.first_requested_action is None:self.first_requested_action=decision
  self._last_selected=action

def _adapter(case):return ConfirmationCase(case.case_id,commanded_release=case.release)

def _transport(capture:R24Capture,case:Case):
 sim=capture.sim; sim.action_index+=1; sim._active_control_sequence=[]; sim.lifecycle_before_action("transport_to_target"); capture.set_phase("transport_to_target","transport_to_target"); target=np.array([sim.spec.target_x,sim.spec.target_y,.80]); sim._advance((sim.data.mocap_pos[0]+target)/2,controls=20)
 if case.release:
  capture.physical_loss_started=True; capture.commanded_release=True; capture.perform("open_gripper","commanded_release_during_transport"); return
 capture.physical_loss_started=True; capture.loss_start_order=capture.capture_order; sim.disable_weld_for_intervention(); capture.add_event_reference("physical_contact_loss_started")
 if case.signal_mode=="partial":
  for _ in range(5):sim.physics_step()
  sim._attach(); capture.add_event_reference("partial_slip_contact_restored")
 else:
  sim.set_object_force(np.array([27.0,27.0,-5.4]))
  for _ in range(20):sim.physics_step()
  sim.clear_object_force()
  for _ in range(100):
   sim.physics_step()
   if capture.first_requested_action is not None:break
 sim.lifecycle_after_action("transport_to_target"); capture.capture_frame(action_end=True)

def run_rollout(root:Path,family:str,fseed:int,rseed:int,case:Case,method:str):
 root.mkdir(parents=True,exist_ok=False); sim=R24Sim(family_spec(family,"normal_pick_place",fseed,rseed),rseed,fault=case.controller_fault); cap=R24Capture(sim,root,family_id=family,family_seed=fseed,rollout_seed=rseed,case_id=case.case_id,requested_effect="RELEASE_OBJECT" if case.release else "HOLD_OBJECT",case=_adapter(case),method=method,r24_case=case)
 try:
  pre=_prehold(cap,50)
  if not pre["pre_hold_verified"]:raise RuntimeError("PREHOLD_UNVERIFIED")
  _transport(cap,case); contact_proxy_observed=any(r.get("contact_present") is False and int(r["capture_order"])>=int(cap.loss_start_order or 10**9) for r in cap.online_rows); signal_observed=cap.raw_trigger_observed; requested=cap.first_requested_action; executions=[]
  if requested and method!="RECOVERY_DISABLED":
   executions.append(_execute_recovery(cap,requested["selected_action"]))
   if case.controller_fault=="secondary_loss":
    later=next((x for x in cap.action_edges if x["physical_time_ns"]>requested["physical_time_ns"]),None)
    if later:executions.append(_execute_recovery(cap,later["selected_action"]))
  elif not case.recovery_required and not case.release:_finish_task(cap)
  final_success=bool(case.release and requested is None) if case.release else bool(sim.oracle_snapshot()["goal_stable"])
  false_action=not case.recovery_required and not case.release and requested is not None and method!="RECOVERY_DISABLED"
  if case.recovery_required and not signal_observed:failure="SIGNAL_NOT_OBSERVED"
  elif case.recovery_required and requested is None:failure="TRIGGERING_ERROR"
  elif executions and not executions[-1]["success"]:failure=executions[-1]["failure_stage"]
  elif false_action:failure="TRIGGERING_ERROR"
  elif case.recovery_required and executions and executions[-1]["success"] and not final_success:failure="TASK_RECOVERY_ERROR"
  else:failure=None
  outcome={"schema":"l2rar2_r24_outcome_v1","family_id":family,"family_seed":fseed,"rollout_seed":rseed,"case_id":case.case_id,"method":method,"recovery_required":case.recovery_required,"physical_loss_exists":not case.release,"contact_proxy_observed":contact_proxy_observed,"signal_observed":signal_observed,"candidate_requested_action":requested["selected_action"] if requested else None,"executed_recovery_cycles":len(executions),"recovery_executions":executions,"final_task_success":final_success,"false_executed_action":false_action,"failure_stage":failure,"candidate_error":failure=="TRIGGERING_ERROR","signal_not_observed":failure=="SIGNAL_NOT_OBSERVED"}
  cap.write_l3(); write_json(root/"outcome.json",outcome); write_json(root/"metadata.json",{k:outcome[k] for k in ("family_id","family_seed","rollout_seed","case_id","method")}); write_json(root/"termination.json",{"status":"COMPLETE","time":float(sim.data.time)}); return outcome
 finally:cap.close()

def _task(a):return run_rollout(Path(a[0]),*a[1:])
def collect(output:Path,workers=2):
 output.mkdir(parents=True,exist_ok=True); tasks=[]
 for f,s,b in FAMILIES:
  for i,c in enumerate(CASES):
   for m in METHODS:
    r=output/f"{f}__{c.case_id}__{m}"
    if not r.exists():tasks.append((str(r),f,s,b+i,c,m))
 done=[];fail=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as p:
  futures={p.submit(_task,a):a for a in tasks}
  for x,a in list(futures.items()):
   try:done.append(x.result())
   except Exception as e:fail.append({"rollout":Path(a[0]).name,"error":f"{type(e).__name__}:{e}"})
 result={"schema":"l2rar2_r24_collection_v1","status":"PASS" if len(list(output.glob("*/outcome.json")))==96 and not fail else "FAIL","rollouts":len(list(output.glob("*/outcome.json"))),"failures":fail};write_json(output/"collection_status.json",result);return result
