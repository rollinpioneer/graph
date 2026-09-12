from __future__ import annotations
import math,statistics
from collections import Counter
from pathlib import Path
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json,write_csv,write_json
from . import CLP3_ID,METHODS

def evaluate(root:Path,output:Path):
 rows=[read_json(p) for p in sorted(root.glob("*/outcome.json"))]
 metrics=[]
 for m in METHODS:
  a=[r for r in rows if r["method"]==m]; durations=[x["duration_s"] for r in a for x in r["recovery_executions"] if x.get("duration_s") is not None]
  metrics.append({"method":m,"rollouts":len(a),"final_task_success":sum(r["final_task_success"] for r in a),"signal_not_observed":sum(r["signal_not_observed"] for r in a),"candidate_errors":sum(r["candidate_error"] for r in a),"false_executed_actions":sum(r["false_executed_action"] for r in a),"actual_recovery_success":sum(any(x["success"] for x in r["recovery_executions"]) for r in a),"recovery_cycles":sum(r["executed_recovery_cycles"] for r in a),"recovery_duration_median_s":statistics.median(durations) if durations else None,"failure_stages":dict(Counter(r["failure_stage"] for r in a if r["failure_stage"]))})
 c=next(x for x in metrics if x["method"]==CLP3_ID); cases=lambda prefix:[r for r in rows if r["method"]==CLP3_ID and r["case_id"].startswith(prefix)]
 gates={"rollouts_96":len(rows)==96,"clp3_signal_missing_4":sum(r["signal_not_observed"] for r in cases("R24C2_"))==4,"clp3_missing_not_candidate_error":sum(r["candidate_error"] for r in cases("R24C2_"))==0,"clp3_delayed_recovery_4":sum(r["final_task_success"] for r in cases("R24C1_"))==4,"clp3_partial_false_action_0":sum(r["false_executed_action"] for r in cases("R24C3_"))==0,"clp3_multisource_signal_missing_4":sum(r["signal_not_observed"] for r in cases("R24C4_"))==4,"clp3_multisource_not_candidate_error":sum(r["candidate_error"] for r in cases("R24C4_"))==0,"clp3_release_false_action_0":sum(r["false_executed_action"] for r in cases("R24C5_"))==0,"clp3_relocation_error_4":sum(r["failure_stage"]=="RELOCATION_ERROR" for r in cases("R24C6_"))==4,"clp3_regrasp_recovered_4":sum(r["final_task_success"] for r in cases("R24C7_"))==4,"clp3_secondary_recovered_4":sum(r["final_task_success"] for r in cases("R24C8_"))==4}
 result={"schema":"l2rar2_r24_evaluation_v1","status":"PASS" if all(gates.values()) else "FAIL","gates":gates,"metrics":metrics,"candidate_frozen":True,"parameter_search":False,"signal_not_observed_excluded_from_candidate_error":True}
 output.mkdir(parents=True,exist_ok=True);write_csv(output/"outcomes.csv",rows);write_csv(output/"method_metrics.csv",metrics);write_json(output/"summary.json",result);return result
