from __future__ import annotations

import subprocess, sys
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json
from upgrade_v2.l2r_logical_observation_clock.guard import INTERCEPT_REASON
from .guard import CanonicalTimeGuard


def _row(ns:int,order:int,contact:bool,**extra:Any)->dict:
    return {"time":ns/1e9,"physical_time_ns":ns,"capture_order":order,"attempt_id":1,"contact_present":contact,
            "gripper_command":"closed","requested_effect":"HOLD_OBJECT","context_valid":True,"attempt_end":False,"attempt_end_reason":None,**extra}


def _proposal(trigger=False)->dict:
    return {"selected_action":"recover_object" if trigger else "none","reason_code":INTERCEPT_REASON if trigger else "no_action","hold_state":"historical_hold_established"}


def run(repo:Path,output:Path)->dict:
    cases=[]
    def check(name,rows,proposals,expected):
        guard=CanonicalTimeGuard(); result=[guard.step(row,p) for row,p in zip(rows,proposals)]; actual=(result[-1]["guard_state"],result[-1]["selected_action"])
        cases.append({"case_id":name,"passed":actual==expected,"expected":list(expected),"actual":list(actual),"records":result})
    check("C01_same_time_false_false_stays_pending",[_row(1_000_000_000,10,False),_row(1_000_000_000,11,False)],[_proposal(True),_proposal()],("PENDING","none"))
    check("C02_same_time_then_one_ns_confirms",[_row(1_000_000_000,10,False),_row(1_000_000_000,11,False),_row(1_000_000_001,12,False)],[_proposal(True),_proposal(),_proposal()],("CONFIRMED","recover_object"))
    check("C03_same_time_false_true_clears",[_row(1_000_000_000,10,False),_row(1_000_000_000,11,True)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C04_one_ns_confirms",[_row(1_000_000_000,10,False),_row(1_000_000_001,11,False)],[_proposal(True),_proposal()],("CONFIRMED","recover_object"))
    check("C05_exact_100ms_confirms",[_row(1_000_000_000,10,False),_row(1_100_000_000,11,False)],[_proposal(True),_proposal()],("CONFIRMED","recover_object"))
    check("C06_100ms_plus_one_ns_clears",[_row(1_000_000_000,10,False),_row(1_100_000_001,11,False)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C07_open_clears",[_row(1_000_000_000,10,False),_row(1_050_000_000,11,False,gripper_command="open")],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C08_release_request_clears",[_row(1_000_000_000,10,False),_row(1_050_000_000,11,False,requested_effect="RELEASE_OBJECT")],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C09_attempt_end_clears",[_row(1_000_000_000,10,False),_row(1_050_000_000,11,False,attempt_end=True)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C10_attempt_change_clears",[_row(1_000_000_000,10,False),_row(1_050_000_000,11,False,attempt_id=2)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C11_equal_order_invalid",[_row(1_000_000_000,10,False),_row(1_050_000_000,10,False)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C12_backwards_time_invalid",[_row(1_000_000_000,10,False),_row(999_999_999,11,False)],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C13_missing_ns_invalid",[_row(1_000_000_000,10,False),{**_row(1_050_000_000,11,False),"physical_time_ns":None}],[_proposal(True),_proposal()],("CLEAR","none"))
    check("C14_nontrigger_no_pending",[_row(1_000_000_000,10,False),_row(1_000_000_000,11,False)],[_proposal(),_proposal()],("PASS_THROUGH","none"))
    source="upgrade_v2/l2r_canonical_time_confirmation/guard.py"; blob=subprocess.run(["git","-C",str(repo),"hash-object",source],capture_output=True,text=True,check=True).stdout.strip()
    result={"schema":"l2rar2_clp3_canonical_time_conformance_v1","status":"PASS" if all(c["passed"] for c in cases) else "FAIL","cases_passed":sum(c["passed"] for c in cases),"cases_total":len(cases),"guard_blob":blob,"mujoco_imported": "mujoco" in sys.modules,"physical_executions":0,"cases":cases}
    write_json(output,result); return result
