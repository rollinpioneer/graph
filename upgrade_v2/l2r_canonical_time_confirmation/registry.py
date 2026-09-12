from __future__ import annotations

from upgrade_v2.l2r_logical_clock_confirmation.registry import CASES as R20_CASES

CASE_INDEXES=(0,2,3,6,10,11)
CASES=tuple(R20_CASES[index] for index in CASE_INDEXES)
FAMILIES=tuple((f"L2RAR2_R22_CANON_{index:02d}_{890000+index}",890000+index,89100000+index*100) for index in range(4))


def registry()->dict:
    return {"schema":"l2rar2_r22_registry_v1","families":[{"family_id":f,"family_seed":s,"rollout_seed_base":b} for f,s,b in FAMILIES],"case_order":[c.case_id for c in CASES],"rollouts":24,"replacement_allowed":False,"rerun_allowed":False}
