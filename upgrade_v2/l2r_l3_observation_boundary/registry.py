from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Case:
    case_id: str
    signal_mode: str
    controller_fault: str | None = None
    release: bool = False
    recovery_required: bool = True

CASES = (
    Case("R24C1_delayed_contact_loss", "delayed"),
    Case("R24C2_loss_signal_missing", "missing"),
    Case("R24C3_partial_slip", "partial", recovery_required=False),
    Case("R24C4_multisource_missing_then_loss", "multisource"),
    Case("R24C5_active_release_conflict", "release", release=True, recovery_required=False),
    Case("R24C6_relocation_error", "normal", "relocation_error"),
    Case("R24C7_regrasp_disturbance", "normal", "regrasp_disturbance"),
    Case("R24C8_secondary_loss", "normal", "secondary_loss"),
)
FAMILIES = tuple((f"L2RAR2_R24_BOUNDARY_{i:02d}_{896000+i}",896000+i,89700000+i*100) for i in range(4))

def registry():
    return {"schema":"l2rar2_r24_registry_v1","families":[{"family_id":f,"family_seed":s,"rollout_seed_base":b} for f,s,b in FAMILIES],"cases":[c.__dict__ for c in CASES],"methods":["O_C3_CLP3_CANONICAL_TIME","O_C3_RAW","RECOVERY_DISABLED"],"paired_groups":32,"physical_rollouts":96,"parameter_search":False,"seed_replacement":False}

