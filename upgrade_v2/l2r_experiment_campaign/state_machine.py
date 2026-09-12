from __future__ import annotations
from dataclasses import dataclass, field
from .protocol import STAGES
@dataclass
class CampaignState:
    state: str = "PREPARING"; current_stage: str|None = None; completed_stages: list[str]=field(default_factory=list); physical_budget_used: int=0; stop_reason: str|None=None
def transition(s: CampaignState, event: str) -> CampaignState:
    if s.state.startswith("STOPPED_") or s.state in {"COMPLETED","READY_WAITING_SINGLE_APPROVAL"}: return s
    if event == "ready": s.state="READY_WAITING_SINGLE_APPROVAL"
    elif event == "approved": s.state="RUNNING"; s.current_stage="A2"
    elif event == "stage_complete":
        if s.current_stage: s.completed_stages.append(s.current_stage)
        i=STAGES.index(s.current_stage)+1 if s.current_stage in STAGES else len(STAGES)
        s.current_stage=STAGES[i] if i<len(STAGES) else None; s.state="COMPLETED" if s.current_stage is None else "RUNNING"
    elif event.startswith("stop:"): s.state=event.split(":",1)[1]; s.stop_reason=s.state
    else: raise ValueError(event)
    return s
