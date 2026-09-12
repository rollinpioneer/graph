from __future__ import annotations
import json
from pathlib import Path
from .approval import validate_approval
from .stage_grants import grants
from .state_machine import CampaignState, transition
class CampaignOrchestrator:
    def __init__(self, root: Path, lock: dict): self.root,self.lock,self.state=root,lock,CampaignState()
    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True); transition(self.state,"ready")
        (self.root/"campaign_state.json").write_text(json.dumps({"schema":"l2rar2_fast_campaign_state_v1","state":self.state.state,"physical_budget_used":0},indent=2)+"\n")
        return self.state
    def authorize(self, approval: dict):
        errors=validate_approval(approval,self.lock)
        if errors: raise PermissionError("approval rejected: "+", ".join(errors))
        transition(self.state,"approved"); return grants(approval)
