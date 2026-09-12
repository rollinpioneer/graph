from __future__ import annotations
import json
from pathlib import Path
from .approval import validate_approval
from .stage_grants import grants
from .state_machine import CampaignState, transition
from .baseline_stage import classify_attempt, preflight_or_block
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

    def preflight_and_dispatch(self, command: list[str], *, cwd: Path, grant: dict, output_root: Path) -> dict:
        """Recover setup failures without consuming scientific budget or reapproval."""
        preflight_result = preflight_or_block()
        if preflight_result["status"] != "PASS":
            return {**preflight_result, "state": "BLOCKED_PREPHYSICS_ENVIRONMENT", **classify_attempt(output_root)}
        from .baseline_stage import invoke_frozen_runner
        completed = invoke_frozen_runner(command, cwd=cwd, grant=grant)
        attempt = classify_attempt(output_root)
        return {"status": "PASS" if completed.returncode == 0 else "FAILED", "returncode": completed.returncode, **attempt}
