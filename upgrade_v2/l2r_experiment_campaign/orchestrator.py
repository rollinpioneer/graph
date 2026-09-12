from __future__ import annotations
import json
from pathlib import Path
from .approval import validate_approval
from .stage_grants import grants
from .state_machine import CampaignState, transition
from .baseline_stage import classify_attempt, preflight_or_block
from .protocol import RECOVERABLE_PREPHYSICS
from .protocol import BUDGET
class CampaignOrchestrator:
    def __init__(self, root: Path, lock: dict): self.root,self.lock,self.state=root,lock,CampaignState()
    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True); transition(self.state,"ready")
        self._persist()
        return self.state
    def _persist(self):
        payload={"schema":"l2rar2_fast_campaign_state_v1.1","state":self.state.state,
                 "current_stage":self.state.current_stage,"completed_stages":self.state.completed_stages,
                 "grant_attempts":self.state.grant_attempts,"prephysics_failures":self.state.prephysics_failures,
                 "physical_budget_used":self.state.physical_budget_used,"physical_budget_remaining":sum(BUDGET.values())-self.state.physical_budget_used,
                 "stop_reason":self.state.stop_reason}
        (self.root/"campaign_state.json").write_text(json.dumps(payload,indent=2)+"\n")
    def authorize(self, approval: dict):
        errors=validate_approval(approval,self.lock)
        if errors: raise PermissionError("approval rejected: "+", ".join(errors))
        transition(self.state,"approved"); self.approval=approval; self._persist(); return grants(approval)

    def preflight_and_dispatch(self, command: list[str], *, cwd: Path, grant: dict, output_root: Path) -> dict:
        """Recover setup failures without consuming scientific budget or reapproval."""
        preflight_result = preflight_or_block()
        if preflight_result["status"] != "PASS":
            transition(self.state, "attempt_prephysics_failed")
            transition(self.state, "block:" + preflight_result["status"])
            self._persist()
            return {**preflight_result, "state": preflight_result["status"], "retryable": True, **classify_attempt(output_root)}
        from .baseline_stage import invoke_frozen_runner
        completed = invoke_frozen_runner(command, cwd=cwd, grant=grant)
        attempt = classify_attempt(output_root)
        if attempt["physical_instance_started"]:
            transition(self.state, "attempt_physics_started")
            if completed.returncode != 0:
                transition(self.state, "stop:STOPPED_A2_FAILED" if self.state.current_stage == "A2" else "stop:STOPPED_INTERNAL_ERROR")
        else:
            transition(self.state, "attempt_prephysics_failed")
            transition(self.state, "block:BLOCKED_PREPHYSICS_VALIDATION")
        self._persist()
        retryable = not attempt["physical_instance_started"]
        return {"status": "PASS" if completed.returncode == 0 else "FAILED", "returncode": completed.returncode, "retryable": retryable, **attempt}

    def recover_prephysics(self, approval: dict, *, retry_index: int, output_root_factory):
        """Resume the same stage after an engineering/setup block.

        A fresh nonce and output root are derived; no second human approval is consumed.
        """
        if self.state.state not in RECOVERABLE_PREPHYSICS:
            raise ValueError("campaign is not in a recoverable pre-physics state")
        if getattr(self, "approval", approval).get("campaign_nonce") != approval.get("campaign_nonce"):
            raise PermissionError("approval identity changed")
        transition(self.state, "recover_prephysics"); self._persist()
        fresh = grants(approval, retry_index=retry_index)
        stage = self.state.current_stage or "A2"
        grant = next(g for g in fresh if g["stage"] == stage)
        output_root = Path(output_root_factory(grant["single_use_nonce"]))
        output_root.mkdir(parents=True, exist_ok=True)
        return grant, output_root
