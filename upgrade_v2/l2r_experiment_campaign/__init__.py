"""Bounded, auditable orchestration for the L2RAR2 fast campaign."""
from .protocol import CAMPAIGN_ID, BUDGET, STAGES
from .approval import validate_approval
from .state_machine import CampaignState, transition

__all__ = ["CAMPAIGN_ID", "BUDGET", "STAGES", "validate_approval", "CampaignState", "transition"]
