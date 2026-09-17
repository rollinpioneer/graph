"""Critical-state diagnostics. Oracle stays in the analysis process."""
from __future__ import annotations
from pathlib import Path
from .errors import IncompletePanel
from .io_utils import write_new
from .test_panel import formal_training_complete

def run_critical_state_panel(*, ledger, campaign_root, backend="fake"):
    ok, reason = formal_training_complete(ledger)
    if not ok:
        raise IncompletePanel(f"critical-state panel refused: {reason}")
    if backend != "real":
        raise IncompletePanel("critical-state panel refused: fake backend")
    out = Path(campaign_root) / "diagnostics" / "critical"
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "panel.json", {
        "schema": "P2CRL_CRITICAL_STATE_PANEL_V1",
        "policy_inputs": ["observation", "action_mask"],
        "oracle_process": "analysis_only",
    })
    return out
