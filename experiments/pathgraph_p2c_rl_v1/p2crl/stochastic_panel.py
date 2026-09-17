"""Secondary stochastic panel. Same completeness gate as main test."""
from __future__ import annotations
from pathlib import Path
from .errors import IncompletePanel
from .io_utils import write_new
from .test_panel import formal_training_complete

def run_stochastic_panel(*, ledger, campaign_root, backend="fake"):
    ok, reason = formal_training_complete(ledger)
    if not ok:
        raise IncompletePanel(f"stochastic panel refused: {reason}")
    if backend != "real":
        raise IncompletePanel("stochastic panel refused: fake backend")
    out = Path(campaign_root) / "diagnostics" / "stochastic"
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "panel.json", {
        "schema": "P2CRL_STOCHASTIC_PANEL_V1",
        "expected_rows": 60 * 64 * 4,
        "sampling_seeds": [2026091802, 2026091803, 2026091804, 2026091805],
    })
    return out
