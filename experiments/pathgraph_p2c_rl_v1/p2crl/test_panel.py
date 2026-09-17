"""Main test panel. Refuses unless 60/60 formal jobs are TRAINING_COMPLETE."""
from __future__ import annotations
from pathlib import Path
from .data_access import load_test_contracts
from .errors import IncompletePanel
from .io_utils import load_json, write_new
from .release import assert_no_test_payload

def formal_training_complete(ledger):
    jobs = ledger.jobs(kind="formal")
    if len(jobs) != 60:
        return False, f"formal count {len(jobs)}"
    for j in jobs:
        if j["status"] != "COMPLETE":
            return False, f"{j['job_id']} {j['status']}"
        if int(j.get("environment_steps") or 0) != 524288:
            return False, f"{j['job_id']} steps"
        complete = Path(j["directory"]) / "complete.json"
        rec = load_json(complete)
        if rec.get("status") != "TRAINING_COMPLETE":
            return False, f"{j['job_id']} not TRAINING_COMPLETE"
    return True, "ok"

def run_main_test(*, ledger, data_root, campaign_root, backend="fake", allow_incomplete=False):
    ok, reason = formal_training_complete(ledger)
    if not ok:
        raise IncompletePanel(f"main test refused: {reason}")
    if backend != "real":
        raise IncompletePanel("main test refused: fake backend cannot evaluate policies")
    contracts = load_test_contracts(data_root, allow=True)
    if len(contracts) != 512:
        raise IncompletePanel(f"expected 512 test contracts, got {len(contracts)}")
    out = Path(campaign_root) / "test" / "main"
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "panel.json", {
        "schema": "P2CRL_MAIN_TEST_PANEL_V1",
        "n_contracts": len(contracts),
        "n_jobs": 60,
        "expected_rows": 60 * 512,
    })
    return out
