"""Campaign orchestration. NOT_ACTIVE never constructs models or calls learn()."""
from __future__ import annotations
from pathlib import Path
from .campaign_ledger import CampaignLedger
from .constants_b import CAMPAIGN_ID, PARALLEL_JOBS_MAX
from .errors import GateError, IncompletePanel
from .io_utils import load_json, sha256_file, write_new
from .release import ReleaseRejected, validate_release
from .train_job import execute_job

def _reject(campaign_root, code, detail):
    if campaign_root:
        root = Path(campaign_root)
        log = root / "control" / "reject.json"
        if not log.exists():
            write_new(log, {"schema": "P2CRL_REJECT_V1", "code": code, "detail": str(detail)})
    raise ReleaseRejected(code, detail)

def smoke_gate(ledger):
    jobs = ledger.jobs(kind="smoke")
    if len(jobs) != 5:
        raise GateError("smoke count")
    hashes = []
    for j in jobs:
        if j["status"] != "COMPLETE":
            raise GateError(f"smoke {j['job_id']} {j['status']}")
        rec = load_json(Path(j["directory"]) / "complete.json")
        if rec.get("status") != "TRAINING_COMPLETE":
            raise GateError(f"smoke {j['job_id']} not trained")
        if int(rec.get("environment_steps") or 0) != 512:
            raise GateError(f"smoke {j['job_id']} steps")
        if int(rec.get("invalid_selected") or 0) != 0 or int(rec.get("nonfinite") or 0) != 0:
            raise GateError(f"smoke {j['job_id']} invalid/nonfinite")
        init = load_json(Path(j["directory"]) / "initialization.json")
        hashes.append(init.get("policy_sha256"))
    hashes = [h for h in hashes if h]
    if hashes and len(set(hashes)) != 1:
        raise GateError("smoke initial policy hash mismatch")
    return True

def complete_panel_gate(ledger):
    jobs = ledger.jobs(kind="formal")
    if len(jobs) != 60:
        raise GateError("formal count")
    for j in jobs:
        if j["status"] != "COMPLETE":
            raise GateError(f"formal {j['job_id']} {j['status']}")
        rec = load_json(Path(j["directory"]) / "complete.json")
        if rec.get("status") != "TRAINING_COMPLETE" or int(rec.get("environment_steps") or 0) != 524288:
            raise GateError(f"formal {j['job_id']} incomplete")
    return True

def _plan_jobs(plan):
    return list(plan.get("smoke_jobs") or []) + list(plan.get("formal_jobs") or [])

def execute_campaign(
    *,
    repo=None,
    release,
    plan,
    data_root=None,
    campaign_root,
    protocol_path=None,
    source_lock_path=None,
    dataset_manifest_path=None,
    amendment_path=None,
    parallel_jobs=2,
    backend="real",
    zero_gradient_drill=False,
    require_clean=False,
    require_detached=False,
):
    campaign_root = Path(campaign_root)
    control = campaign_root / "control"
    try:
        validate_release(
            release,
            repo=repo,
            protocol_path=protocol_path,
            plan_path=None,
            source_lock_path=source_lock_path,
            dataset_manifest_path=dataset_manifest_path,
            amendment_path=amendment_path,
            require_active=True,
            require_clean=require_clean,
            require_detached=require_detached,
        )
    except ReleaseRejected as e:
        _reject(campaign_root, e.code, e.detail)

    if zero_gradient_drill:
        backend = "fake"
    if backend == "real" and zero_gradient_drill:
        _reject(campaign_root, "FAKE_BACKEND_REQUIRED", "drill")
    if int(parallel_jobs) not in (1, PARALLEL_JOBS_MAX):
        _reject(campaign_root, "PARALLEL_JOBS", str(parallel_jobs))

    jobs = _plan_jobs(plan)
    control.mkdir(parents=True, exist_ok=True)
    ledger_path = control / "campaign.sqlite3"
    ledger = CampaignLedger(ledger_path)
    release_sha = sha256_file(control / "release.used.json") if (control / "release.used.json").exists() else "fixture"
    if not (control / "release.used.json").exists():
        write_new(control / "release.used.json", release)
        release_sha = sha256_file(control / "release.used.json")
    ledger.init_campaign(release_sha, jobs)

    if backend == "fake" or zero_gradient_drill:
        smoke_n = formal_n = 0
        learn_called = False
        for job in plan.get("smoke_jobs") or []:
            execute_job(
                repo=None,
                release=release,
                plan_job=job,
                campaign_root=campaign_root,
                ledger=ledger,
                backend="fake",
                register_only=True,
            )
            smoke_n += 1
        for job in plan.get("formal_jobs") or []:
            execute_job(
                repo=None,
                release=release,
                plan_job=job,
                campaign_root=campaign_root,
                ledger=ledger,
                backend="fake",
                register_only=True,
            )
            formal_n += 1
        test_refused = False
        test_reason = None
        try:
            from .test_panel import run_main_test
            run_main_test(ledger=ledger, data_root=data_root, campaign_root=campaign_root, backend="fake")
        except IncompletePanel as e:
            test_refused = True
            test_reason = str(e)
        camp = ledger.campaign()
        report = {
            "schema": "P2CRL_ZERO_GRADIENT_DRILL_V1",
            "campaign_id": CAMPAIGN_ID,
            "backend": "fake",
            "learn_called": bool(learn_called),
            "gradient_updates": 0,
            "formal_steps": int(camp.get("formal_steps_used") or 0),
            "smoke_steps": 0,
            "smoke_registered": smoke_n,
            "formal_registered": formal_n,
            "test_refused": test_refused,
            "test_reason": test_reason,
            "models_constructed": True,
        }
        write_new(control / "zero_gradient_drill.json", report)
        ledger.close()
        return report

    # Real execution path. Not used in B0.
    for job in plan.get("smoke_jobs") or []:
        try:
            execute_job(
                repo=repo,
                data_root=data_root,
                release=release,
                plan_job=job,
                protocol_path=protocol_path,
                source_lock_path=source_lock_path,
                dataset_manifest_path=dataset_manifest_path,
                amendment_path=amendment_path,
                campaign_root=campaign_root,
                ledger=ledger,
                backend="real",
                register_only=False,
            )
        except Exception as e:
            ledger.mark_failed(job["job_id"], str(e))
            ledger.close()
            raise GateError(f"SMOKE_FAILED_STOP: {e}")
    smoke_gate(ledger)
    for job in plan.get("formal_jobs") or []:
        try:
            execute_job(
                repo=repo,
                data_root=data_root,
                release=release,
                plan_job=job,
                protocol_path=protocol_path,
                source_lock_path=source_lock_path,
                dataset_manifest_path=dataset_manifest_path,
                amendment_path=amendment_path,
                campaign_root=campaign_root,
                ledger=ledger,
                backend="real",
                register_only=False,
            )
        except Exception as e:
            ledger.mark_failed(job["job_id"], str(e))
            ledger.close()
            raise
    complete_panel_gate(ledger)
    from .test_panel import run_main_test
    from .stochastic_panel import run_stochastic_panel
    from .critical_state_panel import run_critical_state_panel
    from .analysis import confirmatory_statistics
    run_main_test(ledger=ledger, data_root=data_root, campaign_root=campaign_root, backend="real")
    run_stochastic_panel(ledger=ledger, campaign_root=campaign_root, backend="real")
    run_critical_state_panel(ledger=ledger, campaign_root=campaign_root, backend="real")
    confirmatory_statistics([], {"formal_jobs": 60, "validation_panels": 360, "main_rows": 30720, "stochastic_rows": 15360}, campaign_root / "analysis" / "primary_comparisons.json")
    ledger.close()
    return {"schema": "P2CRL_CAMPAIGN_RESULT_V1", "status": "EXECUTED"}
