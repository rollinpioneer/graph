"""Write final V6 report artifacts. confirmation_passed is not default-true."""
from __future__ import annotations
import json
from pathlib import Path
from .util import require_new, sha256_file, write_csv, write_json, write_text


def write_final(art: Path, extra: dict | None = None) -> dict:
    art = Path(art)
    final = art/"final"
    final.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    dev = json.loads((art/"02_development_v1/run_summary.json").read_text(encoding="utf-8"))
    gates = json.loads((art/"03_semantics_v1/semantic_gates.json").read_text(encoding="utf-8")) if (art/"03_semantics_v1/semantic_gates.json").is_file() else {}
    backend = {}
    if (art/"05_backend_v1/backend_receipt.json").is_file():
        backend = json.loads((art/"05_backend_v1/backend_receipt.json").read_text(encoding="utf-8"))
    per_q = {}
    if (art/"08_confirm_analysis_v1/per_question_decision.json").is_file():
        per_q = json.loads((art/"08_confirm_analysis_v1/per_question_decision.json").read_text(encoding="utf-8"))
    confirm_ran = bool(per_q)
    confirmation_passed = bool(per_q.get("confirmation_passed")) if confirm_ran else False
    backend_pending = backend.get("status") in (None, "BACKEND_NOT_IMPLEMENTED", "BACKEND_UNAVAILABLE")
    execution = "DONE"
    if backend_pending and not confirm_ran:
        execution = "DEVELOPMENT_DONE_PHYSICS_BACKEND_PENDING"
    elif confirm_ran and not confirmation_passed:
        execution = "CONFIRMATION_EXECUTED_NOT_PASSED"
    elif not confirm_ran and backend.get("status")=="PASS":
        execution = "ENGINEERING_OR_CONFIRMATION_INCOMPLETE"
    decision = {
        "execution_status": execution,
        "phi_accounting": per_q.get("Q1_phi_accounting") or ("PASS_DEVELOPMENT" if gates.get("phi_loss_clawback",{}).get("passed") else "NOT_EVALUATED"),
        "label_only_credit": per_q.get("Q2_label_only_credit") or ("PASS_DEVELOPMENT" if gates.get("label_neutrality",{}).get("passed") else "NOT_EVALUATED"),
        "old_reward_cycle_claim": per_q.get("Q3_old_reward_cycle_claim") or "NOT_EVALUATED_ON_NEW_PHYSICS",
        "new_reward_cycle_claim": per_q.get("Q3_new_reward_cycle_claim") or "NOT_EVALUATED_ON_NEW_PHYSICS",
        "development_method_ready": all(gates.get(k,{}).get("passed") is True for k in
            ("phi_loss_clawback","label_neutrality","genuine_progress_not_negative","loss_completion_event_pairing","baseline_fairness","raw_provenance","input_leakage")),
        "independent_verification_passed": bool(per_q.get("independent_verification_passed")),
        "confirmation_passed": confirmation_passed,
        "policy_gain_claimed": False,
        "real_robot_claimed": False,
        "visual_grounding_claimed": False,
        "new_simulations": extra.get("new_simulations", 0),
        "new_physical_robot_runs": 0,
        "new_training": 0,
        "llm_calls": 0,
        "v5r1_preserved": {
            "scientific_status": "CYCLE_CLAIM_NOT_ESTABLISHED",
            "exact_returns": 0, "approx_returns": 40, "object_losses": 112,
            "engine_parity_note": "35200/35248 passed, max_error=0, 48 cap-flag not all-PASS on original V5R1 scoring",
        },
        "v6_development_engine_parity": dev.get("legacy_engine_parity"),
        "backend": {"status": backend.get("status"), "backend_id": backend.get("backend_id"),
                    "mujoco_version": backend.get("mujoco_version")},
    }
    write_json(final/"decision.json", decision)
    rows = [
        {"claim": "V6_CAP_POTENTIAL is a new method not an in-place engine patch", "evidence": str(art/"04_frozen_candidate_v1/candidate_lock.json") if (art/"04_frozen_candidate_v1/candidate_lock.json").is_file() else "missing", "status": "DEVELOPMENT"},
        {"claim": "Q1 cross-node phi via potential, not extra subtract-all-phi", "evidence": str(art/"03_semantics_v1/q1_phi_clawback.csv"), "status": gates.get("phi_loss_clawback",{}).get("passed")},
        {"claim": "Q2 LOST->RECOVERING label-only credit is 0", "evidence": str(art/"03_semantics_v1/q2_label_neutrality.json"), "status": gates.get("label_neutrality",{}).get("passed")},
        {"claim": "Q3 measured loop on new physics", "evidence": str(art/"08_confirm_analysis_v1/per_question_decision.json") if confirm_ran else "not_run", "status": per_q.get("Q3_new_reward_cycle_claim","NOT_RUN")},
        {"claim": "G1 does not drive the controller", "evidence": "reward_shadow_only", "status": True},
        {"claim": "episode-positive FULL is not a cycle", "evidence": "full_episode_positive_not_used_as_cycle", "status": True},
        {"claim": "V5 B-tier physical exploit proven", "evidence": "explicitly_not_claimed", "status": False},
    ]
    write_csv(final/"claim_to_evidence.csv", rows)
    write_csv(final/"versions_and_failed_runs.csv", extra.get("failed_runs") or [{"note": "see collection_summary"}])
    write_csv(final/"external_artifacts.tsv", [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(art.rglob("*")) if p.is_file() and p.stat().st_size < 2_000_000][:400])
    files = {str(p.relative_to(art)): sha256_file(p) for p in sorted(art.rglob("*")) if p.is_file()}
    write_json(final/"result_manifest.json", {"n_files": len(files), "confirmation_passed": confirmation_passed, "execution_status": execution})
    report = f"""# PathGraph P1 V6 three-issue report

Execution status: {execution}
confirmation_passed: {confirmation_passed}
independent_verification_passed: {decision['independent_verification_passed']}

## Frozen facts
- Parent: 39b87fe234b4ac9765357f2243b02726dbbb2332
- V5R1: CYCLE_CLAIM_NOT_ESTABLISHED; 0 exact / 40 approx / 112 object-level losses (not 112 R6 cycles)
- V5R1 engine: 35200/35248 passed, max_error=0, 48 cap-flag remaining on original scoring
- V6 development live-engine comparisons: {dev.get('legacy_engine_parity')}

## Q1
New candidate uses Psi = -C_cap + G with r=Delta Psi. Failure across nodes withdraws transport credit because G becomes 0, not because an extra -phi term was patched onto the frozen engine.
Development clawback gate: {gates.get('phi_loss_clawback')}

## Q2
LOST and RECOVERING share capability units=5. Label-only change yields 0. Real rehold is 5->3 units.
Development label gate: {gates.get('label_neutrality')}
Confirmation label: {per_q.get('Q2_label_only_credit')}

## Q3
Old V5/V5R1 cycle claim is preserved as not established on simplified-state replay.
New physics claim: {per_q.get('Q3_new_reward_cycle_claim', 'NOT_RUN')}
Old physics-on-new-data claim: {per_q.get('Q3_old_reward_cycle_claim', 'NOT_RUN')}
Full-episode positive return is not treated as cycle farming.

## Backend
{json.dumps(backend, ensure_ascii=False, indent=2)}

## Non-claims
No visual grounding, no policy training, no LLM calls, no real robot, G1 does not drive control.
"""
    write_text(final/"report.md", report)
    return decision
