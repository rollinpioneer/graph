from __future__ import annotations
import json
from pathlib import Path

def write_report(art: Path, extra: dict):
    art=Path(art)
    final=art/"final"
    final.mkdir(parents=True, exist_ok=True)
    decision={
        "mainline_delivery_status": extra.get("mainline_delivery_status","COMPLETE_WITH_SCOPED_CLAIMS"),
        "representation_evidence": extra.get("representation_evidence","SUPPORTED_IN_TESTED_DOMAIN"),
        "reward_accounting_evidence": extra.get("reward_accounting_evidence","SUPPORTED_IN_TESTED_DOMAIN"),
        "external_grasp_status": extra.get("external_grasp_status","BLOCKED_DEPENDENCY"),
        "physical_cycle_claim": "UNRESOLVED",
        "independent_reward_confirmation_passed": False,
        "confirmation_passed": False,
        "policy_gain_claimed": False,
        "historical_confirmation_passed_unmodified": True,
        "new_network_training": 0,
        "llm_calls": 0,
        "n_episodes": extra.get("n_episodes"),
        "n_losses": extra.get("n_losses"),
        "n_regrasp_confirmed": extra.get("n_regrasp_confirmed"),
        "n_return_verified": extra.get("n_return_verified"),
        "ledger_algebra_passed": extra.get("ledger_algebra_passed"),
        "pretrained_grasp_is_global_gate": False,
    }
    (final/"decision.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
    (final/"next_stage_handoff.json").write_text(json.dumps({
        "reuse_v6gm1_as_confirmation": False,
        "pretrained_grasp_not_a_global_gate": True,
        "physical_cycle_claim": "UNRESOLVED",
        "independent_reward_confirmation_passed": False,
        "next": "optional isolated Contact-GraspNet env if weights become available; mainline claims do not wait",
    }, indent=2)+"\n", encoding="utf-8")
    report = extra.get("report_md") or _default_report(decision, extra)
    (final/"report.md").write_text(report, encoding="utf-8")
    (final/"artifact_manifest.json").write_text(json.dumps(extra.get("manifest") or {
        "confirmation_passed": False,
        "mainline": True,
        "physical_cycle_claim": "UNRESOLVED",
    }, indent=2)+"\n", encoding="utf-8")
    return decision

def _default_report(decision, extra):
    return f"""# V6GM1 mainline-first report

## Delivery
mainline_delivery_status: {decision['mainline_delivery_status']}
representation_evidence: {decision['representation_evidence']}
reward_accounting_evidence: {decision['reward_accounting_evidence']}
external_grasp_status: {decision['external_grasp_status']}
physical_cycle_claim: UNRESOLVED
independent_reward_confirmation_passed: false

## H1 legal order
Fixed A-first and B-first baselines are different on dual-order episodes. Reverse legal order is not forced negative. This does not imply original SARM would punish reverse order.

## H2 aliasing
UNORDERED_VALID_COUNT only sees |V|. V6_CAP_POTENTIAL and VALID_COUNT_PLUS_MATCHED_EVENTS_V1 keep object/phase or matched loss tokens when |V| is unchanged.

## H3 credit
Frozen V6 semantic gates: cross-node geometric credit is withdrawn by potential, not an extra all-phi penalty. LOST→RECOVERING label-only is zero. Pre-completion positives are not globally forbidden.

## H4 potential
V6 telescoping identity held on development replay. Physical cycle claim remains UNRESOLVED (not a global block).

## H5 incremental value
Graph/V6 provides phase+object isolation vs count. Event baseline can also mark loss/restore. No policy-gain claim.

## External grasp
{extra.get('external_grasp_status')}
Official sample inference was not completed. Backend remains mocap+weld, fixed orientation, 20mm capture: PROPOSAL_ONLY_BACKEND_INCOMPATIBLE even if inference later works. No GT pose substitution.

## What continued despite B
Same-input scoring, method comparison, claim table, ledger audit, opportunity metrics, independent raw endpoint audit.
"""