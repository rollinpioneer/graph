# V6GM1 mainline-first report

## Delivery
mainline_delivery_status: COMPLETE_WITH_SCOPED_CLAIMS
representation_evidence: SUPPORTED_IN_TESTED_DOMAIN
reward_accounting_evidence: SUPPORTED_IN_TESTED_DOMAIN
external_grasp_status: BLOCKED_DEPENDENCY
physical_cycle_claim: UNRESOLVED
independent_reward_confirmation_passed: false
confirmation_passed: false
pretrained_grasp_is_global_gate: false

## Same-input scoring
- episodes: 112 (dual_order=48, recovery=64)
- evidence_tier: EXISTING_SIMPLIFIED_STATE_SIMULATION_REPLAY
- states: 17984
- V6_CAP_POTENTIAL ledger algebra: 112/112 [('PASS_ALGEBRA_ONLY', 112)]
- max telescoping_error: 2.220446049250313e-16
- methods scored on every adjacent transition, no per-episode method picking

## H1 legal order
LINEAR_A_FIRST_R1 vs LINEAR_B_FIRST_R1 differ on 8/48 dual-order episodes.
Differing IDs: P1V5_F00_920000__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F01_920001__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F02_920002__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F03_920003__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F04_920004__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F05_920005__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F06_920006__D6_repeat_completed_subgoal_then_terminal_failure, P1V5_F07_920007__D6_repeat_completed_subgoal_then_terminal_failure
Clean A-then-B and B-then-A successes both receive the same signed return on both linear baselines; reverse legal order is not forced negative. This does not imply original SARM would punish reverse order.

## H2 aliasing
Episodes where the same valid-count appears with more than one object-phase tuple: 112/112.
UNORDERED_VALID_COUNT only sees |V|. V6_CAP_POTENTIAL keeps object phase; VALID_COUNT_PLUS_MATCHED_EVENTS_V1 keeps matched loss/restore tokens. This does not imply that no non-graph method can represent the same state.

## H3 credit
Frozen V6 semantic gates are used as-is: cross-node geometric credit is withdrawn by potential, not by an extra all-phi penalty. LOST->RECOVERING label-only is zero. Pre-completion positives are not globally forbidden. Old vs new backends remain layered; V6/V6RC1/V6RG1 paths are unmodified versus fdc5acd1bbec7906fb1b2dffb091d2c82979d033.

## H4 potential
V6 telescoping identity held on this development replay (algebra-only). Observed-task returns: 40 of 112 losses (104 regrasp confirmed). Independent raw pose/velocity measurement was run for declared return boundaries; complete physical closure (quat/attachment/constraint) is not certified. physical_cycle_claim remains UNRESOLVED and is not a global block.

## H5 incremental value
Graph/V6 provides phase+object isolation versus count. The matched-event baseline can also mark loss/restore, so incremental value is audit/semantic isolation rather than a unique event detector. No policy-gain claim.

## Opportunity denominators
unit: object_level_recovery_opportunity
planned/valid: 112/112
observed_losses: 112
regrasp_confirmed: 104
return_verified: 40
p(regrasp|loss): 0.9285714285714286
p(return|regrasp,loss): 0.38461538461538464
p(return|loss): 0.35714285714285715
No-return opportunities are retained. 40 bounded observed returns are not treated as independent confirmation.

## Endpoint audit
rows: 112
legacy_return_reported: {'false': 72, 'true': 40}
independent_endpoint_verified: {'false': 72, 'true': 40}
independent_endpoint_status: {'NOT_EVALUATED_NO_RETURN_BOUNDARY': 72, 'BOUNDED_OBSERVED_TASK_RETURN': 40}
legacy-true but independent not closed: 0
complete_physical_closure_certified: false for all rows

## External grasp (track B, non-blocking)
provider_status: BLOCKED_DEPENDENCY
model_id: NVlabs/contact_graspnet
license_status: NOT_FOUND_NO_LICENSE_CHECK_PERFORMED
official_demo_evidence: NOT_RUN_NO_CHECKPOINTS_OR_REPO
compatibility: PROPOSAL_ONLY_BACKEND_INCOMPATIBLE
found_paths: []
notes: ['contact_graspnet_import:ModuleNotFoundError', 'anygrasp_import:ModuleNotFoundError']
No Contact-GraspNet/AnyGrasp repo, import, or checkpoint was found. Official sample inference was not run. No GT pose substitution. Backend remains mocap+weld, fixed orientation, 20 mm centroid capture.

## What continued despite B
Same-input scoring, method comparison, claim table, ledger audit, opportunity metrics, independent raw endpoint audit, frozen-path check, and this report. Mainline is not gated by grasp deployment.

## Frozen paths versus BASE
{
  "base": "fdc5acd1bbec7906fb1b2dffb091d2c82979d033",
  "upgrade_v2/p1_reward_repair_v6": "CLEAN",
  "upgrade_v2/p1_return_controller_v6rc1": "CLEAN",
  "upgrade_v2/p1_closed_loop_regrasp_v6rg1": "CLEAN"
}

## Large artifacts not committed
{
  "committed": false,
  "files": [
    {
      "name": "ledger.csv",
      "present": true,
      "bytes": 25808567,
      "sha256": "487bc6cd81ec482d56f80980ce839d42b3335861bb11e80f699469d096ca7467",
      "committed": false,
      "path": "/home/__compress_data/xushijie/graph_pathgraph_p1_v6gm1_worktree/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6gm1_mainline_first_v1/mainline/ledger.csv"
    },
    {
      "name": "states.jsonl",
      "present": true,
      "bytes": 14758545,
      "sha256": "2f1400490783ab003a86184d1045c7af8a7998064ee97b41fb4751df4f9bc9e5",
      "committed": false,
      "path": "/home/__compress_data/xushijie/graph_pathgraph_p1_v6gm1_worktree/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6gm1_mainline_first_v1/mainline/states.jsonl"
    }
  ]
}

## Limits
This is development replay of existing V5 simplified-state simulation, not a held-out confirmation set. Algebraic potential identity is not a physical cycle proof. Engineering completion is not independent reward confirmation.
