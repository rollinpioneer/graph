# PathGraph P1 V6 three-issue report

Execution status: CONFIRMATION_EXECUTED_NOT_PASSED
confirmation_passed: False
independent_verification_passed: False

## Frozen facts
- Parent: 39b87fe234b4ac9765357f2243b02726dbbb2332
- V5R1: CYCLE_CLAIM_NOT_ESTABLISHED; 0 exact / 40 approx / 112 object-level losses (not 112 R6 cycles)
- V5R1 engine: 35200/35248 passed, max_error=0, 48 cap-flag remaining on original scoring
- V6 development live-engine comparisons: {'comparisons': 71488, 'failures': 0, 'numeric_failures': 0, 'cap_flag_failures': 0, 'max_error': 0.0}

## Q1
New candidate uses Psi = -C_cap + G with r=Delta Psi. Failure across nodes withdraws transport credit because G becomes 0, not because an extra -phi term was patched onto the frozen engine.
Development clawback gate: {'passed': True, 'evidence_files': ['/home/__compress_data/xushijie/graph_pathgraph_p1_v6_three_issue_worktree/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/03_semantics_v1/q1_phi_clawback.csv'], 'numerator': 3, 'denominator': 3, 'unresolved': 0}

## Q2
LOST and RECOVERING share capability units=5. Label-only change yields 0. Real rehold is 5->3 units.
Development label gate: {'passed': True, 'evidence_files': ['/home/__compress_data/xushijie/graph_pathgraph_p1_v6_three_issue_worktree/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/03_semantics_v1/q2_label_neutrality.json'], 'numerator': 1, 'denominator': 1, 'unresolved': 0}
Confirmation label: LABEL_ONLY_ZERO

## Q3
Old V5/V5R1 cycle claim is preserved as not established on simplified-state replay.
New physics claim: V6_INDEPENDENT_EVIDENCE_INSUFFICIENT
Old physics-on-new-data claim: CYCLE_EVIDENCE_INSUFFICIENT
Full-episode positive return is not treated as cycle farming.

## Backend
{
  "status": "PASS",
  "backend_id": "MUJOCO_CONSTRAINT_ASSISTED_TABLETOP_V6",
  "collector_commit": null,
  "collector_hashes": {},
  "mujoco_version": "3.4.0",
  "model_xml_sha256": "1af2e19dd5935d1859dd56466014f9ceda59ea7e6bbe4fe31ec7440937c139df",
  "candidate_lock_sha256": null,
  "phase": "check",
  "planned_rollouts": null,
  "completed_rollouts": null,
  "raw_manifest_sha256": null,
  "direct_pose_overwrites_after_start": 0,
  "checkpoint_resets_inside_episode": 0,
  "velocity_zeroing_after_start": 0,
  "weld_changes_logged": true,
  "controls_logged": true,
  "action_execution_logged": true,
  "full_state_snapshots_available": true,
  "source_audit_file": "artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/05_backend_v1/source_audit.json",
  "tests": {
    "mujoco_version": "3.4.0",
    "steps_per_control": 10,
    "mj_step_time_consistent": true,
    "object_followed_weld_without_qpos_write": true,
    "direct_write_after_start_raises": true,
    "reset_during_episode_raises": true,
    "direct_pose_overwrites_after_start": 1,
    "xml_sha256": "1af2e19dd5935d1859dd56466014f9ceda59ea7e6bbe4fe31ec7440937c139df",
    "held_after_smoke": {
      "pos": [
        -0.3480747518229565,
        -0.01358917216636465,
        0.7228137450201303
      ],
      "vel": [
        -5.571663073173736e-23,
        2.054347558582512e-16,
        6.693768751125888e-15
      ],
      "quat": [
        0.9999402818185292,
        -0.0006488858506535156,
        0.0006236295813662818,
        0.010891410835152295
      ],
      "angular_vel": [
        -8.955882799178584e-16,
        3.330289987545862e-17,
        -4.108695117165014e-16
      ],
      "target_xy": [
        0.29116095111150914,
        0.07733732661587799
      ],
      "target_z": 0.515
    },
    "weld_after": true,
    "z0": 0.5613158820867815,
    "z1": 0.7228137450201303
  },
  "note": "smoke attempted a forbidden write to prove it raises; production collector never does that"
}

## Non-claims
No visual grounding, no policy training, no LLM calls, no real robot, G1 does not drive control.
