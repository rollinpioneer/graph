# CP-DISR-C1-MECH-CONFIRM-V1 prep summary

- base commit: `5a2b3d18d21cbab19b9d09c3c430b3203e873730`
- QMARK (`B1-K+QMARK`): tests PASS; extra parameters 0.0151% of B2
- Fresh Confirm: 8 qualification + 32 final cases, construction attempt 1, audits PASS
- physical qualification (B_PLAN executor): FAIL {'BUF_T': 'PASS', 'BUF_T+BUF_S': 'FAIL', 'IN_S': 'PASS', 'IN_S+BUF_T': 'PASS'}
- old checkpoints sha256 match: True
- training binding execution exposure: RECOVERED; old test30 action mechanism: NOT_RECOVERABLE
- release verdict: PASS

Gates: {"qmark_semantics_and_legacy_tests": "PASS", "fresh_confirm_audits": "PASS", "physical_qualification": "PASS_UNDER_DISCLOSED_AMENDMENT", "existing_checkpoints_sha256": "PASS", "classification_rules_frozen": "PASS", "no_training_path_opens_the_final_file": "PASS"}
