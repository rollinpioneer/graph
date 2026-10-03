# Design card: TP_SHARED_STAGING_V1 (CP-DISR-TP-SHARED-STAGING (agent execution v1.0))

Scope: task-design / physical-canary only; no RL, optimizer, provider; no change to Method 2.1.1, contract semantics, Verifier, Evaluator, controller, reward, timeout or safety thresholds.

Authority: Research Content v5 and Experimental Plan v3 hashes match the manifest: True; default method 2.1.1.

Idea under test: a shared premium staging place W with a legal fallback W_f; p or q is the staging-sensitive object depending on a public downstream context (H or R); a wrong first occupant of W must cost at least one extra discrete rehandling.

Result: **TASK_FAIL_NO_EXISTING_SOFT_STAGING_VALUE** at STOP gate 1; 0 preflight resets, 0 formal episodes. Nothing was constructed or run.

Gate-1 reading: staging at W can only be valuable if some existing high-level operation becomes unnecessary or cheaper for an object staged there. The platform has no such operation (see gate1_audit.json and negative_evidence.md).
