# R14B A1 gate forensics and V2 handoff

## Outcome

The consumed A1 remains `STOP_AFTER_EXECUTION_1` with its original `13/14` result and SHA256 `8dd9cbc660a9bf6a0dd1ce3b62451c80557ca738eab5e33e3a9479aed8949eac`. Zero-physics forensics paired all 8 callback/return pairs: physical exact `8/8`, semantic exact `8/8`, expected record-identity relation `8/8`, and legacy full-record hashes differed `8/8`. The conclusion is `GATE_DOMAIN_BUG_CONFIRMED`; A1 was not rewritten.

## V2 engineering freeze

- Branch: `research/l2rar2-r14b-state-hash-domains-v2`
- Runner commit: `15d1b12999adad5172c2dacf8e8a5488d79df6bc`
- Base commit: `45a3544c68c666ac09796407ae4aee31717bd453`
- Protocol: `L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V2`
- Protocol SHA256: `6c630ed542e369d9653cdd7c533cfe5fed1e90bbcc09f64afa94cbce0065f2e3`
- Source-lock canonical SHA256: `e0f2b10403b108090d7e0b575d25197b79261627f7fb43f7b1a03f87e8a71da0`
- Expected main gates: `16` (pairing, physical, semantic are explicit)

The runner now separates `physical_state_sha256`, `semantic_state_sha256`, and `record_sha256`; legacy `state_sha256` is retained only as a deprecated record-integrity alias. Cross-run comparison remains byte/exact with zero numeric tolerances and reports IDENTITY/PHYSICAL/SEMANTIC/RECORD domains.

## Validation and accounting

The package verifier and output validator passed. The V2 suite has 113 pure tests; compileall, AST parsing, and `git diff --check` passed. This round used zero physical executions, generated no nonce, did not create an A2 execution root, and left B/C/R16 at zero.

## Next gate

`baseline_A2_application.json` is `PENDING_HUMAN_AUTHORIZATION`; the authorization template remains `NOT_AUTHORIZED` with all human identity, nonce, expiry, and output-root fields null. A real human reviewer must decide whether to issue a new single-use A2 authorization against this frozen runner/protocol. No automatic approval or execution is performed.
