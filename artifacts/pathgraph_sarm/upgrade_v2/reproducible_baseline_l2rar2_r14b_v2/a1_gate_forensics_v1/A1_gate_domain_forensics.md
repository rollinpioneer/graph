# Baseline A1 callback-return gate-domain forensics

- Status: `GATE_DOMAIN_BUG_CONFIRMED`
- Checkpoints: `54`
- Pairs: `8/8`
- Physical exact: `8/8`
- Semantic exact: `8/8`
- Expected record identity relation: `8/8`
- Old state hash differs: `8/8`

## Interpretation

All paired physical and semantic fields are exact. The legacy full-record hash differs because record identity fields differ.
The consumed A1 result remains STOP_AFTER_EXECUTION_1 and is not rewritten.
A new runner/protocol is required before a new A execution can be authorized.
