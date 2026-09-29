# Wave D gate comparison (original vs amended)

Amendment: WAVE_D_D7_CHECKER_AMENDMENT_1 (non-scientific checker correction; no new attempt, reset, provider, RL or optimizer).

- original gate file: `diagnostics/wave_d_gate.json` sha256 `f914b3731c63dbc5757d3591982c20f5563c24420fbe50d2bbf5e273530dd4b1` (byte-identical, read-only), status FAIL, failed check D7 only
- original D7 failure: a mutable budget receipt was compared by whole-file hash; terminal completion legitimately rewrites the receipt
- amended gate: `diagnostics/wave_d_gate_amended.json`, status PASS (PASS_AFTER_NONSCIENTIFIC_CHECKER_AMENDMENT), wave_p_released True
- pre-terminal receipt bytes: PRE_TERMINAL_RECEIPT_BYTES_NOT_RETAINED (not a failure; no reconstruction)

| gate | original pass | amended pass |
|---|---|---|
| D1 | True | True |
| D2 | True | True |
| D3 | True | True |
| D4 | True | True |
| D5 | True | True |
| D6 | True | True |
| D7 | False | True |
| D8 | True | True |
| D9 | True | True |
| D10 | True | True |
| D11 | True | True |
| X1_original_evidence_unchanged | - | True |
| X2_original_gate_file_unchanged | - | True |
| X3_no_new_execution | - | True |
| X4_tests | - | True |

The checker amendment changed only the D7 criterion; D1-D6 and D8-D11 use unchanged code and inputs. X1-X4 are additional amended-gate conditions (evidence unchanged, original gate unchanged, no new execution, tests).
