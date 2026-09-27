# P1 — B2@Original vs B2@DK-Zero

Exact same R1 B2 checkpoint and frozen T_B dev10 order. DK-Zero replaces only the second input to `phi_K(context, D_K)` with a same-shape zero; masks, facts, candidate features, reward, hidden state, and weights are unchanged. A shadow forward is evaluated before each real Original trajectory step without advancing the environment or hidden state.

| case | Original success | DK-Zero success | return Δ (DK−Original) | Original duration (s) | DK-Zero duration (s) |
|---|---:|---:|---:|---:|---:|
| T_B_dev_00 | True | False | -0.501503 | 22.999999999997577 | None |
| T_B_dev_03 | True | False | -0.485948 | 24.049999999996995 | None |
| T_B_dev_04 | True | False | -0.500751 | 23.04999999999755 | None |
| T_B_dev_08 | True | False | -0.503010 | 22.899999999997632 | None |
| T_B_dev_09 | True | False | -0.501503 | 22.999999999997577 | None |
| T_B_dev_11 | True | False | -0.503765 | 22.84999999999766 | None |
| T_B_dev_12 | True | False | -0.505279 | 22.749999999997716 | None |
| T_B_dev_14 | True | False | -0.490342 | 23.74999999999716 | None |
| T_B_dev_15 | True | False | -0.512148 | 22.299999999997965 | None |
| T_B_dev_19 | True | False | -0.495519 | 23.399999999997355 | None |

- Episodes: 20 (10 cases × 2 conditions).
- Success: Original 10/10; DK-Zero 0/10.
- Decision-level action flips: 30; flips with ≥2 legal candidates: 30.
- Mean decision-level probability TV: 0.10997059183729255.
- Full task duration is recorded only from episode start to independent first success; terminal skill duration is retained separately in the JSONL.
