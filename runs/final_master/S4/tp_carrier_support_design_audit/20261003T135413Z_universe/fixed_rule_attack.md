# Fixed-rule attack (R1-R6)

A rule is refuted by a scenario in which the action it prescribes needs strictly more skills than the other first action (skill counts from the declared route model; MOVE_CARRIER time is unestablished and bracketed).

| rule | statement | counterexample scenarios | refuted |
|---|---|---|---|
| R1 | if A on B: move B first | none | False |
| R2 | if same destination: move B first | none | False |
| R3 | if different destinations: pick A first | none | False |
| R4 | if A near carrier edge: pick A first | S3_CO_TRANSPORT_UNSTABLE | True |
| R5 | always pick A first | S1_CO_TRANSPORT_STABLE, S3_CO_TRANSPORT_UNSTABLE | True |
| R6 | always move carrier first | none | False |

## Scenario dominance (skills)

- S1_CO_TRANSPORT_STABLE: carrier-first ever worse = False (vs card route False); object-first ever worse = True
- S2_SEPARATE_DESTINATIONS: carrier-first ever worse = False (vs card route False); object-first ever worse = False
- S3_CO_TRANSPORT_UNSTABLE: carrier-first ever worse = False (vs card route False); object-first ever worse = True
- S4_NO_SUPPORT_NEUTRAL: carrier-first ever worse = False (vs card route False); object-first ever worse = False

Unrefuted single-variable rules: R1, R2, R3, R6

Result: a fixed single-variable heuristic is sufficient (TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT)
