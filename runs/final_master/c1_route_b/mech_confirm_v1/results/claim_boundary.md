# Claim boundary (CP-DISR-C1-MECH-CONFIRM-V1)

**Allowed now (M3, one seed, 8 cases per cell)**
- A QMARK control that never builds a successor state, with the same relational encoder as B2, reached the same Fresh Confirm total as B2 (27/32) and exceeded ABS (23/32). Explicit successor construction is therefore not shown to be necessary for this structural generalization.
- The old +E and NC controls fail on goal bindings/compositions absent from training goals (+E 0/32, NC 8/32).
- ABS fails the double AtBuffer cell by opening the container first (8/8 failures), not by destroying a completed goal.
- A symbolic contract search (B_PLAN) reaches 26/32 at about 1 ms per decision on this suite.

**Not allowed**
- "Unseen execution": the test placements were executed in training as non-goal actions (target->buffer thousands of times, second_object->container hundreds, public effect verified ~99%). Only unseen goal bindings / goal compositions can be claimed.
- That the difference representation (B2) is uniquely responsible (M2 not met), that absolute successor is sufficient (B2 vs ABS unresolved), or that QMARK is a contributed method.
- Any claim from 32 episodes as 32 independent trials; the unit is seed x cell and there is one seed.
- Any claim about another task family, other perception conditions, or that the nominal successor equals the real observed state. The qualification raw FAIL (1/8) is preserved; the status is only PASS_UNDER_DISCLOSED_AMENDMENT.
- Any use of QMARK seeds 1 and 2 as results (trained, unevaluated).
- Any statement that contract search cannot solve the suite.
