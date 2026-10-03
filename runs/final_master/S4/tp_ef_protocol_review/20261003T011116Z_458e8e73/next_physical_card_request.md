# Next physical card request — CP-DISR-TP-EF-PROTOCOL-REVIEW-1

Requested PICK(target) vs PICK(second_object) protocol: **NOT_FROZEN_OFFLINE_GATE_FAILED**.

**No physical card is requested.** The requested candidate A is infeasible under the registered hard contracts (reason in `scripted_continuation_spec.json`).

Decision options for the user (none is started by this card):

1. OPEN vs PICK(second_object) with a fully scripted continuation (A: OPEN, PICK second, PLACE_BUFFER second, PICK target, PLACE target; B: PICK second, PLACE_BUFFER second, OPEN, PICK target, PLACE target). Offline status: {'CONTRACT_TIED_SCRIPTABLE': 20}; all five skills have observed NORMAL_TERMINATION in DISCOVERY-1. It is contract-tied, not contract-revealed, so it is not excluded by the CONTRACT_REVEALED_PAIR rule; it removes the stepwise-planner dependency but not the unexplained NO_PLAN-after-PICK(second) observations (facts were not saved), so any new card must log post-skill facts.
2. Treat the T_B template as exhausted for candidate diversity (single contract-tied pair) and move to a different decision structure; requires a research decision and probably new task content.
3. Stop T_P discovery here and take the result to the S4 research decision.

Provider: remains 0. A provider/prompt revision is requested only after a new physical card establishes two reliable utility witnesses.
