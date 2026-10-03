# CP-DISR-TP-EF-POST-OPEN-AUDIT-1 — final summary

Zero environment, zero provider, zero RL, zero skill execution. Test/holdout paths were not opened.

- Recoverable post-OPEN T_B snapshots found: **0** (need >= 2). T_B logs scanned: 4 runs, 7281 episode summaries, transition files present: False, files with the Open fact: 0.
- DISCOVERY-1 logged Verifier-confirmed OPEN postconditions in 4 journals over cases ['T_B_dev_03', 'T_B_dev_05', 'T_B_dev_18'], but stored no snapshot (no GripperEmpty/OnTable facts, no restore state), so none counts.
- Hypothetical post-OPEN contract audit: PICK(target) legal=True, PICK(second) legal=True; routes T/S fully legal = True/True; classification `CONTRACT_TIED_SCRIPTABLE` (min skills after candidate: {'PICK_target': 4, 'PICK_second': 4}).
- Scope limit recorded: the 'PICK(target) first is a dead end' statement applies to closed-container initial states only.
- Relation expressibility: universe 152, contract-redundant 20, non-redundant 132; entry for the pair exists: True (syntactic only; truth/utility UNKNOWN).
- Outcome: no state or protocol frozen; a budget request for at most 2 setup-only episodes is in `next_physical_card_request.md`.
