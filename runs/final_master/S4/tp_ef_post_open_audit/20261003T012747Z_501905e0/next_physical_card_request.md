# Next physical card request — CP-DISR-TP-EF-POST-OPEN-AUDIT-1

**Trigger:** fewer than two recoverable post-OPEN T_B snapshots exist in saved logs (qualifying = 0). This is a budget request only; nothing is started by this card.

## Requested budget (maximum)

- Setup-only episodes: **2** (each: 1 bundle, 1 start_case, 1 explicit reset, exactly **1 skill** = OPEN(container), then stop).
- Environment constructions <= 4; skill executions <= 2; continuation skills = 0; provider = 0; RL/optimizer = 0; test = 0.

## Pre-registered state rule (no utility information used)

Cases whose DISCOVERY-1 journal already logged a Verifier-confirmed OPEN postcondition: ['T_B_dev_03', 'T_B_dev_05', 'T_B_dev_18']. Take the first 2 by the DISCOVERY-1 rank key (ascending sha256 of the frozen namespace + case id): **['T_B_dev_03', 'T_B_dev_05']**. If a setup-only episode fails its OPEN postcondition it is reported, not replaced or retried.

## Required record per episode

- post-skill FactStore (all fact ids with truth values)
- candidate ids and mask
- qpos hash
- public observation hash
- restore receipt without episode counter
- controller exit and Verifier postcondition for OPEN
- no continuation skill

## Qualifying test applied to the recorded state (decided after recording, rule fixed now)

- Verifier-confirmed Open(container)=TRUE
- GripperEmpty=TRUE
- OnTable(target)=TRUE
- OnTable(second_object)=TRUE
- PICK(target) legal
- PICK(second_object) legal
- restorable (full fact record + replay/restore identity)

Replay recoverability rests on the DISCOVERY-1 determinism result (repeated branches identical to numerical precision); the setup-only record adds the facts and identity hashes needed to check it later.

## Offline basis

- Routes T and S (spec in `scripted_continuation_spec.json`) are contract-tied on hypothetical post-OPEN facts: classification `CONTRACT_TIED_SCRIPTABLE`, both fully legal.
- Relation entry exists structurally: True (62 non-redundant relations touch the pair); candidate-relative difference: True. Truth/utility unassessed.

## Not requested

No continuation episodes, no PICK-vs-PICK branches, no repeats, no provider call. A subsequent paired-branch card (states, continuation, repeat, epsilon, stop rule, denominator) would be requested separately after the setup-only record is reviewed.
