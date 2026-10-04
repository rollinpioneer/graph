# CP-DISR-TP-BEHAVIOR-STATIC-1 - report

**Verdict: ENGINEERING_BLOCKED** (engineering status FAIL; blocker OFFICIAL_SYMBOLIC_API_UNAVAILABLE). No scientific negative is claimed: the eligibility gates F1-F7 were never evaluated.

## Source

- remote https://github.com/StanfordVL/BEHAVIOR-1K.git, tag v3.9.3-post1, commit bd049de3119acdcdf2334fe9e1ebe060fa20c108, dirty tracked files 0; MIT licence; no dataset or scene asset downloaded.

## What ran (Stage A and B)

- Official BDDL3 `Conditions` parsed 1016 of 1016 problem instances across 1016 activities.
- Primitive inventory (AST of the official source): GRASP, PLACE_ON_TOP, PLACE_INSIDE, OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF, SOAK_UNDER, SOAK_INSIDE, WIPE, CUT, PLACE_NEAR_HEATING_ELEMENT, NAVIGATE_TO, RELEASE
- Diff against the card's expected primitive list: {"expected_not_in_source": [], "in_source_not_expected": [], "identical": true}
- Official recipe loaders: {"load_cooking_recipes": 32, "load_machine_recipes": 39, "load_mixing_recipes": 14, "load_substance_cooking_recipes": 340, "load_washer_rule": 1}
- OmniGibson rule classes (AST): 14
- Static coverage screen (necessary-condition, not execution-verified): FULL 495, PARTIAL 263, UNSUPPORTED 102, UNKNOWN 156.
- Canonical state hash permutation test: {"instances_tested": 1016, "mismatches": 0, "seed": 20261004}

## Why the scan stops here

the official primitives teleport simulator objects into post-condition states by reading and writing OmniGibson object states, and the official transition rules are OmniGibson classes over simulator objects and particle systems; applying them needs a live OmniGibson (Isaac Sim) environment, which this card does not authorise, and writing a substitute would be a re-implementation the card forbids

Evidence: OmniGibson importable = False, Isaac Sim importable = False; primitive module imports ['torch', 'omnigibson']; transition-rule module imports omnigibson = True and torch = True; object states read or written by the primitives: ['Contains', 'Covered', 'HeatSourceOrSink', 'Open', 'ParticleRemover', 'ParticleSource', 'Saturated', 'ToggledOn'].

## Denominator

| item | value |
|---|---|
| activities | 1016 |
| problem_instances | 1016 |
| parseable | 1016 |
| coverage_FULL | 495 |
| coverage_PARTIAL | 263 |
| coverage_UNSUPPORTED | 102 |
| coverage_UNKNOWN | 156 |
| states_examined | 1016 |
| states_with_ge2_legal_actions | NOT_REACHED |
| pairs_examined | NOT_REACHED |
| dual_legal | NOT_REACHED |
| public_consequence | NOT_REACHED |
| both_side_recoverable | NOT_REACHED |
| discrete_utility | NOT_REACHED |
| non_geometric | NOT_REACHED |
| non_contract_redundant | NOT_REACHED |
| final_eligible | NOT_REACHED |
| matched_context_groups | NOT_REACHED |
| reversal_witnesses | NOT_REACHED |
| magnitude_only_witnesses | NOT_REACHED |

## Decision for the user

Stages C-H need the official primitives and transition rules to execute. That requires a live OmniGibson (Isaac Sim) environment, which this card forbids (no simulator, no GPU, no physics). Writing a symbolic re-implementation is also forbidden. Options: authorise a headless OmniGibson symbolic-mode environment for a follow-up card, or stop.
