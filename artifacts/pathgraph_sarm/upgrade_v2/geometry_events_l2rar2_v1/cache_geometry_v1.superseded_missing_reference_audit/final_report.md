# R15-A cache geometry baseline - final report

Entry commit: `93f89d49317a026db09f21acc8686eb82cd5c668`
Scientific status: `L2RAR2_PARTIAL_KEEP_G1` (selected_candidate_id = null)

## Scope

Existing 32 development rollouts only. No physics, no training, no model API,
no key reads. R14 physical replay was not run (authorization absent).

## Method agreement against the frozen legacy windows

| method | tier | correct | none | wrong | unknown | routes | hold-before-event | false-recovery |
|---|---|---|---|---|---|---|---|---|
| O_B2 | O | 4 | 24 | 4 | 0 | - | 11 | 0 |
| O_C3 | O | 4 | 24 | 4 | 0 | - | 20 | 0 |
| S_G_H | S | 4 | 28 | 0 | 0 | HEIGHT_THRESHOLD=456 | 20 | 0 |
| S_G_R | S | 4 | 28 | 0 | 0 | RELATIVE_CO_MOTION=428 | 20 | 0 |
| S_G_HR | S | 4 | 28 | 0 | 0 | LOW_HEIGHT_CO_MOTION=428 | 20 | 0 |

## Availability

- rollouts loaded: 32 / 32
- root families: L2RAR2_REPAIR_00_840000, L2RAR2_REPAIR_01_840001, L2RAR2_REPAIR_02_840002, L2RAR2_REPAIR_03_840003
- orientation available: False
- RGB present for 32 rollouts

## Physical loss status

- planned-loss rows carried to the physical table: 12
- all remain `unresolved`; recall is `NOT_ESTIMABLE`
- the geometry candidates never generate their own truth labels

## Limits

- S-tier results are state-assisted diagnostics, not pure-vision results and
  not an independent upper bound.
- The cache stores no object/gripper quaternion, so only world-frame relative
  translation is used; rotation compensation is out of scope.
- All 32 rollouts were previously inspected during development: this is not a
  confirmation set.
