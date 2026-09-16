# P2A forensics: why V6 lost (no new training)

Overall V6-BC = -4.6875 pp, equal-weighted over 8 test conditions.

## Conditions
There is no `gripper_reopen` test condition; RELEASE is an action. Mapping used:
FREE_ORDER=natural, A_STARTED=A_first, B_STARTED=B_first,
A_VALID_B_LOSS/B_VALID_A_LOSS/LATE_LOSS/TWO_LOSSES=drop_regrasp,
INVALIDATE_FIRST=invalidate_other.

Largest holes: A_STARTED -8.75 pp, INVALIDATE_FIRST -6.25 pp, A_VALID_B_LOSS -5.0 pp.
Every condition is negative. drop_regrasp is not the unique source
(4 conditions contribute -1.875 pp of the overall -4.69 pp).

On A_STARTED, V6 has more invalid actions (mean 7.7 vs BC 2.8; any-invalid 24% vs 9%)
and longer episodes with less waiting.

## Weights
All methods have global mean weight = 1. V6 ESS=70% of N, max=2.81, min=0.563 (floor).
V6 downweights the majority TRANSPORT/ADVANCE mass (~65% of rows, mean w 0.72 / 0.85-0.88)
and WAIT/START_RECOVERY/LOST/VALID (all at floor 0.563).
It upweights PLACE (~2.0), REGRASP (~1.96), TERMINAL (~2.81), HELD/ACQUIRE (~1.3-1.7).

## Shuffle
Shuffle-V6 = +3.63 pp point estimate; 97.5% CI [-3.8, +12.0] includes 0.
4/5 seeds favor shuffle (seed 17 = +10.4 pp). Same weight multiset, broken pairing.
This is compatible with V6 semantic pairing hurting, but not a locked significance claim.

## Scale
Frozen seed-17 same batch: V6 weighted/unweighted grad-norm ratio = 0.50 (BC = 1.00).
Weighted loss < CE because mass sits on easy completion tokens.

## Zero-progress actions
WAIT, START_RECOVERY, object-switch ACQUIRE: V6 mean w = 0.563 = 56% of BC.
RELEASE/reopen: 0.616. These have no positive immediate V6 potential, so the floor map
starves them. ADVANCE is small positive per step and is also compressed by the 95% quantile
of large PLACE/TERMINAL rewards.