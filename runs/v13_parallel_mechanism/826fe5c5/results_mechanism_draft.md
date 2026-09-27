# CP-DISR v1.3 mechanism evaluation draft

## Scope and separation

This document separates the frozen R1 independent-training comparison, the P1 same-weight B2 contract-difference input intervention, and the P2 historical Full prior intervention. It does not replace R1/R2 and does not modify the active B0 run.

## Frozen R1 context

At the common N=8192 development point, B1-K was 0/10 and B2 was 10/10 (mean starting discounted return 0.4999766845). The selected common dev20 records are B1-K at selected N=0 with 0/20 and B2 at selected final N=14464 with 20/20 and mean return 0.5003397511. These are single-seed, post-selection development results, not test or multi-seed claims.

## P1: B2 contract-difference channel

P1 was authorized as B2@Original versus B2@DK-Zero with the same frozen B2 final checkpoint, where only the second input to phi_K(context, D_K) is zeroed. The required checkpoint artifact was not accessible and its expected SHA-256 could not be verified. No substitute checkpoint was used; therefore the planned 20 episodes, paired forward records, action-flip/TV measures, returns, success, and complete durations are `NA`, not zero. This cannot be called B1-K.

## P2: historical Full prior channel

The historical Full manifest and startup evidence confirm a prior-bearing D0 run (H=23.1, d_ref=4.55, N=8192 training, five nonempty prior train opportunities in the startup audit), but the required `n_008192.pt` artifact is absent. Exact architecture, candidate encoding, loaded-weight hash, and hidden-prefix replay cannot be verified, so neither offline forward nor Full@Original/Full@Absent episodes were run. Full@Absent is not an independent B2.

## Evidence limits

No causal claim about general training benefit, Full versus B2, or robustness is supported by this diagnostic. B0 remains `RUNNING/NOT_YET_AVAILABLE` and its score is intentionally not filled. No learning curve is generated because no new training occurred and no mechanism episodes were completed.
