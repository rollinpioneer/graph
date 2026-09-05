# D2/D3 result summary

D2 constructed 30 free/collision pairs (60 states). Each pair has exactly the same distance to the goal; the free route has positive obstacle clearance and the collision route intersects the obstacle.

D3 executed 12 independent seeds per state (720 continuations). All seed replicates of a pair stayed together, leaving 6 root families and 144 continuations for heldout evaluation. Heldout accuracy was 0.5 for both distance-only geometry and the constant baseline, and 1.0 for q-only and D-only. The validation-selected q+D mixture assigned q weight 1.0, so it matched q-only.

The result is strictly simulator-scoped. It clears the data-bridge prerequisite for a U2 stochastic boundary prototype only; it leaves physical generalization and claims about the original robot task unsupported.
