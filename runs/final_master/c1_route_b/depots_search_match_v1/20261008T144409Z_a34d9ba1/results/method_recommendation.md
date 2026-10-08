# C1-DEPOTS-SEARCH-MATCH-V1: method recommendation (one item, plan section 15)

**Recommendation: class B. Spend the next card on the cost of the learned state score, not on its quality, a gate, REL variants or more training.**

Evidence from this card:
- Quality per expansion is already good: on Struct and Joint the frozen DENSE-G reaches the goal in 24 expansions on average (median 21) with plans 1.095 times the optimum (72 of 128 optimal), against 133 expansions (median 54) and 1.22 for WL-GOOSE inside the same engine; on IPC it also needs fewer expansions where both finish (12 of 12).
- Cost is the measured bottleneck on the public problems: 3.5 to 4.4 ms per state against 0.05 to 0.17 ms; the neural rows stop after a few hundred to ~15,000 expansions on the 13 larger IPC problems (8 / 8 / 7 `TIMEOUT_BEFORE_NODE_LIMIT`), while H_WL expands 25,000 to 100,000 nodes in 40 to 230 s and solves 18.
- The previous Track B coverage loss was an execution effect (all three scorers solve 128 of 128 under search), so a scorer-quality or gate project is not indicated by this evidence.

Concrete shape of the next card (a proposal, not started, no training):
1. Profile one frozen DENSE-G evaluation on IPC p15 / p21 / p22 into host state conversion, host-device transfer and GPU message passing; this card only measured their sum.
2. Test cost reductions that keep the scorer fixed and the engine identical: batching across several OPEN nodes is excluded by the protocol, so use incremental or shared encodings of the template graph, cheaper state conversion, and lazy / two-level scoring (a cheap evaluator orders most generated states, the network scores only states that reach the front of the queue). Each variant is compared with the present engine at equal wall clock and at equal expansions, with expansions, scored states, forward calls and time reported side by side.
3. Evaluate on IPC22 plus a new independent problem set with larger topologies for the final claim; the present 182 problems stay development material.

Not recommended from this card: restoring the VOC gate (the best three-way oracle gain was a fixed per-problem choice, not evidence for a per-action gate); relation-restriction rules; larger-topology training (the IPC failures coincide with size and cost, which this card cannot separate from generalisation; if the cost card removes the time limit and failures remain, class E becomes the next question).
