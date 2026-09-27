# Results Draft: T_B Existing-Task R1

Two matched seed-0 policies were trained from scratch on the frozen T_B train64 split with the same source commit, runtime, budget rule, and absent prior. B1-K reads the current contract graph without successor intervention; B2 applies explicit candidate-wise contract intervention. Both jobs stopped at the predeclared simulation-time cap (Tcap=68,812.8 s), before the 16,384-transition ceiling. B1-K used 13,623 transitions and 13 complete plus one fragment PPO update; B2 used 14,464 transitions and 14 complete plus one fragment update.

On the frozen common dev10, B1-K returned zero success and zero mean start-discounted return at N=0, 4,096, 8,192, and its final N=13,623. B2 was also zero at N=0 and 4,096, then reached 10/10 success with mean return 0.499977 at N=8,192 and again at its final N=14,464. The last B2 point is outside the common budget interval, which ends at N=13,623; no B2 evaluation was made exactly at that boundary.

The pre-frozen selection rule chose B1-K's N=0 checkpoint (a tie resolved toward earlier N) and B2's final N=14,464 checkpoint. On one common dev20 evaluation per method, B1-K succeeded in 0/20 cases (mean return 0.0) and B2 in 20/20 (mean return 0.500340); B2 successful episodes had mean success time 2.920 simulated seconds. Dev20 includes dev10, so this is not an independent holdout or unbiased estimate after selection.

These results are descriptive for one seed on one existing task. They support a within-R1 learning-curve difference, not a cross-seed claim, a VLM benefit, or an A_CAT advantage. No new test IDs were used. Earlier D0 and T_C results remain in a separate historical evidence table and are not combined with this source/profile.
