# Claim boundary — CP-DISR-C1-BW-IMITATION-A02

Permitted: under matched exact-planner action supervision (identical D0/D1/D2, full optimal-action set, same loss and schedule), B2-CACHED and QMARK passed the frozen in-distribution gate while ASNET-READOUT
missed the decision-perfect threshold by one episode (32/36 vs 33/36). The PPO conclusion ("ASNET-READOUT under the frozen PPO objective: high ID task success, unstable final decision optimality", `PPO_ID_GATE_FAIL`)
and this imitation conclusion are separate and are not merged into one ranking.

Not permitted: that the ASNet-style representation is worse than or cannot generalize like B2/QMARK; that B2 or QMARK are stronger out of distribution (no OOD slice was evaluated); anything about the planner baseline;
that a mid-training ASNET checkpoint suffices; that the PPO ranking equals a representation ranking; any REP label; any statement about A0/A1/A2/B. One seed per method; a one-episode margin on 36 dev cases is not a significance test.
