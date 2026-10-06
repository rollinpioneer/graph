# Claim boundary — CP-DISR-C1-BW-FROZEN-EVAL-A03

**Permitted.** (1) The three A02 final imitation checkpoints, fixed after training, produced the listed ID and test scores on the pre-materialised A0/A1/A2/B suites, evaluated once each, with the exact planner solving every problem optimally.
(2) Under these fixed models: B2 was perfect on the colour-reversal slice while QMARK and ASNET failed the same ~14–15 of 32 cases; all three models were weak on non-isomorphic goals and failed essentially every n = 7/8 problem, almost always by a state-action loop after a first wrong decision.
(3) A0 renamed problems reproduced each model's own dev trajectory (24/24 per model), so low A0 scores are ID-level behaviour, not a renaming fault.
(4) The PPO result (`PPO_ID_GATE_FAIL`) and the imitation result (`IMITATION_ID_GATE_FAIL`) remain separate and are not merged into one ranking.

**Not permitted.** No matched-ID `REP_*` label (the ID gate was not passed by all three; the label stays `NOT_ISSUED`); no statement that any score difference is a pure representation effect (the ID gaps differ, one seed per method, one training run each, test and ID are different distributions, no ID-normalised ranking is made);
no statement that B2, QMARK or ASNET is generally better or worse; no claim that the 33/36 threshold was met or that ASNET passed; no use of mid-training or PPO checkpoints; no significance claim; no claim about hop > 4 or receptive-field limits (`NOT_TESTABLE_STRUCTURALLY_EMPTY`);
no claim that learned policies are faster or cheaper than the planner (the planner is trivially fast here and the timings are single-run wall times); no claim that this suite is a fully blind test (`PREMATERIALIZED_SUITE_WITH_DISCLOSED_SMOKE_EXPOSURE`; protocol selection used the ID results), and it must not serve as the unseen final confirmation for any method designed from these results.
