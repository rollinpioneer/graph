# R15-B geometry-event fair re-evaluation - final report

Engineering status: `R15B_CACHE_REEVALUATION_COMPLETE`
Scientific status: `L2RAR2_PARTIAL_KEEP_G1`; selected candidate = null; L3 closed.

## O-layer parity with the frozen Round-10 interface

- rows compared: 64; matched: 64; mismatched: 0

## Method summary

| method | tier | correct | accuracy | K1 recall | negative false emergency | hold transitions | routes |
|---|---|---|---|---|---|---|---|
| O_B2 | O | 20 | 0.625 | 1.0 | 0.0 | 0 | {} |
| O_C3 | O | 20 | 0.625 | 1.0 | 0.0 | 0 | {} |
| S_G_H | S | 20 | 0.625 | 1.0 | 0.0 | 0 | {} |
| S_G_R | S | 20 | 0.625 | 1.0 | 0.0 | 0 | {} |
| S_G_HR | S | 20 | 0.625 | 1.0 | 0.0 | 0 | {} |

## Weld transition audit (reference layer only)

| rollout | case | transition | order status | event relation |
|---|---|---|---|---|
| L2RAR2_REPAIR_00_840000:K4_regular_hold_loss | K4_regular_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_00_840000:K5_brief_hold_loss | K5_brief_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_00_840000:K6_long_gap_after_loss | K6_long_gap_after_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_01_840001:K4_regular_hold_loss | K4_regular_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_01_840001:K5_brief_hold_loss | K5_brief_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_01_840001:K6_long_gap_after_loss | K6_long_gap_after_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_02_840002:K4_regular_hold_loss | K4_regular_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_02_840002:K5_brief_hold_loss | K5_brief_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_02_840002:K6_long_gap_after_loss | K6_long_gap_after_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_03_840003:K4_regular_hold_loss | K4_regular_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_03_840003:K5_brief_hold_loss | K5_brief_hold_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |
| L2RAR2_REPAIR_03_840003:K6_long_gap_after_loss | K6_long_gap_after_loss | True | STRICTLY_LATER_TIME | EVENT_INSIDE_TRANSITION_INTERVAL |

## Drift anchors

Candidate-anchor drift and last-weld-on-anchor drift are reported separately in
`relative_drift_anchor_audit.csv`; `candidate_relative_exit_triggered` only means
the frozen candidate threshold fired on saved positions.

- L2RAR2_REPAIR_00_840000:K4_regular_hold_loss / K4_regular_hold_loss / S_G_R: candidate anchor drift 0.0101 m, last-on anchor drift 0.007866 m, exit triggered False
- L2RAR2_REPAIR_00_840000:K4_regular_hold_loss / K4_regular_hold_loss / S_G_HR: candidate anchor drift 0.0101 m, last-on anchor drift 0.007866 m, exit triggered False
- L2RAR2_REPAIR_00_840000:K5_brief_hold_loss / K5_brief_hold_loss / S_G_R: candidate anchor drift 0.0101 m, last-on anchor drift 0.007866 m, exit triggered False
- L2RAR2_REPAIR_00_840000:K5_brief_hold_loss / K5_brief_hold_loss / S_G_HR: candidate anchor drift 0.0101 m, last-on anchor drift 0.007866 m, exit triggered False
- L2RAR2_REPAIR_00_840000:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_R: candidate anchor drift 0.0168 m, last-on anchor drift 0.014659 m, exit triggered False
- L2RAR2_REPAIR_00_840000:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_HR: candidate anchor drift 0.0168 m, last-on anchor drift 0.014659 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K4_regular_hold_loss / K4_regular_hold_loss / S_G_R: candidate anchor drift 0.0106 m, last-on anchor drift 0.007988 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K4_regular_hold_loss / K4_regular_hold_loss / S_G_HR: candidate anchor drift 0.0106 m, last-on anchor drift 0.007988 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K5_brief_hold_loss / K5_brief_hold_loss / S_G_R: candidate anchor drift 0.0106 m, last-on anchor drift 0.007988 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K5_brief_hold_loss / K5_brief_hold_loss / S_G_HR: candidate anchor drift 0.0106 m, last-on anchor drift 0.007988 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_R: candidate anchor drift 0.0175 m, last-on anchor drift 0.014977 m, exit triggered False
- L2RAR2_REPAIR_01_840001:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_HR: candidate anchor drift 0.0175 m, last-on anchor drift 0.014977 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K4_regular_hold_loss / K4_regular_hold_loss / S_G_R: candidate anchor drift 0.0099 m, last-on anchor drift 0.007028 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K4_regular_hold_loss / K4_regular_hold_loss / S_G_HR: candidate anchor drift 0.0099 m, last-on anchor drift 0.007028 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K5_brief_hold_loss / K5_brief_hold_loss / S_G_R: candidate anchor drift 0.0099 m, last-on anchor drift 0.007028 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K5_brief_hold_loss / K5_brief_hold_loss / S_G_HR: candidate anchor drift 0.0099 m, last-on anchor drift 0.007028 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_R: candidate anchor drift 0.0159 m, last-on anchor drift 0.013042 m, exit triggered False
- L2RAR2_REPAIR_02_840002:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_HR: candidate anchor drift 0.0159 m, last-on anchor drift 0.013042 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K4_regular_hold_loss / K4_regular_hold_loss / S_G_R: candidate anchor drift 0.0106 m, last-on anchor drift 0.008448 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K4_regular_hold_loss / K4_regular_hold_loss / S_G_HR: candidate anchor drift 0.0106 m, last-on anchor drift 0.008448 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K5_brief_hold_loss / K5_brief_hold_loss / S_G_R: candidate anchor drift 0.0106 m, last-on anchor drift 0.008448 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K5_brief_hold_loss / K5_brief_hold_loss / S_G_HR: candidate anchor drift 0.0106 m, last-on anchor drift 0.008448 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_R: candidate anchor drift 0.0179 m, last-on anchor drift 0.015872 m, exit triggered False
- L2RAR2_REPAIR_03_840003:K6_long_gap_after_loss / K6_long_gap_after_loss / S_G_HR: candidate anchor drift 0.0179 m, last-on anchor drift 0.015872 m, exit triggered False

## Limits

- S tier uses saved oracle 3D positions: state-assisted diagnostic, not RGB.
- No candidate exit under the frozen thresholds says nothing about whether the
  physical loss occurred; independent physical loss stays UNRESOLVED.
- R14 was not executed (authorization absent).
