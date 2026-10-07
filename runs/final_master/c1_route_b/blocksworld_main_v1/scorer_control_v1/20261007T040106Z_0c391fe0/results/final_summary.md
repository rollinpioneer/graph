# C1-BW-SCORER-CONTROL-V1 — final summary

Card: 4 conditions x 112 fresh problems (448 episodes) + 112 planner references. New training runs 0, optimizer steps 0, dev reruns 0. verify.json: PASS (448/448, 0 technical gaps, planner 112/112 optimal, all weights byte-identical before/after evaluation, old result trees and both checkpoints unchanged).
Registration commit `fb1db209cd8459a4111d3c239430094312777aa7` was pushed before any policy evaluation. Fresh set `C1-BW-SCORER-CONTROL-FRESH-v1`: 692 earlier problem classes excluded, overlap 0, no generation shortfall (all cells 8/8, colour pairs 8+8).

## Main table (success / all-steps-optimal, n in brackets)

| condition | T-MONO | T-BREAK | H4-MONO | H4-BREAK | COLOR-MONO | COLOR-BREAK | ALL |
|---|---|---|---|---|---|---|---|
| B_C3 (B2 + C3) | 20 / 1 (32) | 23 / 10 (32) | 5 / 0 (8) | 7 / 0 (8) | 16 / 16 (16) | 15 / 15 (16) | 86 / 42 (112) |
| B_G1C3 (B2 + G1+C3) | 32 / 25 (32) | 32 / 27 (32) | 6 / 2 (8) | 8 / 5 (8) | 16 / 14 (16) | 15 / 15 (16) | **109 / 88** (112) |
| MG_C3 (M1-GOAL + C3) | 32 / 23 (32) | 31 / 26 (32) | 8 / 2 (8) | 6 / 2 (8) | 16 / 15 (16) | 15 / 14 (16) | 108 / 82 (112) |
| MG_G1C3 (M1-GOAL + G1+C3) | 28 / 20 (32) | 30 / 26 (32) | 7 / 3 (8) | 6 / 2 (8) | 16 / 13 (16) | 15 / 14 (16) | 102 / 78 (112) |

Cost (all 112 episodes; actions = executed actions including failed episodes; excess = extra actions of successful episodes): B_C3 2258 actions / excess 364 / 26 failures (25 step-cap, 1 no-unvisited-successor); B_G1C3 1514 / 82 / 3; MG_C3 1528 / 122 / 4; MG_G1C3 1608 / 162 / 10.

## Paired results (ALL 112; S = success, D = all-steps-optimal; direction labels use the registered decision-aid thresholds, not significance)

| comparison | S: both / only first / only second / neither | D: both / only first / only second / neither | label S / D |
|---|---|---|---|
| MG_C3 vs B_C3 (scorer swap, C3) | 84 / 24 / 2 / 2 | 38 / 44 / 4 / 26 | A_AHEAD / A_AHEAD |
| MG_G1C3 vs B_G1C3 (scorer swap, G1+C3) | 101 / 1 / 8 / 2 | 74 / 4 / 14 / 20 | B_AHEAD / B_AHEAD |
| MG_G1C3 vs MG_C3 (G1 added, new scorer) | 102 / 0 / 6 / 4 | 75 / 3 / 7 / 27 | B_AHEAD / no clear |
| B_G1C3 vs B_C3 (G1 added, old scorer) | 84 / 25 / 2 / 1 | 40 / 48 / 2 / 22 | A_AHEAD / A_AHEAD |
| MG_C3 vs B_G1C3 (deployment) | 106 / 2 / 3 / 1 | 72 / 10 / 16 / 14 | no clear / B_AHEAD |

Colour twins (16 pairs, success both / RED only / BLUE only / neither): all four conditions 15/1/0/0. All-steps-optimal: B_C3 15/1/0/0, B_G1C3 14/1/0/1, MG_C3 13/3/0/0, MG_G1C3 12/3/0/1.

## Reading against plan section 9
- Under plain C3 the new scorer is far better than the old one (24 vs 2 only-one successes; 44 vs 4 optimal). That gain is real on this suite, and it is the same direction as the previous card.
- Adding G1 helps the old scorer a lot (+23 net successes, +46 net optimal) and does not help the new scorer (net -6 successes; optimal not clearly different). With G1+C3 the new scorer is behind the old one (101/1/8/2 on success).
- The new combinations do not exceed B_G1C3: MG_C3 is level on success (108 vs 109, label no clear) and behind on all-steps-optimal (82 vs 88, 16 vs 10 discordant); MG_G1C3 is behind on both.
- Branches that apply: "G1 helps the old scorer and hurts or does not help the new scorer" and "the new combinations do not exceed B_G1C3". B_G1C3 stays the best execution system on this task family; the card ends the local read-out / loss / rule tuning loop here.
- H4 is still hard for everyone (best H4 all-steps-optimal 7/16, B_G1C3). Several H4 differences are 1-3 problems of 8 or 16 and carry no label.
