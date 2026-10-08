# C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3: generated numbers (interpretation is in final_summary.md / claim_boundary.md)

## struct: coverage / mean plan length of solved (node budgets 1,000 / 10,000 / 100,000 within 300 s)

| condition | <= 1,000 | <= 10,000 | <= 100,000 | ratio to proven optimum | optimal plans | mean expansions at solution | statuses (100k) |
|---|---|---|---|---|---|---|---|
| ALT_DENSE_ADD | 32/32 (14.844) | 32/32 (14.844) | 32/32 (14.844) | 1.0637 | 16 | 19.62 | {'SOLVED': 32} |
| ALT_WL_ADD | 32/32 (16.75) | 32/32 (16.75) | 32/32 (16.75) | 1.2133 | 11 | 44.09 | {'SOLVED': 32} |
| OLD_DENSE | 32/32 (14.219) | 32/32 (14.219) | 32/32 (14.219) | 1.0218 | 25 | 14.31 | {'SOLVED': 32} |
| OLD_WL | 32/32 (16.156) | 32/32 (16.156) | 32/32 (16.156) | 1.1735 | 13 | 36.84 | {'SOLVED': 32} |
| OLD_HADD | 32/32 (18.188) | 32/32 (18.188) | 32/32 (18.188) | 1.2957 | 7 | 78.09 | {'SOLVED': 32} |

## joint: coverage / mean plan length of solved (node budgets 1,000 / 10,000 / 100,000 within 300 s)

| condition | <= 1,000 | <= 10,000 | <= 100,000 | ratio to proven optimum | optimal plans | mean expansions at solution | statuses (100k) |
|---|---|---|---|---|---|---|---|
| ALT_DENSE_ADD | 128/128 (23.047) | 128/128 (23.047) | 128/128 (23.047) | 1.1523 | 40 | 33.41 | {'SOLVED': 128} |
| ALT_WL_ADD | 123/128 (23.862) | 128/128 (24.375) | 128/128 (24.375) | 1.2498 | 19 | 207.89 | {'SOLVED': 128} |
| OLD_DENSE | 128/128 (21.773) | 128/128 (21.773) | 128/128 (21.773) | 1.0945 | 72 | 24.41 | {'SOLVED': 128} |
| OLD_WL | 124/128 (23.315) | 128/128 (23.688) | 128/128 (23.688) | 1.2186 | 18 | 132.9 | {'SOLVED': 128} |
| OLD_HADD | 110/128 (26.927) | 128/128 (29.641) | 128/128 (29.641) | 1.4849 | 8 | 509.6 | {'SOLVED': 128} |

## ipc: coverage / mean plan length of solved (node budgets 1,000 / 10,000 / 100,000 within 300 s)

| condition | <= 1,000 | <= 10,000 | <= 100,000 | ratio to proven optimum | optimal plans | mean expansions at solution | statuses (100k) |
|---|---|---|---|---|---|---|---|
| REF_DENSE | 13/22 (37.692) | 14/22 (37.357) | 14/22 (37.357) | 1.1285 | 0 | 207.07 | {'SOLVED': 14, 'TIMEOUT_BEFORE_NODE_LIMIT': 8} |
| FAST_DENSE | 13/22 (37.769) | 14/22 (37.429) | 14/22 (37.429) | 1.1345 | 0 | 207.07 | {'SOLVED': 14, 'TIMEOUT_BEFORE_NODE_LIMIT': 8} |
| ALT_DENSE_ADD | 14/22 (34.286) | 15/22 (36.533) | 15/22 (36.533) | 1.065 | 2 | 626.47 | {'SOLVED': 15, 'TIMEOUT_BEFORE_NODE_LIMIT': 7} |
| ALT_WL_ADD | 12/22 (35.583) | 14/22 (42.571) | 18/22 (45.111) | 1.0898 | 2 | 7805.22 | {'SOLVED': 18, 'UNSOLVED_AT_NODE_BUDGET': 3, 'TIMEOUT_BEFORE_NODE_LIMIT': 1} |
| EAGER_WL | 13/22 (32.769) | 16/22 (37.688) | 18/22 (40.5) | 1.0551 | 2 | 5035.56 | {'SOLVED': 18, 'UNSOLVED_AT_NODE_BUDGET': 4} |
| EAGER_HADD | 5/22 (23.8) | 7/22 (27.714) | 10/22 (45.8) | 1.1993 | 2 | 9293.6 | {'SOLVED': 10, 'UNSOLVED_AT_NODE_BUDGET': 10, 'TIMEOUT_BEFORE_NODE_LIMIT': 2} |
| OLD_DENSE | 13/22 (37.923) | 14/22 (37.571) | 14/22 (37.571) | 1.1345 | 0 | 206.64 | {'SOLVED': 14, 'TIMEOUT_BEFORE_NODE_LIMIT': 8} |
| OLD_WL | 13/22 (32.769) | 16/22 (37.688) | 18/22 (40.5) | 1.0551 | 2 | 5035.56 | {'SOLVED': 18, 'UNSOLVED_AT_NODE_BUDGET': 4} |
| OLD_HADD | 5/22 (23.8) | 7/22 (27.714) | 10/22 (45.8) | 1.1993 | 2 | 9293.6 | {'SOLVED': 10, 'UNSOLVED_AT_NODE_BUDGET': 10, 'TIMEOUT_BEFORE_NODE_LIMIT': 2} |

## IPC: solved within wall-clock seconds

| condition | 10 s | 60 s | 300 s |
|---|---|---|---|
| REF_DENSE | 11 | 14 | 14 |
| FAST_DENSE | 11 | 14 | 14 |
| ALT_DENSE_ADD | 12 | 14 | 15 |
| ALT_WL_ADD | 13 | 16 | 18 |
| EAGER_WL | 16 | 16 | 18 |
| EAGER_HADD | 9 | 10 | 10 |
| OLD_DENSE | 11 | 14 | 14 |
| OLD_WL | 16 | 16 | 18 |
| OLD_HADD | 9 | 10 | 10 |

## paired comparisons at the primary milestone (10000 nodes)

| first | second | set | solved | only first / only second | common: shorter / longer / same (p) | fewer / more expansions (p) |
|---|---|---|---|---|---|---|
| ALT_DENSE_ADD | FAST_DENSE | ipc | 15 / 14 | 2 / 1 | 7 / 2 / 4 (0.18) | 6 / 7 (1) |
| ALT_DENSE_ADD | REF_DENSE | ipc | 15 / 14 | 2 / 1 | 6 / 2 / 5 (0.289) | 6 / 7 (1) |
| ALT_DENSE_ADD | ALT_WL_ADD | struct | 32 / 32 | 0 / 0 | 16 / 4 / 12 (0.0118) | 29 / 1 (0) |
| ALT_DENSE_ADD | ALT_WL_ADD | joint | 128 / 128 | 0 / 0 | 78 / 32 / 18 (1.4e-05) | 123 / 4 (0) |
| ALT_DENSE_ADD | ALT_WL_ADD | ipc | 15 / 14 | 4 / 3 | 7 / 3 / 1 (0.344) | 11 / 0 (0.000977) |
| ALT_DENSE_ADD | EAGER_WL | ipc | 15 / 16 | 2 / 3 | 4 / 7 / 2 (0.549) | 12 / 1 (0.00342) |
| ALT_DENSE_ADD | EAGER_HADD | ipc | 15 / 7 | 8 / 0 | 4 / 1 / 2 (0.375) | 5 / 2 (0.453) |
| ALT_DENSE_ADD | OLD_DENSE | struct | 32 / 32 | 0 / 0 | 0 / 9 / 23 (0.00391) | 0 / 29 (0) |
| ALT_DENSE_ADD | OLD_DENSE | joint | 128 / 128 | 0 / 0 | 15 / 64 / 49 (0) | 11 / 116 (0) |
| ALT_DENSE_ADD | OLD_DENSE | ipc | 15 / 14 | 2 / 1 | 8 / 2 / 3 (0.109) | 6 / 7 (1) |
| ALT_WL_ADD | EAGER_WL | ipc | 14 / 16 | 2 / 4 | 2 / 8 / 2 (0.109) | 10 / 2 (0.0386) |
| ALT_WL_ADD | EAGER_HADD | ipc | 14 / 7 | 7 / 0 | 4 / 1 / 2 (0.375) | 2 / 4 (0.688) |
| ALT_WL_ADD | OLD_WL | struct | 32 / 32 | 0 / 0 | 2 / 5 / 25 (0.453) | 8 / 21 (0.0241) |
| ALT_WL_ADD | OLD_WL | joint | 128 / 128 | 0 / 0 | 18 / 43 / 67 (0.00187) | 35 / 91 (1e-06) |
| ALT_WL_ADD | OLD_WL | ipc | 14 / 16 | 2 / 4 | 2 / 8 / 2 (0.109) | 10 / 2 (0.0386) |
| FAST_DENSE | REF_DENSE | ipc | 14 / 14 | 0 / 0 | 1 / 2 / 11 (1) | 0 / 0 (1) |
| FAST_DENSE | EAGER_WL | ipc | 14 / 16 | 2 / 4 | 1 / 8 / 3 (0.0391) | 12 / 0 (0.000488) |
| REF_DENSE | OLD_DENSE | ipc | 14 / 14 | 0 / 0 | 2 / 0 / 12 (0.5) | 0 / 1 (1) |

## Joint protection line (engineering trade-off, not a statistical claim)

ALT_DENSE_ADD solved 128/128, OLD_DENSE 128/128; mean ratio to optimum 1.1523 vs 1.0945; difference +0.0578 (line: <= +0.05 when all 128 are solved).

## FAST vs REF on IPC22 (same GPU, adjacent, order by case_id hash)

- per-problem scoring throughput (states per scoring-second) ratio FAST/REF: median 1.83, min 1.37, max 2.25 over 22 problems; aggregate 1.74 (states 1098185 vs 643675, scoring seconds 2414 vs 2466)
- both timed out (8 problems): expansions reached in 300 s FAST/REF median 1.75 (min 1.46, max 2.12)
- both solved (14 problems): wall ratio REF/FAST median 1.55; problems with REF wall >= 5 s (3): median 1.76; solved-set total wall REF 95.6 s vs FAST 56.1 s
- solved sets identical: True; same expansion count on the 14 common-solved problems: 14; plan length equal: 11
- order check, REF first (12 problems): median throughput ratio 1.79
- order check, FAST first (10 problems): median throughput ratio 1.89

## IPC: which DENSE failures are guidance, which are truncation (FAST run, 300 s)

| case | FAST status / expansions | WL solved at | h_add solved at | ALT_DENSE | class |
|---|---|---|---|---|---|
| ipc_p06 | TIMEOUT / 28913 | 593 | None | TIMEOUT / None | GUIDANCE_DISADVANTAGE_VS_WL(DENSE explored more expansions than WL needed) |
| ipc_p11 | TIMEOUT / 29127 | 1556 | 19068 | SOLVED / 8452 | GUIDANCE_DISADVANTAGE_VS_WL(DENSE explored more expansions than WL needed) |
| ipc_p12 | TIMEOUT / 21787 | 978 | None | TIMEOUT / None | GUIDANCE_DISADVANTAGE_VS_WL(DENSE explored more expansions than WL needed) |
| ipc_p14 | SOLVED / 1289 | None | None | SOLVED / 83 | DENSE_SOLVED_WL_NOT_SOLVED |
| ipc_p15 | SOLVED / 804 | None | None | TIMEOUT / None | DENSE_SOLVED_WL_NOT_SOLVED |
| ipc_p17 | TIMEOUT / 13559 | 588 | None | TIMEOUT / None | GUIDANCE_DISADVANTAGE_VS_WL(DENSE explored more expansions than WL needed) |
| ipc_p19 | TIMEOUT / 25951 | None | None | TIMEOUT / None | NO_WL_REFERENCE(WL also unsolved) |
| ipc_p20 | TIMEOUT / 9041 | None | None | TIMEOUT / None | NO_WL_REFERENCE(WL also unsolved) |
| ipc_p21 | TIMEOUT / 1804 | 25037 | 139 | SOLVED / 46 | TRUNCATED(DENSE explored fewer expansions than WL needed) |
| ipc_p22 | TIMEOUT / 512 | 48652 | None | TIMEOUT / None | TRUNCATED(DENSE explored fewer expansions than WL needed) |

## ALT cost accounting on common-solved problems (counts; wall only for same-day IPC pairs)

| set | first vs second | common | expansions (first / second) | first DENSE states / second states | first h_add states | encoder forwards (first / second) |
|---|---|---|---|---|---|---|
| struct | ALT_DENSE_ADD vs OLD_DENSE | 32 | 628 / 458 | 2198 / 1694 | 2198 | 624 / None |
| struct | ALT_DENSE_ADD vs ALT_WL_ADD | 32 | 628 / 1411 | 2198 / 4049 | 2198 | 624 / None |
| struct | ALT_DENSE_ADD vs OLD_HADD | 32 | 628 / 2499 | 2198 / 6168 | 2198 | 624 / None |
| struct | ALT_DENSE_ADD vs OLD_WL | 32 | 628 / 1179 | 2198 / 3648 | 2198 | 624 / None |
| joint | ALT_DENSE_ADD vs OLD_DENSE | 128 | 4276 / 3124 | 23175 / 17954 | 23175 | 4274 / None |
| joint | ALT_DENSE_ADD vs ALT_WL_ADD | 128 | 4276 / 26610 | 23175 / 101033 | 23175 | 4274 / None |
| joint | ALT_DENSE_ADD vs OLD_HADD | 128 | 4276 / 65229 | 23175 / 267533 | 23175 | 4274 / None |
| joint | ALT_DENSE_ADD vs OLD_WL | 128 | 4276 / 17011 | 23175 / 69980 | 23175 | 4274 / None |
| ipc | ALT_DENSE_ADD vs FAST_DENSE | 13 | 899 / 2095 | 11097 / 21337 | 11097 | 899 / 2095 |
| ipc | ALT_DENSE_ADD vs ALT_WL_ADD | 14 | 9314 / 65638 | 56292 / 384681 | 56292 | 9403 / None |
| ipc | ALT_DENSE_ADD vs EAGER_HADD | 10 | 8859 / 92936 | 49174 / 291606 | 49174 | 8948 / None |
| ipc | ALT_DENSE_ADD vs EAGER_WL | 14 | 9314 / 39829 | 56292 / 787038 | 56292 | 9403 / None |
