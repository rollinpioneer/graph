# C1-DEPOTS-SEARCH-MATCH-V1: generated numbers (interpretation is in final_summary.md / claim_boundary.md)

## struct: coverage / mean plan length of solved problems (shared engine)

| scorer | nodes <= 1,000 | nodes <= 10,000 | nodes <= 100,000 (300 s) | mean expansions at solution (100k) | statuses (100k) |
|---|---|---|---|---|---|
| V_DENSE | 32 / 32 (14.219) | 32 / 32 (14.219) | 32 / 32 (14.219) | 14.31 | {'SOLVED': 32} |
| V_MG | 32 / 32 (14.844) | 32 / 32 (14.844) | 32 / 32 (14.844) | 15.25 | {'SOLVED': 32} |
| V_REL | 32 / 32 (14.312) | 32 / 32 (14.312) | 32 / 32 (14.312) | 14.69 | {'SOLVED': 32} |
| H_WL | 32 / 32 (16.156) | 32 / 32 (16.156) | 32 / 32 (16.156) | 36.84 | {'SOLVED': 32} |
| H_ADD | 32 / 32 (18.188) | 32 / 32 (18.188) | 32 / 32 (18.188) | 78.09 | {'SOLVED': 32} |
| H_COUNT | 32 / 32 (20.062) | 32 / 32 (20.062) | 32 / 32 (20.062) | 161.53 | {'SOLVED': 32} |

## joint: coverage / mean plan length of solved problems (shared engine)

| scorer | nodes <= 1,000 | nodes <= 10,000 | nodes <= 100,000 (300 s) | mean expansions at solution (100k) | statuses (100k) |
|---|---|---|---|---|---|
| V_DENSE | 128 / 128 (21.773) | 128 / 128 (21.773) | 128 / 128 (21.773) | 24.41 | {'SOLVED': 128} |
| V_MG | 128 / 128 (22.398) | 128 / 128 (22.398) | 128 / 128 (22.398) | 26.98 | {'SOLVED': 128} |
| V_REL | 128 / 128 (21.539) | 128 / 128 (21.539) | 128 / 128 (21.539) | 25.66 | {'SOLVED': 128} |
| H_WL | 124 / 128 (23.315) | 128 / 128 (23.688) | 128 / 128 (23.688) | 132.9 | {'SOLVED': 128} |
| H_ADD | 110 / 128 (26.927) | 128 / 128 (29.641) | 128 / 128 (29.641) | 509.6 | {'SOLVED': 128} |
| H_COUNT | 88 / 128 (24.727) | 120 / 128 (28.575) | 128 / 128 (29.906) | 1852.17 | {'SOLVED': 128} |

## ipc: coverage / mean plan length of solved problems (shared engine)

| scorer | nodes <= 1,000 | nodes <= 10,000 | nodes <= 100,000 (300 s) | mean expansions at solution (100k) | statuses (100k) |
|---|---|---|---|---|---|
| V_DENSE | 13 / 22 (37.923) | 14 / 22 (37.571) | 14 / 22 (37.571) | 206.64 | {'SOLVED': 14, 'TIMEOUT_BEFORE_NODE_LIMIT': 8} |
| V_MG | 11 / 22 (40) | 14 / 22 (50.571) | 14 / 22 (50.571) | 912.5 | {'SOLVED': 14, 'TIMEOUT_BEFORE_NODE_LIMIT': 8} |
| V_REL | 13 / 22 (37.846) | 15 / 22 (40.8) | 15 / 22 (40.8) | 585.13 | {'SOLVED': 15, 'TIMEOUT_BEFORE_NODE_LIMIT': 7} |
| H_WL | 13 / 22 (32.769) | 16 / 22 (37.688) | 18 / 22 (40.5) | 5035.56 | {'SOLVED': 18, 'UNSOLVED_AT_NODE_BUDGET': 4} |
| H_ADD | 5 / 22 (23.8) | 7 / 22 (27.714) | 10 / 22 (45.8) | 9293.6 | {'SOLVED': 10, 'UNSOLVED_AT_NODE_BUDGET': 10, 'TIMEOUT_BEFORE_NODE_LIMIT': 2} |
| H_COUNT | 3 / 22 (18) | 3 / 22 (18) | 6 / 22 (39.167) | 20021.33 | {'SOLVED': 6, 'UNSOLVED_AT_NODE_BUDGET': 16} |

## paired comparisons at the primary milestone (10000 nodes)

| first | second | set | solved first / second | only first / only second (p) | common | first shorter / longer / same (p) | first fewer / more expansions (p) |
|---|---|---|---|---|---|---|---|
| V_DENSE | H_WL | struct | 32 / 32 | 0 / 0 (1) | 32 | 17 / 1 / 14 (0.000145) | 31 / 0 (0) |
| V_DENSE | H_WL | joint | 128 / 128 | 0 / 0 (1) | 128 | 90 / 16 / 22 (0) | 125 / 3 (0) |
| V_DENSE | H_WL | ipc | 14 / 16 | 2 / 4 (0.688) | 12 | 1 / 8 / 3 (0.0391) | 12 / 0 (0.000488) |
| V_DENSE | H_ADD | struct | 32 / 32 | 0 / 0 (1) | 32 | 24 / 0 / 8 (0) | 30 / 0 (0) |
| V_DENSE | H_ADD | joint | 128 / 128 | 0 / 0 (1) | 128 | 112 / 5 / 11 (0) | 128 / 0 (0) |
| V_DENSE | H_ADD | ipc | 14 / 7 | 8 / 1 (0.0391) | 6 | 3 / 3 / 0 (1) | 3 / 3 (1) |
| V_DENSE | H_COUNT | struct | 32 / 32 | 0 / 0 (1) | 32 | 22 / 1 / 9 (6e-06) | 32 / 0 (0) |
| V_DENSE | H_COUNT | joint | 128 / 120 | 8 / 0 (0.00781) | 120 | 100 / 5 / 15 (0) | 120 / 0 (0) |
| V_DENSE | H_COUNT | ipc | 14 / 3 | 11 / 0 (0.000977) | 3 | 0 / 1 / 2 (1) | 3 / 0 (0.25) |
| V_DENSE | V_MG | struct | 32 / 32 | 0 / 0 (1) | 32 | 10 / 1 / 21 (0.0117) | 17 / 2 (0.000729) |
| V_DENSE | V_MG | joint | 128 / 128 | 0 / 0 (1) | 128 | 59 / 23 / 46 (8.7e-05) | 79 / 21 (0) |
| V_DENSE | V_MG | ipc | 14 / 14 | 2 / 2 (1) | 12 | 8 / 1 / 3 (0.0391) | 10 / 0 (0.00195) |
| V_DENSE | V_REL | struct | 32 / 32 | 0 / 0 (1) | 32 | 2 / 2 / 28 (1) | 4 / 4 (1) |
| V_DENSE | V_REL | joint | 128 / 128 | 0 / 0 (1) | 128 | 29 / 30 / 69 (1) | 32 / 34 (0.902) |
| V_DENSE | V_REL | ipc | 14 / 15 | 2 / 3 (1) | 12 | 7 / 1 / 4 (0.0703) | 4 / 6 (0.754) |
| V_REL | V_MG | struct | 32 / 32 | 0 / 0 (1) | 32 | 12 / 3 / 17 (0.0352) | 17 / 4 (0.0072) |
| V_REL | V_MG | joint | 128 / 128 | 0 / 0 (1) | 128 | 55 / 22 / 51 (0.000217) | 68 / 27 (3.1e-05) |
| V_REL | V_MG | ipc | 15 / 14 | 3 / 2 (1) | 12 | 5 / 5 / 2 (1) | 7 / 3 (0.344) |
| H_WL | H_ADD | struct | 32 / 32 | 0 / 0 (1) | 32 | 14 / 5 / 13 (0.0636) | 24 / 6 (0.00143) |
| H_WL | H_ADD | joint | 128 / 128 | 0 / 0 (1) | 128 | 92 / 15 / 21 (0) | 119 / 9 (0) |
| H_WL | H_ADD | ipc | 16 / 7 | 10 / 1 (0.0117) | 6 | 4 / 1 / 1 (0.375) | 2 / 4 (0.688) |
| H_WL | H_COUNT | struct | 32 / 32 | 0 / 0 (1) | 32 | 17 / 6 / 9 (0.0347) | 32 / 0 (0) |
| H_WL | H_COUNT | joint | 128 / 120 | 8 / 0 (0.00781) | 120 | 75 / 22 / 23 (0) | 118 / 2 (0) |
| H_WL | H_COUNT | ipc | 16 / 3 | 13 / 0 (0.000244) | 3 | 2 / 0 / 1 (0.5) | 3 / 0 (0.25) |
| H_ADD | H_COUNT | struct | 32 / 32 | 0 / 0 (1) | 32 | 12 / 10 / 10 (0.832) | 28 / 4 (1.9e-05) |
| H_ADD | H_COUNT | joint | 128 / 120 | 8 / 0 (0.00781) | 120 | 52 / 51 / 17 (1) | 86 / 33 (1e-06) |
| H_ADD | H_COUNT | ipc | 7 / 3 | 4 / 0 (0.125) | 3 | 2 / 0 / 1 (0.5) | 3 / 0 (0.25) |
