# C1-PUBLIC-DEPOTS-REL-V1: auto-generated numbers (interpretation is in final_summary.md / claim_boundary.md)

## struct

| method | solved / n | failure-penalised cost ratio | cost ratio (solved) | total cost (solved) | excess over reference (solved) | mean wall s |
|---|---|---|---|---|---|---|
| MG-G/one_step | 32 / 32 | 1.1424 | 1.1424 | 515 | 70 | 0.397 |
| MG-G/two_step | 32 / 32 | 1.1057 | 1.1057 | 495 | 50 | 1.168 |
| MG-G/gated | 32 / 32 | 1.0664 | 1.0664 | 478 | 33 | 0.753 |
| DENSE-G/one_step | 32 / 32 | 1.0288 | 1.0288 | 459 | 14 | 1.233 |
| DENSE-G/two_step | 32 / 32 | 1.0427 | 1.0427 | 465 | 20 | 2.613 |
| DENSE-G/gated | 32 / 32 | 1.0421 | 1.0421 | 463 | 18 | 0.355 |
| REL-G/one_step | 32 / 32 | 1.0578 | 1.0578 | 469 | 24 | 1.48 |
| REL-G/two_step | 32 / 32 | 1.0292 | 1.0292 | 461 | 16 | 2.688 |
| REL-G/gated | 32 / 32 | 1.0382 | 1.0382 | 462 | 17 | 0.872 |
| LAMA-first | 32 / 32 | 1.2274 | 1.2274 | 559 | 114 | 0.009 |
| LAMA-anytime | 32 / 32 | 1.0 | 1.0 | 445 | 0 | 0.014 |
| WL-GOOSE | 32 / 32 | 1.1735 | 1.1735 | 517 | 72 | 0.662 |

## joint

| method | solved / n | failure-penalised cost ratio | cost ratio (solved) | total cost (solved) | excess over reference (solved) | mean wall s |
|---|---|---|---|---|---|---|
| MG-G/one_step | 119 / 128 | 1.3248 | 1.2546 | 2914 | 593 | 0.927 |
| MG-G/two_step | 123 / 128 | 1.2271 | 1.1853 | 2857 | 451 | 1.137 |
| MG-G/gated | 126 / 128 | 1.1781 | 1.1621 | 2858 | 401 | 1.031 |
| DENSE-G/one_step | 125 / 128 | 1.164 | 1.1396 | 2802 | 374 | 2.286 |
| DENSE-G/two_step | 125 / 128 | 1.1448 | 1.1197 | 2768 | 336 | 1.723 |
| DENSE-G/gated | 127 / 128 | 1.1191 | 1.1108 | 2806 | 324 | 1.839 |
| REL-G/one_step | 119 / 128 | 1.1971 | 1.1198 | 2643 | 343 | 2.696 |
| REL-G/two_step | 116 / 128 | 1.221 | 1.1152 | 2553 | 301 | 5.805 |
| REL-G/gated | 119 / 128 | 1.17 | 1.0886 | 2555 | 236 | 1.356 |
| LAMA-first | 128 / 128 | 1.3498 | 1.3498 | 3458 | 947 | 0.03 |
| LAMA-anytime | 128 / 128 | 1.007 | 1.007 | 2536 | 25 | 0.022 |
| WL-GOOSE | 128 / 128 | 1.2106 | 1.2106 | 3009 | 498 | 0.601 |

## ipc

| method | solved / n | failure-penalised cost ratio | cost ratio (solved) | total cost (solved) | excess over reference (solved) | mean wall s |
|---|---|---|---|---|---|---|
| MG-G/one_step | 4 / 22 | 2.0258 | 1.5826 | 150 | 60 | 31.35 |
| MG-G/two_step | 9 / 22 | 1.7244 | 1.1554 | 406 | -26 | 80.032 |
| MG-G/gated | 8 / 22 | 1.7431 | 1.1023 | 254 | -90 | 66.971 |
| DENSE-G/one_step | 8 / 22 | 1.8132 | 1.2796 | 329 | 69 | 23.027 |
| DENSE-G/two_step | 10 / 22 | 1.6529 | 1.1119 | 326 | 17 | 114.961 |
| DENSE-G/gated | 11 / 22 | 1.5974 | 1.0885 | 404 | 29 | 74.085 |
| REL-G/one_step | 7 / 22 | 1.901 | 1.4569 | 245 | 80 | 28.349 |
| REL-G/two_step | 11 / 22 | 1.631 | 1.1602 | 396 | 47 | 70.809 |
| REL-G/gated | 12 / 22 | 1.6065 | 1.1983 | 622 | 130 | 91.948 |
| LAMA-first | 20 / 22 | 1.4241 | 1.3613 | 1270 | 343 | 34.081 |
| LAMA-anytime | 21 / 22 | 1.0489 | 1.0024 | 939 | 2 | 14.737 |
| WL-GOOSE | 20 / 22 | 1.0683 | 0.9688 | 798 | -175 | 44.8 |

Wall-clock columns: internal controllers = policy time (scoring-only label time excluded); WL-GOOSE = whole planner run; LAMA-first = time of its first plan; LAMA-anytime = time of ITS first plan inside the 300 s run (the full run lasts up to 300 s, see compute_costs.csv). The machine and its GPUs were shared with other users, so wall times are indicative only.

## primary and secondary paired comparisons (set ALL = struct + joint + ipc; per set in paired_costs.csv)

| first | second | set | n | first solved | second solved | only first | only second | common | first cheaper | first costlier | same | mean diff (common) | sign p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REL-G/one_step | DENSE-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 4 | 24 | 0.312 | 1.0 |
| REL-G/one_step | DENSE-G/one_step | joint | 128 | 119 | 125 | 1 | 7 | 118 | 29 | 27 | 62 | 0.195 | 0.893853 |
| REL-G/one_step | DENSE-G/one_step | ipc | 22 | 7 | 8 | 1 | 2 | 6 | 3 | 3 | 0 | 2.333 | 1.0 |
| REL-G/one_step | MG-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 17 | 4 | 11 | -1.438 | 0.007197 |
| REL-G/one_step | MG-G/one_step | joint | 128 | 119 | 119 | 8 | 8 | 111 | 61 | 22 | 28 | -2.063 | 2.2e-05 |
| REL-G/one_step | MG-G/one_step | ipc | 22 | 7 | 4 | 3 | 0 | 4 | 2 | 0 | 2 | -4 | 0.5 |
| DENSE-G/one_step | MG-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 18 | 2 | 12 | -1.75 | 0.000402 |
| DENSE-G/one_step | MG-G/one_step | joint | 128 | 125 | 119 | 7 | 1 | 118 | 71 | 16 | 31 | -1.975 | 0.0 |
| DENSE-G/one_step | MG-G/one_step | ipc | 22 | 8 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | -11.25 | 0.125 |
| REL-G/two_step | DENSE-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 4 | 24 | -0.125 | 1.0 |
| REL-G/two_step | DENSE-G/two_step | joint | 128 | 116 | 125 | 2 | 11 | 114 | 25 | 22 | 67 | 0.105 | 0.770867 |
| REL-G/two_step | DENSE-G/two_step | ipc | 22 | 11 | 10 | 1 | 0 | 10 | 3 | 3 | 4 | 2.1 | 1.0 |
| REL-G/two_step | MG-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 16 | 1 | 15 | -1.062 | 0.000275 |
| REL-G/two_step | MG-G/two_step | joint | 128 | 116 | 123 | 4 | 11 | 112 | 46 | 33 | 33 | -0.714 | 0.176611 |
| REL-G/two_step | MG-G/two_step | ipc | 22 | 11 | 9 | 3 | 1 | 8 | 5 | 1 | 2 | -3.75 | 0.21875 |
| DENSE-G/two_step | MG-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 16 | 3 | 13 | -0.938 | 0.004425 |
| DENSE-G/two_step | MG-G/two_step | joint | 128 | 125 | 123 | 4 | 2 | 121 | 51 | 33 | 37 | -0.826 | 0.062972 |
| DENSE-G/two_step | MG-G/two_step | ipc | 22 | 10 | 9 | 2 | 1 | 8 | 5 | 0 | 3 | -6.125 | 0.0625 |
| REL-G/gated | DENSE-G/gated | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 5 | 23 | -0.031 | 1.0 |
| REL-G/gated | DENSE-G/gated | joint | 128 | 119 | 127 | 0 | 8 | 119 | 22 | 24 | 73 | -0.067 | 0.882996 |
| REL-G/gated | DENSE-G/gated | ipc | 22 | 12 | 11 | 3 | 2 | 9 | 3 | 3 | 3 | 1.444 | 1.0 |
| REL-G/gated | MG-G/gated | struct | 32 | 32 | 32 | 0 | 0 | 32 | 9 | 2 | 21 | -0.5 | 0.06543 |
| REL-G/gated | MG-G/gated | joint | 128 | 119 | 126 | 2 | 9 | 117 | 51 | 29 | 37 | -1.009 | 0.018316 |
| REL-G/gated | MG-G/gated | ipc | 22 | 12 | 8 | 4 | 0 | 8 | 5 | 2 | 1 | 22 | 0.453125 |
| DENSE-G/gated | MG-G/gated | struct | 32 | 32 | 32 | 0 | 0 | 32 | 11 | 2 | 19 | -0.469 | 0.022461 |
| DENSE-G/gated | MG-G/gated | joint | 128 | 127 | 126 | 2 | 1 | 125 | 56 | 29 | 40 | -0.88 | 0.004512 |
| DENSE-G/gated | MG-G/gated | ipc | 22 | 11 | 8 | 4 | 1 | 7 | 3 | 2 | 2 | -2 | 1.0 |
| MG-G/two_step | MG-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 12 | 4 | 16 | -0.625 | 0.076813 |
| MG-G/two_step | MG-G/one_step | joint | 128 | 123 | 119 | 7 | 3 | 116 | 57 | 30 | 29 | -1.716 | 0.005014 |
| MG-G/two_step | MG-G/one_step | ipc | 22 | 9 | 4 | 5 | 0 | 4 | 4 | 0 | 0 | -11.5 | 0.125 |
| MG-G/gated | MG-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 10 | 1 | 21 | -0.531 | 0.011719 |
| MG-G/gated | MG-G/two_step | joint | 128 | 126 | 123 | 3 | 0 | 123 | 49 | 22 | 52 | -0.61 | 0.00182 |
| MG-G/gated | MG-G/two_step | ipc | 22 | 8 | 9 | 1 | 2 | 7 | 7 | 0 | 0 | -9.286 | 0.015625 |
| MG-G/gated | MG-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 13 | 0 | 19 | -1.156 | 0.000244 |
| MG-G/gated | MG-G/one_step | joint | 128 | 126 | 119 | 8 | 1 | 118 | 61 | 17 | 40 | -2.136 | 1e-06 |
| MG-G/gated | MG-G/one_step | ipc | 22 | 8 | 4 | 5 | 1 | 3 | 3 | 0 | 0 | -8.667 | 0.25 |
| DENSE-G/two_step | DENSE-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 4 | 24 | 0.188 | 1.0 |
| DENSE-G/two_step | DENSE-G/one_step | joint | 128 | 125 | 125 | 1 | 1 | 124 | 25 | 38 | 61 | -0.371 | 0.129918 |
| DENSE-G/two_step | DENSE-G/one_step | ipc | 22 | 10 | 8 | 2 | 0 | 8 | 7 | 0 | 1 | -7.625 | 0.015625 |
| DENSE-G/gated | DENSE-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 1 | 1 | 30 | -0.062 | 1.0 |
| DENSE-G/gated | DENSE-G/two_step | joint | 128 | 127 | 125 | 3 | 1 | 124 | 39 | 10 | 75 | -0.435 | 3.8e-05 |
| DENSE-G/gated | DENSE-G/two_step | ipc | 22 | 11 | 10 | 1 | 0 | 10 | 6 | 1 | 3 | 1.7 | 0.125 |
| DENSE-G/gated | DENSE-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 3 | 25 | 0.125 | 1.0 |
| DENSE-G/gated | DENSE-G/one_step | joint | 128 | 127 | 125 | 3 | 1 | 124 | 29 | 27 | 68 | -0.839 | 0.893853 |
| DENSE-G/gated | DENSE-G/one_step | ipc | 22 | 11 | 8 | 3 | 0 | 8 | 8 | 0 | 0 | -5.25 | 0.007812 |
| REL-G/two_step | REL-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 4 | 5 | 23 | -0.25 | 1.0 |
| REL-G/two_step | REL-G/one_step | joint | 128 | 116 | 119 | 6 | 9 | 110 | 27 | 29 | 54 | -0.482 | 0.893853 |
| REL-G/two_step | REL-G/one_step | ipc | 22 | 11 | 7 | 4 | 0 | 7 | 6 | 1 | 0 | -5.286 | 0.125 |
| REL-G/gated | REL-G/two_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 3 | 4 | 25 | 0.031 | 1.0 |
| REL-G/gated | REL-G/two_step | joint | 128 | 119 | 116 | 3 | 0 | 116 | 39 | 9 | 68 | -0.655 | 1.5e-05 |
| REL-G/gated | REL-G/two_step | ipc | 22 | 12 | 11 | 2 | 1 | 10 | 6 | 0 | 4 | -2.1 | 0.03125 |
| REL-G/gated | REL-G/one_step | struct | 32 | 32 | 32 | 0 | 0 | 32 | 3 | 4 | 25 | -0.219 | 1.0 |
| REL-G/gated | REL-G/one_step | joint | 128 | 119 | 119 | 6 | 6 | 113 | 30 | 20 | 63 | -1.106 | 0.202639 |
| REL-G/gated | REL-G/one_step | ipc | 22 | 12 | 7 | 5 | 0 | 7 | 7 | 0 | 0 | -7.571 | 0.015625 |
