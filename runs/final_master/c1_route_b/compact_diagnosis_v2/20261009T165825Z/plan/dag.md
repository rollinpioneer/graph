# CP-DISR-C1-COMPACT-DIAGNOSIS-V2: realised DAG (plan section 9)

Nodes are sub-commands of `scripts/c1_compact_diagnosis_v2.py`; independent nodes were launched as separate background processes (one GPU each for training / timing tasks, CPU tasks with <= 2 threads).

| node | command | depends on | resource |
|---|---|---|---|
| A1 identities / data roles | `prepare` | - | CPU |
| fixtures (training arms: identity, isolation, C0, ledger) | `fixtures-train` | A1 | 1 GPU, seconds |
| registration 1 | git commit (config, D0 / C0 definition, ledger, panels, thresholds) | fixtures-train | - |
| **D0, C0** | `train --arm D0|C0` | registration 1 | 1 GPU each, parallel |
| W1 data + fixture + fit | `fixture-w1`, `w1-fit` | registration 2 | CPU, 1 fit |
| library B (build, labels, pairs) | `fixture-b`, `lib-build` | registration 3 | CPU |
| library scoring and equivariance, old four models | `lib-score`, `lib-equiv` | lib-build | 1 GPU + CPU |
| paired task changes (build, fixture, old DENSE / old WL searches) | `variants-build`, `fixture-variants`, `panel --set variants` | registration 4 | CPU + 1 GPU |
| part-A tables | `report-a` | all of the above (read only) | CPU |
| **T1 selection** | `trigger-s` (library B, old models only) -> registration 5 | lib-score | CPU |
| **T1 (S)** | `fixture-s`, `train-t1` | registration 5 | 1 GPU, parallel to D0 / C0 |
| Dev24 checkpoint selection | `devsel --arm D0|C0|T1` | the arm's training | 1 GPU, as soon as the arm ends |
| W1 panel (Joint32; IPC22 not comparable) | `panel --arms W1` | W1 fit | CPU |
| D0 / C0 panels (Joint32, IPC22), same GPU per problem | `panel --arms D0,C0` | both selections | GPUs, 4 shards |
| T1 panel | `panel --arms T1` | T1 selection | GPU |
| library scoring and equivariance of the new models | `lib-score`, `lib-equiv` | selections | GPU |
| tables, verify, accounting | `report-b`, `report-c`, `report-panel`, `verify`, `accounting` | all | CPU |

Waiting only happened at true dependencies. The T1 decision never used D0 / C0 / W1 results: it was taken from library B with the old models before any new model was evaluated (registration 5).
