# Implementation receipt — FLOW-CRITIC-DUAL-HOST-PREP-V1

Status: **ZERO_TRAIN_VERIFIED**. GFP + floq Critic and DFQL + floq Critic implementation and zero-training functional acceptance are complete, ready for a separately authorized training design.

**This card executed 0 optimizer/parameter updates, 0 training runs and no online learning.** GPU work consisted solely of forward, loss, gradient and JIT checks. New random-state tests use initialized state step 1, unchanged afterward; restored historical model step values are also unchanged.

Server root: `/home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014`.
Local archive branch: `codex/flow-critic-dual-host-prep-v1-20261014`. Preparation initially completed without push, PR or merge. The subsequent user instruction “推送” authorizes publication of this registered branch only: https://github.com/rollinpioneer/graph/tree/codex/flow-critic-dual-host-prep-v1-20261014. PR/merge and training remain unauthorized. Original primary repository branch, HEAD and working status are unchanged.

Completed: required pinned sources and private host environments, official data loading/relabeling for four tasks, existing-asset identity audit, namespaced floq expected-value core, host adapters, 12 complete G0/G1/D0/D1 task configurations and six differences, and Z0–Z8 verification. Original host files and network utilities are unchanged; source snapshot files are read-only.

| Task | Train transitions | Observation / action dims | Actual rewards |
|---|---:|---|---|
| humanoidmaze-medium-navigate-singletask-task1-v0 | 2,000,000 | 69 / 21 | [-1.0, 0.0] |
| cube-double-play-singletask-task2-v0 | 1,000,000 | 37 / 5 | [-2.0, 0.0] |
| puzzle-3x3-play-singletask-task4-v0 | 1,000,000 | 55 / 5 | [-9.0, 0.0] |
| antmaze-large-navigate-singletask-task1-v0 | 1,000,000 | 29 / 8 | [-1.0, 0.0] |

Host GPU suites: GFP 24 checks, DFQL 26 checks, all PASS. Original scalar path controls compare outputs, losses and all parameter gradients exactly (atol=rtol=0). Both suites check real 256-transition batches. Flow/scalar/Actor/VaBC gradient routes are isolated; targets have zero parameter gradients. Synthetic tests cover noise SUM, terminal masking, aggregation ordering, current/next state branches, multi-step TD policy, clipped-endpoint action derivative and target EMA maps. Additional pinned-source critic-loss controls and original floq value mapping are included.

44 old checkpoint files are byte-hash checked before and after. Existing final FQL/GFP seeds 0/1 and floq 1M/2M seeds 0/1 restore read-only in their original environments; Q and action derivatives remain finite. Both new randomly initialized combinations serialize and restore independently. Existing weights are not treated as valid new-host value functions.

GFP CSV matching uses the actual official routine after OmegaConf parsing, no CLI overrides. Humanoid η=0.001 agrees with the card. The earlier generic YAML parse failure is retained as an engineering failure receipt and occurred before parameter updates. floq's actual time input is cos(t), one channel, with inherited inert time_embed_dim=64. See the source/environment manifest for all differences and resolutions.

Unavailable: the two named prior report files. They are logged as MISSING_USER_ACCESS; the card and pinned sources supply the implemented contract. DFQL trained weights are NOT_YET_TRAINED. New-task Baselines are not reproduced. Dataset/weight files and full logs stay on the server and are not committed as lightweight artifacts.

This acceptance makes no claim about performance, superiority, training stability, full-training cost, paper reproduction of the combinations or novelty. The next decision must separately authorize Baseline admission, training/evaluation protocols and any scope or seed expansion. The current preparation stops here.
