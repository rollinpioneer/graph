# R14-B 新 Agent 资源接手说明 V1.0

请先完整阅读同包 `MANUAL.md`。本轮默认物理授权为0，不得直接运行A/B/C。

## 固定Git入口

```text
repository:
rollinpioneer/graph

base branch:
maintenance/l2rar2-r14-ordinary002-forensics-v1

base commit:
ab024c71d315100f0e375a844856807114bedaab

formal main:
234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2

new branch:
research/l2rar2-r14b-reproducible-baseline-v1
```

## 本机优先寻找

```text
shared clone:
/home/__compress_data/xushijie/graph_github_upload

forensic worktree:
/home/__compress_data/xushijie/graph_l2ra_r2_r14_forensics_v1_worktree

controlled-drop worktree:
/home/__compress_data/xushijie/graph_l2ra_r2_forced_drop_v1_worktree

recommended new development worktree:
/home/__compress_data/xushijie/graph_l2ra_r2_r14b_baseline_v1_worktree

recommended fixed execution worktree:
/home/__compress_data/xushijie/graph_l2ra_r2_r14b_exec_v1_worktree

recommended external output root:
/home/__compress_data/xushijie/graph_l2ra_r2_r14b_baseline_v1_data
```

## 必须读取

```text
artifacts/pathgraph_sarm/upgrade_v2/replay_consistency_l2rar2_r14_v1/
forensics/ordinary_002_first_divergence_v1/
  final_report.md
  review_decision.json
  source_provenance.json
  next_stage_handoff.json

upgrade_v2/l2r_replay_consistency/
upgrade_v2/l2r_forced_drop/
upgrade_v2/l2r_task_context/collector.py
upgrade_v2/l2r_hold_evidence/probe_adapter.py
upgrade_v2/visual_refine_l2/dynamic_simulator.py
upgrade_v2/visual_refine_l2/repaired_simulator.py
upgrade_v2/visual_refine_l2/renderer.py
upgrade_v2/visual_refine_l2/vision.py
```

## 固定实验

```text
root_family_id = L2RAR2_REPRO_BASE_00_870000
family_index   = 0
family_seed    = 870000
rollout_seed   = 87100002
case_id        = K3_normal_hold_pause_resume
control_variant= v0_short
actions        = 8
controls       = 29
renders        = 37
physics steps  = 145
```

## 当前不得执行

```text
ordinary A
ordinary B
instrumented C
R16 calibration
R16 development
任何无授权的MuJoCo model/data/renderer构造
任何旧nonce复用
任何容差放宽
```

先完成：

1. `upgrade_v2/l2r_reproducible_baseline/`
2. 至少60项纯测试；
3. runner commit并推送；
4. detached execution worktree；
5. protocol/source/environment锁；
6. A授权申请模板，authorized=0。

用户签署A后只运行A；A人工通过后才准备B授权；B完全通过后才准备C授权。
