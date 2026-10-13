# FLOQ-MIN-REPRO-V2 final run receipt

Status: COMPLETE; B - FUNCTIONAL_REPRODUCED. Both seeds reached 2M with exit0, 101 native evaluations and verified 1M/2M checkpoints.

Official SHA: 3d60e638b42bbef018c56cb1199ff37c3470520d. Branch: codex/floq-min-repro-v2-20261013. ROOT: /home/xushijie3/work/floq_min_repro_v2_20261013.

Frozen native 2M success: seed0 98.0% / seed1 94.0%; mean 96.0%, population std 2.00pp, sample std 2.83pp. Author aggregation remains unverified; no claim of paper statistical reproduction.

See REPRO_IDENTITY.md, PERFORMANCE_REPORT.md, COST_REPORT.md and RESEARCH_ADMISSION.md. All original curves and artifact identities are retained.

## P1 工程证据

独立 5k 更新、50 回合原生评价（第1步）、checkpoint save/restore 通过；参数/优化器/JAX RNG树逐字节身份一致，固定采样 key 的动作完全一致。所有 Actor/Flow/Scalar Critic 梯度有限，Target Flow 梯度为0；目标更新按原版一拍延迟实现，误差0。试跑不计入正式成绩。

P1 GPU 驻留 0.06664 小时，进程生命期 0.06871 小时，显存峰值 980 MiB，进程树 RAM 峰值 2.055 GiB。CPU 真值见 p1/cpu_time.txt；GPU 驻留来自约2秒轮询，存在采样边界误差。

P1 原生评价耗时 104.96 秒；5k 训练循环日志耗时 153.16 秒，扣除该评价后 48.20 秒，约 103.7 更新/秒。此数值仍包含循环日志/验证等开销，非纯内核计时。以此粗估每条2M更新 5.36 小时，加101次原生评价按P1失败回合估计约 2.94 小时；共享资源与策略回合长度会变化，不作为预算或停止阈值。

The earlier P1 cost projection is retained as a historical estimate. Measured final costs are in COST_REPORT.md.

All four prior FQL/GFP runs and Matched evaluations remain complete. Their old score artifacts/branch are not pushed here. Q-Flow remains blocked.

Heartbeat lab is PAUSED after a fresh account-wide completion inspection. This does not authorize new seeds/tasks or module replacement training.
