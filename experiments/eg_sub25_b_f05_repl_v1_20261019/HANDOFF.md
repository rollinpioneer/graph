# EG-SUB25-B-F05-REPL-V1运行交接
分支codex/eg-sub25-b-f05-repl-v1-20261019，继续沿用。
研究根目录/aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019；gpu04 aita_xushijie@172.19.164.161。
当前同一seed20261015实验，第7轮后从GPU6迁到GPU1；新训练PID2849958，迁移/续训控制器2806952，依赖流水线PID2849959，禁止重复启动。旧1815858/1812557/1894651已按迁移计划退出，不再恢复。
状态入口results/backbone_status.json、pipeline_status.json、GPU_migration_status.json。原始迁移现场及重放说明见results/GPU_migration_receipt.json和GPU_migration_report.md；旧日志和权重保留。
后续标签最多32CPU、评价24任务、GPU推理最多8进程；启动时按实时资源准入。无定时监控。训练完成后流水线自行生成标签、训练3头、冻结后评价C3。
COMPLETED_PENDING_GIT_ARCHIVE仅表示计算验证完成，仍须最终结果Git归档与远端SHA核验。TECHNICAL_STOP按日志处理，不自动追加训练。
