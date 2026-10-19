# 运行交接
分支 codex/eg-sub25-b-f05-repl-v1-20261019；沿用，不另起分支。
根目录 /aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019
计算 gpu04 / aita_xushijie@172.19.164.161。
已有正式训练PID1815858，控制器1812557；依赖控制器1894651；输入准备队列1892632。禁止重复启动。
状态入口 results/backbone_status.json、results/pipeline_status.json、results/input_preparation_complete.json；正式日志 logs/train_B.log 与 runs/B/train/batch_metrics.jsonl、epoch_metrics.jsonl。
用户新并行授权见 resource_parallel_amendment.json；既有P1运行不受修改，后续按新manifest保护代码。
流水线完成状态 COMPLETED_PENDING_GIT_ARCHIVE 表示计算与必要验证结束，仍须按本分支归档最终轻量结果、核验本地/远端SHA，再更新交接。TECHNICAL_STOP需读具体日志，不能当负面科研结果，不自动追加训练。
无定时监控/自动任务。本轮结束需退出SSH，保留nohup工作。
