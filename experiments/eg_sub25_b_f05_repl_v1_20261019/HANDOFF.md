# EG-SUB25-B-F05-REPL-V1当前进度
根目录/aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019；分支codex/eg-sub25-b-f05-repl-v1-20261019继续沿用，禁止另起分支或重复训练。
EG-B已完成10epoch/16000更新，最终权重/aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019/runs/B/train/economicgrasp_epoch10.tar，SHA256 533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98；迁移/续训控制器2806952已自然退出。
流水线PID2849959现处P2_LABELS_PARALLEL，32 CPU worker；B候选32场景×8帧已导出完成。当前标签进度以results/labels_*.json和logs/labels_*.log为准。
后续自动训练3个固定1000步头，再冻结所有权重和C3方法，执行同C3的A/B共16配置评价。不得重训EG或用C3调参。
状态入口results/pipeline_status.json；结果COMPLETED_PENDING_GIT_ARCHIVE后仍须同分支归档最终轻量结果和SHA核验。TECHNICAL_STOP应先读日志，不能将技术失败视为科学阴性。
无定时监控。资源上限GPU8h、CPUworker50h、25GiB；非训练并行已由用户改为CPU32、评价24、GPU导出最多8进程，实际按实时资源准入。
