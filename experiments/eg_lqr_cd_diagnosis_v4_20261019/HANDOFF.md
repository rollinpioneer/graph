# V4交接：有界CD退化诊断与固定F05条件实验

用户已授权执行。入口aita_xushijie@172.19.164.161（gpu04）。根目录/aita_data/home/aita_xushijie/work/grasp/eg_lqr_cd_diagnosis_v4_20261019；source activate_gpu04.sh；沿用V3独立环境，V3结果只读。

分支codex/eg-lqr-cd-diagnosis-v4-20261019，父结果4bdf15a9b3a844929ccea5c14317c5f495a2768b，卡登记72979ff1860bfb327a174df21ca2014a4cc1527a。实验卡先于新计算推送。无新训练，不改姿态，不搜系数。严格固定F05=.5z(EG)+.5z(Q)，标准差/精度处理沿用V3。

先读取C1完整指标做CD效应分解，候选诊断固定608帧。诊断实现初次将CD候选转float64，0102边界检查失败；最终全部608帧用原始dtype统一重放，旧尝试保留。此前元数据available池被审计误判为暴露，已根据原清单明确说明修复并保留失败审计。当前C2预留来自未有已知性能/训练使用的场景：Seen128/105/127/112，Similar147/142/143/131，Novel165/166/168/184；已核验全部原始文件。预留不是放行，未打开新性能。

诊断条件已满足：0136三个头223/207/180帧EG更高AP且实际Top1不同。候选诊断成功汇总后执行固定F05完整C1探索，再按卡内冻结的四项条件决定C2；不事后切换S或调系数。run_pipeline.py完成后写exploration_decision、C2_method_lock（如放行）、confirmation_decision、pipeline_status；build_report.py生成研究报告。pipeline内禁止以C1探索作为独立确认。

计算上限8进程：7CPU评价+1控制；C2导出阶段GPU1+CPU6+控制。每任务1线程低优先级，不干扰其他进程。0训练，CPU worker20小时、GPU导出0.75进程小时、产物12GiB。无自动监控。当前D统一重放在运行，F05与C2尚未执行；后续可检查results/diagnosis_status.json、pipeline_status.json及logs/。

正式F05运行前相关代码身份写code_manifest.json。大候选CSV、质量NPZ、输入及日志留服务器；最终补充轻量结果、哈希与报告到本分支，不覆盖原结果。

资源补充（首个F05性能前）：112逻辑CPU，忙31.42%，内存可用451GiB；按用户能并行就拉满的持续授权，F05/C2改为12场景评价+控制共13进程，导出时GPU1+CPU11+控制。D峰值8，科学标准/预算不变，见resource_amendment.json。只停止无子任务的等待调度器1066154并以同配方新资源恢复。

F05探索完整3072帧已完成：相对EG+2.141003AP，描述性95%CI[1.472968,2.842484]，三头+2.065991/+2.085615/+2.271404，12/12场景正向。0136从S的-7.201563变为F05的+0.140414。F05相对S平均+0.708194但95%CI跨0，Novel低1.002741AP，不能声称处处更好。四个冻结探索闸门均通过。
C2_method_lock已在新导出/性能前写入，锁定F05和既定12场景；C2原始标注3072份、相机矩阵和58物体模型可用性检查PASS。C2 GPU1新鲜快照空闲8798MiB、利用率0，导出少量占用，未操作其他任务。C2当前仅输入导出和就绪场景CPU评价在进行，结果尚未知，不把探索CI作为独立确认。控制PID1082824，运行日志logs/pipeline_resource_amended.log；export日志logs/C2_export.log。
