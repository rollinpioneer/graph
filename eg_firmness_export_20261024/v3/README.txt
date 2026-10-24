V3：全部 split 的冻结 A100 预测、同次前向特征和独立直接姿态标签

本目录为本轮有效导出。上级目录 v1/v2 的特征和旧标签不参与本轮。
protocol.json 固定 784 帧（probe 640，cal/C3/P2 各 48）。
模型为官方 Kinect epoch-10，固定 SHA256；eval 模式、is_training=False、TF32 开启。
没有训练、参数更新、视角覆盖、换角选择或换角收益分析；不访问 C4/C5。

infer.py：所有 split 重新按 N0 数据构造与种子前向；两个钩子抓取 vp/ang，
logit 为前 12 个角度类别，和本次原始预测一起缓存。
首先 C3_0100_0000 运行两次，以数组原始字节比较 pred/vp/ang/logit。
通过后正式运行；首次已生成帧不重复推理。
evaluate.py：按 N0 的 owner、后处理选出候选，生成 12 个角度变体。
使用 N0 的 float64 评价口径，原角度格保留原预测值，无旋转运算。
直接调用 eg_p2_i1_mu_20261024/evaluate_p2.py 的 evaluate 函数，不改实现。
额外直接评价缓存原姿态，逐项核验变体原角度的 collision/empty。
R1 使用整幅有效深度点云（depth>0），不使用分割作为 R1 输入；
EG 原检测器 voxel=.005、approach_dist=.05、collision_thresh=.01，24 姿态一批，
return_ious=True 的第 0 个比例。预测及候选生成仍沿用 N0 原有输入预处理。

features/：仅含本次候选的 candidate_id、original_angle、vp、ang、logit。
eval/：本次对应姿态的 candidate_id、original_angle、object_id、pred、
mu、collision、empty、I1、R1、unknown_reason、contact_reason。
candidate_id/original_angle 为 (n,)，pred 为 (n,12,17)，
object_id/mu/collision/empty/I1/R1/原因字段为 (n,12)。object_id 按 12 个角度重复所属物体。
features 与 eval 的 ID、角度、logit argmax 逐帧核验。
receipts/ 保留每帧输入输出 SHA、核验结果与时间。
NaN 保持未知或接触缺失，不填补；mu=-1 与 NaN 区别保留。

复现（在相同服务器环境与历史依赖目录存在时）：
激活 eg_sub25_b_f05_repl_v1_20261019/activate.sh 后再进入本目录。
BLAS/OMP/MKL 线程为 1；infer.py 使用 A100，evaluate.py 禁用 CUDA。
先 python infer.py pilot；通过后 python infer.py 0 2 和 python infer.py 1 2。
三个服务器分别 python evaluate.py 0 4 / 1 4 / 2 4，共 12 个 CPU 进程。
本轮实际执行中，245 的进程池在内存接近耗尽时退出，248 出现大量换页。
停止这两台的本轮 CPU 任务后，在 161 上以 python evaluate.py recover 12
继续它们未完成的帧；161 原来的 4 个进程继续运行，总计 16 个 CPU 进程。
已完成收据与结果不重算；未完成文件如存在则保留在 interrupted_outputs/。
详见 resource_recovery.json。这是运行资源调整，未改变候选或评价标准。
恢复池完成后，将原始 shard 0 尚未完成的帧交给 12 个进程继续，
保留全部已完成结果；详见 resource_recovery_shard0.json。
完成后 python finalize_schema.py，再 python summarize.py。本轮初始工作进程
先保存一维 object_id；finalize_schema.py 仅将其重复成规定的 (n,12) 并更新 SHA，
不改变姿态、特征或物理评价。已是 (n,12) 的文件不改写。
已存在评价结果拒绝覆盖，失败标记会停止后续帧。

summary.json 仅含用户指定的各 split 数据核验计数和耗时。
原角度干净=无碰撞且非空，可行=干净且 0<mu<=.4，弱=干净且 mu>.4。
timing_seconds 中 GPU forward 为各前向累计时间（含第二次一致性核验）；
inference_frame_wall、evaluation_frame_wall 是并行进程逐帧墙钟时间的和，
不是端到端时间；evaluation_cpu 为实际 CPU 时间之和。
逐帧 CPU 和墙钟累计时间仅包括有完成收据的成功任务，不含被中断尝试。
batch_elapsed_from_first_frame_start 为首帧开始至末帧收据的墙钟跨度，
按服务器收据时间估算，不含模型加载、下载及 Git 上传。

Git 保存特征、评价结果、收据、核验汇总及代码；完整原始预测/点云缓存与
1024 行中间特征保留在服务器 v3/cache/、v3/raw_features/，不上传模型或大型缓存。
