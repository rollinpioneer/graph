# EconomicGrasp 完整复现与 Bad Case 诊断实验方案 V1

- 实验编号：EG-FULL-REPRO-V1。
- 类型：官方 Baseline 完整复现与有界失败案例分析。
- 执行授权：用户于 2026-10-15 直接要求执行本方案；常见工程排障、独立环境安装、非破坏性适配、断点恢复、结果整理及本卡 Git 推送已授权。
- 状态：准备阶段执行中；已确认数据下载阻塞，尚未通过正式训练放行门。
- 论文：An Economic Framework for 6-DoF Grasp Detection，ECCV 2024。
- 官方代码：https://github.com/iSEE-Laboratory/EconomicGrasp 。
- 固定上游提交：`4119bdcd6bf5d3712a110f78ca87504dd359eec0`。
- 本卡分支：`codex/eg-full-repro-v1-20261015`。
- 分支链接：https://github.com/rollinpioneer/graph/tree/codex/eg-full-repro-v1-20261015 。
- 执行服务器：`lab-xushijie3`。
- 隔离工作根目录：`/home/xushijie3/work/eg_full_repro_v1_20261015`。
- Git 归档检出：上述根目录下的 `archive_worktree`；本地交接目录为本文件所在目录。
- 原始产物根目录：上述根目录下的 `runs/EG_FULL_V1`。数据、依赖和外部源码分别独立保存，不进入 Git。

本文件登记用户提供的完整执行要求。不得以安装成功、官方权重可用或少量数据冒烟替代本卡的完整训练与评价。

## 1. 目标与边界

完成官方运行环境、完整训练数据及监督标签准备，从头训练 EconomicGrasp，完成 Kinect 的 Seen、Similar、Novel 全部官方测试，保存可核验 AP、训练成本和初步 Bad Case。完整保存原始数据、训练日志、各轮权重和评价产物，建立后续研究可信 Baseline。

本卡不开发新模块、不改变算法设计、不进行大规模超参数搜索，不因预期性能调整设置，不追加训练种子。完成后先评价 Baseline 可信度、成本、失败案例稳定性及后续研究所缺证据，不自动开展新方法训练。

## 2. 资源和独立环境

启动前实测 GPU 型号、显存与占用、驱动/CUDA/PyTorch、CPU/内存、磁盘和其他任务。不得依赖历史空闲状态、终止或干扰其他任务、覆盖历史结果或改变当前项目之外的工程资产。

使用本项目独立 Conda 环境。优先 Python 3.10、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1，安装支持 CUDA 12 的 MinkowskiEngine、PointNet2、KNN 和 GraspNetAPI。官方 README 是主要安装依据；所有兼容性变动须记录具体版本、补丁和可比性影响，不改共享环境。

优先并行独立数据准备或评测作业。正式训练保留官方单实例 batch=4 及更新次数；不为使用多卡擅自改变全局 Batch Size、优化目标或引入新的训练策略。推理/评价按 Seen、Similar、Novel 顺序安排；各自独立的内部准备可并行。作者 RTX3090 的 8.3 小时和 5.81GB 显存仅作规划参考，不代表实测承诺，也不含下载、标签和完整评价。

## 3. 数据和监督

- 数据：GraspNet-1Billion，仅 Kinect。
- 训练：scene_0000–scene_0099，每场景 256 帧，共 25,600 帧，完整训练集，从头训练。
- Seen：scene_0100–scene_0129，30 场景、7,680 帧。
- Similar：scene_0130–scene_0159，30 场景、7,680 帧。
- Novel：scene_0160–scene_0189，30 场景、7,680 帧。
- 合计完整测试：90 场景、23,040 帧。不得同时增加 RealSense 训练。

先检查现有数据；只复用通过完整性核验的数据，不能以目录存在判定完整。缺少内容按官方渠道下载并确认使用条件，避免重复下载。需要 RGB、Depth、Segmentation/label、Meta、场景姿态与生成监督所需 annotations、训练原始抓取标签、Collision Labels、评测物体模型及官方 API。

按官方语义先运行 `dataset/generate_graspness.py --dataset_root ... --camera_type kinect`，再运行 `dataset/generate_economic.py --dataset_root ... --camera_type kinect`。按实际代码设置执行目录和模块路径。

已知非语义适配：移除 graspness 固定 GPU6 的强制选择，尊重作业的实际 GPU 映射；economic 仅处理 100 个训练场景。可以按场景分片以并行，但必须证明场景集合无重叠遗漏，保持计算和筛选语义。

监督生成后核对数量、形状、数值范围、点/对象/场景对应关系，发现 NaN、Inf、空标签或异常索引必须定位根因，不能静默跳过。Economic Supervision 的筛选原则和阈值保持官方实现；只允许不改变含义的工程优化。完整缓存留存，不重复生成已核验标签。

## 4. 正式训练放行门

环境、数据、监督全部通过后做最小 Smoke Test：依赖导入、CUDA 扩展实际执行、DataLoader 读取数据和标签、模型前向、全部损失有限、反向与参数更新、Checkpoint 保存和重载、显存检查。少量样本和更新只用于工程检查，独立目录保存，不算正式结果。

上述检查全部通过才允许完整训练。任何会影响正确性的未解决问题应停止受影响阶段，保留证据，不强行继续。

## 5. 固定训练协议

| 项目 | 固定值 |
|---|---|
| 相机/数据 | Kinect；GraspNet 完整 Train |
| 模型 | 官方 `economicgrasp`，随机初始化 |
| Epochs / Batch Size | 10 / 4 |
| Optimizer / LR | Adam / 0.001 |
| Num Points / Voxel Size | 20,000 / 0.005 |
| Schedule | 官方按 Epoch 的 Cosine Schedule |
| 预期更新数 | 25,600 / 4 × 10 = 64,000 |
| 本次固定种子 | 20261015（本卡执行前固定） |

TDUnet、GraspableNet、ViewNet、Cylinder Grouping 与特征交互、Grasp Head、损失及监督机制均保留。完整默认参数须按上游 `utils/arguments.py` 实际核对并写入 `train_config.json`。

官方调用形式：`CUDA_VISIBLE_DEVICES=<实际卡> python train.py --model economicgrasp --camera kinect --log_dir <独立训练目录> --max_epoch 10 --batch_size 4 --dataset_root <已核验数据根>`。

固定并记录 Python/NumPy/Torch/CUDA 随机种子；如修复上游每轮无参重新播种行为，保留补丁和说明。断点恢复使用本次实验 Checkpoint，并恢复优化器、Epoch 和必要随机状态，不因中途指标差更换超参数。

保存每轮完整 Loss 与 Objectness、Graspness、View、Angle、Depth、Score、Width 等实际分项，学习率、耗时、GPU 峰值显存、每轮 Checkpoint、异常与处理。保留原始分项名称和聚合口径，不伪造不存在的指标。

OOM 优先采取不改变模型与目标的工程措施。若必须改变 Batch Size 或使用梯度累积等改变训练行为的措施，完整披露并明确称为环境适配版，不能声称完全相同官方配方。不得自动开启第二套训练配方、种子或网络版本。

## 6. 官方测试和评价

完整训练完成后先验证最终 Checkpoint 可加载且能产生有效抓取。随后使用官方 `test.py` 与 GraspNetAPI 完整运行 Seen、Similar、Novel，分别独立输出目录并预建父目录。仅更换 `test_mode` 和输出路径，不改 AP 定义。可以按资源调整评测进程数。

调用形式：`python test.py --model economicgrasp --camera kinect --checkpoint_path <本次最终权重> --dataset_root <数据根> --save_dir <独立划分目录> --test_mode <seen|similar|novel> --inference`。

主要指标 AP Seen/Similar/Novel；辅助报告 AP0.4、AP0.8、推理耗时、每场景 AP、有效抓取数、训练总耗时和峰值显存。官方参考 AP 为 62.59/51.73/19.54，仅作比较，不据此挑种子、改规则或调参。明显差异先审计数据、监督、Checkpoint、64,000 更新、补丁和评价配置。

GraspNet AP 不等于真实机械臂抓取成功率。没有执行的评价必须为 NOT_RUN/null，不能填写官方参考数作为实测结果。

## 7. 有界 Bad Case 分析

完整测试后进行一次有界探索性分析，不据此修改网络重训。检查抓取点遗漏/聚集、方向/角度/深度、评分排序与碰撞风险、遮挡/密集度/可见面积、Novel 未见物体表现。

保存代表性 RGB/Depth、预测抓取姿态图、候选评分、碰撞/质量评价、场景 AP，结合实际网络中间输出及官方评价辨认可能环节。区分观察表现和已证实机制，测试后选择的困难场景明确标为探索性，不能当无偏因果证据。

有界登记：每个测试划分选择最低 AP 的 3 个场景和一个中位 AP 对照场景；每场景选最低帧 AP 的 2 帧及一个中位帧 AP 对照。相同 AP 时按编号排序；最多 12 场景、36 帧，无追加训练。这是执行端的产物选择规则，不修改测试或成功标准。

## 8. 变更、阻塞和归档

授权独立环境安装与版本修复、路径与格式修复、确定性工程错误修复、非语义并行度、官方权重工程检查、同次实验恢复及报告。所有实质补丁保存原实现或相对固定提交的差异，并解释是否影响论文可比性。

不能改变方法/损失、训练测试划分、模型主要参数、AP 定义、成功标准、历史资产、其他用户任务和项目外代码。不能在上述范围内解决的问题停止相应阶段并报告阻塞；不采取改变科学结论的替代方法。

本轮复用一个保活 SSH 长会话；结束后确认本轮 SSH 客户端退出。断线后如必须接管，记录连接故障和接管过程，不因此重启仍在运行的训练。

Git 全程沿用上方本卡分支。先登记并推送，再归档每阶段相关代码和轻量结果；使用明确文件白名单，不提交大型权重、数据、视频和完整日志。每次推送核对实际分支与远端 SHA；不合并、不 force-push，失败或停止也如实归档。

## 9. 交付与最终验收

独立实验目录保存 `environment_report.md`、`data_report.md`、`preflight_report.md`、`train_config.json`、`train_log.txt`、`checkpoints/`、`evaluation_results.json`、`per_scene_results.csv`、`bad_case_report.md`、`final_reproduction_report.md`。另留源码提交、数据身份/哈希、完整补丁、资源记录和原始大产物索引。

- [ ] 环境和主要 CUDA 扩展实际通过。
- [ ] 完整训练数据及两种监督完整且通过内容检查。
- [ ] Smoke Test 全部通过。
- [ ] 原始模型完成 10 Epoch、64,000 次更新。
- [ ] 最终模型 Checkpoint 可重载。
- [ ] Seen、Similar、Novel 完整官方评价。
- [ ] 得到可核验 AP 与每场景指标。
- [ ] 保存训练时间、资源消耗、日志和权重。
- [ ] 完成一次有界 Bad Case 整理。
- [ ] 提交完整复现报告。

未达到全部判据只能报告部分完成或技术阻塞，不得称完整复现成功。完成后仅提出研究决策依据，不提前承诺新算法贡献。
