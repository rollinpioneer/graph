# CUPID 研究方案总览：从最小复现到稳定过滤验证

版本日期：2026-07-23  
工作根目录：`/home/xushijie/CUPID`  
研究状态：Square-MH 分支完成并停止；Transport-MH 预注册实验正在运行

## 1. 一句话概括

本研究不是直接假设 CUPID 的最低分 Demo 就是“坏数据”，而是依次验证：

1. 能否复现 CUPID 的训练、rollout、TRAK 和 Demo 评分链路；
2. CUPID 排名及过滤名单在有限 rollout 下是否稳定；
3. Bootstrap 稳定核心是否比简单按分数排序更可靠；
4. 只有前述离线证据强通过，才从头进行过滤数据后的配对重训练，检验删除这些 Demo 是否真的改善策略性能。

当前完整主线为：

```text
Square-MH 最小复现
  -> Square-MH 排名与过滤稳定性诊断
  -> Square 固定过滤分支失败并停止
  -> Transport-MH 基础策略
  -> 固定 100 条 rollout
  -> 原始 CUPID/TRAK 评分
  -> 独立稳定性诊断
  -> 强通过才冻结过滤 ID
  -> 未过滤/过滤配对重训练
  -> 通过后再扩展多种子
```

## 2. 研究问题

### 2.1 核心问题

CUPID 基于策略表现和训练样本影响给 Demo 排序。真正需要验证的不是“能否计算出分数”，而是：

- 这些分数是否能够稳定区分低价值或有害 Demo；
- 排名最低集合是否对 rollout 随机性敏感；
- 复杂的 Bootstrap 成员选择是否优于一次完整分数排序；
- 删除被选 Demo 后，重新训练的策略是否在未参与筛选的评估上变好。

### 2.2 研究假设

- `H1`：CUPID 的整体 Demo 排名会随 rollout 数量增加而稳定。
- `H2`：整体排名稳定不等于固定 Top/Bottom 集合稳定，边界附近仍可能高度不确定。
- `H3`：若存在真实、强烈的低分信号，Bootstrap 应能找到跨 rollout 子池稳定的低分核心。
- `H4`：Bootstrap 稳定核心必须优于“直接选相同数量最低分 Demo”，否则复杂方法没有额外价值。
- `H5`：只有稳定核心通过独立诊断后，过滤重训练才可能构成可信的因果验证。
- `H6`：影响信号可能具有任务依赖性；Square-MH 失败不应直接外推为所有任务上的失败，因此需要 Transport-MH 验证。

## 3. 作者方法与本研究的关系

### 3.1 CUPID 原始操作流程

本项目复现的 CUPID 核心流程是：

1. 训练基础策略；
2. 收集带有成功/失败标签的在线 rollout；
3. 使用 TRAK 计算训练样本对 rollout 决策的影响；
4. 聚合样本影响到 Demo 级别；
5. 根据 rollout 表现构造 performance influence；
6. 得到每条训练 Demo 的净影响分数；
7. 按分数排序，尝试一个或多个过滤比例；
8. 删除对应 Demo 后重新训练策略并比较性能。

在本项目的实现中，官方净分数的操作性重建为：

```text
成功 rollout 的影响加总
失败 rollout 的影响减去
```

重建分数必须与官方输出对齐，之后才允许进行稳定性分析。

### 3.2 本研究增加的控制

作者流程能够产生排序，但仅凭一次排序不能证明过滤名单可靠。本研究增加：

- 固定 rollout seed 和完整文件审计；
- rollout 预算稳定性分析；
- 互不重叠 rollout 子池压力测试；
- 固定比例边界平坦度分析；
- Bootstrap 删除概率和稳定核心；
- 相同选择数量的简单最低分基线；
- 离线强门槛和预注册停止规则；
- validation/holdout 不变的配对重训练；
- 哈希、PID、状态文件和恢复谱系。

这些控制的目的不是改变 CUPID 分数，而是判断该分数是否足以支持“删除哪些 Demo”的决策。

## 4. 数据、任务和统一协议

两个任务均使用 RoboMimic MH、低维输入和 Diffusion Policy。

### 4.1 统一数据划分

```text
原始 Demo：300
训练 / validation / holdout：192 / 12 / 96
dataset seed：0
train_ratio：0.64
val_ratio：0.04
uniform_quality：true
```

过滤只允许改变 192 条训练 Demo。12 条 validation 和 96 条 holdout 必须逐位保持不变。

### 4.2 Square-MH

Square-MH 用于最小复现、完整影响链路验证和第一轮过滤稳定性诊断。

### 4.3 Transport-MH

Transport-MH 用于检验影响信号是否具有任务依赖性。其固定输入为：

```text
数据集：repo/data/robomimic/datasets/transport/mh/low_dim_abs.hdf5
数据 SHA-256：2034f404d1e9dd04514c443f9b2fb2bda99f320b46acd9ffc9983fdbec0f9d95
原始时序步：195800
输入 / 动作维度：59 / 20
模型参数：17181204
训练样本：125324
每 epoch 训练 batch：490
rollout 最大步数：700
```

## 5. 已完成阶段：Square-MH 最小复现

### 5.1 完整链路结果

Square-MH 后续监督恢复实验已经完整跑通：

- 训练 epoch：1751（0--1750）；
- 正式训练时间：39788 秒，约 11 小时 3 分；
- 固定 rollout：100 条，seeds 100000--100099；
- 成功 / 失败：71 / 29；
- rollout 时间：5670 秒，约 1 小时 34 分；
- TRAK 时间：4326 秒，约 1 小时 12 分；
- TRAK 原始矩阵：`75129 x 4066`；
- Rollout-Demo 矩阵：`100 x 192`；
- 官方分数重建最大误差：`7.1316965e-06`。

注意：`results/CUPID_minrep_final_report.md` 记录的是早期第一次正式训练因 DataLoader worker `SIGKILL` 而停止的历史结果。后续按用户授权加入监督恢复并完成了全链路；当前 Square 结果应以 `reports/minimal_reproduction_summary_20260722.md` 和冻结后的 Stage 2/2B 报告为准。

### 5.2 Rollout 预算结论

与完整 100-rollout 排名比较：

| Rollout 预算 | 随机 Spearman | 随机 Top-20% Jaccard |
|---:|---:|---:|
| 5 | 0.214 | 0.156 |
| 10 | 0.332 | 0.190 |
| 25 | 0.540 | 0.280 |
| 50 | 0.768 | 0.431 |
| 100 | 1.000 | 1.000 |

结论：

- 少量 rollout 不足以恢复可靠排序；
- 50 条 rollout 已能较好恢复整体排名；
- 但固定 Top/Bottom 20% 成员集合仍明显不稳定；
- “整体相关性较高”不能替代“具体过滤名单稳定”。

## 6. 已完成阶段：Square-MH 过滤诊断

### 6.1 固定最低 38 条的来源

训练集有 192 条 Demo，作者式固定底部 20% 对应：

```text
floor(192 x 0.20) = 38
```

这个数量来自预设比例，不代表客观上恰好存在 38 条有害 Demo。

### 6.2 Stage 2 序贯分支

Stage 2 同时检查：

- 子集对完整 100 条的开发诊断；
- 两个不重叠子池的一致性；
- 60 条顺序池对 40 条独立池的压力测试；
- Bootstrap 提前停止原型。

最终决策：

```text
FAIL_STOP_SEQUENTIAL_BRANCH
```

完整池开发诊断和独立池压力测试均未通过，因此停止 Square 序贯分支。

### 6.3 Stage 2B 固定过滤诊断

底部 20% 边界结果：

```text
第 38/39 条分数差：0.000200
配对分数差标准误差：0.067776
边界差 / 标准误差：0.002956
边界一倍标准误差内 Demo：57
边界两倍标准误差内 Demo：121
```

边界差远小于噪声，说明固定切出最低 38 条缺乏自然分界。

50 条 rollout 下：

```text
固定目标数量：38
p >= 0.90 的稳定低分 Demo：5
稳定保留 Demo：114
模糊 Demo：73
```

50/50 独立池关键结果：

| 方法 | 平均选择数 | 独立精度 |
|---|---:|---:|
| 固定最低 38 条 | 38.00 | 0.4221 |
| Bootstrap 稳定核心 `p >= 0.90` | 5.29 | 0.5265 |
| 相同数量直接最低分 | 5.29 | 0.6316 |

Bootstrap 核心虽然比固定最低 38 条略好，但比相同数量的简单最低分集合低约 0.105。因此没有证明复杂的 Bootstrap 成员选择带来额外价值。

最终决策：

```text
FAIL_SQUARE_FIXED_FILTER_BRANCH_CONSIDER_TRANSPORT
```

Square 结论是“当前证据不足以支持过滤重训练”，不是“CUPID 在所有任务上必然无效”。

## 7. 当前阶段：Transport-MH 预注册实验

### 7.1 为什么转向 Transport

Square 的失败可能有两种解释：

1. CUPID 影响分数本身不适合稳定识别过滤对象；
2. Square 任务上的影响边界过于平坦，缺少足够强的区分信号。

Transport 是更复杂的双臂搬运任务。保持 CUPID/TRAK 方法不变并更换任务，可以检验影响信号是否具有任务依赖性，而不是在 Square 上不断增加补丁。

### 7.2 基础策略配置

```text
physical GPU：1（进程内 cuda:0）
training seed：0
num_epochs：1751
resume：true
checkpoint_every：50
rollout_every：50
top-k checkpoint：3
n_envs：8
dataloader workers：0
W&B：offline
```

训练由独立 supervisor 管理：

- 非零退出后等待 30 秒；
- 使用 `latest.ckpt` 恢复；
- 连续快速失败达到 5 次才标记不可恢复；
- 训练、GPU 监控、邮件提醒、rollout、analysis 和 filter-ID freezer 使用独立进程；
- Codex 会话终止不会终止训练。

### 7.3 运行恢复谱系

第一次正式 attempt 在 epoch 1100 后以返回码 137 退出。监督器在 30 秒后启动 attempt 2，并从 checkpoint 恢复。

epoch 1100 因恢复语义出现两次：

```text
恢复前：global_step 539489，test mean score 0.46
恢复后：global_step 539978，test mean score 0.62
```

这是一次受监督恢复和 epoch 重放事件，最终报告必须保留，不得把两条记录静默合并。

### 7.4 当前快照

截至 2026-07-23 19:26 +08:00：

```text
状态：RUNNING
attempt：2
restart_count：1
当前 epoch：1399 / 1750
当前 global_step：686259
邮件监控 PID：3715914
```

已审计里程碑包括：

| Epoch | 测试成功率 | 审计 |
|---:|---:|---|
| 1100（恢复后） | 0.62 | 768 个张量有限 |
| 1150 | 0.46 | 768 个张量有限 |
| 1200 | 0.40 | 768 个张量有限 |
| 1250 | 0.50 | 768 个张量有限 |

周期性 50-seed 分数只用于运行健康检查，不作为最终研究结论。正式结论必须使用最终冻结的 100-rollout 池。

## 8. Transport 后续自动流程

### 8.1 最终训练审计

到达 epoch 1750 后要求：

- 日志中的数值全部有限；
- `latest.ckpt` 可加载；
- checkpoint epoch、global step 与日志一致；
- 模型、EMA 模型和优化器状态存在；
- 所有 checkpoint 张量有限；
- checkpoint、数据集和划分哈希冻结。

不通过则停止，不收集最终 rollout。

### 8.2 固定 100 条 rollout

```text
checkpoint：最终 latest.ckpt
num_episodes：100
seeds：100000--100099
device：cuda:0（物理 GPU 1）
```

要求保存 episode、视频、seed、成功标签、决策点和哈希。

类别平衡门槛：

```text
成功至少 5 条
失败至少 5 条
```

不通过则停止，不补采样、不更换 seed 窗口、不运行 TRAK。

### 8.3 TRAK smoke 和资源门槛

完整设置：

```text
proj_dim：4000
proj_max_batch_size：32
lambda_reg：0
use_half_precision：0
loss_fn：square
num_timesteps：64
batch_size：128
seed：0
featurize_holdout：1
finalize_scores：1
```

先以 4096 个 source sample 和 128 个 target 决策点运行同配置 smoke。只有数值、显存、磁盘和时间外推可接受才进入正式 TRAK。

只有 CUDA OOM 时允许将 `proj_max_batch_size` 从 32 降到 16。不得降低投影维度、时间步数或模型规模来换取通过。其他错误停止并诊断。

### 8.4 原始 CUPID 分数

正式 TRAK 完成后：

1. 导出 rollout 对训练 Demo 的影响矩阵；
2. 使用原始 CUPID performance influence 生成 Demo 分数；
3. 重建官方净分数；
4. 最大重建误差必须不高于 `2e-3`；
5. 冻结矩阵、分数、manifest、split 和哈希。

## 9. Transport 独立稳定性诊断

### 9.1 诊断内容

- rollout 预算：10、25、50、75；
- 过滤比例：5%、10%、20%、30%；
- 固定最低 38 条；
- 50/50 不重叠 rollout 池；
- Bootstrap 删除概率；
- `p >= 0.90` 稳定核心；
- 相同数量简单最低分基线。

完整 100 条 rollout 只能称为有限池参考，不能称为真实 Demo 排名。

### 9.2 “强通过”的精确定义

必须同时满足：

```text
稳定核心非空率 >= 0.80
平均选择数量 >= 5
独立池 precision >= 0.70
相对固定最低 38 条的配对 precision 增益 >= 0.15
独立池平均排名百分位 <= 0.25
50-rollout 下稳定 Demo 数量 >= 5
相对相同数量简单最低分集合的配对 precision 增益 >= 0.03
官方分数重建最大误差 <= 2e-3
```

只有决策文本严格等于：

```text
PASS_STABILITY_WEIGHTED_CORE_CANDIDATE
```

才称为强通过。

普通 PARTIAL 只能支持继续离线研究，不能启动过滤训练。最终 FAIL 则停止 Transport 过滤分支。

## 10. 强通过后的过滤重训练

### 10.1 冻结过滤集合

从完整 100-rollout 的底部 20% Bootstrap 删除概率中选择：

```text
p >= 0.90
```

冻结：

- 有序原始 Demo ID；
- 过滤数量；
- 来源矩阵、manifest、split 和分数哈希；
- 精确 Hydra `filter_episode_ids` override；
- 输出文件哈希。

不得用固定比例近似替代可变大小稳定核心。

### 10.2 配对训练设计

先运行同一 seed 的配对实验：

| 条件 | 训练集 |
|---|---|
| 未过滤对照 | 原 192 条训练 Demo |
| 过滤实验 | 192 条减去冻结稳定核心 |

两者必须保持：

- 相同模型和优化器；
- 相同训练 epoch 和 batch 设置；
- 相同训练 seed；
- 相同 validation/holdout；
- 相同评估 seed 和 rollout 数；
- 相同 checkpoint 选择与审计规则。

过滤模型必须从头训练，不能从已见过被删 Demo 的基础 checkpoint 微调。

### 10.3 因果判定

过滤方法是否有效，最终由配对重训练的独立评估决定，而不是由离线 precision 单独决定。

至少报告：

- 最终 100-rollout 成功率和配对差；
- 成功数/失败数；
- validation loss；
- 不同 checkpoint 的波动；
- 训练成本；
- 过滤 Demo 数量和 ID；
- 是否优于未过滤对照。

单 seed 配对结果只作为候选证据。只有效果方向明确，才扩展到 seeds 0、1、2，并报告均值、离散程度和逐 seed 结果。

## 11. 停止规则

实验在以下任一条件出现时停止对应分支：

1. 最终训练或 checkpoint 审计失败；
2. 固定 100 rollout 不完整；
3. 成功或失败少于 5 条；
4. TRAK smoke 出现非 OOM 错误、非有限值或不可接受资源外推；
5. 正式 TRAK 或官方分数重建失败；
6. 稳定性诊断不是强通过；
7. 冻结 ID 不属于训练集、重复、越界或改变 validation/holdout；
8. 配对重训练无法保持控制变量；
9. 强通过后的重训练没有改善独立评估，则不扩展多种子。

不得通过新增 rollout、移动 seed 窗口、降低 TRAK 配置、反复试阈值或引入交叉拟合/主动采样/贝叶斯模块来事后挽救失败。

## 12. 预期结果与可发表结论

### 12.1 Transport 也失败

可支持的结论：

- CUPID 分数能够计算，但固定比例过滤边界可能不足以稳定识别具体 Demo；
- 整体排名稳定不能保证成员选择稳定；
- Bootstrap 不必然优于简单最低分排序；
- 在进入昂贵重训练前进行独立稳定性门控是必要的。

不能声称 CUPID 在所有任务和设置上无效。

### 12.2 Transport 强通过但重训练无改善

可支持的结论：

- 离线稳定成员选择不等于因果有害数据识别；
- 稳定性是重训练的必要条件，但不是充分条件。

### 12.3 Transport 强通过且重训练改善

可支持的候选结论：

- CUPID 过滤信号具有任务依赖性；
- Bootstrap 稳定核心可作为比固定比例过滤更可靠的候选；
- 经过独立门控的可变大小过滤集合可能改善数据筛选。

正式结论仍需多种子确认，不能由 seed 0 单次结果直接推广。

## 13. 时间和资源估算

Transport 经验估计：

```text
基础训练总计：24--36 小时
固定 100 rollout：2--4 小时
TRAK smoke：几十分钟到数小时
正式 TRAK：1--4 天
离线诊断：分钟到小时
强通过后的单轮重训练：约 24--36 小时
```

截至当前快照，基础训练剩余约 350 epoch。按恢复后实测速度，预计仍需约 8--10 小时。完整路线的主要不确定性来自正式 TRAK。

## 14. 自动化、恢复和邮件

- 训练由独立 supervisor 运行；
- `latest.ckpt` 每 50 epoch 保存并用于恢复；
- 非零退出自动等待 30 秒后重试；
- rollout 和 analysis waiter 自动等待上游 PASS；
- 每个阶段有 `.running`、`.pass`、`.fail` 或 `.stop` 状态文件；
- 邮件监控独立于训练和 Codex；
- 邮件事件包括启动、重启、完成、终止、不可恢复错误和 supervisor 丢失；
- 当前过滤 ID freezer 只冻结 ID，不自动启动过滤训练。

## 15. 主要证据与入口文件

研究总览：

- 本文件：`CUPID_research_plan_summary_20260723.md`

Square 结果：

- `reports/minimal_reproduction_summary_20260722.md`
- `results/stage2_bottom_filter_sequential_v1/stage2_decision_report.md`
- `results/stage2b_fixed_filter_diagnosis_v1/stage2b_decision_report.md`

Transport 预注册与执行：

- `CUPID_stage3_transport_mh_preregistered_runbook_20260722.md`
- `stage3_transport_env.sh`
- `tools/run_stage3_transport_train_supervisor.sh`
- `tools/run_stage3_transport_rollout_pipeline.sh`
- `tools/run_stage3_transport_analysis_pipeline.sh`
- `tools/stage3_transport_stability_diagnosis.py`
- `tools/run_stage3_transport_filter_id_freezer.sh`

运行状态：

- `status/stage3_transport/`
- `pids/stage3_transport/`
- `logs/stage3_transport/`

冻结协议：

- `frozen/stage3_transport/execution_protocol_20260722_revision6/`

## 16. 最终研究原则

本研究坚持三层证据：

```text
可计算：能得到 CUPID 分数
  !=
可选择：能稳定确定应过滤的 Demo
  !=
有效：过滤后从头训练能改善独立评估
```

只有三层依次通过，才能把 CUPID Demo 评分解释为可用于数据过滤的方法证据。
