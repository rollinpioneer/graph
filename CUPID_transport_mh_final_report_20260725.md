# CUPID Transport-MH 最终实验报告

版本日期：2026-07-25  
工作目录：`/home/xushijie/CUPID`  
实验名称：`CUPID Transport-MH`  
最终状态：完整流程执行完成；稳定性强门槛未通过；未启动过滤重训练

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-25
- Verification Status: ANALYZED
- Version Label: transport_mh_final_report_v1
- Evidence Scope: 本地训练、rollout、TRAK、影响分数和稳定性诊断产物
- Reproducibility Scope: 输入与结果已哈希审计；未进行独立全流程复跑

## 1. 摘要

本实验在 Transport-MH 任务上保持 CUPID/TRAK 评分定义不变，依次完成基础策略训练、固定 100 条 rollout、正式 TRAK、Demo 级影响分数重建、50/50 独立池压力测试和 Bootstrap 稳定性诊断。

Transport-MH 的低分信号明显强于此前 Square-MH：在 50 条 rollout 下可稳定识别 18 条底部 20% Demo，`p >= 0.90` 稳定核心的独立池精度达到 `0.8708`。但是，相同数量的简单最低分集合精度为 `0.8822`，比 Bootstrap 稳定核心高 `0.01135`。因此，本实验没有证明 Bootstrap 成员选择相对直接分数排序具有额外价值。

最终决策为：

```text
PASS_VARIABLE_K_DIAGNOSIS_BOOTSTRAP_MEMBERSHIP_NOT_PROVEN
```

该决策表示“Transport 中存在较强的可变数量低分信号”，但不表示“Bootstrap 已证明应删除哪些 Demo”，更不表示“这些 Demo 已被证明有害”。按照预注册停止规则，本轮未冻结过滤集合，也未启动过滤重训练。

## 2. 研究问题

本实验主要回答以下问题：

1. CUPID/TRAK 在 Transport-MH 上能否形成可审计的 Demo 影响分数？
2. 固定最低 20%（38 条）是否存在清晰、稳定的自然边界？
3. Bootstrap 是否能从有限 rollout 中识别稳定低分核心？
4. Bootstrap 稳定核心是否优于“直接选择相同数量最低分 Demo”？
5. 当前离线证据是否足以授权过滤后的策略重训练？

## 3. 实验设计

### 3.1 数据划分

| 项目 | 数量 |
|---|---:|
| 原始 Demo | 300 |
| 训练 Demo | 192 |
| Validation Demo | 12 |
| Holdout Demo | 96 |

过滤诊断只针对 192 条训练 Demo。Validation 和 holdout 在本阶段保持不变。

### 3.2 基础策略与 rollout

| 项目 | 设置或结果 |
|---|---|
| 任务 | RoboMimic Transport-MH，low-dimensional |
| 训练 seed | 0 |
| 最终 epoch | 1750 / 1750 |
| 最终 global step | 858,478 |
| 训练恢复次数 | 1 |
| 成功恢复后的训练时间 | 46,369 秒（12.88 小时） |
| 固定 rollout seeds | 100000--100099 |
| Rollout 数量 | 100 |
| 成功 / 失败 | 44 / 56 |
| 平均成功率 | 0.44 |
| 决策点数量 | 7,969 |
| Rollout 时间 | 11,798 秒（3.28 小时） |

成功和失败样本均远高于预注册的最少 5 条类别平衡门槛，因此允许继续计算 net influence。

### 3.3 TRAK 配置

| 参数 | 值 |
|---|---:|
| `proj_dim` | 4,000 |
| `lambda_reg` | 0 |
| `loss_fn` | square |
| `num_timesteps` | 64 |
| `use_half_precision` | 0 |
| `proj_max_batch_size` | 32 |
| 正式计算 `batch_size` | 64 |
| 模型梯度参数数 | 17,181,204 |

资源调整记录：原始 `batch_size=128` 在共享 GPU 环境中 OOM，峰值已预留约 `20.34 GiB` 后仍需额外分配 `640 MiB`。经用户授权，主批量改为 64；`proj_dim`、时间步、损失函数、样本集合和评分定义均未改变。

TRAK smoke 结果：

| 指标 | 数值 |
|---|---:|
| Source 样本 | 4,096 |
| Target 样本 | 64 |
| 峰值 allocated 显存 | 15,503.4 MiB |
| 峰值 reserved 显存 | 15,886.0 MiB |
| Smoke 时间 | 773.68 秒 |
| 正式 TRAK 记录墙钟时间 | 36,690 秒（10.19 小时） |
| TRAK 结果磁盘占用 | 约 12 GiB |

正式 TRAK 期间发生 6 次无终态子进程退出；分析守护器根据 `_is_featurized` 标记自动续跑，共记录 8 次恢复检测，最终完成全部样本和打分。该恢复行为保留了已经完成的 per-sample 特征。

## 4. 输出审计

| 审计项 | 结果 |
|---|---:|
| 原始影响矩阵形状 | `185942 x 7969` |
| Rollout-Demo 矩阵形状 | `100 x 192` |
| 官方分数重建最大绝对误差 | `1.468116e-06` |
| 官方分数重建 RMSE | `2.304112e-07` |
| 允许的最大重建误差 | `2e-3` |

最大误差约为允许门槛的 `0.073%`，说明后续稳定性分析使用的重建分数与官方分数充分对齐。

## 5. 固定底部 20% 边界

训练集 192 条 Demo 的底部 20% 对应 38 条。完整 100-rollout 池的边界结果为：

| 指标 | 数值 |
|---|---:|
| 第 38/39 条边界分数差 | 0.000471 |
| 配对边界差标准误差 | 0.017961 |
| 边界差 / 标准误差 | 0.026208 |
| 一倍标准误差内 Demo | 41 |
| 两倍标准误差内 Demo | 75 |
| Bootstrap 稳定删除数（`p >= 0.90`） | 20 |
| Bootstrap 稳定保留数（`p <= 0.10`） | 130 |
| 模糊 Demo 数 | 42 |

边界差只相当于约 `0.026` 个标准误差，因此固定切出最低 38 条仍缺乏清晰自然分界。Transport 的边界比 Square 更有信号，但“固定比例”仍不能被解释为真实存在恰好 38 条有害 Demo。

## 6. Rollout 预算稳定性

在 50 条 rollout、底部 20% 设置下：

| 分类 | Demo 数 |
|---|---:|
| 稳定删除（`p >= 0.90`） | 18 |
| 稳定保留（`p <= 0.10`） | 130 |
| 模糊（`0.10 < p < 0.90`） | 44 |
| 强模糊（`0.20 < p < 0.80`） | 33 |

这说明 Transport 中确实存在一小组在有限 rollout 下反复落入低分区的 Demo，但约 23% 的训练 Demo 仍处于模糊区间。

## 7. 50/50 独立池结果

每次将 100 条 rollout 分成互不重叠的 50/50 子池，一侧选择 Demo，另一侧评估，共形成 200 个方向性评价。

| 方法 | 平均选择数 | 独立精度 | 精度标准差 | Recall | 平均排名百分位 | 相对固定底部 20% 配对增益 | 相对同数量最低分增益 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定底部 10% | 19.000 | 0.8100 | 0.0800 | 0.4050 | 0.1365 | 0 | 0 |
| 固定底部 20% | 38.000 | 0.6213 | 0.0543 | 0.6213 | 0.2068 | 0 | 0 |
| 同数量最低分，匹配 `p >= 0.90` | 12.845 | 0.8822 | 0.0974 | 0.2970 | 0.1064 | 0 | 0 |
| Bootstrap 稳定核心 `p >= 0.80` | 18.585 | 0.8157 | 0.0894 | 0.3971 | 0.1301 | 0.1943 | 0.00005 |
| Bootstrap 稳定核心 `p >= 0.90` | 12.845 | 0.8708 | 0.0972 | 0.2930 | 0.1076 | 0.2495 | -0.01135 |
| Bootstrap 稳定核心 `p >= 0.95` | 9.215 | 0.8807 | 0.1140 | 0.2128 | 0.1030 | 0.2594 | -0.02926 |

主要观察：

1. 固定底部 20% 选择过多，独立精度只有 `0.6213`。
2. Bootstrap 稳定核心显著缩小了候选集合，并把独立精度提高到约 `0.87`。
3. 但是，选择相同数量时，直接最低分基线达到 `0.8822`，高于 Bootstrap 核心的 `0.8708`。
4. `p >= 0.80` 时两种方法几乎完全相同；阈值提高到 0.90 和 0.95 后，Bootstrap 相对简单基线反而略差。

这里的“精度”是相对于另一独立 rollout 子池所定义的有限池参考一致性，不是真实有害 Demo 的 ground truth 准确率。

## 8. 预注册强门槛判定

| 门槛 | 要求 | 实际 | 判定 |
|---|---:|---:|---|
| 官方分数重建最大误差 | `<= 0.002` | `1.468e-06` | 通过 |
| 稳定核心非空率 | `>= 0.80` | `1.00` | 通过 |
| 平均选择数量 | `>= 5` | `12.845` | 通过 |
| 独立池精度 | `>= 0.70` | `0.8708` | 通过 |
| 相对固定最低 38 条增益 | `>= 0.15` | `0.2495` | 通过 |
| 独立池平均排名百分位 | `<= 0.25` | `0.1076` | 通过 |
| 50-rollout 稳定 Demo 数 | `>= 5` | `18` | 通过 |
| 相对同数量简单最低分增益 | `>= 0.03` | `-0.01135` | **未通过** |

八项门槛中七项通过。唯一失败项恰好是证明“Bootstrap 成员选择具有额外价值”的必要门槛，因此不能把其余七项通过解释为整体强通过。

## 9. 与 Square-MH 的对照

| 指标 | Square-MH | Transport-MH |
|---|---:|---:|
| Rollout 成功 / 失败 | 71 / 29 | 44 / 56 |
| 固定底部 20% 独立精度 | 0.4221 | 0.6213 |
| `p >= 0.90` 稳定核心平均数量 | 5.29 | 12.85 |
| `p >= 0.90` 稳定核心独立精度 | 0.5265 | 0.8708 |
| 同数量直接最低分精度 | 0.6316 | 0.8822 |
| Bootstrap 相对同数量基线增益 | 约 -0.105 | -0.01135 |
| 50-rollout 稳定低分 Demo | 5 | 18 |

Transport 的低分信号强度和独立一致性均明显好于 Square，支持“影响信号具有任务依赖性”。但两个任务得到相同的关键结论：Bootstrap 成员选择没有优于同数量直接最低分排序。

## 10. 结论

### 10.1 数据直接支持的结论

1. Transport-MH 上的 CUPID/TRAK 全链路已成功完成，分数重建误差远低于门槛。
2. Transport 中存在比 Square 更强、更稳定的低分小集合信号。
3. 固定底部 20% 仍不是自然边界；可变数量核心比固定 38 条更合理。
4. 简单按独立子池分数直接选择最低约 13 条，表现不低于 Bootstrap 稳定核心。
5. 当前证据不支持声称 Bootstrap 能更准确地识别具体过滤成员。

### 10.2 数据尚未证明的结论

1. 低分 Demo 是否在因果意义上有害。
2. 删除这些 Demo 是否会改善重新训练后的策略性能。
3. 当前结果是否能跨训练 seed、rollout seed 或任务泛化。
4. 完整 100-rollout 池是否足以充当真实价值标签。

### 10.3 最终操作决策

```text
稳定性强通过：否
冻结过滤 ID：否
过滤重训练授权：否
本轮自动下游训练：未启动
```

## 11. 统计与方法风险检查

总体置信度：`CAUTION`。计算链路和预注册门槛判断证据充分，但有限 rollout 池不是外部 ground truth，也未完成过滤重训练的因果验证。

Fallacy Scan 覆盖：`11/11`

| 风险类型 | 级别 | 本实验判断 |
|---|---|---|
| Simpson's paradox | NOTE | 未进行会产生总体/分组方向反转的群组聚合推断；当前数据不足以专门检验。 |
| Ecological fallacy | NOTE | 结论限定在 Demo 和 rollout 层级，没有外推到个体层级。 |
| Berkson's paradox | CAUTION | MH 数据集和固定训练划分属于选择后的样本，结论不能直接外推到全部示范分布。 |
| Collider bias | NOTE | 成功/失败标签用于构造影响分数，但本报告不据此进行因果控制变量解释。 |
| Base-rate neglect | NOTE | 已明确报告 44/56 成功失败比例；独立精度不被解释为部署场景诊断准确率。 |
| Regression to the mean | CAUTION | 研究对象是极端低分 Demo；独立 50/50 池和同数量基线降低了风险，但有限池噪声仍存在。 |
| Survivorship bias | NOTE | 100 条固定 rollout 全部纳入，未发现只保留成功完成项的分析。 |
| Look-elsewhere effect | CAUTION | 诊断检查了多个比例和阈值；最终强门槛预注册且固定，但其他阈值结果应视为描述性。 |
| Garden of forking paths | NOTE | 主要门槛已预注册；资源批量调整有明确日志且未改变评分定义。 |
| Correlation != causation | CAUTION | 离线低分一致性不能证明 Demo 有害；本轮没有进行删除后的配对重训练。 |
| Reverse causality | NOTE | 本实验未提出方向性因果关系，因此不适用。 |

未报告传统 p 值或置信区间，因此本报告不使用“统计显著”措辞。精度标准差描述 200 次方向性评价的变异，不等同于独立重复实验的置信区间。

## 12. 后续建议

1. 冻结并保留本轮产物，将其作为“Transport 离线稳定性诊断完成、Bootstrap 成员优势未证明”的正式结果。
2. 若继续研究，应新建协议，重点比较可变 `K` 的简单最低分策略，而不是默认继续 Bootstrap。
3. 任何过滤重训练都应同时包含未过滤基线、相同 seed、固定 validation/holdout 和独立 rollout seeds。
4. 在获得配对重训练证据前，继续使用“低分候选”而不是“有害 Demo”的表述。
5. 若目标是确认小幅差异，应增加独立 rollout 池或多 seed，而不是在当前 100 条 rollout 上继续调阈值。

## 13. 关键产物

| 产物 | 路径 |
|---|---|
| Rollout 审计 | `manifests/stage3_transport/rollout100/rollout_summary.json` |
| Demo 影响层 | `results/stage3_transport/influence_layers/` |
| 稳定性审计 | `results/stage3_transport/stability_diagnosis/audit.json` |
| 稳定性决策报告 | `results/stage3_transport/stability_diagnosis/stage2b_decision_report.md` |
| 独立池汇总 | `results/stage3_transport/stability_diagnosis/cross_pool_conservative_core_summary.csv` |
| 最终分析状态 | `status/stage3_transport/analysis_pipeline.pass` |
| 停止状态 | `status/stage3_transport/stability_diagnosis.stop` |
| 过滤决策 | `status/stage3_transport/filter_id_freezer.skip` |

## 14. 一句话结论

Transport-MH 证明了 CUPID 分数可以稳定找到一个高一致性的低分小集合，但没有证明 Bootstrap 比直接选择相同数量最低分 Demo 更好，因此当前证据不足以授权过滤重训练。
