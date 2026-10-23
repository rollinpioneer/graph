# EG 模块级受控机制诊断（2026-10-23）

本分支归档已完成的 E0–E4 零参数更新诊断：16 个训练帧、48 个 C3 帧、40,489 次新增解析评价。未运行 E3-2/E5/E6，未访问 C4/C5，未修改 EG/G 基准代码或权重。

## 结果入口

- [中文完整报告](mechanism_report.html)（下载后用浏览器打开）、[结论正文](narrative.html)、[汇总图](mechanism_summary.png)。
- [统计与场景区间](statistics.json)、[逐候选干预](intervention_results.csv)、[姿态/分数四组合](counterfactual_composition.json)、[AD 网格](local_AD_grid.csv)。
- [梯度路径](gradient_routes.csv)、[梯度夹角](gradient_cosines.csv)、[损失台账](loss_population.csv)、[零质量宽度审计](width_target_audit.json)。
- [冻结协议](protocol.json)、[重放凭据](baseline_replay.json)、[执行凭据](execution_receipt.json)、[零训练凭据](training_receipts.json)、[源码映射](source_map.md)。

方法开发结论为 **HOLD**。姿态梯度并未与质量特征分支断开，关闭注意力没有稳定的完整流程收益。137 个失败查询中，59 个在 Top-2×2 中存在可行组合，14 个仅更低排名格可达，64 个在固定位置、方向、宽度下全 48 格失败。零质量样本占有效样本 7.26%，贡献本次平均宽度损失的 64.9%；这是下一步 E6 掩码对照的具体假设，不是已证明的因果机制。

## 文件范围与完整性

原交付目录的全部 339 个文件均按原始字节保留，包括 `light_results.zip` 及解包的 `data/`。`data/cache/` 是 64 帧已导出的轻量姿态/logit/特征缓存；`data/evaluation/` 为 64 帧逐姿态结果及哈希凭据，`data/gradient/` 和 `data/receipts/` 保留逐帧台账。新增归档项包括本轮全部 Python 脚本、先导失败与修正日志、旧直接解析评价器依赖快照以及归档清单。

完整内部激活缓存、原始 GraspNet 数据和模型权重仍在原服务器；它们不是本次本地交付包的一部分。完整激活缓存的原始 SHA-256 已保存在每帧 receipts 中；轻量缓存因有意删去内部数组，不能拿它的文件哈希与完整缓存直接比较。全套已交付文件和新增归档文件均列于 `ARCHIVE_MANIFEST.json`。

`protocol.json` 和 `execution_receipt.json` 中的 `allow_git_write:false` / `git_write:false` 描述诊断执行当时的权限与状态，未回写历史记录。用户随后明确要求“把所有结果和代码提交git”，本次提交由该后续授权执行。

## 代码与复核

- `code/probe.py`：冻结权重的 E0–E3 前向/梯度探针，包含 E1、定向 attention 及不变量检查；无优化器。
- `code/evaluate_module.py`：逐实际姿态的官方解析评价、E4 网格及限定的完整后处理；不使用最近离散 GT 质量替代。
- `code/analyze.py` / `code/report.py`：读取已保存结果，生成统计与报告；`code/finalize.py` 记录交付完整性。
- `code/package.py`：生成逐帧轻量缓存；`code/check_label.py` / `code/grad_summary.py`：先导定位与数值摘要；`code/dispatch.py` 仅将命令写入既有 SSH 队列，不含凭据。
- `code/dependencies/previous_direct_evaluate.py`：上一轮直接解析评价器原样快照。新脚本仍按原实验目录导入 `evaluate`。

代码保留原执行路径配置：服务器 `~/work/grasp/eg_module_controlled_diagnosis_20261023`，旧缓存 `~/work/grasp/eg_pose_quality_diagnosis_v1_20261022`，官方源码 `~/work/grasp/eg_full_repro_v1_20261015/runtime/EconomicGrasp`；本地统计脚本的 `O` 为原交付目录。迁移复核时只需配置这些根路径和既有环境，不应更改冻结候选、阈值、匹配或质量口径。

历史前向为 TF32，标签与梯度数值核验使用 FP32。重放中官方视图映射重复写入造成的差异已审计，ADWQ 保留历史匹配和标签。报告中的 Oracle 与 GT 辅助基底不代表可部署收益；解析可行率不等于机器人成功率。
