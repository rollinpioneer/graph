# SAW-GOALPROBE-05 复现报告

**执行完成；H0未获预登记支持。**

分支：https://github.com/rollinpioneer/graph/tree/codex/saw-goalprobe-05-20261013

来源冻结提交：176ad8d17ec9d71eaca9ed55d87a8ee0bf837a28；正式CPU查询源码：71fcd50a8a4a3689763ee99f690ff6972950747c。最终提交与远端一致性记录在独立发布回执，避免自引用提交哈希。

| Seed | P1 A/B | P1差百分点及95%区间 | P2 A/B |
|---|---|---|---|
| 0 | 100.00% / 95.83% | 4.17 [0.00, 12.50] | 100% / 100% |
| 1 | 97.92% / 95.83% | 2.08 [-6.25, 12.50] | 100% / 100% |
| 2 | 97.92% / 93.75% | 4.17 [-6.25, 18.75] | 100% / 100% |

P1是近处分岔、匹配BFS距离的不同通道目标；P2是同通道近目标对照。在这些场景中，B大多能区分并进入正确初通道。P3原官方远目标下，A均16/16进入初通道；B为seed0 3/16、seed1 3/16、seed2 0/16。Task1/3各8状态，替换后的目标单列。不能把P3两种目标合并后解释成新的官方成绩，也不能据P1/P2的高分断言只有后段路线执行失败。

结论与下一步：停止依据本轮证明“普遍忽略目标”的修复设计；目标身份/距离条件下的动作质量与状态支持仍值得设计新卡核验。没有追加状态、目标、窗口、完整评价或训练。seed2区间较宽，不能称A/B等价，也不能排除全部随机目标机制。

身份与一致性：6份1M权重SHA及来源索引核验；12项CPU历史动作最大绝对误差0；64状态完整integration、qpos/qvel、目标、所有记录RNG和wrapper恢复/重复一致；768条保存轨迹独立重算首次通道、裁剪和累计路径，最大路径误差8.89e-16。原MECH工作区仍干净、HEAD未变；原权重、数据、配置及旧成绩未覆盖。

实际资源：诊断窗口1145.774秒（0.318271小时），含准备、导出修复、依赖补全、统计与审计；允许CPU核心72–75，硬上限4小时。GPU驻留0、显存0、新训练更新0。所有物理控制步21249/25000，其中正式18024、准入2969、首次恢复128、导出修复重复恢复128。准备阶段CPU秒未单独测量，不能把守卫采样CPU秒冒充完整CPU消耗；完整墙钟窗口覆盖该成本。

技术状态：所有正式模型及历史检查通过。导出阶段的NumPy/NaN编码、缺失统计包和报告预检错误均保留失败代码/日志/退出回执，修复没有重新选择样本、重新查询模型或改变判据；唯一新增的128步恢复自检计入硬预算。统计/图已完成后的NaN导出故障使用只读报告恢复，原CSV和图SHA保持一致。

结果入口：HYPOTHESIS_VERDICT.md（机制与边界）、NEXT_DECISION.md（停止/后续卡）、PAIRED_CONTRASTS.csv（配对区间）、NEGATIVE_CONTROLS.csv（失败/运动）、results/SOURCE_STRATIFIED_RESULTS.csv（来源）、results/P3_GOAL_ROLE.csv（官方目标与干预）、results/OFFLINE_ALIGNMENT.csv（严格数据动作）、receipts/FINAL_AUDIT.json（身份/轨迹审计）。

四张结果图已逐张可视检查，布局和标注清晰。CBR_BY_SEED_AND_CONTROL.png的误差条是簇重采样区间；P3柱包含两个目标角色，只作诊断混合集描述。GOAL_ACTION_DIRECTION_DISTRIBUTION.png展示方向分布；GOAL_SENSITIVITY_CONTROLS.png展示目标对动作差；REPRESENTATIVE_TRACES.png展示每个类别按字典序选择的第一条轨迹。

完整初态、每步integration/qpos/qvel、裁剪前后动作、目标、参考路线与终止记录位于本目录runs；完整日志在logs。RAW_ARTIFACT_INDEX.json登记路径、大小、SHA。原大权重和数据路径见MODEL_IDENTITY.json与BANK_FROZEN.json；Git仅归档代码、卡片、配置、轻量表/回执/图。
