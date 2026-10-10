# 05 冲突与缺失证据

格式：旧说法 — 新证据 — 版本/时间 — 冲突原因是否已知。

## 冲突
1. **上一卡 `conflicts_and_missing.md` §4 / 报告第 6 节**：“wlplan 的 `Problem(..., statics)` 只会加孤立节点” — 新证据：`include/planning/problem.hpp` 与 `src/main.cpp:328–353` 表明四参构造是 `(domain, objects, positive_goals, negative_goals)`，第四个位置参数是**负目标**；ILG 完全不使用 `statics`（`ilg.cpp` 无引用）。上一卡传入的“静态原子”其实成了负目标节点（无边的孤立节点）。 — 本卡读取官方源码，2026-10-10 — 原因已知（对构造函数签名的误读）。**影响**：上一卡把静态原子改为状态原子的做法仍然正确，候选 16 的结论不受影响；但“wlplan 对 statics 的行为”的解释需更正。
2. **上一卡 `A1 报告 §6`**：“文献中的 ILG 包含静态事实与对象类型” — 新证据：P1 Def 3.1 与 `ilg.cpp:48–52`：ILG 对象统一色、不含类型；静态事实仅在作为状态原子出现时才进入（P2 的推断；官方 `state_representation=all` 选项）。 — 原因已知（概括过宽）。更正：包含它们是官方的**可选**数据表示，不是 ILG 定义。
3. **上一卡“a-m 剪枝是首个丢失环节（23/23）”** — 新证据：`bulk_pruners.cpp:11–113` 按设计在训练集上保持列等价划分；而上一卡 `pruning_probe.json` 称 202 个未剪列里只有 22 列与保留列相等（按设计应 ≥147）。两者矛盾。 — 原因**未知**：分析假象 / wheel 与源码差异 / 真实行为，皆未排除。结论降为“本项目观察，来源 UNRESOLVED_ORIGIN”。
4. **论文 L 与默认 L**：P1 实验 L=4；`classic.toml` 与项目为 2。 — 配置差异，已知。
5. **聚合方式**：P1 抽取结果为邻居多重集；`classic.toml` `hash="set"`。 — 页码待核。
6. **作者归属**：用户文档把 ICAPS 2024 Expressiveness 论文与 Ståhlberg 等人的工作相邻列出；实际作者是 Horčík & Šír（ICAPS 34(1) 281–289）。Ståhlberg–Bonet–Geffner 是另一篇（arXiv 2109.10129）。 — 已知。
7. **历史数字复核**：112/36、24→23、836、449 的分母与上一卡一致（Scope 449 = 338+111 与登记 339+110 的 1 行差异沿用上一卡的说明）。
8. **IPC/Joint 搜索事实**：D0R 与 D0 逐题展开一致、计划长度 3 题不同 — 与“确定性重跑只排除评测噪声”的限定一致，不把 D0/C0 差异全归训练噪声。

## 未获取/缺口
- 论文时的官方代码（分支 `icaps24`、Zenodo 包）、论文页码；P1 的 rank-SVM 细节段；P5 正文；P6；官方运行记录。
- 官方 Scorpion/FD 搜索侧的并列、目标检查、重复检测细节（未读 `plan.py`/搜索配置）。
- wheel 与 `ext/wlplan` 源码逐字节一致性（只确认由该子模块 `uv pip install` 构建）。
- Hao 与 Chrestien 的官方实现入口；D0/T1 搜索轨迹；IPC 的库 B 覆盖。
- 无法解析的 PDF 已用 arXiv HTML 或第三方文本镜像代替，镜像文本的公式符号有损。
