# 来源与真实阅读状态

本轮补充包根目录：`C:/Users/jackx/Desktop/CP-DISR-HANDOFF/Original_Sources_20261010`。`FINAL_INVENTORY.json`登记的56个文件已逐项检查大小与SHA256，无不一致；这只证明资料完整性，不意味着56份内容全部已读。读取范围和逐条页码/代码行号以A1/B1报告为准。

| 来源 | 状态 | 范围与限制 |
|---|---|---|
| 指定实验卡 | READ_PRIMARY | 实际阅读权限、输入、两路审计与交付要求 |
| papers/01_Return_to_Tradition_2403.16508 | READ_PRIMARY | 分页正文第2–4、6、9页；未运行原生系统 |
| papers/02_WL_Hyperparameter_Study_2508.18515 | READ_PRIMARY | 第1、4–6、12页，剪枝与配置研究 |
| papers/03_WLPlan_2411.00577 | READ_PRIMARY | 第1–4页，ILG/νILG与工具身份 |
| papers/04_Expressiveness_ICAPS2024 | READ_PRIMARY | 第1–3、6–8页；定理与图7，协调者另实际查看第7页图像 |
| papers/05_Effective_Data_Generation_ECAI2025 | INDEX_ONLY | 包完整性核验；本轮无据其正文新增裁决 |
| papers/06_Domain_Independent_2312.11143 | INDEX_ONLY | 同上 |
| papers/07_Guiding_GBFS_IJCAI2024 | READ_PRIMARY | 正式论文第3–7页、相关理论和协议 |
| papers/08_Optimize_Rank_NeurIPS2023_2310.19463 | READ_PRIMARY | arXiv v1正文/附录指定范围；B1另核对正式会议版本及Table1 |
| GOOSE icaps-24 tag / icaps24 branch | READ_PRIMARY | 指定训练、ILG、WL、CPU搜索代码；不同commit，未运行 |
| WLPlan v2.1.0 tag | READ_PRIMARY | 指定图、特征、邻居容器、MaxSAT与映射代码；未安装补充wheel |
| OptRank后续公开Zenodo包12575733 | READ_PRIMARY | 数据、模型、训练、部署和搜索入口静态审计；非论文原受限包 |
| Chrestien作者仓库e7cd65b及内嵌NeuroPlanner | READ_PRIMARY | 指定损失/训练/结果制表代码；一个已保存表格单元格重统计 |
| 论文时期Zenodo 10757383 / 11107790原包 | UNAVAILABLE | 元数据可读，原受限包字节未取得 |
| 服务器安装WLPlan、W1输入与旧诊断表 | READ_PRIMARY | 输入字节/身份绑定；本次A-V获准调用collect/embed |
| 服务器当前WLPlan源码9d9fc99 | READ_PRIMARY | 六份相关源码/测试静态范围，manifest登记；未证明对应安装二进制 |
| Claude既有报告、脚本及173行B1表 | READ_PRIMARY | 下载原字节；173行重统计，不冒充原模型评分重跑 |
| 用户粘贴的Claude总结 | USER_REPORTED → 部分已核实/更正 | 原总结只作线索，具体核实或冲突见05文件 |
| D0/T1真实OPEN事件、对应历史完整日志 | UNAVAILABLE | 没有补造轨迹；全部B1变化对仍E0 |

`prior_claude/`与`prior_diagnostic/`是明确选取文件的原字节副本，不是两目录完整镜像。所取得且出现在旧output_index的18份文件已核对旧索引大小/SHA。论文引用页码和源码版本见 `A1/original_method_audit.md`、`B1/original_method_audit.md`；原作者结果统计仅核验一个单元格，未取得完整原生运行身份。
