# 方法与版本身份

| 系统 | 固定身份 | 本轮证据层级 |
|---|---|---|
| CP-DISR服务器 | codex/cp-disr-c1-compact-diagnosis-v2，cc7d79d6fcb053bf0c16033e765a2e808de2aaac | 真实仓库、分支/HEAD/remote/工作区核查；已有未跟踪checkpoint资产 |
| 本地CP-DISR目录 | Git master尚无提交、无可解析HEAD、无remote | 报告工作目录，不是服务器实际代码checkout；本轮未初始化/克隆 |
| Claude最新审计 | /home/xushijie3/work/cp_disr_a1_b1_audit/CP_DISR_A1_B1_AUDIT_20261010T052625Z | 独立目录，不能称已包含于CP提交 |
| 旧A1/B1诊断 | /home/xushijie3/work/cp_disr_a1_b1/CP_DISR_A1_B1_20261010T021755Z_cc7d79d6f | 已保存诊断资产，与最新审计分开 |
| 2024 Return to Tradition代码tag | 1b1f57f0c1654049605758a8ddd52ea2d657267c（icaps-24） | OFFICIAL_CODE_AUDITED，指定静态范围，无原生运行 |
| 同名近似历史branch | 56a025daae4a6aca576899589ad87181b0ebea71（icaps24） | 不等同tag；训练划分不同 |
| 补充WLPlan v2.1.0 tag | 4aa84678d7a9dde58bac4fe0af1a16d50e5be3b6 | 静态审计；未安装、未替换服务器扩展 |
| 服务器WLPlan源码 | 9d9fc99165135bfe8e2817145888ede162e4a97a | 受控文件无修改，相关源码已读；构建对应关系未知 |
| 服务器安装WLPlan | METADATA=2.2.0；扩展SHA256=0089e22796904c5ee73d0d5c863e3ce4e82b54561393e9e86b43d37e3932587f | 固定构建6次特征流水线已运行；不是2024原生复现 |
| OptRank(GOOSE) | IJCAI2024；后续公开记录12575733，归档MD5 ae64b6490faff699ce11318082c4020c | 指定官方代码静态审计；原论文记录11107790未取得 |
| Chrestien等 | arXiv2310.19463v1 / NeurIPS2023正式版；仓库e7cd65baa820dc8b9267749ff57e384d41bea02a及内嵌包 | 指定静态代码审计；一个作者已保存Table1单元格统计核实 |

2024 WL原论文是L4、多重集、h*回归；CP当前后续配方为ILG/L2/set/a-m/rank-SVM。OptRank(GOOSE)是LLG神经排序模型，不是上述WL/rank-SVM。研究年份、配置一致性与构建字节一致性分别登记，不混用。

服务器origin为 `git@github.com:rollinpioneer/graph.git`。本轮未fetch/pull/push；remote配置本身不是远端所有refs与本机版本相同的证明。
