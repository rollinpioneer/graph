# 02 原作者方法身份与还原程度

## 1 方法身份
| 方法 | 论文/版本 | 官方代码依据 | 固定版本 | 身份标签 |
|---|---|---|---|---|
| WL-GOOSE（A1） | Return to Tradition，ICAPS 2024（arXiv 2403.16508 v1） | 作者 DillonZChen 的仓库，README 引用该论文；论文给出 Zenodo 代码包 | 本机 main `04dbe8f`（2026-03-27，**晚于论文约两年**）+ wlplan `9d9fc99`（v2.1.0-12）+ scorpion `4c16ce05`；论文时分支 `icaps24` 与 Zenodo 包未取得 | 论文 `READ_PRIMARY`；代码 `OFFICIAL_CODE_AUDITED`（静态，**未运行**）；无 `OFFICIAL_EXISTING_RUN_VERIFIED` |
| 项目的旧 WL（typed，goose/train_typed） | 同上 | 项目内用官方训练入口在 Depots typed 数据上训练（`classic.toml` 副本，opts 与 `classic.toml` 一致：ilg、2 轮、a-m、set、rank-svm） | 同上本机副本 | `CP_DISR_ADAPTATION`（数据与领域）；训练命令行未在本卡重新核实 |
| W1 | — | 拟合脚本复制 `rank_classifier.py` 的估计器，监督换成 D0 对齐 | `scripts/c1_compact_w1_fit.py` @ cc7d79d6f | `CP_DISR_ADAPTATION`（监督与权重）+ 官方特征库 |
| Opt Rank（B1） | Hao 等，IJCAI 2024 §3–4；其中 “Opt Rank(GOOSE)” 把 GOOSE 的 WL 模型接到该排序框架 | 未找到官方实现入口 | — | `PAPER_ONLY` |
| Chrestien 等排序损失（B1） | NeurIPS 2023 | 未找到官方实现入口 | — | `PAPER_ONLY` |
| D0 / T1-S | 项目 | `compact/{models,train,crossparent}.py` @ cc7d79d6f | — | `CP_DISR_ADAPTATION`（非任何论文方法的复现） |

## 2 还原程度（八项；某一项一致不代表端到端还原）
| 项 | WL-GOOSE（官方静态） vs 旧 WL / W1 | 说明 |
|---|---|---|
| 算法 | 一致（同一 wlplan 库：ILG、WL、a-m、LinearSVC 估计器） | 仅有 2 轮（现行默认）对 4 轮（论文实验）的配置差 |
| 数据 | **不同** | 官方：每个训练题一条最优计划上的状态；项目：Depots typed、Train96 的 457 条轨迹（96 条专家+361 条偏离） |
| 监督 | 旧 WL 近似官方；W1 **不同** | 官方：计划后继 vs 兄弟（“maybe”）+ 父状态（“bad”）；W1：精确 d* 严格序对，无父子对 |
| 训练 | 估计器一致；W1 多了样本权重与相同差向量合并 | `rank_classifier.py:21,57` 无权重；`c1_compact_w1_fit.py:81–83` 有权重 |
| 模型选择 | 无（单次凸拟合） | — |
| 推理 | 特征与 `predict` 一致；取整一致 | 官方在 C++ 里 `round`；项目 Python `WlEval` 同语义（半数远离零） |
| 搜索 | **不同** | 官方 Scorpion/FD 里的 GBFS；项目共享的 eager GBFS（`engine.py`：键 (h, 序号)、生成时检查目标、无 reopen）；FD 侧细节本卡未核 |
| 计时 | **不同/不可比** | 共享 GPU/CPU、300 s、10 万节点；不与作者计时拼表 |
轨道：作者原生轨道（未运行、无记录）与 CP-DISR 匹配轨道（共同 GBFS）严格分开，本卡不拼表。
