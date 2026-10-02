# 授权请求（仅请求，未启动）：R-TB-E-1 = B1-K+E seed 1，占用 ELASTIC-01

生成于 2026-10-02T02:28:32Z。本文件不是授权：没有启动任何 run，没有签发 token，没有改动账本，没有分配 elastic 槽位。

## 依据
- Plan v3 §3.1：B1-K+E s0 的预先固定 post-update dev10 评价点出现完整任务成功 → NONTRIVIAL_LEARNING（登记见 `registration_NONTRIVIAL_LEARNING_R-TB-E-0.json`，绑定 `eval_final.json`（10/10）与 `final_n_014512_u14.pt`）；→ 申请同 profile s1；该 s1 从 elastic4 扣 1，必须另有授权（§3.1、§7.2）。该规则按原文执行，不以"是否击败 B2"为条件，也不事后改规则。
- elastic 余量：全仓库（本 worktree 与 15 个同级 worktree）未发现任何 ELASTIC-0x 分配记录或统一 elastic 总账；S0 manifest、S1 五份预算账本、本卡 T_B 账本均记 elastic 已用 0（三条 T_B 为基础 MUST，不占 elastic）。按已有记录**余 4/4**；对外部 Master Ledger 的核对 **UNVERIFIED**（服务器上没有）。若批准，余 3。

## 请求的 run
| 项 | 内容 |
|---|---|
| Run | R-TB-E-1，T_B，B1-K+E，seed 1，ELASTIC-01 |
| profile | 与 R-TB-E-0 相同，仅 seed 与 run id 不同：同源码 cac2fa5b3（算法/运行时文件不变）、同 dev10 切分、Ncap 16384、Tcap 68812.8、H 23.1、评价点 0/4096/8192/final、同 PPO/optimizer、1 worker、1 张独立 GPU |
| attempt 成本 | elastic 1/4；失败也计数；续跑不得清零 N/T/墙钟 |
| 墙钟（估计） | 约 8.9 h（R-TB-E-0 实测 8.89 h；三条 run 在共享服务器上 7.9–11.9 h） |
| 存储（估计） | E-0 实际 2224 MiB（2.17 GiB）；×1.25 约 2.7 GiB，再加 2 GiB 预留 → 启动前需 ≥ 约 4.7 GiB 空闲；guard 停止阈值应 ≥ 2 GiB |
| 评价 | 4 点 × 10 = 40 个 dev episode；provider 0；formal test 0 |

## 代码里发现的前置条件（需要你决定）
- `src/cp_disr/final_tb.py` 的 `PLAN_TABLE` 写死三条 plan，且 `MAX_NEW_RL_ATTEMPTS = 3`，当前 launch 入口无法登记第 4 条 run。
- 因此跑 R-TB-E-1 需要对 launch 入口做最小改动（加 plan 行与 elastic 记账）→ 新 prep 提交 → HEAD 不再等于 cac2fa5b3 → 现有 smoke receipt 和 release token（绑定 HEAD == prep）失效 → 需要新登记和 smoke 决定。
- 累计 smoke 额度已用尽：episode 3/3、construction 6/6、reset 6/6，skill 11/24。
- smoke 两个选项：A1（保守）在新增 smoke 额度下对 1 个 dev case 做最小 smoke（本轮实测 1 case = 1 episode、2 constructions、2 resets、5 skill calls）；A2 由你批准把现有 TB_RUNTIME_SMOKE_PASS 重新绑定到新 prep（算法与运行时文件逐字节不变，差异仅限 launch 入口 plan 表）。我不默认选 A2。

## 需要你回复的决定
1. 是否批准 ELASTIC-01 → R-TB-E-1（B1-K+E seed 1）。
2. smoke 走 A1 还是 A2。
3. 是否允许最小 launch 入口改动与新 prep 提交。
4. 本 run 的 guard 停止阈值是否定为 ≥ 2 GiB。
5. 启动时间窗/GPU。

## 不在本请求内
T_P、provider、formal test、B1-H 或其他 elastic 用途、重跑 R-TB-E-0/DK-1/K-1、Method 2.1.1 的任何改动、Family B / RoboCasa 工程。
