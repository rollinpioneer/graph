# CP-DISR-TB-INDEP-HOLDOUT-01 — frozen T_B prematerialized independent holdout

- 测试标签：`PREMATERIALIZED_INDEPENDENT_HOLDOUT`（不是 strict blind test）。`GLOBAL_S4 = NOT_COMPLETE`，`GLOBAL_S5 = NOT_COMPLETE`；本次不是 GLOBAL S6。
- R = absent；deterministic argmax；isolated RNG；optimizer_steps = 0；provider = 0；每个 model×case 一次评测。
- planned slots：210；MEASURED：210；TECHNICAL_NOT_MEASURED：0；NOT_EXECUTED_RELEASE_STOPPED：0。
- 全部 210 个 planned slot 正常完成：**是**。

## A. CORE（current 2.1.1 执行类，4×30=120）

| slot | method | seed | success | mean G | time-to-first-success mean / median (s, 成功局) | skills mean / median | duration median (s) | valid / planned / technical |
|---|---|---|---|---|---|---|---|---|
| TB-M3 | B1-K | 1 | 0/30 | 0.000 | n/a / n/a | 10.30 / 10.0 | 59.95 | 30 / 30 / 0 |
| TB-M5 | B2 | 1 | 26/30 | 0.435 | 22.99 / 22.97 | 4.83 / 5.0 | 22.92 | 30 / 30 / 0 |
| TB-M6 | B1-K+E | 0 | 27/30 | 0.442 | 23.68 / 23.50 | 4.90 / 5.0 | 23.42 | 30 / 30 / 0 |
| TB-C1 | B1-K+E | 1 | 28/30 | 0.469 | 22.96 / 22.87 | 4.93 / 5.0 | 22.82 | 30 / 30 / 0 |

- TB-M3 失败原因分布：{"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 27, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-7": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-20": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-21": 1}
- TB-M5 失败原因分布：{"NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-7": 1, "INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-20": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-21": 1}
- TB-M6 失败原因分布：{"NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-7": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-20": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-21": 1}
- TB-C1 失败原因分布：{"NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-23": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-26": 1}

## B. HISTORICAL_EXTENSION（ARCHIVED_EXECUTION_CLASS，2×30=60）

| slot | method | seed | success | mean G | time-to-first-success mean / median (s, 成功局) | skills mean / median | duration median (s) | valid / planned / technical |
|---|---|---|---|---|---|---|---|---|
| TB-M2 | B1-K | 0 | 0/30 | 0.000 | n/a / n/a | 15.43 / 17.0 | 59.95 | 30 / 30 / 0 |
| TB-M4 | B2 | 0 | 28/30 | 0.469 | 22.96 / 22.87 | 4.93 / 5.0 | 22.82 | 30 / 30 / 0 |

- TB-M2 失败原因分布：{"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 27, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-6": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-11": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-28": 1}
- TB-M4 失败原因分布：{"NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-23": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-26": 1}

## C. SYSTEM_REFERENCE（ARCHIVED_EXECUTION_CLASS，1×30=30）

| slot | method | seed | success | mean G | time-to-first-success mean / median (s, 成功局) | skills mean / median | duration median (s) | valid / planned / technical |
|---|---|---|---|---|---|---|---|---|
| TB-M1 | B0 | 0 | 0/30 | 0.000 | n/a / n/a | 3.80 / 4.0 | 24.10 | 30 / 30 / 0 |

- TB-M1 失败原因分布：{"NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-1": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-2": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-3": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-4": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-5": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-6": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-7": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-8": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-9": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-10": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-11": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-12": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-13": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-14": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-15": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-16": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-17": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-18": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-19": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-20": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-21": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-22": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-23": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-24": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-25": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-26": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-27": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-28": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-29": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-30": 1}

## CORE 同 case 配对差异（描述性；每格一个训练 seed，30 个配对 case）

| 比较 | 配对 n | a成功b失败 | a失败b成功 | 均值 ΔG (a−b) | SE | 精确双侧 sign test p |
|---|---|---|---|---|---|---|
| B2 s1 vs B1-K s1 | 30 | 26 | 0 | 0.435 | 0.032 | 0.0000 |
| B2 s1 vs B1-K+E s0 | 30 | 0 | 1 | -0.008 | 0.016 | 1.0000 |
| B2 s1 vs B1-K+E s1 | 30 | 2 | 4 | -0.034 | 0.040 | 0.6875 |
| B1-K+E s0 vs s1 | 30 | 2 | 3 | -0.026 | 0.037 | 1.0000 |

## 解释边界（C1）

- 开发期学习过程（0 / 4096 / 8192 / final）与独立 holdout 的 final 模型表现是两个不同的证据层。
- 本 holdout 不能直接证明“B2 的 early-learning 优势可泛化”；它只回答 final 模型在这 30 个冻结 test case 上的表现。
- 若 B2 与 +E 的 final holdout 都高，不删除 early-learning 开发期结果；若 +E 或 B2 的 holdout 下降，如实报告 endpoint generalization gap，不更换 checkpoint。
- 没有选择 best-test checkpoint，没有调参；selected-dev checkpoint 分数不在主 test 表中。

## 执行类与限制

- CORE 为 CURRENT_2_1_1 执行类；HISTORICAL_EXTENSION / SYSTEM_REFERENCE 为 ARCHIVED_V13 执行类，结果始终标注 ARCHIVED_EXECUTION_CLASS。7 个模型不是同源排名，历史模型未转换进当前 Policy，未改 state_dict。
- 沿用 prep 卡的未决缺口 UC2 / UC3 / UC4 / UC5 / UC7（带入，不在此修复）。
- `success_seconds` 仅是最终 skill 时长（FINAL_SKILL_DURATION_ONLY）；episode 级时间使用 `time_to_first_confirmed_success_s`。

## 事后数据完整性观察（描述性）

- 7 个模型两两比较同 case 行（success、G、steps、reason、每个动作 id 与仿真时钟全部相同才算相同），见 `pairwise_row_identity.json`。
- TB-C1 vs TB-M4：30/30 行完全相同（checkpoint SHA256 不同；执行类 CURRENT / ARCHIVE）。
- TB-M1 vs TB-M5：3/30 行完全相同（checkpoint SHA256 不同；执行类 ARCHIVE / CURRENT）。
- 解读边界：这是确定性策略在同一冻结场景上选择了同一条动作序列的描述性事实；不同 SHA256 的 checkpoint 可以做出相同决策。它不改变任何模型的计分，也不能据此推出权重或训练等价。
