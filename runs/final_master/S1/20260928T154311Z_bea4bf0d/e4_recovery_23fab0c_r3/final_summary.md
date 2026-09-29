# E4 恢复运行摘要（工程记录，非科学结论）

- 恢复 attempt 已预留 8/8；状态 {'RESERVED': 0, 'STARTED': 0, 'COMPLETED': 8, 'FAILED': 0, 'UNKNOWN': 0}；累计（原8+恢复）16/16。
- 新 provider / RL / optimizer / elastic：0 / 0 / 0 / 0。S2、S3、正式 test：NOT_RUN。
- E1–E6 中六类证据表未在本轮补齐，均标 NOT_ESTABLISHED；tp_training_authorized=false。

## 分支结果

| case | candidate | repeat | state | exit | status | reason | success |
|---|---|---|---|---|---|---|---|
| T_A_dev_14 | a:OPEN:container:v1 | 0 | COMPLETED | NORMAL_TERMINATION | TERMINATED | TASK_SUCCESS | True |
| T_A_dev_14 | a:PICK:second_object:v1 | 0 | COMPLETED | NORMAL_TERMINATION | NO_PLAN | NO_PLAN | False |
| T_A_dev_14 | a:OPEN:container:v1 | 1 | COMPLETED | NORMAL_TERMINATION | TERMINATED | TASK_SUCCESS | True |
| T_A_dev_14 | a:PICK:second_object:v1 | 1 | COMPLETED | NORMAL_TERMINATION | NO_PLAN | NO_PLAN | False |
| T_A_dev_18 | a:OPEN:container:v1 | 0 | COMPLETED | NORMAL_TERMINATION | TERMINATED | TASK_SUCCESS | True |
| T_A_dev_18 | a:PICK:second_object:v1 | 0 | COMPLETED | NORMAL_TERMINATION | NO_PLAN | NO_PLAN | False |
| T_A_dev_18 | a:OPEN:container:v1 | 1 | COMPLETED | NORMAL_TERMINATION | TERMINATED | TASK_SUCCESS | True |
| T_A_dev_18 | a:PICK:second_object:v1 | 1 | COMPLETED | NORMAL_TERMINATION | NO_PLAN | NO_PLAN | False |

## 配对恢复与配对结果

- T_A_dev_14 repeat 0: restore=PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE；paired_effect=OUTCOME_DIFFERENCE
- T_A_dev_14 repeat 1: restore=PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE；paired_effect=OUTCOME_DIFFERENCE
- T_A_dev_18 repeat 0: restore=PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE；paired_effect=OUTCOME_DIFFERENCE
- T_A_dev_18 repeat 1: restore=PAIRED_RESTORE_VERIFIED_PUBLIC_AND_SIM_STATE；paired_effect=OUTCOME_DIFFERENCE

每 case 汇总：{'T_A_dev_14': 'REPEATS_AGREE_OUTCOME_DIFFERENCE', 'T_A_dev_18': 'REPEATS_AGREE_OUTCOME_DIFFERENCE'}

吞吐见 throughput_report.json（单 worker 基线 NOT_MEASURED，不声称加速）。
