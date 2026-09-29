# E4 恢复运行摘要（工程记录，非科学结论）

- 恢复 attempt 已预留 2/8；状态 {'RESERVED': 0, 'STARTED': 0, 'COMPLETED': 0, 'FAILED': 0, 'UNKNOWN': 2}；累计（原8+恢复）10/16。
- 新 provider / RL / optimizer / elastic：0 / 0 / 0 / 0。S2、S3、正式 test：NOT_RUN。
- E1–E6 中六类证据表未在本轮补齐，均标 NOT_ESTABLISHED；tp_training_authorized=false。

## 分支结果

| case | candidate | repeat | state | exit | status | reason | success |
|---|---|---|---|---|---|---|---|
| T_A_dev_14 | a:OPEN:container:v1 | 0 | UNKNOWN |  |  |  |  |
| T_A_dev_14 | a:PICK:second_object:v1 | 0 | UNKNOWN |  |  |  |  |
| T_A_dev_14 | a:OPEN:container:v1 | 1 | NOT_STARTED |  |  |  |  |
| T_A_dev_14 | a:PICK:second_object:v1 | 1 | NOT_STARTED |  |  |  |  |
| T_A_dev_18 | a:OPEN:container:v1 | 0 | NOT_STARTED |  |  |  |  |
| T_A_dev_18 | a:PICK:second_object:v1 | 0 | NOT_STARTED |  |  |  |  |
| T_A_dev_18 | a:OPEN:container:v1 | 1 | NOT_STARTED |  |  |  |  |
| T_A_dev_18 | a:PICK:second_object:v1 | 1 | NOT_STARTED |  |  |  |  |

## 配对恢复与配对结果

- T_A_dev_14 repeat 0: restore=PAIRED_RESTORE_NOT_ESTABLISHED；paired_effect=NOT_ESTABLISHED
- T_A_dev_14 repeat 1: restore=PAIRED_RESTORE_NOT_ESTABLISHED；paired_effect=NOT_ESTABLISHED
- T_A_dev_18 repeat 0: restore=PAIRED_RESTORE_NOT_ESTABLISHED；paired_effect=NOT_ESTABLISHED
- T_A_dev_18 repeat 1: restore=PAIRED_RESTORE_NOT_ESTABLISHED；paired_effect=NOT_ESTABLISHED

每 case 汇总：{'T_A_dev_14': 'NOT_ESTABLISHED', 'T_A_dev_18': 'NOT_ESTABLISHED'}

吞吐见 throughput_report.json（单 worker 基线 NOT_MEASURED，不声称加速）。
