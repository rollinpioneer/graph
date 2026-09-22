# Stage 0A — Environment, Interface and Time Binding

## Scope / Purpose
绑定真实基础设施，并完成时间reference校准。

## Inputs / Preconditions
原source可读且目标执行已获授权。共同输入为本包research规范、interfaces manifest及Experiment Plan v1.1；目标runtime不得有执行关键MUST_BIND。

## Frozen / Tunable
所有方法定义与权限按共同合同冻结。除1B的有限队列以及明确记录的资源分块，不能当场新增优化变量。消融只改变自己的字段。

## Run Matrix / Budget
不产生默认新训练。实际evaluation/reference/API预算见本阶段细则，不能把它们计作RL训练run。

## Evaluation / Required Logging
0是本阶段指定evaluation默认episode数（0表示不做策略评价）；按主计划精确condition/split执行。每个真实run记录配置/source/Git/cache/H、实际时间、mask与所选动作、损失与D/Delta；无训练量不填伪造0。

## Extra protocol
每训练family最多10次reference尝试目标5个合法成功，确定H及d_ref；不是RL/BC数据。

## Output Directory / Required Files
`runs/stage_0a/`；阶段级`reports/stage_0a_summary.md`、`status/stage_0a.json`、实际run/source清单与异常/调参记录。训练阶段另含train/eval/decision日志、checkpoints、raw metrics和图表；诊断阶段含条件适用性与原始episode；打包阶段含source/hash/selection。

## PASS / STOP
缺实际安全/资产/入口/软件则BLOCKED；生成文档不等于目标机通过。。状态采用NOT_STARTED/RUNNING/PASS/PASS_WITH_NOTES/NEEDS_RERUN/BLOCKED；NOT_APPLICABLE和SKIPPED放execution_reason，不另造状态。

## Failures / Rerun / Writing
安全/接口/数据污染即停并保留证据；正常负结果不删除。重跑新attempt不覆盖旧run；Full对照只有所有匹配字段一致才复用。所有Stage不阻塞论文方法写作，缺结果只限制具体经验claim。

## Agent Execution Template
1. 确认授权范围为Stage 0A及其phase，读取本卡和主计划，不默认跨Stage。
2. 验证前置报告、source/hash和执行关键runtime绑定；缺失写BLOCKED。
3. 解析本阶段真实run/evaluation矩阵，区别planned ID、已有control和新任务。
4. 冻结本轮task/split、seed、H、mask权限、配置与条件清单。
5. 只在已授权实际工作树调用真实入口；本包不假装包含训练器。
6. 按本阶段预算执行并持续写原始日志；失败、安全和不适用条件分别处理。
7. 按既定规则选择checkpoint/聚合指标，保留全部有效失败与预算分母。
8. 生成本阶段summary、必要图表与source引用；未生成材料写NOT_CREATED。
9. 更新stage状态与实际成本账本，记录改动和对照复用。
10. 提出下一Stage建议后停止；没有明确授权不自动继续。
