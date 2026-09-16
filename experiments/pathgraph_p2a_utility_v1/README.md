# PathGraph P2A — State-conditioned weighted-BC utility

本包把研究从“表示与核算是否自洽”推进到“奖励重加权是否有助于策略学习”。

**证据层级：`ACTION_CONDITIONED_ABSTRACT_SKILL_MDP_NOT_PHYSICS`。**

提供：参考环境、状态—实际执行动作数据生成、字节锁P1奖励接口、9方法权重、PyTorch策略训练、自主rollout、配对统计和测试。

不提供：第三方抓取模型、物理机器人、策略增益保证、已经完成的正式45个训练任务。

入口：

```bash
python -B -m unittest discover -s tests -v
python -B tools/smoke.py --out NEW_SMOKE_DIR --v6-tools FROZEN_V6_TOOLS --extra-methods FROZEN_EXTRA_METHODS
PY=/path/to/python bash tools/run_formal.sh REPO V6_TOOLS EXTRA_METHODS NEW_OUTPUT_ROOT
```

运行前阅读`MANUAL.md`、`NEW_AGENT_HANDOFF.md`。本包依赖numpy、torch；无需MuJoCo、视觉库或新授权流程。
