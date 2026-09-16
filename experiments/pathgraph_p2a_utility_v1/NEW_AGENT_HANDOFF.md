# P2A 新 Agent 接手说明

**执行目标：训练策略，再让策略自己执行；比较不同奖励重加权是否提高任务完成率。不是再做一次公式回放。**

固定BASE：`8e86b246e900e1e12dbb21c88b95046a06c8e4c5`。
新分支：`research/pathgraph-p2a-state-weighted-bc-utility-v1`。

先读`MANUAL.md`，再验证包。包内`p2a/`是一套可执行的抽象技能MDP、混合质量示范生成、冻结P1奖励接入、训练、自主测试及配对统计。

## 路径

```text
repo: /home/__compress_data/xushijie/graph_github_upload
new worktree: /home/__compress_data/xushijie/graph_pathgraph_p2a_utility_worktree
package in repo: experiments/pathgraph_p2a_utility_v1/
external outputs: /home/__compress_data/xushijie/graph_pathgraph_p2a_utility_data/run_v1
existing V6 tools: /home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools
existing extra methods: upgrade_v2/p1_mainline_grasp_v6gm1/extra_methods.py
```

## 核心执行

1. BASE建树，把本包复制到新目录；不修改任何P1历史或冻结方法。
2. 校验P1四个文件字节哈希；本包不重新实现V6潜势。
3. 跑37项单测和单独smoke。确认动作有因果效果、policy真的控制环境。
4. 将P2A源、配置和协议提交成runner；从固定执行树运行。
5. 生成1920条新训练示范；锁定数据哈希；训练9方法×5seed。
6. 使用最后5000步checkpoint，各跑256条validation和1024条test自主episode。
7. 补齐轻量结果索引、逐loss/condition报告和来源清单，完整模型与rollout留外置目录。
8. 对BC、count+events做主配对比较；同分或更差是正常科学结果。

命令详见手册和`tools/run_formal.sh`。不要把“文件生成成功”“训练loss降低”或“总reward更高”写成policy utility。

## 绝对不扩张的工程支线

无需抓取网络、相机、EGL、MuJoCo、反应式拦截、真机、LLM图生成。参考环境的REGRASP是声明的抽象技能动作，不代表解决了真实重抓。

当前state模型对所有策略同样可见；只比较示范权重。不能给PathGraph策略更多观测，不能让基线缺失历史再比较。

## 必须保留的解释

- 参考MDP结果不是机器人策略性能结果。
- 原SARM文章只作动机来源；当前线性基线不是原视频SARM模型。
- P1闭环代数正确不保证P2加权BC有效。
- count+events或shuffle不差时，要收窄图增益主张。
- `confirmation_passed=false`不阻塞P2收口；本轮独立使用`policy_utility_evidence`。
