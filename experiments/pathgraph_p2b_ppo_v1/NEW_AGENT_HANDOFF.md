# P2B 新Agent接手说明

## 一句话任务

保留P2A负结果。把被冻结的V6势函数作为折扣PBRS接入官方PPO，在同一抽象技能环境内，与五个奖励对照做真实on-policy训练和自主评估。

## 固定入口

```text
BASE: 577db67c64343850965fa4e0f6bc7cac0c15674d
branch: research/pathgraph-p2b-graph-ppo-utility-v1
repo: /home/__compress_data/xushijie/graph_github_upload
worktree: /home/__compress_data/xushijie/graph_pathgraph_p2b_ppo_worktree
install package: $WORK/experiments/pathgraph_p2b_ppo_v1
data: /home/__compress_data/xushijie/graph_pathgraph_p2b_ppo_data/run_v1
V6 tools: /home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools
frozen P2A env: $WORK/experiments/pathgraph_p2a_utility_v1/p2a/env.py
```

确认路径存在后再运行；这不是聊天sandbox地址。分支或main后来前进不影响使用固定BASE，不改main。

## 必读文件

`MANUAL.md`、`ENVIRONMENT_AND_REWARD_CHANGES.md`、`contracts/protocol.json`、`contracts/method_registry.json`、`contracts/source_lock.json`、`LOCAL_VALIDATION.json`。

## 首要注意

1. 不再训练加权BC；不使用P2A示范、不加载BC模型。
2. 不再接视觉/抓取模型；无MuJoCo前置条件。
3. 用真实SB3，不用自写“简化PPO”替代。
4. V6数值势函数不改；训练reward用gamma*Phi_next-Phi_current。
5. 真正终端effective Phi=0；外部截断保留末势并bootstrap；buffer结束不是episode结束。
6. 当前wrapper将已声明的有限task deadline映射成terminated，保存legacy truncated，不改历史。
7. 同seed actor/critic初始化相同；on-policy动作/轨迹不要求相同。
8. 不用reward总分给方法排名，比较最终成功率及强非图对照。
9. 32train/8val/32test family互斥；test在48模型最终checkpoint封存后运行。
10. 本包本地未完成SB3真实smoke，原因是依赖不可下载。服务器先跑真实smoke并记录，不把52个纯/适配测试当成训练成功。

## 执行

隔离Python环境安装锁定SB3/Gymnasium；保留一套兼容PyTorch，pip freeze。设置P2B_SOURCE_REPO、P2B_V6_TOOLS后运行`tools/run_checks.sh`。

完成源码/数据契约/终止API测试，提交runner，执行`tools/run_formal.sh`。该脚本可以自动推进训练和评价，无人工nonce流程。

Agent需要补齐运行环境锁、外置历史family扫描、人可读报告、奖励组件、leave-one-seed-out及限制；包内参考训练代码不能代表这些已执行。

## 完成标准

48job、真实PPO自主策略、同环境和输入、终止核算正确、全部test保留、强基线配对比较完成。科学结果可以为正、混合或负；禁止为了PASS修改test。
