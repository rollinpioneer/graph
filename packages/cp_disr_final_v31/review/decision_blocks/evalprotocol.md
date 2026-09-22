## 29.3 Seed、checkpoint与评价：小规模但可解释

Smoke和探索pilot用seed0；主比较补齐0/1/2，而非默认新增完全独立Presentation组。配置/任务未改变且hash可追溯时复用pilot；必要的开发改动记录后，只补跑受影响的比较组，不静默混profile。最多补到5seed是明确可选项，不反复加到获胜。

主文定量对照默认包含全部3个预定有效训练seed，显示逐seed点及均值/样本SD；零成功但数值正常属于有效结果。Private Paper View仍可单独选seed/task作illustration，但标selected exploratory view，不取代“整体方法更好”所需的完整对照表。基础设施无效run及其重试均在Raw中保存。

dev与test分离。每checkpoint：smoke10个dev episode，主训练20个dev；默认从step0开始评价。按最近3个dev窗口平均score选checkpoint，并列取窗口最小值更大，再并列取更早step。默认score为**实际起点时长折扣return**，同时报告success；这避免所有方法success饱和时按偶然尖峰/最早step选择而忽略真实时间差。规则在训练前统一，test前写checkpoint/hash；每task/seed最终30个独立test episode，可共同扩到50，不只补赢家。

主结果同时给success、discounted return、raw learning AUC和成功条件完成时间/成功分母。不同gamma的return不能直接跨family混成同指标，默认共同suite H避免该问题。图显示可用窗口3 trailing moving average，但表与AUC只用原始值，保存raw版；所有选择/平滑写manifest。

Actor-dependence与闭环分开。推荐≤100个真实snapshot，比较DK0、DP0、候选间DP置换和合法先验变化；闭环每条件20episode。same-checkpoint先验测试Original、No-prior、Deletion、Target-swap各25case，Full与A_CAT使用同seed的冻结checkpoint和配对case；Irrelevant仅有原定义合法独立组件时运行。无需新训练。

No-prior正式称**missing-prior / prior-dependence evaluation**：整份输入缺省机制在训练见过，但不能保证关先验后整条闭环状态分布都在训练分布内。80/20对Full和CAT相同不是“该正则完全没有作用”的证明；不做Full-clean，因为本论文不独立归因dropout收益。
