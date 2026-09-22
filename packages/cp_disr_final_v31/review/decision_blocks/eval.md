## 29.2 Revised Experimental Route v1.1

保留已有Stage ID语义大体兼容，并增加0D。**0A/0B/0C/0D只阻塞依赖它们的真实运行，不阻塞Problem/Method写作。**本轮只生成计划与代数文件验收，不宣称已经执行这些Stage。

| Stage | 目标 | 默认新增训练 |
|---|---|---:|
| 0A | 真实接口/软件/权限/任务绑定；每训练family最多10次reference尝试取得5次成功，确定H与d_ref | 0 |
| 0B | 原生产不变量＋A_CAT null、B1无后继、Set permutation、unweighted Actor及时长单位检查 | 0 |
| 0C | D0/TB/TC各8个初始dev场景，固定VLM协议和cache质检 | 0 |
| 0D | 每family100个uniform masked episodes，低成功时扩到200；必要时50个contract-aware诊断 | 0 |
| 1A | D0，B2/Full，seed0，16,384转移上限 | 2 |
| 1B | 仅学习/数值明显异常时，有限单项优化，最多6个额外job | 条件性，非默认 |
| 2A | TB/TC × B0/B1-K/B2/Full × seed0，65,536/运行 | 8 |
| 3A-pilot | TC × A_CAT × seed0；Full复用2A | 1 |
| **第一暂停点** | **11个默认训练；形成探索Results；不等3seed全部完再写** | **累计11** |
| 2B | 原两任务四方法补seed1、2；不重跑已匹配seed0 | 16 |
| 3A-completion | TC A_CAT补seed1、2；Full继续复用 | 2 |
| 2C | 按明确source生成主表/主曲线 | 0 |
| 3B | TC A_Q，0/1/2 | 3 |
| 3C | A_B仅在bounded经验收益claim需要时，TC，0/1/2 | 可选3 |
| 4A/4B | 同checkpoint Actor依赖与先验变化；Full/A_CAT配对 | 0 |
| 5A | 一种已有source模型对应的实际未见组合 | 0；无source则不自动增训 |
| 6A/6B/6C | Raw审计、展示manifest与论文包 | 0 |

第一批11个，上限622,592训练transition；补齐两任务三seed和A_CAT后累计29个，上限1,802,240；再含A_Q为32个，上限1,998,848；A_B也做则35个，上限2,195,456。表内不含评价、API和条件性重跑，均另记成本。不同stage复用同一Full只计一次。

**Tier1：**B0/B1-K/B2/Full及A_CAT；先单seed再补齐。**Tier2：**A_Q及同checkpoint诊断，必要的一种组合评价。**Tier3：**A_B、B1-H、A_DD及资源/表示诊断。不存在7方法×所有task×多seed一次性全跑，也取消无条件新建36个Presentation训练的要求。

总训练预算在旧记录存在冲突，本版**重新明确选择**上述16,384/65,536，未声称它们本来就是唯一冻结旧值。smoke每4096评价/保存，主训练每8192评价/保存；rollout1024，因此16/64个PPO updates上限。smoke最早8个updates后满足可用性可停；主对比预算完成或明确数值/安全错误才停止，不因Full先赢单独提前停。

每family预算还设$T_{cap}=N_{cap}d_f^{ref}$，$d_f^{ref}$为0A参考执行中实际每技能时长中位数。所有同family方法共用Ncap/Tcap；先达到一个cap即停，最后不足buffer用有效mask处理。实际任务秒数、技能数、control ticks、wall time分别保存。主学习曲线在共同实际时间区间计算raw AUC，并附skill-transition横轴；不得向未运行尾部外推。
