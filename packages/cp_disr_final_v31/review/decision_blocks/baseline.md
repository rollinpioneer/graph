# 24. Final Baseline Matrix

| ID | 最终定义 | prior | 名义后继 | 默认位置 |
|---|---|---|---|---|
| B0 | 两层SAB的非显式图集合策略；完整原始合同/事实/目标/候选字段 | 无 | 无 | Tier1主阶梯 |
| B1-K（表内仍可简写B1） | 同B2上下文和head，$u_i^K=\phi_K(c_i,0)$ | 无 | 不计算 | Tier1主阶梯 |
| B2 | 同一$\phi_K(c_i,D_i^K)$，独立训练 | 始终空 | 合同后继 | Tier1主阶梯 |
| Full | 现有合同干预＋纯$D_i^P$＋零锚有界残差 | episode80/20 | 四视图 | Tier1主方法 |
| A_CAT | Full合同分支不变，prior读四视图拼接及合同重复anchor | episode80/20 | 同四视图 | 核心必跑，但先1seed |
| A_Q | Full仅置$\lambda_Q=0$ | episode80/20 | 同四视图 | Tier2训练机制 |
| A_B | Full仅移除tanh，保留$\beta$尺度和zero anchor | episode80/20 | 同四视图 | Tier3，仅有界收益claim需要 |
| B1-H | 旧B1：只读当前GK/GH的静态图策略 | episode80/20 | 无 | 可选，不默认跑 |
| A_DD | 旧非空DH、空prior显式0消融 | episode80/20 | 同四视图 | 可选，被A_CAT替代必跑位置 |

**B1的精确匹配。**$c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)]$与B2一致；B1将DK输入slot置零，而不把ZK或额外action-local embedding塞入该slot。这样B1→B2只增加已定义的候选变化特征及其计算。B1不是宣称最强的任意current-graph网络，而是匹配的no-intervention对照。换一种更丰富的current-graph readout是另一个架构实验，不默认加入。

**归因等级。**B1→B2是该实现下的受控干预信息对照；B0→B1是非显式图attention与图编码的系统比较，不声称仅“是否有图”一个自由度。B2→Full检验先验交互路径与其训练协议的整体增量；Full→A_CAT检验纯交互限制与四视图融合，不是同一输入信息。共同数据预算不意味着相同轨迹；相同结构不等于所有参数完全一样。

除A_Q，全部主配置保持匹配的实际动作Q监督、同reward/感知/合同/mask/时钟与训练预算。所有使用prior的方法共享相同cache和80/20协议。报告有效trainable参数、前向视图数、延迟和内存；不使用死参数填充冒充等容量，不给baseline额外oracle或剥夺原始合同字段。

## 24.1 B0：最终两层Set Transformer实施合同

输入为当前事实（含T/F/U与goal sign）、签名目标、全体grounded技能及其原始PRE_POS/PRE_NEG/ADD/DEL/UNKNOWN文字项记录、当前候选角色和同源实际观测。每条合同记录包含owning action、predicate、参数的有序角色/类型及一致对象绑定信息；不得只保留效果类别而丢掉“谁影响谁”。任意显示ID不当作可学习语义标签；对象特征来自所有方法共同可见的对象表/观测。

所有token通过共享字段投影进入$d=128$。使用两个标准SAB：multi-head self-attention，4 heads，逐token FFN128→256→128、ReLU、残差、LayerNorm，dropout0，无集合位置编码。候选query对全部编码token做一层cross-attention，输出128，再与共同obs/GRU及候选上下文用MLP256→128融合。V/Q/Actor各自head与其他方法保持相同输出角色。SAB/MAB属于现有Set Transformer组件。[W15]

attention只使用padding mask，不输入graph adjacency、PRE/ADD稀疏attention mask、最短路、层级答案或名义赋值后的事实；不调用NOMINAL_PATCH。token包含原始合同关联，因此它可以通过attention学习关系；“非图”指无显式graph message-passing拓扑，不是禁止一切关系计算。ADD/DEL原字段也不等于已经应用当前T/F/U、条件效果和全局frame后的名义后继。

实现前冻结宽度/层数，不按Full领先与否搜索B0容量；参数差异透明报告。若已有更强且信息匹配的标准attention基线，可在未跑任何主结果前替代一次，但不能增加另一项必跑。第一版采用上述唯一实现。
