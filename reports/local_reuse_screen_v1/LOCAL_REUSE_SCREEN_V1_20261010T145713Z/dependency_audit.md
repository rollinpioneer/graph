# 实际前向依赖与增量更新规则

冻结实现：FastValue.pack → encode_nodes（4×RGCN mean/root/bias/ReLU/LayerNorm）→ SerialModel._features（动作全局mean、命题全局mean、目标H、目标真值、目标符号）→ phi → GeneralGoalAttention.pair_features/Attention → sum → rho。记录到 manifest 的源文件完整哈希；历史 FAST 511b117 与科研基点 cc7d79d 的这些核心文件无差异，服务器当前 HEAD 的归档提交也未改变它们。

图边及关系固定，命题 TRUE/FALSE 一热值变化。初始影响集是实际翻转命题对应的节点；额外动态输入只有目标真值及当前为真的二元命题关系特征，它们在目标读出中完整重算。静态事实、对象类型、目标标记在任务内不变。

设关系 r 在目标节点 v 的入度为 d_r(v)，父状态第 l 层的非线性前量为 P_l(v)。改变前一层节点表示的差量为 ΔH。精确实数更新是：

P'_l(v) = P_l(v) + ΔH(v)W_root + Σ_r mean_{u→v,r}(ΔH(u))W_r。

空关系入度按原 PyG mean 的零聚合处理；有边关系按各自固定入度归一化。bias 已在父 P 中，不重复增加。然后对受影响输出节点完整执行 ReLU 和逐节点 128 维 LayerNorm；其他行复用父表示。不通过小数值阈值删去影响。下一层集合是本层输入影响集与沿实际有向边的目标集并集，root 项确保原节点仍受影响。

全局汇总不能当作静态。本轮原型沿用完整 _features、phi、pair_features、DENSE Attention 和 rho，全量重算全局mean与目标读出，以保持评分函数和计入真实成本。没有缓存整个 OPEN；父缓存只在请求组存活，交错流只用登记的容量2 LRU。

affected_work.csv 的节点、边、root 和关系目标计数基于上述运算依赖，是保守依赖集，不是实际张量值变化率。group_work.csv 的 MAC 估计包含一次父全量重算，尚不包含访存、索引、缓存及读出成本。不得把该估计写成实际加速。缓存9×N×128×4字节是 H0、四层 H、四层 P 的核心张量估计，正式峰值还必须包括静态边、模型及瞬时子状态张量。

通用增量对照 I 与分支共享 S 在目前规则中共用同一父表示、同一差量内核，没有辨识出额外独立机制，预登记合并为 I/S。后续若门控允许性能比较，只比较 F 与 I/S；未用第三套系统制造弱基线。
