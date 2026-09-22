# 14. Contract–Prior Interaction Contrast

## 14.1 四视图与同一个表示空间

在固定实际观测、目标、节点索引、关系R和编码器参数下，记

$$
Z_{00}=E(F,\varnothing),\quad Z_{10}=E(\widetilde F_i,\varnothing),\quad
Z_{01}=E(F,R),\quad Z_{11}=E(\widetilde F_i,R).
$$

第一个下标是当前/名义后继，第二个是无/有软关系；K合同始终存在，空集合仅指没有VLM软边。

$$
D_i^K=Z_{10}-Z_{00},\qquad D_i^H=Z_{11}-Z_{01},
$$

$$
\boxed{D_i^P=Z_{11}-Z_{01}-Z_{10}+Z_{00}=D_i^H-D_i^K.}
$$

相减对象是目标对齐的表示，不是邻接矩阵、真值表或真实环境结果。该定义与v2.1逐项相同；正式名称用**representation interaction contrast**，实现继续沿用D_K/D_H/D_P。

## 14.2 对称解释

仅为推导定义$S_R(F)=E(F,R)-E(F,\varnothing)$，则

$$
\boxed{D_i^P=S_R(\widetilde F_i)-S_R(F).}
$$

因此可以读成“软关系怎样改变候选变化的编码响应”，也可以读成“候选名义变化怎样改变软关系的编码作用”。S不是新网络、训练目标或缓存，也不必另造prior signature品牌名。

这是共享网络计算的2×2有限交互对比，不是从观察数据识别处理效应的causal difference-in-differences；没有平行趋势假设，也没有将表示坐标解释为世界因果变量。事实patch本身不受R影响。

## 14.3 有意限制与不能过度宣称的性质

若$E(F,R)=A(F)+B(R)$，则$D_i^P=0$。消去的是对所比较变化可加分离的表示成分，不是所有拓扑静态边，也不是所有可能有用的行动先验。

在线性诊断$E(F,R)=M_Rx(F)+b_R$中，

$$D_i^P=(M_R-M_\varnothing)[x(\widetilde F_i)-x(F)],$$

边不变仍可改变变化消息的传播。SOFT_ORDER在数学上也可能非零，主schema删除它是因order-only职责不合，而非必被消去。

完整$(Z^K,D^K)$与$(Z^K,\widetilde Z^K)$可逆，但当前Actor只把Pool$(Z^K)$放进上下文，并继续压缩候选特征。因此不能据上述可逆恒等式声称整个网络对当前/后继信息无损；也不能因$D^H=D^K+D^P$而称A_DD与Full必然同信息。

目标局部没有路径时，全局非线性读出仍可能混合不同组件。非零DP不是关系有用性的证明；DP大小与动作概率变化也不是单调关系。
