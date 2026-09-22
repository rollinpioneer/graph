## 15.5 Properties Box：适用条件与精确结论

共同条件：固定参数、相同真实观测与历史前缀、相同候选与执行mask、相同goal/node索引、相同确定性编码；只改变明确比较的输入。softmax只在至少一个合法候选上定义。

**P1 Exact null.** 若$R=\varnothing$，复用合同编码：$D_i^P=u_i^P=\Delta_i=0$对全部候选成立，因而$\pi_\theta=\pi_K^\theta$。$\pi_K^\theta$是**同参数**关闭先验修正的策略，不是独立训练B2。

**P2 Empty nominal patch.** 某候选$i$的全部编码输入未变时，$D_i^K=D_i^H=D_i^P=\Delta_i=0$。这只使该候选无直接残差；其他候选的非零残差仍可经softmax分母改变它的概率。只有全部候选满足null时才得到整项policy equality。effect_fact_ref对应p不变不等于整个patch为空。

**P3 Additive separability.** 若对当前与名义输入有$E(F,R)=A(F)+B(R)$，则$D_i^P=0$。不要求线性E，也不推得静态edge一律无影响。

**P4 Bounded direct deviation.** $\Delta_i=\beta\tanh(w_P^\top u_i^P)$，$\beta\ge0$，合法动作有

$$
\frac{\pi_\theta(a_i\mid X)}{\pi_K^\theta(a_i\mid X)}
=\frac{e^{\Delta_i}}{\sum_j\pi_K^\theta(a_j\mid X)e^{\Delta_j}}
\in[e^{-2\beta},e^{2\beta}].
$$

因为分子、加权分母都在$[e^{-\beta},e^{\beta}]$。默认$\beta=0.5$时界为$[e^{-1},e]\approx[0.367879,2.718282]$。这是对合同reference的单步界，不是任意两个先验之间同样的界（后者只能直接给出$e^{\pm4\beta}$粗界），也不保证轨迹回报或argmax不变。mask为0的动作不计算0/0比值。

**P5 Candidate permutation equivariance.** 在逐候选共享网络、集合读出置换不变、候选和mask同步置换、无候选位置编码时，$\pi(P\mathcal A,Pm)=P\pi(\mathcal A,m)$。若先执行argmax，并列需按canonical ID解决；分布等变不自动使按数组首项打破并列的动作选择等变。对象重命名是另一项更强检查，不能由候选重排直接推出。

P1/P4/P5也适用于按本版定义的A_CAT；A_CAT不要求P2/P3成立，因为它允许静态表示成分。A_B保留null而移除P4。全部是代数/架构条件，不作为全局安全或有效性定理。
