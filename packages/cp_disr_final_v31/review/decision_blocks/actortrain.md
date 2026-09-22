## 17.2 PPO目标：有效transition均值，不加episode前缀权重

$$
\varrho_t(\theta)=\frac{\pi_\theta(a_t\mid X_t,h_{t-1}^{\theta})}{\pi_{old}(a_t\mid X_t,h_{t-1}^{old})},
$$

令$v_t\in\{0,1\}$标记非padding的有效transition，

$$
\boxed{L_{clip}=\frac{1}{\sum_t v_t}\sum_t v_t
\min\{\varrho_t\widehat A_t,
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)\widehat A_t\}.}
$$

$$
\mathcal L_{total}=-L_{clip}+c_V\mathcal L_V+\lambda_Q\mathcal L_Q
-c_H\mathbb E_{valid}[\mathcal H(\pi_\theta)].
$$

**Final Actor Weighting Rule：**`actor_episode_discount_weight=false`；Actor/V/Q/entropy均按有效transition平均。$w_t=\prod_{j<t}\Gamma_j$保留为实际起点折扣回报核算和诊断字段，不进入Actor loss，不用它重采样，不在别处隐式补乘。GAE的$\lambda$仍按技能决策，$\Gamma$按实际时间；不是把时长折扣删除。

我们采用当前on-policy采样分布上的经验PPO surrogate，并使用时长折扣的优势与价值target。它**不是**episode起点折扣目标$J$的无偏精确梯度，也不能简单重解释成无折扣J的精确梯度。[W01，W02] 每批冻结的surrogate本身可微；Nota–Thomas指出的是省略外层折扣的一般期望更新场通常不是某个固定总体目标的梯度，不是说代码里没有loss。

选择有效样本均值是减少额外时间位置重加权的工程决定。原来prefix加权不属于P0错误；归一化batch中的共同尺度可抵消，V/Q也未因此失去监督，所以不能声称300秒时后段“完全学不到”。统一采用此规则后，旧weighted checkpoint只能标历史profile，不与新profile混作同设置。

其余默认保留：Adam单参数owner，lr3e-4、clip0.2、GAE0.95、cV0.5、cQ0.1、entropy0.01、gradclip0.5、4 epochs、1024转移/rollout、seq16、minibatch64。Actor advantage在有效batch标准化，不标准化V/Q标签；不加入新reward、teacher、pretraining或课程。
