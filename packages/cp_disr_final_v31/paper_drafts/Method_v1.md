# 38. Method v1

## 4.1 Overview

CP-DISR针对每个可执行候选计算名义结构变化，以固定语义关系对变化的编码响应调制合同策略。信息来源与表示学习分离，目标是技能选择而非关系修订。

## 4.2 Matched structural views

当前合同图$G^K$具有Action/Proposition两类节点、PRE_POS/PRE_NEG/ADD/DEL及其反向消息边。增强图$G^H$只增加合法软关系，不改当前事实、goal索引和节点集合。effect_fact_ref是已有后果入口的admission metadata，不是单效果神经门控。

## 4.3 Candidate nominal interventions

采用已知动作模型后继状态思想，[W03] 对候选a_i仅在只读事实副本中应用合同名义成功效果，形成$\widetilde G_i^K,\widetilde G_i^H$。两个后继使用同一patch，既不预测实际成功概率，也不取得真实未来；实际结果另由执行及Verifier更新。

## 4.4 Contract–prior interaction contrast

同一个E输出固定global/goal行Z。定义

$$D_i^K=\widetilde Z_i^K-Z^K,\quad D_i^H=\widetilde Z_i^H-Z^H,\quad D_i^P=D_i^H-D_i^K.$$

若记$S_R(F)=E(F,R)-E(F,\varnothing)$，则$D_i^P=S_R(\widetilde F_i)-S_R(F)$。这一2×2表示交互不是因果DiD估计。它消去对该名义变化可加分离的先验成分，而不意味着一切静态边无影响。相同编码器和索引提供共同坐标；不声明每一维有固定语义。

## 4.5 Prior-modulated contract policy

令$c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)]$，

$$u_i^K=\phi_K(c_i,D_i^K),\qquad b_i=w_K^\top u_i^K+b_K,$$

$$u_i^P=\phi_P(c_i,u_i^K,D_i^P)-\phi_P(c_i,u_i^K,0),$$

$$\Delta_i=\beta\tanh(w_P^\top u_i^P),\quad \ell_i=b_i+\Delta_i,\quad \pi=\operatorname{masked\_softmax}(\ell).$$

当前ZH不能绕过DP进入Full context。无先验时残差为0，同参数policy退化为合同reference；对合法动作$e^{-2\beta}\le\pi_i/\pi_{K,i}\le e^{2\beta}$。空patch保证该候选直接残差0，但其他候选仍可改变其softmax概率。其他性质和置换条件见Properties Box。

## 4.6 Outcome-grounded training

真实执行数据构造时长TD误差$\delta_t=r_t+\Gamma_t(1-terminated_t)V_{old}(X_{t+1})-V_{old}(X_t)$，GAE为$\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1}$。独立V以固定$V_{old}+\widehat A$回归，Auxiliary Candidate Q仅gather实际动作并回归$\operatorname{sg}(r_t+\Gamma_t(1-terminated_t)V_{old}(X_{t+1}))$。Q不是关系真假监督，candidate-set context仍可使梯度影响其他候选特征。

总loss为有效transition均值的clipped PPO＋$c_VL_V+\lambda_QL_Q-c_HH(\pi)$；Actor不额外乘episode-prefix折扣。它是当前采样分布下的经验surrogate，不宣称精确J梯度。[W01–W02] 旧targets和先验版本在PPO重算期间固定，四视图和GRU前缀按当前参数重算。VLM/合同/Verifier/低层控制器不接受梯度。

训练在episode起点按80%原始、20%整份空prior抽样并保持episode内固定，模式标签不送网络。不训练人工错边，不加入奖励塑形。H按训练suite的真实reference时长标定并固定，详见Setup。

## 4.7 Implementation and complexity

使用标准4层128维R-GCN、4bases、mean、ReLU与逐节点LayerNorm、dropout0。观测前端固定，后接可训练融合器与GRU。$N_t$候选通常需要$2+2N_t$个共享图视图，空prior复用为$1+N_t$；报告实测延迟/显存，不跨参数更新缓存可学习表示。具体runtime与完整层表从授权实际项目绑定。
