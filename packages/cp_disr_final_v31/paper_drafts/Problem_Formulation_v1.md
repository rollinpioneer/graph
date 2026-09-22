# 37. Problem Formulation v1

给定固定技能库$\mathcal U$、经过接口校验的合同$\mathcal K$、独立任务$\mathcal T$与每episode缓存的软关系$R_e$。第t次高层决策利用截至实际时刻$\tau_t$已到达的观测、三态事实及历史，形成snapshot $X_t$；允许策略依赖完整可得历史，不假设有限摘要必为Markov状态。候选数记$N_t$，执行mask由实际能力、合同、安全与参数绑定独立确定，R不改变mask。

技能$a_t$持续$d_t>0$秒。指定训练suite共同折扣半衰期H，$\Gamma_t=2^{-d_t/H}$；技能内实际事件奖励按发生偏移折扣为$r_t$。环境仅在首次独立确认全部目标时给事件奖励1；子目标与名义图不另给分。实际任务成功/失败、deadline耗尽和外部截断分别管理，外部有效末状态用于bootstrap。

任务评价准则为$J(\theta)=\mathbb E[\sum_t w_t r_t]$，$w_0=1,w_{t+1}=w_t\Gamma_t$。Method使用有效transition平均的经验PPO surrogate及duration-aware targets，不宣称其是J的无偏精确梯度。[W01] 关系语义是否成立与当前候选是否值得执行不同；本文学习后者的结构输入用途，而不输出逐边正确率。
