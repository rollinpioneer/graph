## 7.2 时间与奖励：共同折扣半衰期

**Final Duration Discount Rule：**以秒为唯一日志换算单位，指定半衰期$H>0$，

$$
\boxed{\Gamma_t=2^{-d_t/H},\qquad \bar\gamma_{1s}=2^{-1/H}.}
$$

技能内真实奖励事件发生在偏移$\epsilon_k$时，$r_t=\sum_k2^{-\epsilon_k/H}\rho_k^{env}$。控制tick先用实际tick秒数换算，不将高层技能次数冒充物理秒。事件奖励只在首次独立确认整项任务成功时为1，其他事件0；子目标、恢复、图变化与VLM判断均不另给分。

**H的确定算法而非无限调参：**Stage0A在每个正式训练family取得5条合法成功的固定reference执行，最多尝试10条，使用独立development案例和已授权固定技能；取实际完成时间中位数$T_f^{ref}$。该校准不向PPO buffer、BC或VLM few-shot提供轨迹。一个比较suite使用共同

$$H=\max_{f\in\mathcal F_{train}}T_f^{ref}.$$

因此每个family典型reference完成时的保留率至少为0.5。同family全部方法、seed与消融共用H；跨family训练和组合holdout也继承该suite H。只有一个family时H等于其reference中位数。reference无法执行时暴露真实接口/任务绑定问题，不猜H，也不阻塞Method写作。

0.5是可解释的时间偏好，不是理论最优阈值。Claude的$\bar\gamma^{T}\ge0.3$同样只是工程规则，不能从理论推出。运行开始前冻结H和来源hash，不根据Full成绩调gamma；新family需要新suite时登记版本，旧run不静默重算奖励。独立family使用不同H在各自匹配比较中不违反公平，但其discounted return不能直接跨family平均作为同一物理时间偏好，故本版默认共同H。

若某一任务独立标定为30/60/120/180/300秒，对应每秒gamma为0.97715997/0.98851402/0.99424042/0.99615659/0.99769218。该表是公式示例，不是实际已测的任务时长。
