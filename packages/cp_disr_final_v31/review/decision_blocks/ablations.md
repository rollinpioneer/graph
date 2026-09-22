# 25. Core Ablations

## 25.1 A_CAT：四视图融合替代纯交互输入

按goal/global对应行，沿**feature维**拼接，不能把goal行平铺成固定任务长度：

$$
C_i=[Z^K,\widetilde Z_i^K,Z^H,\widetilde Z_i^H],\qquad
C_i^0=[Z^K,\widetilde Z_i^K,Z^K,\widetilde Z_i^K].
$$

$$
\boxed{u_i^{P,CAT}=\phi_{CAT}(c_i,u_i^K,C_i)-\phi_{CAT}(c_i,u_i^K,C_i^0),}
$$

$$\Delta_i^{CAT}=\beta\tanh(w_P^\top u_i^{P,CAT}).$$

合同FK、c、E、名义patch、goal mask、shared candidate output、V/Q/Actor功能、优化参数和数据预算均与Full相同。CAT必须从头独立训练；“相同head”指下游架构/宽度，不表示两个独立run共享已经训练好的参数。

输入首投影4d→d，而Full为d→d；之后使用相同候选attention/MLP/head。若只此处增宽，CAT额外参数为$3d^2=49,152$（d128，不含相同bias）。实际层表不同则重新精确计数；不谎称严格等参数。Full是在共同输入上固定使用$[I,-I,-I,I]$对比的受限归纳偏置，CAT可学出这一映射，但有限训练不保证自动学会。

无prior时C=C0，残差严格0。空patch或可加分离的静态prior响应则**可以非零**，这是比较目的，不能额外用DP=0 gate把CAT变回Full。锚定到全零而非合同重复视图不是本消融定义。CAT不允许从effect_fact_ref额外读几何或语义标签。

本消融回答：明确丢弃可加分离静态成分是否值得？它不单独隔离“减法计算”与容量、完整信息保留的全部区别。若CAT在同预算主任务/后续评价中稳定更好，纯交互优越性不成立，应报告、缩小claim或在未来v2.2重新选择融合；不能为了保住Full挑选seed/删task。写作仍可继续，但结果结论须改变。

## 25.2 A_Q和A_B

A_Q仅将$\lambda_Q$设为0，保留初始化顺序/网络接口，不把Q输出送Actor。它是保留Q作为正式训练设计时最小必要的训练机制对照，默认只在TC用3seed。

A_B仅使用$\Delta_i=\beta w_P^\top u_i^P$，无tanh、无补回clamp或归一化。零锚定保留。若论文只声明代数上界而不声称其性能/鲁棒收益，A_B不列必跑；一旦主张bounded带来行为收益，就需要它。

A_DD仍可定义：R非空用DH，R为空用0；但不与A_CAT一起强制运行。旧B3、static-prior独立模块、Full-clean、effect-anchored图等不列当前默认任务。

## 25.3 对照复用

Full可复用先前完全匹配的task/split、code行为、runtime/软件、cache、seed、common hyperparameters、预算、时间H与checkpoint规则的run。复用写control_reuse_manifest，不复制成新训练。新参数profile改变后，旧seed0不能与新seed1/2凑成同profile三seed；受影响范围显式补跑并计成本。其他Stage不得自动扩实验。
