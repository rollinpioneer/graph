# PC-P123-03 实际结果

执行跨越2026-10-11/12（Asia/Taipei），沿用[最新实验卡分支](https://github.com/rollinpioneer/graph/tree/codex/pc-p123-03-20261011-card)。原卡设计状态被用户执行请求覆盖，执行不另建分支。完成24个正式完整回合和5个固定发现集补测，均200步；另有6x10及2x5步资源/读出自检。

## 结果

| 项目 | 条件 | 平均达标比例 | 全部达标 | 落桌 | 无进展 | 平均XY距离m |
|---|---|---:|---:|---:|---:|---:|
| P1 | 6_sparse | 0.7292 | 1/8 | 1/8 | 6/8 | 0.03994 |
| P1 | 6_crowded | 0.7292 | 1/8 | 1/8 | 6/8 | 0.05503 |
| P3 | B0 | 0.7083 | 0/4 | 1/4 | 3/4 | 0.03994 |
| P3 | B1 | 0.4167 | 0/4 | 2/4 | 2/4 | 0.09183 |

P1稀疏减拥挤平均主指标差 0.0000，8对探索性bootstrap95%区间 [-0.125, 0.12500000000000006]，两个块平均均接近零、逐对方向混合。旧稀疏优势未稳定复现，不能证明布局无影响。此次固定6C，不能估计数量的因果效应。

P3 B1减B0平均主指标差 -0.2917，4对探索性区间 [-0.5833333333333333, 0.0]。须同时看落桌与任务推进；小样本未见损害不代表无损害。同种子不代表恢复同状态，不能宣布数量差距已缩小。所有失败、落桌、成功回合进入主分析。

本轮普通动作减半未降低落桌且降低任务推进，当前固定规则未通过双重判定；不外推为所有限速方法均无效。实际配对目标误差最高1.115mm，每个场景均通过预登记的请求坐标误差<1mm门；精确同状态比较不成立。另见pairing_actual_errors.json。

## 解释与限制

新场景保持旧密度区间与逐物体8cm平移，以新种子生成环/网格布局；逐物体桌边/可达性几何代理误差<=0.030m。旧场景排除，样本在结果不可见的成本试运行后冻结。只适用于这一受控场景分布。P1-B接近/切换仅为观察标签，DLP粒子不是物体身份，未确证遗漏或顺序错误。

原四个无进展窗口控制目标、关节与末端均有响应，动作没有±1限幅。官方缩放和实际桌面限位重算目标与日志一致，新12个窗口也一致（最大误差约5.7e-8m）。因此接口错误证据弱；无进展发生在末端响应后，接触是否有效、是否阻挡、推动方向是否合适仍未知。

五场固定补测由四个旧无进展初态和一个旧落桌初态组成，各一次完整重跑；四个旧无进展均未复现，另出现三次落桌。五槽批量改变策略随机流，不能冒充原轨迹或反复挑选复现。P2-B没有足够定位依据而科学停止，不表示干预无效。

历史/上轮全部10个首次落桌事件去重并按轮次、物体数和可用100/200/400步窗口分别登记，不合并发生率。补测三例在首次包围盒跨桌边之前的峰值物体速度约0.476、0.758、0.395m/s；没有使用之后自由落体速度推断成因。直接、次级接触与残余动量路径仍保留未知。

P3在看干预效果前冻结普通动作幅度减半，两个条件各四个新匹配场景；采样槽位预先随机分配，均200模型调用/控制步，无真值触发、额外时间或P2叠加。命令阻尼不是保证实测物理速度的限速器。

最初自检只检查有限值，后来发现官方CC_NEVER导致接触力全零；P1和补测接触力均不可用，零不等于无接触。P3只开启CC_LAST_SUBSTEP读出，非零信号已验证，求解器、dt、两内部子步、重力及接触物理参数一致。每0.01667模拟秒仅最后内部子步的净力，不能识别接触对或覆盖全部瞬态。

## 预算与停止

逐GPU驻留区间并集采样估计 0.7747 GPU小时；停止用逐进程全墙钟x驻留卡数保守上界（含2秒/作业余量）1.0052 GPU小时，可能重复计算共享渲染卡并含CPU加载。合计任务显存峰值5974MiB，首末GPU任务区间墙钟1068.2秒。准备/分析时间见cpu_timing.txt；未单独计时的早期纯CPU用量不伪造。

初次资源试运行因场景文件未生成技术失败，计入预算。几何预检及超32位seed在正式冻结前纠正，未看策略结果调参。24个正式回合无排除或重跑。拥挤GPU共享负载较高，监控接管把稀疏未用预约转给拥挤，P1总2340GPU秒上限不变，worker未重启，源/场景/样本不变，回执已保存。

- **P1_A**：completed: 8 new pairs, 16 full episodes; primary layout advantage not stably replicated
- **P1_B**：descriptive discovery completed; sparse failure repeated in >=2 scenes, proximity only, no sequence error established
- **P2_A**：original four windows plus one rerun each audited; transforms match official OSC; old stalls not reproduced; contact cause unresolved
- **P2_B**：scientific stop / unexecuted: no sufficiently localized repeatable cause
- **P3_A**：10 original first-fall episodes and three rerun falls indexed; pre-edge velocity measured, contact counterpart pathway unknown
- **P3_B**：completed: uniform xyz amplitude x0.5 vs original, 4 independent scene pairs, full 200 steps
- **snapshot_repair**：unexecuted; complete-task alternative chosen
- **new_network**：unexecuted; no automatic development

当前最多保留“可验证的接近/有效接触与动作选择”作为一条后续诊断线索，尚不足以立项新网络。原始数组和完整日志仅留服务器，路径与散列见evidence_inventory.json。
