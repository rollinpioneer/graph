# 24. Figure 1 / Figure 2最终设计规范

## Figure 1 — 2×2核心总览

采用双栏横向版式：左侧20%为任务与合同/soft relation来源；中间50%为**行K/H、列current/nominal-after**的四个对齐图；右侧30%为DK合同评分与DP有界残差、mask、真实skill execution。底部用一条细反馈线从实际执行记录指向可训练网络，标“training on real executions”。VLM/cache标frozen，不必画一串红叉或工程模块。

同节点位置在四格一致；真实事实变化仅在名义列高亮；软关系仅在H行出现，两个H视图内容相同。一个shared E标记覆盖四视图；箭头说明先构造图再编码。图中可以使用蓝色代表合同、橙色代表soft，黑白线型同时区分；不能依赖颜色才可读。

场景图必须来自实际可授权输入，否则明确标schematic，不把示意图称为真实机器人实验。MOVE、effect与对象必须已有注册；不能为了好看新增OnLid/ClearPath事实。每候选重复2×2的批处理用“for each allowed candidate”标注，不画全部候选导致拥挤。

## Figure 2 — 交互代数与状态依赖

四格用同一E的Z00/Z10/Z01/Z11标注，横向箭头DK/DH，纵向S_R(F)/S_R(Ftilde)，中心完整等式DP=DH−DK=S_R(Ftilde)−S_R(F)。S仅临时记号，不画第五个网络。

右侧示意采用两个**确实成立**的对照：R为空时DP=0；整个候选编码patch为空时该候选DP/Delta=0。再放一个“不同candidate/state可产生不同响应”的定性可表达性面板，不画必然增大/减小箭头。

**删除Claude原图要求的两种暗示：**PICK(blue)看似无关所以DP必小；p已TRUE所以MOVE的DP必变小。这些不由当前A→A消息图保证。需要展示它们时，只能用真实checkpoint测得值和来源作为机制结果图，不把随机RGCN数字或未测绿色箭头混成方法性质。

最终出图应保留可编辑向量源；本轮交付的是上述draw specification，不伪造真实场景图、实验效果或latent维度语义。
