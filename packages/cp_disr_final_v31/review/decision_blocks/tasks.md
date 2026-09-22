## 29.1 Task Family最终选择与奖励暴露

默认核心family仅**T_B：Multi-Object Choice**和**T_C：Intermediate Relocation**。D0作多步smoke；T_A保留为合同重复/共享前置参考而非强制prior增量；T_D在确有组合评价需要时绑定；T_E成本较高，后置。模板不是已存在资产，平台/控制器/观测/goal/deadline/split仍MUST_BIND。

主任务应在合法参考轨迹中至少出现两个有≥2合法候选的决策点，并且有实际长期后果差异。仅candidate数量大或mask非空不证明有真正决策。合同保留所有真实必要启动/安全条件，不能故意删掉前置制造VLM优势；额外软关系限于当前抽象未完整建模的场景关联、代价与后续联系，而非隐藏的硬安全许可。

优先采用**错误通常可恢复，但消耗真实时间**的任务：增加探索时仍完成的机会，又由同一终端成功的时间折扣区分绕路与高效执行。它仍是稀疏奖励，不会自动给每次返工产生即时负奖励。少量错误即终止case可以保留为边界，不要求全部可恢复，不放宽安全阈值。

记录从实际改变的Fact到目标/候选的message-path长度；4层是否覆盖由具体图核对。PICK→Held→PLACE→Inside含3条边，从变化起点Held算才是2跳。局部覆盖不等于完整长期规划；不提前加candidate-local row或Graph Transformer。

包含prior有用、当前无用及自然误导case，但不按Full测试分数挑选cache或刻意给训练造错边。原始输出可能为空/错误，这些case照实保留。任务分层与test-independent设计先登记；相同任务下所有方法获得同源可观测线索。
