# 39. Experimental Setup v1

采用同一固定技能库、合同、控制器、观测与Verifier、执行mask及独立终端奖励。默认T_B多对象选择与T_C中间重定位，D0用于smoke；所有模板的真实simulator/robot、资产、camera、时限和safe execution permission在0A绑定，不把符号toy当机器人结果。

0A用固定reference执行校准各family典型完成时间，共同H取中位时长的最大值。0D用100–200个uniform masked episodes检查奖励暴露，低暴露时最多另50个contract-aware random诊断；5%不是硬阈值。所有这类数据不送PPO、BC或VLM few-shot。

B0使用两层Set Transformer；B1-K与B2同context但DKslot为0且不生成后继；B2做合同干预；Full增加DP先验路径。A_CAT使用同四视图行对齐拼接和合同重复锚点，作为纯交互限制的核心对照。A_Q在一个任务检验辅助loss，A_B仅当声明bounded经验增益时运行；旧A_DD/B1-H降为optional。

第一轮D0两个smoke加两family四方法单seed及TC单seed CAT，共11个新训练。配置正常后补seed1/2，累计29；再加TC三seed A_Q为32。smoke16,384/主训练65,536 skill cap，另以reference每技能中位时长定义真实时间cap，先达到者停止。所有失败交互记账，数据预算不能用未测安全5秒代换。

每checkpoint主评价20个dev，smoke10个；按三个dev窗口真实discounted return均值选择模型，test前冻结选择，最终每task/seed30个独立test。定量比较默认完整0/1/2三seed；精选Private View另标并保留Raw。报告success、discounted return、raw AUC、完成时间与分母；显示曲线可窗口3平滑，表和AUC不平滑。

相同checkpoint比较先验缺省、每边p=.5删除和1条合法targetswap；Irrelevant仅在已有独立合格组件时测试。Full与CAT配对评价，不分别重训。缺省模式训练见过，不写成完全未知OOD；语义targetswap不自动等于错误关系。具体数据、软件hash、总秒数与所有结果仍待实际运行。
