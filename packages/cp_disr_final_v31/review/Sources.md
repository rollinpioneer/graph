# 来源与核验等级

[U] 当前用户复核指令；[C] Claude审查稿；[V3] 原v3.0研究规范；[EP0] 原实验Agent Prompt。四者不混为外部研究证据。原始字节和hash见sources。

**[W01] Nota & Thomas, Is the Policy Gradient a Gradient?** — AAMAS 2020；FULL_TEXT_METHOD_CHECKED。

依据：外层折扣省略与起点折扣目标不等价；不能据此要求所有实现必须加权，也不能称去权仍无偏。

地址：`https://arxiv.org/html/1906.07073`

**[W02] Stable-Baselines3 PPO source** — version 2.7.0；OFFICIAL_SOURCE_CHECKED。

依据：clipped surrogate使用mean而无episode-prefix权重；不据此声称它原生支持本项目GRU/动态候选/SMDP。

地址：`https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/ppo/ppo.html`

**[W03] Ståhlberg, Bonet & Geffner, Learning General Policies with Policy Gradient Methods** — KR 2023；FULL_TEXT_METHOD_CHECKED。

依据：GNN state-transition classifier与策略梯度/actor-critic；arXiv2512.19366是后续deposit，不把编号年份当会议年份。

地址：`https://proceedings.kr.org/2023/63/kr2023-0063-stahlberg-et-al.pdf`

**[W04] Counterfactual Quotient Models: Learning What Actions Change, Not What the World Does** — arXiv 2608.22092, 2026；FULL_TEXT_METHOD_CHECKED。

依据：行动间中心化的未来量/同步反事实rollout目标；不是本项目prior×nominal编码2×2；其定理不移植。

地址：`https://arxiv.org/html/2608.22092`

**[W05] Yan et al., Efficient Reinforcement Learning with Large Language Model Priors** — ICLR 2025; arXiv 2410.07927；FULL_TEXT_METHOD_CHECKED。

依据：状态条件化动作先验p(a|s)参与KL/采样/Q；不能称为状态不变static prior。

地址：`https://arxiv.org/html/2410.07927`

**[W06] Guiding Reinforcement Learning Using Uncertainty-Aware Large Language Models** — arXiv 2411.14457, 2024；FULL_TEXT_METHOD_CHECKED。

依据：MC dropout与动态熵调节guidance，并涉及LLM定制/微调；不是本项目固定图对照。

地址：`https://arxiv.org/html/2411.14457`

**[W07] Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents / LLM4Teach** — IJCAI 2024; arXiv 2311.13373；FULL_TEXT_METHOD_CHECKED。

依据：不完美teacher软动作指导/蒸馏与衰减；不作四视图结构交互。

地址：`https://arxiv.org/html/2311.13373`

**[W08] The Mechanism Matters: When Knowledge Graphs Help Reinforcement Learning** — arXiv 2607.19616, 2026；FULL_TEXT_METHOD_CHECKED。

依据：比较feature/mask/shaping及KG质量；有限实验或最优策略不变性不证明本项目坏prior无害。

地址：`https://arxiv.org/html/2607.19616`

**[W09] Leveraging Approximate Symbolic Models for Reinforcement Learning via Skill Diversity / ASGRL** — ICML 2022；FULL_TEXT_METHOD_CHECKED。

依据：近似符号模型、landmarks、技能多样性；不完美符号知识+RL是已有设定。

地址：`https://proceedings.mlr.press/v162/guan22c/guan22c.pdf`

**[W10] Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning** — arXiv 2511.20993, 2025；FULL_TEXT_METHOD_CHECKED。

依据：子目标图与critique/refinement路径，与固定源关系使用不同。

地址：`https://arxiv.org/html/2511.20993`

**[W11] Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning** — NeurIPS 2023; arXiv 2305.14909；FULL_TEXT_METHOD_CHECKED。

依据：显式规划模型构造、验证/修正、规划；非本项目软关系残差。

地址：`https://arxiv.org/html/2305.14909`

**[W12] InterPreT: Interactive Predicate Learning from Language Feedback for Generalizable Task Planning** — RSS 2024; arXiv 2405.19758；FULL_TEXT_METHOD_CHECKED。

依据：根据交互语言反馈学习谓词与动作模型；不是冻结关系决策利用。

地址：`https://arxiv.org/html/2405.19758`

**[W13] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Robot Task Planning** — arXiv 2507.21545, 2025；FULL_TEXT_METHOD_CHECKED。

依据：从示范学习/构造可复用PDDL domain，与soft message edges不同。

地址：`https://arxiv.org/html/2507.21545`

**[W14] Vision-Language Models Provide Promptable Representations for Reinforcement Learning / PR2L** — arXiv 2402.02651, 2024；FULL_TEXT_METHOD_CHECKED。

依据：本轮独立补充；VLM提供语义表示，由RL学其用途，因此非动作指令式VLM+RL并非新领域。

地址：`https://arxiv.org/html/2402.02651`

**[W15] Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** — ICML 2019；FULL_TEXT_METHOD_CHECKED。

依据：SAB/MAB/PMA集合注意力与置换性质；本项目仅使用标准两层SAB+candidate query适配。

地址：`https://proceedings.mlr.press/v97/lee19d/lee19d.pdf`

**[W16] High-Dimensional Continuous Control Using Generalized Advantage Estimation** — ICLR 2016; arXiv 1506.02438；FULL_TEXT_METHOD_CHECKED。

依据：GAE/TD(lambda)偏差方差取舍；one-step并非普遍偏差最小。

地址：`https://arxiv.org/html/1506.02438`

**[W17] Reinforcement Learning with Unsupervised Auxiliary Tasks / UNREAL** — ICLR 2017; arXiv 1611.05397；FULL_TEXT_METHOD_CHECKED。

依据：共享表示的辅助任务是已有原则；并非声称本Q严格复现UNREAL。

地址：`https://arxiv.org/html/1611.05397`

**[W18] Residual Reinforcement Learning for Robot Control** — ICRA 2019; arXiv 1812.03201；FULL_TEXT_METHOD_CHECKED。

依据：本轮独立补充；既有控制+学习残差，区别在本项目logit界与交互输入。

地址：`https://arxiv.org/html/1812.03201`

**[W19] Modeling Relational Data with Graph Convolutional Networks** — ESWC 2018; arXiv 1703.06103；FULL_TEXT_METHOD_CHECKED。

依据：关系权重共享、basis与aggregation；不是对当前task门控的证明。

地址：`https://arxiv.org/html/1703.06103`

**[W20] Phasic Policy Gradient** — ICML 2021；ABSTRACT_OR_OFFICIAL_PAGE。

依据：辅助阶段与特征共享干扰的背景；未据摘要细节证明当前lambda=0.1无风险。

地址：`https://proceedings.mlr.press/v139/cobbe21a.html`

**[W21] Time Limits in Reinforcement Learning** — ICML 2018；ABSTRACT_OR_OFFICIAL_PAGE。

依据：终止与时间限制处理背景；本项目具体flag定义由规范负责。

地址：`https://proceedings.mlr.press/v80/pardo18a.html`

**[W22] Planning with Learned Object Importance in Large Problem Instances using GNNs / PLOI** — AAAI 2021；ABSTRACT_OR_OFFICIAL_PAGE。

依据：对象重要性筛选，不是直接候选后继策略的同义工作。

地址：`https://ojs.aaai.org/index.php/AAAI/article/view/17421`

**[W23] LLM+P: Empowering Large Language Models with Optimal Planning Proficiency** — arXiv 2304.11477, 2023；ABSTRACT_OR_OFFICIAL_PAGE。

依据：自然语言到规划问题/外部planner；不把全部LLM-PDDL都概括为学习新的动作模型。

地址：`https://arxiv.org/abs/2304.11477`

**[W24] Graph Learning for Planning: The Story Thus Far and Open Challenges** — arXiv 2412.02136, 2024；ABSTRACT_OR_OFFICIAL_PAGE。

依据：只作综述导航，不以综述取代原创论文核验。

地址：`https://arxiv.org/abs/2412.02136`

**[W25] Afterstate RL for Continuous Control（Claude给出的条目）** — NOT_INDEPENDENTLY_CONFIRMED；UNCONFIRMED_ACCESS_BLOCKED。

依据：访问返回浏览器验证页，本轮未独立确认该题目/作者/正文，不作为已确认核心依据。

地址：`https://openreview.net/forum?id=XO944P8prc`
