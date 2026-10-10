# FAST 与通用增量方法对应关系（预登记静态核查）

已读原始材料：[InkStream 论文 v1](https://arxiv.org/pdf/2309.11071)、[作者发布仓库](https://github.com/WuDan0399/InkStream/tree/release)、[incremental_aggregation_mean / incremental_inference](https://github.com/WuDan0399/InkStream/blob/release/inkstream.py)、[SAGE 适配](https://github.com/WuDan0399/InkStream/blob/release/inkstream_sage.py)。2023论文与2025发布仓库的身份分开；本轮不安装作者修改版 PyG，不运行作者基准。

| 当前运算 | 对应能力 | 兼容判定 |
|---|---|---|
| 固定权重下的 sum/mean 更新 | 作者 mean 累加器与事件传播 | 直接支持数学机制 |
| 命题 TRUE/FALSE 特征变化，图边固定 | 作者公开入口接受边插入/删除；需特征变化事件入口 | 需要适配 |
| 每关系各自入度、basis 合成权重 | 当前 RGCN 有分关系归一化；发布示例为 GCN/SAGE/GIN | 需要适配 |
| root、bias、ReLU、逐节点 LayerNorm | 保留 root 差量与完整非线性；不使用阈值跳过 | 需要适配 |
| 动作／命题全局 mean、phi、目标注意力与 rho | 作者节点表示更新不是完整规划评分器 | 需要适配；本轮完整重算读出 |
| 兄弟状态共用同一个父缓存 | 通用增量内核应用于同一基点多个更新 | 若无不同机制，I 与 S 合并 |
| 本项目 GPU 完整成本与 GBFS 优势 | 没有取得直接匹配的官方实验 | 无法核实；须由本卡门控决定 |

预登记实现是上述通用差量规则的有限适配，不是官方 InkStream 运行复现。“缓存未变表示”和“局部 mean 更新”不构成新颖性证据。尚未证明作者原实现以合理代价完成本项目全部评分与搜索机制，也不能因此宣称候选更优。