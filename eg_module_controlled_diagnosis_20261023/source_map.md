# EG 受控诊断：源码、张量与干预位置

官方版本：`iSEE-Laboratory/EconomicGrasp@4119bdcd6bf5d3712a110f78ca87504dd359eec0`。权重 SHA-256 见 protocol.json。旧诊断归档版本：`rollinpioneer/graph@0b3df6b`。本轮不修改上述源码或权重；探针位于独立目录。

|入口|固定事实与本轮用法|
|---|---|
|`models/economicgrasp.py::economicgrasp.forward`|选择 1024 个查询及各自方向；三种模式标志均为 false，BN/dropout 固定，但保留 autograd。训练路径采用标签旋转，推理路径采用当前方向的标准基底。|
|`Cylinder_Grouping_Global_Interaction.forward`|支持点、查询点均为已选查询；每点 16 邻居，分组输入含 3 维局部坐标及 512 维特征。MLP 后连接查询的世界坐标，邻域 attention 后经 MLP 与 max pool 得到 256 维 F。|
|`Grasp_Head_Local_Interaction.forward`|四个 256→64 特征层；token 顺序为 angle、depth、width、score。单层四-token attention 后分别输出 13、5、1、6 通道。|
|`MultiHeadAttn.attention`|行接收、列提供信息。H1 修改第 3 行前 3 列；H2 修改前 3 行第 3 列。半强度采用 log(0.5) 偏置，关闭采用 -1e9。H3 仅保留对角。|
|`AttentionModule.forward`|原式为 LN(v+MSA)。H4 与 N 分别在头部或邻域设为 LN(v)，保留 LayerNorm；不替换成 Identity。|
|`models/loss_economicgrasp.py`|ADWQ 共享有效掩码；零质量仍参与姿态损失。官方加权系数 A/D/Q=1、W=10、view=100、graspness=10、objectness=1。各项梯度分别测量，并与统一总损失核对。|
|`process_grasp_labels`|最近点距离严格小于 5 mm，并要求方向匹配。变换方向映射有重复目的索引写入，可能改变重放掩码。本轮逐项核实歧义来源后保留历史 ADWQ 标签及掩码；view-graspness 标签由当前同点匹配计算。|
|`pred_decode`|角度仅取前 12 类、深度仅取前 4 类，宽度为 clip(1.2×raw/10)。排序是六类原始 logits 的加权和，不能当成概率。|
|E1 的局部基底干预|将参考物理夹爪轴投影到当前接近轴的正交平面，按 π 对称确定零角；固定邻居及非坐标特征，只变换局部 xyz，并成对改变输出解码基底。纯物理旋转往返先检查。GT 辅助基底不代表可部署收益。|
|旧 `evaluate.py::Direct.score`|复用官方 GraspNet 碰撞、空夹和 Dex-Net force-closure 评价；逐姿态计算，不用最近离散标签。正式可行口径仍为无碰撞且 0<μ≤0.4。|
|旧 `postprocess`|原碰撞过滤→NMS(3 cm/30°)→每物体 Top-10→Top-50。仅 H1_off/H2_off 完整重放流程，其余为固定候选池机制分析。|

历史前向继续使用 TF32；梯度运算采用 FP32，以在不放宽 1e-4 容差的情况下核验加权梯度可加性。FP32 回传并非新增一次 FP32 预测。局部 12×4 网格固定查询位置、接近方向和预测宽度；其原 argmax 单元必须复现旧解析标签。

Q0 为冻结的官方代表标签；Q1 为同评价器重评参考姿态；Q2 为实际预测姿态质量。仅相同有效匹配、评价已知交集参与比较。正 μ 转换为 clip(1.1−μ,0,1)，再按官方 float32 `(*10/2).long()` 取六类；失败为 0，未知不作负例。

这些探针测量已发布权重在固定推理查询上的依赖和损失方向，并非历史训练 batch 的追溯，也不证明任何更新会改善泛化。
