# 兼容性修正与异常

1. 使用独立 Conda 与 CUDA12.1 工具链，避免修改系统 CUDA13/GCC14。CUDA 标签通道曾被系统镜像重写到不存在的地址，改用官方完整通道 URL。首轮编译缺 curand.h，补齐独立工具链 libcurand-dev/NVTX 后完成编译。
2. MinkowskiEngine 按官方 README 引用的 CUDA12 指南补充四个文件的 Thrust 头文件；补丁见 patches/minkowski_cuda12_headers.patch，未修改算子公式。
3. pointnet2_utils 存在裸导入 pytorch_utils；environment.sh 增加精确模块路径。PointNet2/KNN 构建轮子后安装，训练和算子源码未因安装而修改。
4. EconomicGrasp 补丁见 patches/economicgrasp_compatibility.patch：尊重外部 GPU 映射；Economic 仅100个训练场景；graspness 遇非有限/零分母明确报错；固定种子20261015及按 Epoch 派生 NumPy 种子，保存/恢复 RNG；增加逐批/逐轮诊断和有限损失检查。保留10 Epoch、batch4、Adam、LR0.001、官方Cosine、结构、监督阈值和损失。源码语法及补丁一致性已检查；训练路径尚需真实数据冒烟。固定种子不保证 CUDA 算子逐位确定。
5. 依赖版本约束固定 NumPy1.23.4、SciPy1.10.1、protobuf3.20.3、matplotlib3.7.5、OpenCV4.8.1.78 等以兼容官方旧依赖，见 constraints.txt。
6. 工程测试最初错误假定 pred_decode 评分在0–1；核查发现官方 conv_score 输出裸 logits，pred_decode 直接加权求和，未归一化。仅修正测试的额外假设，保留原评分及排序。失败脚本和回执已保存。合成输入评分约[-8.66,3.83]不是实际数据错误率，也不证明性能瓶颈。
7. 初次 Torch 检查的命令引号问题已修正并通过，原失败回执保留。torch.cross 的废弃提示保持记录，本版本尚可运行，不擅自改变方向计算。

原始上游代码（排除文档媒体）在 upstream_source/，286文件、约2.2MB，逐文件散列见 UPSTREAM_SOURCE_MANIFEST.json；完整原仓库仍留服务器。所有实质补丁可追踪，不修改其他项目。


## 数据恢复后阶段的工程补丁

- Graspness加入scene范围/stride、frame范围、独立输出目录、可选场景缓存及带SHA回执的原子写入/恢复。缓存计算仅复用同场景、同对象序号和ID的原始点及官方摩擦/碰撞筛选求和，逐帧姿态、KNN10000点分块与min-max均保持原实现。正式全量前必须以scene0和99前2帧逐元素及NPY文件哈希一致放行；当前尚未执行该实数据比对。
- Economic仅显式100训练场景，增加互不重叠的分片、原子写入和带摘要恢复。所有0.4/0.6/0.1阈值、max-depth/max-angle、topk300、量化和字段表达式保持官方。
- 训练增加实际更新计数、禁止覆盖的原子checkpoint；原先已登记的种子修复、RNG保存恢复及分项日志保留。参数和模型/损失不变。固定种子提高本次可追踪性，不宣称GPU计算逐位确定，也没有多种子误差估计。
- 测试固定同一种子，proc由环境变量设为4，创建父目录，拒绝覆盖预测，记录有限性、抓取数及CUDA同步的forward+decode计时。原始pred_decode及评分加权没有softmax；raw score不限定0到1。AP计算仍直接调用官方API，TOP_K50和摩擦系数集合未改。
- 通过单独flag保留每帧实际采样点/voxel坐标/seg，供测试后有界诊断重放；模型输入与输出未改。推理网络计时不含DataLoader、输入/预测保存和评测，整个测试wall time另由resource receipt记录。
- 大文件写入检查24GiB磁盘余量。新资源执行器在作业启动时实查GPU占用并保留独立进程组RSS/GPU采样和原生退出码；采样峰值不等同连续精确峰值，Torch每epoch峰值另外保存。资源执行器真实CUDA检查PASS，后续实数据阶段尚未执行。

相对固定上游的完整补丁见 patches/economicgrasp_compatibility.patch；原实现源码快照仍在upstream_source。所有实数据测试必须按门执行，语法检查不替代完整Smoke Test。

## 完整数据准备至训练启动的实际适配

隔离CPU Graspness KNN以候选树和原float32 FMA处理边界，原4帧和跨12场景36帧与GPU标签数组/哈希一致，全部监督PASS；抽样对照不宣称全量双后端验证。GPU训练仍用原生CUDA。Economic目录竞态以预建目录和恢复失败分片处理。检查器错误要求每物体非空，5个物体场景组合经全原始标签证实官方筛选保留点为0后修正检查器，未修改标签或跳过场景/帧。

正式训练首次9更新因NOFILE1024失败，无checkpoint。仅将本卡进程软上限提升65536，同种子20261015和原配方恢复；失败日志保留，不改batch/优化器/损失/监督/模型。实质官方源码补丁仍为train_config登记身份。
