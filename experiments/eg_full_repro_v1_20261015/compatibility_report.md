# 兼容性修正与异常

1. 使用独立 Conda 与 CUDA12.1 工具链，避免修改系统 CUDA13/GCC14。CUDA 标签通道曾被系统镜像重写到不存在的地址，改用官方完整通道 URL。首轮编译缺 curand.h，补齐独立工具链 libcurand-dev/NVTX 后完成编译。
2. MinkowskiEngine 按官方 README 引用的 CUDA12 指南补充四个文件的 Thrust 头文件；补丁见 patches/minkowski_cuda12_headers.patch，未修改算子公式。
3. pointnet2_utils 存在裸导入 pytorch_utils；environment.sh 增加精确模块路径。PointNet2/KNN 构建轮子后安装，训练和算子源码未因安装而修改。
4. EconomicGrasp 补丁见 patches/economicgrasp_compatibility.patch：尊重外部 GPU 映射；Economic 仅100个训练场景；graspness 遇非有限/零分母明确报错；固定种子20261015及按 Epoch 派生 NumPy 种子，保存/恢复 RNG；增加逐批/逐轮诊断和有限损失检查。保留10 Epoch、batch4、Adam、LR0.001、官方Cosine、结构、监督阈值和损失。源码语法及补丁一致性已检查；训练路径尚需真实数据冒烟。固定种子不保证 CUDA 算子逐位确定。
5. 依赖版本约束固定 NumPy1.23.4、SciPy1.10.1、protobuf3.20.3、matplotlib3.7.5、OpenCV4.8.1.78 等以兼容官方旧依赖，见 constraints.txt。
6. 工程测试最初错误假定 pred_decode 评分在0–1；核查发现官方 conv_score 输出裸 logits，pred_decode 直接加权求和，未归一化。仅修正测试的额外假设，保留原评分及排序。失败脚本和回执已保存。合成输入评分约[-8.66,3.83]不是实际数据错误率，也不证明性能瓶颈。
7. 初次 Torch 检查的命令引号问题已修正并通过，原失败回执保留。torch.cross 的废弃提示保持记录，本版本尚可运行，不擅自改变方向计算。

原始上游代码（排除文档媒体）在 upstream_source/，286文件、约2.2MB，逐文件散列见 UPSTREAM_SOURCE_MANIFEST.json；完整原仓库仍留服务器。所有实质补丁可追踪，不修改其他项目。
