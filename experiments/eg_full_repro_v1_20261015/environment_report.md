# 环境与工程检查

执行日期：2026-10-15（Asia/Taipei）。服务器 lab-xushijie3/gpu03，8×A100-SXM4-40GB、112 CPU、503GiB 内存；初始 GPU0/1 基本空闲，其余卡存在其他任务。使用当前空闲 GPU0 做有界工程检查；没有干预其他任务。硬件快照见 hardware_inventory.json。

独立 Conda 环境 runtime/conda；独立 CUDA 编译工具链 runtime/cuda121。Python 3.10.22、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.23.4、GCC11.4、nvcc12.1.105；详细冻结版本见 pip_freeze.txt、conda_explicit.txt、cuda_explicit.txt。系统 GCC14/CUDA13 未修改，驱动为610.57.04。

独立安装官方 MinkowskiEngine、PointNet2、KNN、GraspNetAPI；所有源码身份见 SOURCE_IDENTITIES.json。依赖导入和 pip check 通过，ROS 相关可选方法未安装，不用于 GraspNet 离线评价。

CUDA 工程检查通过：Torch 矩阵运算反向有限；KNN 与 torch.cdist 最近邻一致；FPS 与参考实现一致；采样及 Cylinder/Grouping 的前反向与索引参考一致；Minkowski 稀疏卷积前反向有限。具体输出见 receipts/operator_checks.json。

官方 Kinect Checkpoint 严格加载成功，使用固定种子的 20,000 点合成几何前向，输出 1024×17 个有限值抓取表示，宽度在官方范围。见 receipts/model_environment_check.json。该测试仅验证工程；不是数据集评价、真实数据 Smoke Test 或本次完整训练。官方权重身份见 OFFICIAL_CHECKPOINT.json，不能当作本次训练产物。

算子检查 Torch allocated 峰值约16.3MiB；合成模型前向约265.1MiB、reserved约378MiB。这些小型检查不能预测正式训练峰值。未连续采样驱动级 GPU 驻留，工程 GPU 小时与真实驱动峰值未知，不补造。正式训练和正式评价 GPU 小时均为0。
