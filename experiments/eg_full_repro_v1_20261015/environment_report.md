# 环境与工程检查

执行日期：2026-10-15（Asia/Taipei）。服务器 lab-xushijie3/gpu03，8×A100-SXM4-40GB、112 CPU、503GiB 内存；初始 GPU0/1 基本空闲，其余卡存在其他任务。使用当前空闲 GPU0 做有界工程检查；没有干预其他任务。硬件快照见 hardware_inventory.json。

独立 Conda 环境 runtime/conda；独立 CUDA 编译工具链 runtime/cuda121。Python 3.10.22、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.23.4、GCC11.4、nvcc12.1.105；详细冻结版本见 pip_freeze.txt、conda_explicit.txt、cuda_explicit.txt。系统 GCC14/CUDA13 未修改，驱动为610.57.04。

独立安装官方 MinkowskiEngine、PointNet2、KNN、GraspNetAPI；所有源码身份见 SOURCE_IDENTITIES.json。依赖导入和 pip check 通过，ROS 相关可选方法未安装，不用于 GraspNet 离线评价。

CUDA 工程检查通过：Torch 矩阵运算反向有限；KNN 与 torch.cdist 最近邻一致；FPS 与参考实现一致；采样及 Cylinder/Grouping 的前反向与索引参考一致；Minkowski 稀疏卷积前反向有限。具体输出见 receipts/operator_checks.json。

官方 Kinect Checkpoint 严格加载成功，使用固定种子的 20,000 点合成几何前向，输出 1024×17 个有限值抓取表示，宽度在官方范围。见 receipts/model_environment_check.json。该测试仅验证工程；不是数据集评价、真实数据 Smoke Test 或本次完整训练。官方权重身份见 OFFICIAL_CHECKPOINT.json，不能当作本次训练产物。

算子检查 Torch allocated 峰值约16.3MiB；合成模型前向约265.1MiB、reserved约378MiB。这些小型检查不能预测正式训练峰值。未连续采样驱动级 GPU 驻留，工程 GPU 小时与真实驱动峰值未知，不补造。该句记录前期工程时点；现在正式训练已启动，其耗时和资源按实际回执汇总，正式评价尚未执行。

## 本轮准备适配与共享授权

GPU0/1在本卡预处理启动后出现其他任务，原先暂停后改用已通过对照的CPU预处理，保留7,820帧标签；旧进程的-15是本卡有意取消，并非算法失败。随后用户明确允许共享GPU0、1，详见GPU_SHARING_POLICY.json。资源采样保存共享进程显存与GPU总利用率；其他任务未被终止或修改。CPU适配不进入训练进程，不改变网络、损失或主要配方。

正式训练遇到进程NOFILE软上限1024导致Tensor IPC失败。仅在本卡作业父子进程将软上限升至65536（硬上限524288），不修改系统或其他进程；同种子同配方恢复见TRAINING_NOFILE_RECOVERY.json。
