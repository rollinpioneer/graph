# 运行环境

lab-xushijie3/gpu03，A100-SXM4-40GB，112CPU、503GiB内存；复用本研究独立Conda（Python3.10.22、Torch2.5.1+cu121、torchvision0.20.1、NumPy1.23.4）、独立CUDA12.1/GCC11、ME、PointNet2、KNN和GraspNetAPI，系统环境未改。旧数据/环境实际根目录eg_full_repro_v1_20261015，新的代码、输出与回执根目录eg_sub25_v1_20261015。

当前GPU1单卡，用户此前授权共享GPU0/1；每次启动按实查可用显存和负载选择，不干扰其他任务，资源日志记录共享负载。NOFILE软上限65536，原配方seed20261015/10epoch/batch4，16000更新。未启用DDP或额外seed，GPU其他进程保持原状。

所有源码身份见train_config.json与economicgrasp_sub25.patch。补丁继承随机播种、完整指标/RNG/checkpoint和兼容性记录，新增显式冻结划分与空候选明确报错；未定义空集替代策略，正常样本算法未变。固定seed不承诺自定义CUDA/稀疏算子跨GPU位级相同。
