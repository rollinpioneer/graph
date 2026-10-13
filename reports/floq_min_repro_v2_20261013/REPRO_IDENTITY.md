# FLOQ 复现身份与运行回执

状态：两条正式训练均正常完成 2M；分类 B — FUNCTIONAL_REPRODUCED。

官方源码 SHA：`3d60e638b42bbef018c56cb1199ff37c3470520d`；原 checkout 清洁，未修改算法。
源码：[CMU-AIRe/floq](https://github.com/CMU-AIRe/floq/tree/3d60e638b42bbef018c56cb1199ff37c3470520d)。
归档分支：[codex/floq-min-repro-v2-20261013](https://github.com/rollinpioneer/graph/tree/codex/floq-min-repro-v2-20261013)。
服务器 ROOT：`/home/xushijie3/work/floq_min_repro_v2_20261013`；最终归档 SHA 由 `archive_status.json` 的远端核验记录给出。

## 环境、数据和随机数

Python 3.8.19；JAX 0.4.13；jaxlib 0.4.12+cuda11.cudnn86；NumPy 1.22.4；Flax 0.7.2；Optax 0.1.3；OGBench 1.1.2；MuJoCo 3.1.6；dm-control 1.0.20；Gymnasium 1.0.0。完整包表见 `report/installed_environment.json`，与作者 YAML 的差异见 `report/dependency_differences.csv`。

隔离 uv 环境使用 pip CUDA 11.8/cuDNN 8.9.2.26 库及官方 CUDA wheel，未宣称与作者 conda 二进制构建逐字一致。矩阵精度保持官方 JAX 默认 None，x64=false，参数 float32。A100 40GB 上两个种子分别使用 GPU 3/6、各四个 CPU 核；共享卡的总利用率包含其他用户，未干预其进程。

命令任务为 `antmaze-large-navigate-singletask-v0`；OGBench 实际环境 spec 为 `antmaze-large-singletask-v0`。与显式 Task1 的初始 qpos/qvel/observation 一致性已在 P0 验证；见 `report/task_alias_check.json`。训练/验证数据分别 1M/100k transitions，observation 29 维、action 8 维，动作 [-1,1]，episode 上限 1000。Success 使用官方环境回合末 info 的 success，经官方 evaluate 聚合；return 为原版未折扣回合奖励总和。

原始 NPZ 来自官方 OGBench 下载并通过只读链接复用。以下数据文件在收尾再次核验 SHA256：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| antmaze-large-navigate-v0.npz | 242877817 | `2dbe946ac107ca5ac524f82f7a72efa951ceab0be915201d6b32e612d823f2e7` |
| antmaze-large-navigate-v0-val.npz | 24302303 | `e9a4126825faefb57c34c29f1ee423858dbc4b6f71d9a9846a8b8bdb908b44a1` |

种子 0/1 从头独立初始化，配置 SHA 相同，初始网络参数 SHA 不同；seed 0 与 pristine P1 初始化一致，未复用 P1 权重。见 `receipts/formal_config_validation.json`。各种子的初始 Python/NumPy/JAX 身份见 `formal/formal_seed*/initial_state.json`。

官方 main 在设置全局 seed 前初始化环境 RNG；保留该顺序并记录真实状态。原生起点未与 FQL/GFP 匹配。temperature=0 仍有 Gaussian action noise，不能称为确定性策略。每次评价的 policy key、环境 RNG 及逐回合初始观察哈希保留在服务器回执；2M 评价状态和 100 条逐回合轻量表另存 `final/`。

## 完整命令与参数

实际入口是工程包装器导入固定官方 main；仅切换 W&B offline、记录既有数值/RNG、每 100 更新检查有限性，不增加算法项或随机采样。正式训练不运行 P1 探针。以下命令在 `ROOT/source` 执行；路径与环境覆盖原样记录。

Seed 0：

```bash
cd /home/xushijie3/work/floq_min_repro_v2_20261013/source
env HOME=/home/xushijie3/work/floq_min_repro_v2_20261013/home CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cusparse/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/curand/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cufft/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cublas/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cuda_cupti/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/nccl/lib XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=offline /usr/bin/time -v -o /home/xushijie3/work/floq_min_repro_v2_20261013/runs/formal_seed0/cpu_time.txt taskset -c 35,41,48,47 /home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/bin/python /home/xushijie3/work/floq_min_repro_v2_20261013/scripts/native_entry.py --env_name=antmaze-large-navigate-singletask-v0 --agent=agents/floq.py --agent.alpha=10 --offline_steps=2000000 --online_steps=0 --seed=0 --save_dir=/home/xushijie3/work/floq_min_repro_v2_20261013/runs/formal_seed0/outputs --project_name=CP_FLOQ --run_group=FLOQ_MIN_REPRO_V2_P2 --eval_interval=20000 --eval_episodes=50 --save_interval=1000000
```

全部展开模型参数：`formal/formal_seed0/actual_config.json`；全部实际 flags：`formal/formal_seed0/expanded_flags.json`。
Seed 1：

```bash
cd /home/xushijie3/work/floq_min_repro_v2_20261013/source
env HOME=/home/xushijie3/work/floq_min_repro_v2_20261013/home CUDA_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cusparse/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/curand/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cufft/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cublas/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cuda_cupti/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/lib/python3.8/site-packages/nvidia/nccl/lib XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=offline /usr/bin/time -v -o /home/xushijie3/work/floq_min_repro_v2_20261013/runs/formal_seed1/cpu_time.txt taskset -c 111,34,36,44 /home/xushijie3/work/floq_min_repro_v2_20261013/envs/floq/bin/python /home/xushijie3/work/floq_min_repro_v2_20261013/scripts/native_entry.py --env_name=antmaze-large-navigate-singletask-v0 --agent=agents/floq.py --agent.alpha=10 --offline_steps=2000000 --online_steps=0 --seed=1 --save_dir=/home/xushijie3/work/floq_min_repro_v2_20261013/runs/formal_seed1/outputs --project_name=CP_FLOQ --run_group=FLOQ_MIN_REPRO_V2_P2 --eval_interval=20000 --eval_episodes=50 --save_interval=1000000
```

全部展开模型参数：`formal/formal_seed1/actual_config.json`；全部实际 flags：`formal/formal_seed1/expanded_flags.json`。

配置 SHA256：`8ed2f832d1e888f2b42a85d50bb56fa22b6e2db6519d4887fa3a2d643703adc6`。参数来源表 `report/actual_parameter_origins.csv` 标明作者 get_config 默认、官方 main 派生、卡片/CLI 和工程路径字段；种子及输出路径的实际不同值以逐种子 JSON 为准。α10、batch256、lr3e-4、γ.99、τ.005、actor flow10、critic flow8、双 Flow/Scalar ensembles、noise8、coverage.1、mean Q，均保持冻结值。

## 完整性与可追溯产物

P0/P1 已通过，P1 独立 5k：初始化/前向/梯度、目标更新一拍延迟、全参数/优化器/JAX RNG 恢复身份及固定 key 动作一致性通过。正式训练每条 exit0、2M updates、101×50 原生回合、400 行官方训练/验证指标。全记录未发现非有限数值；未重启、延长、调参或挑选最好模型。

1M/2M 四份 checkpoint 文件大小与 SHA 核验通过，索引见 `formal/*/checkpoint_index.jsonl` 和 `final/artifact_index.json`。正式模型未另加恢复评价；保存/恢复工程验证来自 P1。

全 stdout、全部原生逐回合记录、资源采样、完整 RNG 和模型留服务器 `ROOT/runs/`；`final/artifact_index.json` 给出路径、大小与 SHA。Git 保存全部原生曲线、轻量 loss 曲线、最终评价、逐回合指标及成本，权重、数据、全日志未入库。之前明确禁止推送的 FQL/Q-Flow/GFP 分支未推送。
