# P0/P1 回执（2026-10-16）

远端根目录：`/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/`，cwd 为官方源码 `source/gfp`（1abbd5f，未改动）。
Python：`flow_baselines_preflight_20261013/envs/gfp312`（jax 0.6.2 / jaxlib 0.6.2 GPU，flax 0.10.7，optax 0.2.5，ogbench 1.1.0，mujoco 3.3.5，gymnasium 1.1.1，numpy 1.26.4）。GPU 为 A100-SXM4-40GB。

## 脚本（本地与远端 sha256 一致）

| 文件 | sha256 |
|---|---|
| code/common.py | 946453db752c99c2e2356642da1fc431c65525fac25ddd44e6c449f6d1f24180 |
| code/frozen_eval.py | 88b76ccfce4f23986d7d6a36fac4fcf33e450841401e84089f00bcac3d5c64d2 |
| code/ckpt_probe.py | 5837febd19128b11cf895bbfa3516bc8018ab550dd2f2d1ef6942b3de30953b9 |
| code/precision_probe.py | a93f90b8cc72e5cce28b82bfb5b6426691f475924bb15562b5716fbe610834b3 |
| code/coverage.py | 8880221ca911f154db6083f46d400116394e9ad1222d3778611862726729a754 |
| code/aggregate.py | 36eb0f91b2141b082854e22c4fd8bd40cf2e237798030c18debb2add66077b76 |

修正记录（均在正式运行前）：common.py 加入 cwd 到 sys.path；frozen_eval.py 改为在加载 JAX 之前用 spawn 建环境（fork 后 MuJoCo EGL 初始化失败）；coverage.py 按 OGBench 的实际口径改用当前状态的 xy 判成功（首版用 next xy，断言失败）。

## 运行（2026-10-16 03:20 UTC 起，全部 setsid -f，环境变量 `HOME=$R/home MUJOCO_GL=egl OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false`）

| 作业 | GPU | CPU 核 | 内容 |
|---|---|---|---|
| p0a_s0 | 5 | 6-17 | frozen_eval 种子 4242，20 轮×10 = 200 回合：G_task1_s0 500k–1M |
| p0a_ctrl | 7 | 18-29 | 同上：G_task1_s1 600k/700k，F_task1_s0 600k/700k，G_task1_s3 800k/900k |
| p1_rollout | 4 | 30-41 | 5 轮 = 50 回合：16 条训练的 1M |
| probe | 6 | 42-47 | ckpt_probe 50,000 状态：task1 两臂 4 种子 500k–1M，task4 两臂 4 种子 800k–1M，共 72 个检查点 |
| precision | 0 | 48-53 | precision_probe：G_task1_s0 600k/700k/1M，G_task1_s1 1M，G_task4_s0 1M，F_task1_s0 1M |
| coverage | CPU | 0-5 | 全数据集覆盖 |

全部正常结束（日志末行 DONE）。另有一次冒烟评价（G_task1_s0 600k，10 回合，种子 999），不计入结果。

## 输出（远端 sha256）

| 文件 | sha256 |
|---|---|
| out/coverage.json | 902860e7bd86b9d704338a85093cb6d447fddbecdfb2e6f7ad22197e267ba400 |
| out/p0a_s0.json | 8495f60579ce633106572f03b361497cb4505bde31aee11b5f40198acea949d7 |
| out/p0a_ctrl.json | e0c9f8f11e49741f43640790d2387db911f675e276fb17c007b231d43accaa83 |
| out/p1_rollout_1m.json | efa4cc625014f90cf59db03873d4df709856801ac10cfc333ce5f82422fce34a |
| out/precision.json | 55977b038592ce35eea80748401d2dcadd1352f81080e355e349512d601d90f2 |
| out/probe/states.npz | c639a690fe309be266a954808beb99b00964d5d905c595fb830c9a9b304698bd |
| out/summary/frozen_eval_summary.csv（本仓库 results/ 同名文件一致） | b96fea5f2afff30d9a50f4085f55f49b6f46a802fea6fe8e9c7639766ba6a85b |
| out/summary/probe_compact.csv（本仓库 results/ 同名文件一致） | 11da31fec574d82569144409b09e5bec19fe056ba460edfaa5b11921fa8ee7ba |
| out/summary/probe_summary.csv | cda4ebb65223fe3cd3e5e87b919db511fcd86a120049ecd4dc51b66861c69ed7 |
| out/summary/cell_maps.json | eb702f40378901d05875deeee74dd32bc7bfd6ba7ea6472f7e0b725c192f638c |
| out/summary/cell_V_compact.csv | 809fd8518eee8530114992874669290936bf11e0baa69db670ac49c75dae5ece |
| out/summary/rollout_summary.json | 6645b9148be07ce93e4ef27c30adc7332b9f61445ab178e9ffc25694bd75fc95 |
