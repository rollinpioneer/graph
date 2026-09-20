# CP-DISR v2.1 实现清单

生产核心源码已存在；数值运行验收尚未完成。不是只有 Protocol 或建议 CLI。

| 组件 | 实现 | 验证状态 |
|---|---|---|
| __init__ | `src/cp_disr/__init__.py` / `c45f4721bc68ce4db6ffc25e9b1c64b0a366f1f8521ca3252ee8fe24e8e71816` | 编译/静态；相应纯逻辑测试见测试清单 |
| __main__ | `src/cp_disr/__main__.py` / `79ee6f134c5c5193377cf9b189cbdcc3e2095e49ba9cec310114441c15a917c4` | 编译/静态；相应纯逻辑测试见测试清单 |
| adapters | `src/cp_disr/adapters.py` / `e0687ff747287437a89c1a2ba97a0b8c404c8a9291837aa7cee42802985013fd` | 编译/静态；相应纯逻辑测试见测试清单 |
| cli | `src/cp_disr/cli.py` / `52e25350a9882785bdc8323877a1f13a70e36705a0b019dfd391c62609850404` | 编译/静态；相应纯逻辑测试见测试清单 |
| collector | `src/cp_disr/collector.py` / `eae8f807b08460d48ff1e2d6ae2a113393d4aedb37c4d602917fc3dc2fdbbb0b` | 编译/静态；相应纯逻辑测试见测试清单 |
| common | `src/cp_disr/common.py` / `74b6c695d67ee996cf3031091888f275eea12f8fcf3b521837618dff258407c5` | 编译/静态；相应纯逻辑测试见测试清单 |
| contracts | `src/cp_disr/contracts.py` / `4479d3205ef01c57da87ebef87b4127c21a925813832ae0a4d631afac020529b` | 编译/静态；相应纯逻辑测试见测试清单 |
| evaluation | `src/cp_disr/evaluation.py` / `3568056bb25b36cd241352629a53d7846e5250b12276b142e777d361290eefb3` | NOT_RUN_ENVIRONMENT_BLOCKED |
| execution | `src/cp_disr/execution.py` / `5e250244b65610e74899ac6ddb46f471147b271527979d8352ed3cb9c620f33f` | NOT_RUN_ENVIRONMENT_BLOCKED |
| facts | `src/cp_disr/facts.py` / `0d01cdf49c07576f11fa6e5711bd7245827f54144e3515c757c1f181914a0a7d` | 编译/静态；相应纯逻辑测试见测试清单 |
| fixtures | `src/cp_disr/fixtures.py` / `11e1ffc1617daad4cb78571c90e6c000fd0891004b86dd6347c758720ecdf06d` | 编译/静态；相应纯逻辑测试见测试清单 |
| graph | `src/cp_disr/graph.py` / `92becfafd17aa7cfdcf17ba1d8046f4bc54c4288300cfbc8469e56734df2ebe6` | 编译/静态；相应纯逻辑测试见测试清单 |
| neural | `src/cp_disr/neural.py` / `22ed9cdecae882037154bf5e5e5504b48dafaaa75a8384ee7658ace4ff967654` | NOT_RUN_ENVIRONMENT_BLOCKED |
| prior | `src/cp_disr/prior.py` / `078fd2742777c69a5016c363e96d49ddb103583216a03737b4a2958ca6e77b25` | 编译/静态；相应纯逻辑测试见测试清单 |
| rl | `src/cp_disr/rl.py` / `56e28adaece956fe8576e4a01caf1709222570fef5cba300f38ac92b0ed4a8fc` | 编译/静态；相应纯逻辑测试见测试清单 |
| runtime | `src/cp_disr/runtime.py` / `5bc5f091cb447e6eae3ba95f17f004554615706a696e592f5c61e1cac6777bff` | 编译/静态；相应纯逻辑测试见测试清单 |
| skills | `src/cp_disr/skills.py` / `b2c08e6d1b36ed05f3ff9ad3a3b7bf25f50911f280056eacd94caeb30a23bd6a` | 编译/静态；相应纯逻辑测试见测试清单 |
| torch_rl | `src/cp_disr/torch_rl.py` / `73aca522393d7dab2eb016408f8d9436cefa18d0b54a45160724cbdffca01e72` | NOT_RUN_ENVIRONMENT_BLOCKED |
| validation | `src/cp_disr/validation.py` / `fe182efe96e8c98538bbd43db127bd1226b8cace8303077ca9aa59352f5a326e` | 编译/静态；相应纯逻辑测试见测试清单 |
| vlm | `src/cp_disr/vlm.py` / `6afb390798c19efd5085e65f48ade89a4e392eafb00680a5adb685a1b9b48be8` | 编译/静态；相应纯逻辑测试见测试清单 |

## 入口与边界

8 个正式 CLI 均可定位。train/evaluate 遇到未绑定运行资源返回非零退出码和具体 MUST_BIND；generate-cache 本轮仅本地校验能力，未实现/调用外部 API transport，明确拒绝请求。run-unit-tests --scope full 为未来用户授权 Stage 0B 后入口，本轮仅运行 --scope pure 对应的逻辑检查。

Torch 模块包含标准 128×4 层 RGCN、goal readout、七配置、GRU、独立 V/Q、当前参数 prefix 重算、单参数所有者 PPO、执行动作 Q loss、真实 transition collector 和有界资源 cap driver。外部 snapshot builder、controller、perception 等必须由真实 plugin 显式绑定。所有数值代码仅通过编译和静态检查，不能宣称其动态正确性已通过。
