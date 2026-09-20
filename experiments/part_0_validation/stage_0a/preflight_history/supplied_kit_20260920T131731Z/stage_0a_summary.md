# Stage 0A：本地资料预检

**状态：BLOCKED；范围：conversation_sandbox的元数据盘点，不是用户训练机验收。**

已记录三份输入的SHA-256、当前Python与包metadata，并写入缺项表。没有安装依赖，没有进行生产模型前后向、完整单元测试、VLM请求、RL训练或机器人动作。

当前Python为3.13.5；torch为2.10.0+cpu；nvidia-smi路径为未发现。缺失包metadata：torch-geometric, tensorboard, gymnasium, dashscope。这些只描述当前进程所在环境，不能推出用户目标主机的情况。

## 阻塞项

- `runtime.target_host`：实际训练主机及资源授权。
- `runtime.repository`：当前v2.1实现worktree/commit及入口。
- `runtime.simulator_or_robot`：环境及版本。
- `runtime.task_assets`：D0/T_A/T_C真实资产与初始化池。
- `runtime.controller`：冻结技能执行器及参数生成。
- `runtime.camera`：观测来源、标定和冻结感知。
- `runtime.verifier_thresholds`：谓词校准与独立成功评价。
- `runtime.skill_timeouts`：技能时限、任务deadline与安全许可。
- `runtime.interaction_time_unit`：时钟单位、d_ref和耗时计数。
- `vlm.account_region_endpoint`：实际账户、地区、endpoint、SDK调用验证。

## 下一步

在实际项目环境读取已有绑定并完成0A剩余工作；证据齐全后由Agent更新状态。本预检工具从不自动给出PASS，也不执行0B。论文方法写作不受阻塞。
