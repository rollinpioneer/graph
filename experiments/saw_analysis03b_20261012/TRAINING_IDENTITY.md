## 1. 训练身份与科学配置

官方源3535a16c901e5de1233ac5a586401b3ddfa5e30d；RD01输入6462e06c540594acca6bc8630c37a803fe36dd4a；EVAL02输入b0dd744d323b924d02de93f5c487c4afc9565171。正式训练启动源码提交388ded0a23258ed43e12093825b17fe80b7510fc，后续CPU分析提交不改变运行中科学训练代码。

配置B仅actor_p_trajgoal 1→0.5、actor_p_randomgoal 0→0.5。两个独立seed0从头初始化，update0代理参数、目标网络、优化器与代理RNG和A逐数组相同，误差0；不是从A训练后权重继续。完整SGD历史未由相同seed自动推断。实际采样器未修改，random目标分支Waypoint为当前段25步后状态，未来目标分支为min(state+25,goal)。4096样本比例/边界/终止/奖励/Waypoint全部通过，真实比例见SAMPLER_AUDIT.json。

历史RD01 Navigate62.8%、Stitch49.6%与EVAL02 Navigate64.8%、Stitch50.6%分别保留。旧检查点和轻量文件已按原manifest重新核验，READ_ONLY_ASSETS.json为回执。Stitch仍是作者专用协议近似，B不冒称作者配置。
