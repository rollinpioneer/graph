# FACB-Z0 交接

状态：完成，NO_GO_PROXY_ANCHOR_SEPARABILITY。
分支：codex/facb-z0-20261017
链接：https://github.com/rollinpioneer/graph/tree/codex/facb-z0-20261017
服务器：/home/xushijie3/work/facb_z0_20261017

固定c=44.09771900177007，κ=0.22048859500885037。健康整体上界0.164504%，通过≤1%；价值岛s0 700k为26.416016%、s3 400k为39.086914%，均未达到>50%，两套噪声结论一致。健康整体下界1.105027%，近目标下界6.677246%。不得因此调整阈值或自动启动训练。

已完成20主配对+6辅助、8个500k同种子I臂代理锚。代理是同期训练、Target Q两头均值expectile 0.7的V，不等价独立IQL预训练锚。最终计算核验误差0，34个检查点和配置、数据及模型树不变，真实推理退出0。

两次失败检查、编译边界审计、恢复别名修复及无效输出完整保留；正式结果均在独立Target参数映射核验后重算。旧产物未改。总GPU驻留成本保守上界0.188646 GPU·h，训练/新回合/环境构造resetstep/新价值权重初始化均0。

RESULTS.md为完整解释，separability.json为主结果，calibration.json为唯一有效带宽。数组、权重和完整成功日志留服务器；隔离无效输出不用于统计。最终提交与远端核验另见receipts/git_verified.json（提交后记录）。
