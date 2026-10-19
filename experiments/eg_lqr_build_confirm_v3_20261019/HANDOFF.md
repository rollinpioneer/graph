# V3交接：P1完成，锁定S，C1运行

当前入口aita_xushijie@172.19.164.161（gpu04），现有AITA公钥可登录。与旧节点共享10.10.1.122:/aita_data/home，原研究目录直接可用；未复制私钥、大数据或覆盖旧结果。

独立环境environment使用Open3D0.19适配GLIBC2.34，原共享环境不改。P0四帧三头线上/缓存、保存加载逐元素一致；56条官方原生路径与历史R0/S指标一致，CLI输出一致。时延见results/latency.json。

P1完整384帧已完成。S+CD平均+1.471576AP，三头全部为正；F05+1.921163AP，两族均过预算闸门。按预定S优先规则锁定S，不因F05更高更换。method_lock先于C1审计与性能读取生成，171份已知记录审计未发现C1暴露，不保证未知外部历史。

C1控制PID561483，通过nohup运行；只确认S三头，R0/S带无CD共8配置/帧，完整12场景×256帧。导出期间GPU1+CPU评价6+控制1，之后CPU评价7+控制1，总计算上限8。源code/run_confirmation.py在导出完整场景后自动补位；结束自动运行report.py和dedup_sensitivity.py，生成C1_decision、完整指标、去重敏感性和final_report。没有新训练、局部姿态或系数搜索。

实时状态results/C1_status.json与C1_export_status.json，日志logs/C1_launcher.log、C1_export.log和C1_场景.log。当前尚无完整C1结论；不要把P1放行或方法跑通当独立确认。若技术失败保留产物并处理同配置恢复，不换场景、不改头。

继续沿用codex/eg-lqr-build-confirm-v3-20261019。阶段归档包含P0/P1、method_lock、暴露审计和冻结源码；C1结束后补充最终轻量结果与Git核验。大输入、权重、完整日志和质量缓存留服务器。
