# GFP-REFERENCE-SUPPORT-CONTROL-AB-V1 交接

状态：训练、评价和结果分析全部完成；不自动追加实验。

服务器根目录：`/home/xushijie3/work/gfp_reference_support_control_ab_v1_20261016`。

分支：`codex/gfp-reference-support-control-ab-v1-20261016`。

链接：https://github.com/rollinpioneer/graph/tree/codex/gfp-reference-support-control-ab-v1-20261016

最终提交与远端SHA核验见 `receipts/git_verified.json`。该回执在最终提交后生成，避免提交哈希的自引用。

两臂各50k，总100k更新；400评价回合。A/B在25k成功率31%/39%，50k均41%；最终起点滞留14%/18%；均无联合价值异常。实际幅度配对基本通过，但持续更新门槛仅65.2%事件通过，低于90%。结论为未建立数据支持的额外收益，完整准入INCONCLUSIVE/DIAGNOSE。

来源、冻结MDN、训练RNG、计数和保存恢复已核对；四组评价初始观测哈希匹配且真实exit0。最初采样接口错误发生在零更新准备，保留失败日志和48.5秒成本，未重跑训练。

完整分析：`reports/RESULTS.md`。关键曲线：`reports/full_curves.png`。逐回合、匹配、成本和checkpoint索引均在reports目录。权重、全量候选和日志只在服务器；轻量暂存清单见receipts/archive_manifest.json。

下一阶段需新授权；不要再次启动admission、continue_train或evaluation_manager。当前protocol保留运行时冻结内容，其授权预算已全部消耗。
