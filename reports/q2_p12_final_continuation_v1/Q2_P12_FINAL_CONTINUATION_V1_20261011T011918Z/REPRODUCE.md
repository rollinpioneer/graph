# p12 最后一次受控续跑

本卡是阶段C结果后的补充，固定同一个已知p12事件，不是独立测试任务。
用户“执行”批准本卡；随后“能并行吗，能的话就都并行”覆盖卡第7节的串行调度。
四支使用独立进程、不同物理CPU核、各自108步缓存和干净D0@820模型。
单支评分仍为单线程CPU FP32原生单状态确定性评分。全部预算和科学边界保持冻结。

源结果提交为 fabfa1341b057a4c2b67c0b3acb88e35c43b45ac。
checkpoints记录源日志第108条完整展开后的场景；半完成现场和未来缓存没有恢复。
core_semantics.py从冻结phase_c.py抽取规范化函数及普通D0搜索体，删除干预入口。
worker.py以每步无损delta、每128步完整检查点和最后完整场景保存轨迹。
log_codec.py验证重建前后SHA；物理heap、失效条目、父指针、动作、g及缓存全部保存。
orchestrator.py仅调度与观察字节一致性，不向worker提供评分或缓存。

仅在新批准的独立目录运行一次，不能重启同一实验根目录：

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/xushijie3/envs/cpdisr/bin/python identity.py ROOT identity_before.json
/home/xushijie3/envs/cpdisr/bin/python orchestrator.py ROOT
/home/xushijie3/envs/cpdisr/bin/python analyze.py ROOT
/home/xushijie3/envs/cpdisr/bin/python identity.py ROOT identity_after.json
```

以上命令是本次执行记录，不构成另一次实验授权。
analyze.py只读取保存证据，无模型、搜索或距离查询。
完整原始日志和评分字节保存在服务器及本地证据归档，以artifact_manifest.json绑定。
Git分支保存代码、输入身份、恢复检查点和轻量结果；不推送模型权重。

展开账本为事件后108+本次新增展开，任务累计另加事件前2步。
资源账本保留旧运行的所有实际成本，包括回退掉的后缀、失败和中断，以及本次四支和辅助开销。
加载、恢复、日志与清理分列，不能解释为连续端到端速度测试。
CPU协议结果不能直接外推旧GPU部署；H3仍未决。
本卡结束即停止p12计算，不自动扩大预算或追加分支。
