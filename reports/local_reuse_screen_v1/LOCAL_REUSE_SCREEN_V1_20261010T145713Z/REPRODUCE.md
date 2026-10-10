# 本轮复现入口

本卡已经结束；以下是复现说明，不能作为自动再运行授权。所有新运行必须使用新的独立目录并遵守对应授权预算；不覆盖本次产物。

代码基点cc7d79d6f，依赖原服务器/home/xushijie3/work/graph_cp_disr及已冻结的public_depots_rel_v1、search_scope_diagnostic_v1、compact_diagnosis_v2输入。权重不纳入Git，路径和SHA在manifest.json。模型及核心前向与历史FAST 511b117无差异。

解释器~/envs/cpdisr/bin/python；PYTHONPATH=/home/xushijie3/work/graph_cp_disr/src；LD_LIBRARY_PATH=/home/xushijie3/.uv_python/cpython-3.10.19-linux-x86_64-gnu/lib；PYTHONDONTWRITEBYTECODE=1；OMP/MKL/OPENBLAS_NUM_THREADS=2。准备和依赖统计使用CPU，神经正确性与计时使用冻结GPU选择CUDA_VISIBLE_DEVICES=2。外部GPU占用判据必须按登记执行，不能以本次部分有效计时替代新运行负载记录。

阶段入口：screen.py prepare → 冻结manifest/parents并登记 → screen.py audit → 读取G1 → 冻结prototype.py/measure.py和phase2_registration → measure.py correctness → 读取G2 → 冻结timing.py/timing_registration → timing.py。各命令均传--run-root <新独立目录>。目录必须事先建立。不得越过门控；本次G3未通过，没有搜索入口需要运行。

本次原始核查只加载DENSE，无训练、WL拟合、精确oracle或新任务生成。publication_*.json是独立Git发布回执；实际提交SHA避免写入自身提交产生自引用。实验卡原始DRAFT文本保留字节，用户执行授权记录在registration.json。publish_card.ps1是本地凭据桥接发布工具，只在本卡新分支添加归档文件，不改服务器checkout。
