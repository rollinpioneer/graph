# EG-B Global Rank Pilot

GPU服务器独立运行环境复用：

```bash
source /aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019/activate.sh
export RANK_ROOT=/aita_data/home/aita_xushijie/work/grasp/eg_b_global_rank_pilot_v1_20261020
```

`preflight.py`核验B已有256帧标签、动作0映射、归一化和覆盖；`reserve_c4.py`仅根据已知使用史与原始数据预留名单。`train_all.py`执行唯一六个正式M/R训练（各1000步）；现有权重禁止重复训练。

`continue_p2.py`执行O/M/R离线诊断，再按4分片/场景完成C3固定帧开发评价。`evaluate.py`使用旧B导出、缓存质量和CPU FrozenScorer，所有20种配置从1024候选重新执行相同CD/NMS/候选选择。仅新排序触及的未知质量调用Native补算；未知值保留NaN，不标为负例。EG/O的选择与原逐帧指标逐项精确核对。

`report.py`核验384帧、7680条记录、权重身份与资源，再生成R−O、R−M、M−O、R−EG等配对结果及开发闸门。Bootstrap按场景和划分，seed20261020，三头先在场景内平均；C3只是开发材料。

本卡不执行P3。GPU空闲后不追加其他训练；官方质量评价主要使用CPU并行。源缓存、原权重保持只读。
