P2 原始角度变体的 μ 与 I1 数据导出（2026-10-24）

范围：既有 P2 面板，12 个场景 × 帧 32/96/160/224，48 帧。
直接读取 N0 cpu/P2_*.npz 的 pred；不生成、选择或筛除姿态。
仅 CPU；没有训练、模型前向、参数更新或 C4/C5 访问。
不运行风险预算规则、换角规则、救回统计或碰撞性能统计。

评价：evaluate_p2.py 的物理评价核心逐字复制上一轮 evaluate_angles.py。
依赖同工作根目录 eg_pose_quality_diagnosis_v1_20261022/evaluate.py。
输入根目录：eg_n0_nref_main_20261024；同时读取 cpu 与 per_pose。
先在 C3 0100_0000 的全部既有 pred 上回放，再运行 P2。
每帧核验候选 ID、原角度、物体 ID、owner 映射、collision/empty；不一致停止。
每个输出 NPZ 对应一个 JSON 收据，记录输入/输出 SHA256、时耗及核验结果。
implementation_provenance.json 记录原实现及评价核心的 SHA256。
c3_replay.json 记录回放结果和实际使用的官方评价代码 SHA256。
output_manifest.json 记录逐帧输出与收据 SHA256。

输出：evaluation/SSSS_FFFF.npz，含 candidate_id、original_angle、object_id、
mu、collision、empty、I1、unknown_reason、contact_reason、pred。
保留官方评价的 -1（碰撞或当前范围未找到力闭合解）与 NaN（未知）的区别；
具体状态须结合 collision、empty、unknown_reason，不把 -1 当作 NaN。
I1 无接触时保持 NaN，并保留 contact_reason，不补值。

运行：先 bash run_cpu.sh replay；通过后在三台机器分别运行
bash run_cpu.sh 0 4 / bash run_cpu.sh 1 4 / bash run_cpu.sh 2 4。
全部完成后 python summarize.py，仅汇总用户指定的核验计数。
输出已存在时拒绝覆盖；历史输入不作修改。

summary.json 的“原角度干净”指无碰撞且非空；可行指其中 0<μ≤0.4，
弱指其中 μ>0.4。原角度 I1 缺失数以全体候选的原角度为分母。
这些仅是数据交付核验，不构成任何选择规则效果的结论。
