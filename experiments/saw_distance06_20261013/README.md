# SAW-DISTANCE-06

用户批准执行的零训练CPU诊断；独立分支codex/saw-distance-06-20261013，基点19d1b4b411a00b92866b4512150339472a086f0b。

prepare.py复用GOALPROBE05完整状态工具，冻结地图条件并用独立参考控制准入；signal_prepare.py使用官方HGCDataset封存F/R及固定目标邻域。
execute.py只运行既有1M actor，每条件最多25步；signal06.py只前向冻结value/teacher/actor并核验原损失与网络/优化器不变。
behavior_analysis.py按目标→距离→首通道→起点等权，起点聚类5000次；signal_analysis.py按原数据轨迹聚类，保留缺类和近似目标局限。
audit06.py独立复算全部新D和旧P3逐步轨迹，未新增物理步。finalize.py整理裁定，package.py显式暂存轻量材料。

GPU0、参数更新0、总CPU4小时含准备协调修复、4核72–75、RSS16GiB、最多25000物理步；run_phase.py只监控并在预算违规时停止自有phase。
所有大NPZ/PKL与完整日志留服务器runs/和logs/，由RAW_ARTIFACT_INDEX.json定位。六个既有权重不复制、不修改，正式源码版本见D_STARTED与SCIENTIFIC_SOURCE_FREEZE_V2。

附卡、静态预检和原始ZIP的MANIFEST保持用户提供身份。执行前登记的目标排序有一个6格中间条件相对卡片字典序的偏差，见SELECTION_LITERAL_CHECK和SCIENTIFIC_SOURCE_FREEZE；已封存条件未按结果重选，不进入主近远估计。
本轮不自动启动后续训练。结论限于固定模型、有限起点、25步局部窗口和原采样器冻结探针。
