import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/eg_causal_v2_20261023/r3"
s = json.loads((OUT / "R3_scene_paired_statistics.json").read_text())
d = pd.read_csv(OUT / "AD_joint_structure.csv")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
scenes = s["scene_counts"]
x = np.arange(len(scenes))
for key, label, color, marker in [("top2x2_any", "Top-2 x 2", "#2563a6", "o"), ("local4_any", "Argmax + 3 neighbors", "#d77916", "s"), ("random4_expectation", "Uniform random 4 (exact expectation)", "#6b7280", "^")]:
    axs[0].plot(x, [r[key]/r["n"]*100 for r in scenes], marker=marker, label=label, color=color, linewidth=1.6)
axs[0].set(xticks=x, xticklabels=[r["scene"] for r in scenes], xlabel="Training scene (32 queries per scene)", ylabel="Oracle set coverage (%)", ylim=(0, 100), title="Four-action budget at the frozen width")
axs[0].legend(loc="lower right", fontsize=8)
vals = [50, 9, 14, 64]
labels = ["One-class change\nwithin Top-2", "Both classes needed\nwithin Top-2", "Only below Top-2", "No feasible cell\nin frozen 48"]
bars = axs[1].bar(np.arange(4), vals, color=["#2563a6", "#649dd2", "#d77916", "#9ca3af"])
axs[1].bar_label(bars, padding=3)
axs[1].set(xticks=np.arange(4), xticklabels=labels, ylabel="Query count", ylim=(0, 75), title="All 137 original failures")
axs[1].tick_params(axis="x", labelsize=8)
fig.suptitle("R3: model preferences contain information; rescue is not deployment success", fontsize=12)
fig.savefig(OUT / "R3_oracle_controls.png", dpi=180)
plt.close(fig)

report = """# R3：联合决策已有网格诊断

**机制判断：支持模型的类别偏好携带有效信息；仅部分支持存在组合不相容，证据不足以认定它是主要错误机制。方法开发：HOLD。**

本轮只读取本地既有数据，未访问服务器、未重新推理、未新增解析评价、未更新参数、未提交 Git。全部 256 个查询保留，覆盖训练集 8 个场景、16 帧。它们含每帧 1 个原始 Top-1、7 个排名分层和 8 个随机查询，不能当作 256 次部署 Top-1。48 格全部使用原来的位置、接近方向和宽度；主口径为无碰撞、非空夹、0<μ≤0.4。

## 1. 同预算结果

下表为“集合中至少存在一个可行动作”的 **Oracle 覆盖数**，不是模型能够自行选对的次数。随机 4 格的非整数为组合数计算的精确期望。

|查询层|n|原姿态|Top-2×2|原argmax＋3邻居|随机4格期望|原argmax＋随机3格期望|完整48格|
|---|---:|---:|---:|---:|---:|---:|---:|
|raw_top1|16|12|15|15|10.37|14.19|16|
|rank_stratified|112|48|71|67|30.58|54.89|75|
|random|128|59|92|82|37.05|70.92|101|
|全部|256|119|178|164|78.01|139.99|192|

Top-2×2 覆盖率 69.53%。相对随机 4 格高 **39.06 个百分点**，场景配对 bootstrap 95% 区间 [33.38,44.83]，8/8 场景为正。随机 4 格不保证保留原成功格，因此同时加入“保留原格＋随机3格”：Top-2×2 仍高 **14.85 个百分点** [10.39,19.96]，8/8 场景为正。这支持模型偏好并非单纯随机多试。

相对预登记邻居 4 格，Top-2×2 高 **5.47 个百分点** [2.73,8.59]，6/8 场景为正、2/8 相同。但 **raw_top1 层两者均为15/16**，没有证明联合选择在真正高排名样本上超过简单局部扩展。没有按结果调整邻居规则：固定原格、角度±1格（π周期）、深度+1格（最深格则−1格）。

## 2. 59个恢复不等于59个“头不协调”

137 个原失败中，Top-2×2 恢复 59 个；其中 **50/59（84.7%）只需改角度或深度之一，9/59 必须两者都改**。其余失败中，14 个只在更低排名格可行，64 个在当前固定位置／方向／宽度的48格内均不可行。

|可行的次选组合模式|失败查询数|
|---|---:|
|仅(a2,d1)|15|
|仅(a1,d2)|5|
|仅(a2,d2)|9|
|(a2,d1)和(a2,d2)|10|
|(a1,d2)和(a2,d2)|14|
|三者均可行|6|
|三者均不可行|78|

“首选角度在别的深度可用、首选深度在别的角度可用，但首选组合失败”的严格描述性事件有 **14/137（10.2%）**，跨 6/8 场景；其中 raw_top1 2例、排名分层5例、随机7例。它说明联合可行结构确实存在，不能证明独立头导致了这些错误。

若直接把全部查询改成次选，而非用 Oracle 挑选：(a2,d1)恢复31、损害29，净+2；(a1,d2)恢复25、损害53，净−28；(a2,d2)恢复39、损害52，净−13。可选集合的上限收益不能直接变成可部署收益。

所有查询的角度边际可行质量量、深度边际可行质量量、独立组合质量量平均分别为 **0.561、0.624、0.424**。原失败组分别为 0.290、0.345、0.127。以预登记描述性口径“两边际≥0.5、组合<0.25”计，原失败只有3例；连续数值已逐查询保存，不把 softmax 当作校准成功概率。

例如 scene0002/frame0000/candidate700，角度与深度边际为0.999、1.000，组合为0.135，原组合失败而(a1,d2)可行；scene0068/frame0000/candidate53，两边际0.987、0.982，组合0.172，换角度即可恢复。两例支持组合限制存在，但恢复本身仍可解释为普通类别预测误差。scene0068/frame0128/candidate453 是raw_top1：两边际0.888、0.951，组合0.360，Top-2×2仍失败，只有更低排名格可行。

第13角度类概率均值1.35×10⁻⁹、最大1.79×10⁻⁷；第5深度类均值8.78×10⁻¹⁰、最大5.36×10⁻⁸，均无全类argmax落入额外类。正式12/4类条件归一化在本集合不会因丢弃大量额外类概率而改变结论。

## 3. 与宽度主线的联系

64个全格失败查询中，原宽度上限饱和21/64（32.8%）；存在可行格组为39/192（20.3%）。场景配对差为+11.48个百分点 [0.24,24.30]，但仅5/8场景为正。限制到有效标签匹配查询后，饱和为11/30对17/101，场景差+19.96个百分点，区间 **[−3.27,46.38]**，尚不稳定。

有效匹配组中，全格失败的 q_ref=0 为 **13/30**，有可行格组为 **1/101**；场景配对差+31.63个百分点 [12.14,53.97]（7个可比较场景）。参考宽度>100mm为12/30对0/101，参考零宽度1/30对0/101。超限宽度与q_ref=0高度重合，不能当作两份独立机制证据。其余34个全格失败查询没有有效参考匹配，未用其q_ref作负例解释。

这一富集给宽度主线提供了定位线索；未改变宽度、未控制完整物体几何与方向，故不能认为宽度监督造成了全格失败，也不能称这些位置“无解”。

## 4. 可信度与边界

16帧的候选ID、H0姿态、角度／深度logits与旧缓存逐元素一致；全部256查询的原格和Top-2×2标签与上一轮一致，**119原可行／137原失败、178 Top-2覆盖、192全格覆盖**均复验通过。全部12,288格无未知。解码宽度按原公式核对，最大差<2×10⁻⁸m。没有重新匹配或修改标签。

统计以场景等权配对、20,000次固定种子bootstrap为主，只有8个训练场景，区间应谨慎解释。模型可能已见过这些训练场景；这里没有C3、C4、C5新验证。类别间隔、宽度富集及边际质量量都是描述性关联，不能追溯历史有害训练或替代留出干预。

**结论：R3排除了“Top-2恢复只是任意多试4格”的强解释，但大量恢复只需单类别调整，真正首选组合不相容仅占部分失败；在16个raw_top1上尚未超越预登记邻居基线。保留普通分类不确定性和局部几何的竞争解释，联合机制的方法开发维持HOLD，不据此新建联合头或扩大支线。**

证据表：`AD_joint_structure.csv`（逐查询）、`R3_scene_paired_statistics.json`（场景配对区间、原姿态替换损害、来源哈希）、`R3_scene_counts.csv`。规则冻结文件：`R3_manifest.json`。源码路径：既有 `code/probe.py` 的grid_ids/分类输出、`code/evaluate_module.py` 第28–33行的固定宽度网格及原格校验、`code/analyze.py` 第33–37行的历史Top-2对照。本轮未更改这些源文件。

![四格预算与恢复分解](R3_oracle_controls.png)
"""
(OUT / "R3_conclusion.md").write_text(report, encoding="utf-8")
print("Written report and figure", OUT)
