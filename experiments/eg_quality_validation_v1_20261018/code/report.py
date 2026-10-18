import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
s=json.loads((N/'summary.json').read_text());c=json.loads((N/'cost_official_results.json').read_text())
groups=s['groups'];names={'subset/dev':'子集模型 / 开发集','subset/diagnostic_seen':'子集模型 / Seen','official/diagnostic_seen':'官方模型 / Seen','subset/diagnostic_novel':'子集模型 / Novel','official/diagnostic_novel':'官方模型 / Novel'}
order=[k for k in names if k in groups]
lines=['# EconomicGrasp：开发前四项验证结果','',
'实验卡：EG-QUALITY-VALIDATION-V1；日期：2026-10-18。分支：[codex/eg-quality-validation-v1-20261018](https://github.com/rollinpioneer/graph/tree/codex/eg-quality-validation-v1-20261018)。',
'','本轮复用冻结模型和预测，没有训练、换种子或开发新模块。下面的 AP 是抽帧诊断指标；每个场景只取 0/64/128/192 四帧，不能作为完整 256 帧标准场景 AP。Top-1 归因另取每场景 64 个等间隔帧。开发集 10 场景，Seen 119/122、Novel 188/189 各两个场景，测试场景均已作为探索性开发材料。','',
'## 1. 实际抓法质量能否准确获得？','',
f"对 {s['total_verified']:,} 个原候选直接调用原生 collision_detection 与 get_grasp_score，质量与碰撞逐个复现保存的官方结果；所有冻结帧的 precision 复算也通过。这样取得的是连续预测姿态在 GraspNet 几何/SDF 评价协议下的质量，不依赖 5 mm 标签匹配。它不是实机成功率；碰撞仍受官方 8 mm 模型采样、接触与摩擦模型限制。",'',
'近邻原标签仍是近似。下表仅统计四个池诊断帧的候选，限定同一原生分配物体、距离≤5 mm；分母不同于 D2 的全部 1024 个原预测，匹配率不可直接横比。合格定义为 μ≤0.4 且有效、无碰撞。','',
'|模型/范围|池候选数|5 mm 匹配率|标签 vs 实际抓法一致率|仅改宽度后的一致率|还原标签点/姿态/宽度后的一致率|',
'|---|---:|---:|---:|---:|---:|']
for k in order:
    q=groups[k]['quality'];f=q['distance_sensitivity']['5'];lines.append(f"|{names[k]}|{q['pool_candidates']}|{100*q['match_fraction']:.2f}%|{100*f['actual']['agreement']:.2f}%|{100*f['width_only']['agreement']:.2f}%|{100*f['exact_cell']['agreement']:.2f}%|")
lines+=['','**碰撞来源差异：开发场景使用已发布的场景碰撞缓存；测试场景没有这类训练标签，本轮仅对所查询的 48 个原标签姿态，用原生几何计算碰撞。测试场景的标签一致率因此属于另一种条件，不能与开发场景合并。** 原抓法真实质量和几何 Best/Actual Oracle 的定义完全相同。','',
'这三个一致率的差值是受控几何替换下的变化，不能逐项相加成为唯一原因分解。完整混淆矩阵、1/2/5/10 mm 敏感性、物体归属差异、角度与宽度偏差保存在 summary.json。无需为未匹配候选伪造“失败”标签。','',
'## 2. Top-1 排序失败与姿态/宽度失配','',
'每个原 Top-1 独立评价：W 只替换为对应原标签宽度；P 保持中心、接近方向和预测宽度，枚举 12 角度×4 深度；J 在同一中心/接近方向枚举 48 个姿态并使用各自标签宽度。J 是有限网格，不是所有可能抓法的穷尽搜索。各项都使用场景真实几何和原生质量评价。','',
'|模型/范围|Top-1 失败/总帧|仅 W 控制可修复|仅 P 控制可修复|W、P 分别均可修复|仅 J 控制可修复|网格内未修复|',
'|---|---:|---:|---:|---:|---:|---:|']
for k in order:
    t=groups[k]['top1'];d=t['categories'];lines.append(f"|{names[k]}|{t['top1_fail']}/{t['frames']}|{d.get('width_only_control',0)}|{d.get('pose_only_control',0)}|{d.get('both_single_controls',0)}|{d.get('joint_only_control',0)}|{d.get('not_repaired_in_grid',0)}|")
lines+=['','这些是“某种几何改变能够修复”的证据，而非经济监督导致错误的因果证据。宽度单独修复与姿态单独修复可能重叠；表中采用互斥分类。网格内未修复不能被直接归因为抓取点错误，可能涉及接近方向、缺失姿态、宽度或评价几何。','',
'|模型/范围|失败 Top-1 中有后续合格候选|匹配≤5 mm 且最佳标签合格|其中几何网格也确认可行|',
'|---|---:|---:|---:|']
for k in order:
    t=groups[k]['top1'];lines.append(f"|{names[k]}|{t['ranking_fail']}|{t['matched5mm_label_optimistic_fail']}|{t['matched5mm_label_optimistic_verified']}|")
lines+=['','## 3. 固定候选 Oracle 与可改善空间','',
'原生 NMS、每物体筛选以及最终候选集合全部冻结。重排时不插入、删除或改变任何候选姿态；最终 AP 始终按原候选的真实实际质量计算，采用 1–50 固定分母和六档摩擦系数。分数相同保留原排序。','',
'Best-48 表示同一候选中心和接近方向下，48 个标签宽度姿态与原姿态中的最佳几何质量；它为“最好能怎么抓”的有限候选代理。Actual 表示该候选原来实际输出的抓法质量。“仅解 Best 并列”保持不同 Best 质量组的先后，仅在同一组内按 Actual 排序。','',
'Best-48 使用原生评价器的六档摩擦质量和标签坐标系角度格点，是几何潜力代理，不是原 Economic 监督的精确重建，也不保证所有替代角度恰落在原解码器格点上。保留原标签 0.1 质量分辨率的查表代理另列；开发集另做原解码器格点控制。','',
'|模型/范围|原排序|Best-48 排序|仅解 Best 并列|Actual 排序|Actual−Best|Actual−原排序|',
'|---|---:|---:|---:|---:|---:|---:|']
for k in order:
    o=groups[k]['oracle'];lines.append(f"|{names[k]}|{o['original']['ap']:.2f}|{o['best']['ap']:.2f}|{o['best_ties_actual']['ap']:.2f}|{o['actual']['ap']:.2f}|{o['actual']['ap']-o['best']['ap']:.2f}|{o['actual']['ap']-o['original']['ap']:.2f}|")
lines+=['','**Actual−Best 是两种完美已知目标的条件差，不能被表述为修改经济监督后一定可获得的 AP。** Actual−原排序仍然是该冻结池的总体排序上限。只纠正评分失配的可训练收益还未被识别：需要在不接触这些评价真值的条件下训练、固定候选和容量做受控比较，本轮没有执行。','',
'几何 Oracle 使用完整物体模型、SDF 和真实物体姿态，这些不是模型正常推理时可直接获得的信息。','',
'原标签查表代理另列如下。此处只重排≤5 mm 的匹配候选所在槽位，所有未匹配候选保持原排名，避免把缺失标签当作失败。','',
'|模型/范围|最佳标签代理|实际姿态标签代理|实际几何质量（同样只改匹配槽位）|',
'|---|---:|---:|---:|']
for k in order:
    o=groups[k]['oracle'];lines.append(f"|{names[k]}|{o['label_best_matched_slots']['ap']:.2f}|{o['label_actual_matched_slots']['ap']:.2f}|{o['actual_matched_slots']['ap']:.2f}|")
lines+=['','## 4. 保留更多标签的实测成本','',
'从开发场景 8、36 各固定抽取 256 个已保留点，保留全部 300 视角，直接使用现有原标签与碰撞标签。最佳分数、角度、深度、宽度与当前官方缓存逐项一致。下表含共享点、视角和 graspness 字段，使用与官方相同的未压缩 NPZ。','',
'|场景|缓存|体积 MB（十进制）|保存 ms|完整读取中位数 ms|',
'|---|---|---:|---:|---:|']
for r in c['rows']:
    for k,v in r['variants'].items():lines.append(f"|{r['scene']}|{k}|{v['bytes']/1e6:.3f}|{v['write_s']*1000:.2f}|{v['median_read_s']*1000:.2f}|")
lines+=['','uint8 版本沿用官方分数量化和毫米宽度量化；float32 保留原始精度。约 10 倍/39 倍是这一实际布局的样本体积比，不是训练速度比。源标签提取耗时分别为 '+', '.join(f"{r['source_extract_s']:.3f} s" for r in c['rows'])+'。这是缓存生成/读取测量，未声称测得新监督加入训练后的吞吐；也不是重新生成全部力闭合标签的成本。首次读取不能保证系统缓存冷启动，未清空共享服务器缓存。','',
'额外的同精度压缩缓存试验见 cost_results.json；它与官方布局不同，因此其 22–23 倍压缩倍率仅为辅助结果。当前磁盘余量不足以直接扩展全量 richer cache，应先解决容量或设计按需缓存，并重新测量；本轮仅保存小样本。','',
'## 资源、局限与下一步','',
f"共复核 {s['total_verified']:,} 个原候选，执行 {s['total_geometry_calls']:,} 个原/对照姿态的质量评价，另对 {s['additional_label_pose_collision_checks']:,} 个原标签姿态按需检查碰撞。四个低优先级单线程 CPU 工作进程，无 GPU 使用；各工作进程墙钟和合计、采样资源、原始产物散列见随附 JSON。单工作进程峰值 RSS {s['max_worker_rss_gib']:.2f} GiB。",'',
'Novel 188 的 109–255 帧、189 的 170–255 帧是重复观测。主表保留冻结帧权重，summary.json 另给按独立观测等权的敏感性结果；这些不是独立新视角。少量固定场景、单训练 seed 和事后探索性诊断均限制外推。','',
'本轮验证可用于决定监督质量表示与受控实验设计，但不证明经济监督是排序失败的主要原因。尤其不能以几何可修复比例或总体 Oracle 上限代替评分失配的可学习收益。下一步若进入开发，应先冻结无测试真值泄漏的监督构建方案，并设置同候选、同容量、仅替换评分监督的对照；不得自动追加训练。','',
'原始结果根目录：`/home/xushijie3/work/eg_quality_validation_v1_20261018`。大数据和模型继续保留在 EG-SUB25 原目录，全部只读复用。','',
'执行异常：调度脚本最初命名 queue.py，与 Python 标准库同名，在任何正式工作进程启动前报错；仅重命名为 run_validation.py 后继续。一次管理指令的自动审批审核超时，拆成小指令重试成功，未改变实验条件。']
lines+=['','八组测试任务最初在读取不存在的测试场景碰撞缓存时停止，尚未开始评价；保留失败日志。改为按需原生碰撞后先做单帧核验，再恢复相同八组任务。开发集已完成结果未重跑，未以空碰撞标签替代。']
sp=json.loads((N/'storage_projection.json').read_text())
lines+=['',f"50 个冻结训练场景共 {sp['points']:,} 个已保留点；当前缓存实测 {sp['existing_bytes']/1e9:.3f} GB。按相同点、300 视角和本轮布局计算，rich48 uint8 负载约 {sp['rich48_uint8_payload_estimate_bytes']/1e9:.2f} GB，float32 约 {sp['rich48_float32_payload_estimate_bytes']/1e9:.2f} GB，尚未生成，未计 ZIP 头、临时副本及峰值空间。"]
dc=json.loads((N/'decoder_control_summary.json').read_text())
dv=groups['subset/dev'];a=dv['quality']['score_associations_frozen_pool'];o=dv['oracle']
lines+=['',f"开发集固定候选池的逐帧平均 Spearman：模型分数对原生 Actual / Best 分别为 {a['actual']['mean_frame_spearman']:.3f} / {a['best']['mean_frame_spearman']:.3f}，对原标签 Actual / Best 代理分别为 {a['label_actual']['mean_frame_spearman']:.3f} / {a['label_best']['mean_frame_spearman']:.3f}。关系随质量定义变化，不能据此确认稳定的最佳质量偏好，更不能确认为训练机制因果。该统计仅适用于已被原生筛选的候选池。",'',
f"同一开发样本上，仅在≤5 mm 匹配槽位按实际姿态的原标签代理重排，相对原排序仅提高 {o['label_actual_matched_slots']['ap']-o['original']['ap']:.2f} AP 点。这是现有查表代理的实际后处理效果，与 {o['actual']['ap']-o['best']['ap']:.2f} 点的几何 Oracle 目标差应严格区分；二者都不是新模型训练结果。"]
lines+=['','## 原解码器动作格点补充控制','',
f"为检查标签坐标系格点是否高估可修复性，对开发集全部 {dc['failed_top1']} 个 Top-1 失败候选，保持原中心、接近方向、宽度，使用原旋转加 k×15° 和原四档深度，包含原动作本身。原动作质量/碰撞逐个再次复现。原解码器格点可修复 {dc['decoder_pose_repaired']} 个，标签坐标系格点则为 {dc['original_label_frame_pose_repaired']} 个；两者不能混同。",'',
f"只换角度可修复 {dc['angle_only']} 个，只换深度可修复 {dc['depth_only']} 个，其中 {dc['both_single']} 个两种单独控制都能修复；另有 {dc['joint_only']} 个需要同时改变角度和深度。这仍是原生几何的反事实可行性检查，不是经济监督或某个预测头的因果归因。补充计算 {dc['quality_calls']:,} 个姿态。"]
ac=json.loads((N/'cost_actual_quality.json').read_text())
ms=[r['ms_per_candidate'] for r in ac['rows']]
lines+=['',f"补充实际质量生成计时：两个开发场景共 8 帧、{sum(r['n'] for r in ac['rows'])} 个原候选，按物体批处理，在已有场景几何与 SDF 下每候选 {min(ms):.1f}–{max(ms):.1f} ms。全部复现官方质量/碰撞。该成本不能与读取现成标签混同，也未验证在每个训练步中在线使用是否可接受。",'',
'标签代理数值处理说明：原标签摩擦系数以 0.1 为步长，官方 AP 阈值以 0.2 为步长。最终标签代理排序按原始质量保留全部 0.1 分辨率；最初工作进程记录的六阈值分桶代理仅保留在原始文件中，汇总时重新读取原标签求精确最小摩擦系数并重算代理排序。原候选几何质量、原生 AP、几何 Best/Actual Oracle 和合格判定未改变。','',
'## 探索性案例','',
'按开发场景编号、帧号，事后选择每类最早一个案例。−1 表示几何评价未通过；非负摩擦系数越小越好。这些案例用于展示控制结果，不作为独立因果证据。','',
'|类别|场景/帧|原实际 μ|只改宽度 μ|只改姿态最佳 μ|联合最佳 μ|',
'|---|---|---:|---:|---:|---:|']
for r in json.loads((N/'exploratory_cases.json').read_text())['cases']:
    lines.append(f"|{r['category']}|{r['scene']:04d}/{r['frame']:04d}|{r['actual']:.1f}|{r['width_only']:.1f}|{r['pose_only']:.1f}|{r['joint']:.1f}|")
(N/'validation_report.md').write_text('\n'.join(lines)+'\n')
# Static scientific figure; labels use English for portable font rendering.
fig,axes=plt.subplots(1,2,figsize=(12,6.3),layout='constrained')
x=np.arange(len(order)); labels=['Subset dev','Subset Seen','Official Seen','Subset Novel','Official Novel']
for i,(variant,label,color) in enumerate([('original','Original','#64748b'),('best','Best-48 target','#f59e0b'),('best_ties_actual','Best ties -> actual','#10b981'),('actual','Actual target','#2563eb')]):
    axes[0].bar(x+(i-1.5)*.19,[groups[k]['oracle'][variant]['ap'] for k in order],.19,label=label,color=color)
axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].set_ylabel('Sampled-frame AP (%)');axes[0].set_title('Fixed candidates; original geometry evaluated');axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2)
cats=[('width_only_control','Width alone'),('pose_only_control','Pose alone'),('both_single_controls','Either alone'),('joint_only_control','Joint only'),('not_repaired_in_grid','Not repaired')]
bottom=np.zeros(len(order))
for cat,label in cats:
    v=np.array([100*groups[k]['top1']['categories'].get(cat,0)/max(1,groups[k]['top1']['top1_fail']) for k in order]);axes[1].bar(x,v,bottom=bottom,label=label);bottom+=v
axes[1].set_xticks(x,labels,rotation=25,ha='right');axes[1].set_ylabel('Share of Top-1 failures (%)');axes[1].set_title('Label-frame pose grid; not decoder lattice');axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2)
fig.savefig(N/'validation_figure.png',dpi=180)
