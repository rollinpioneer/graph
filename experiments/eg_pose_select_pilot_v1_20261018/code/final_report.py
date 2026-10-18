from core import *
import csv
out=R/'results';status=json.loads((out/'pipeline_status.json').read_text());gate=json.loads((out/'gate_calibration.json').read_text());heads=json.loads((out/'selected_heads.json').read_text());d=json.loads((out/'development_decision.json').read_text());rows=list(csv.DictReader(open(out/'development_per_scene.csv')));tau='infinity (no replacement)' if gate['tau'] is None else str(gate['tau'])
lines=['# EG-POSE-SELECT-PILOT-V1 执行报告','',f"状态：{status['status']}。分支 `codex/eg-pose-select-pilot-v1-20261018`。注册提交 `46309a8436c114d7936835e03656470f5132a6d1`。用户明确批准执行；3个41,729参数小头，0次EG重训。",'', '所有拟合/选模/校准/开发及旧12场景均为已观察材料。新确认只在冻结门槛全部通过后打开，不以开发结果冒充独立确认。','', '## 标签、选模与门槛','', '直接原生几何标注32768组×6动作=196608姿态，包含碰撞、空抓、无力闭合与全失败组，不用5mm经济标签mask。输入共享特征按group256/token64排列；仅fit统计归一化。全部标签身份和覆盖统计见label_manifest.json、label_coverage.csv。','', '|种子|选中更新步|model-val场景等权MSE|常数基线MSE|','|---|---:|---:|---:|']
for h in heads:lines.append(f"|{h['seed']}|{h['selected_step']}|{h['metrics']['scene_equal_mse']:.6f}|{h['metrics']['constant_mse']:.6f}|")
lines += ['',f'共同替换门槛：**{tau}**。四档有限门槛按最小可行值选取，三种子各自需场景平均Damage≤2%且Net>0；完整场景分母与每档结果见gate_calibration.json。门槛不是理论安全保证。','', '## 固定开发应用结果','', '|策略（三种子均值）|AP无CD|AP+CD|','|---|---:|---:|']
for policy in ['R0','A3','D2','S4','J6','S4-zero']:
 vals=[]
 for suffix in ['','_CD']:
  names=[policy+suffix] if policy=='R0' else [f'{policy}_{seed}{suffix}' for seed in SEEDS];vals.append(np.mean([float(x['ap']) for x in rows if x['configuration'] in names]))
 lines.append(f'|{policy}|{vals[0]:.3f}|{vals[1]:.3f}|')
lines += ['',f"主效应 S4+CD−R0+CD：**{d['mean_ap_difference_pp']:+.3f} AP pp**，场景95%CI {d['ci95_pp']}；逐种子 {d['seed_ap_difference_pp']}；正向场景 {d['criteria']['positive_scene_count']}/8。",f"同一原R0（无CD原生列表）Top-1身份：每种子N={d['N_per_seed']}，原成功={d['original_success_per_seed']}；R={d['R_by_seed']}，D={d['D_by_seed']}，Net={d['Net']:.5f}，Damage={d['Damage']}。Net使用同一总数N；Damage按原成功条件分母。Net CI={d['Net_ci95']}，Damage CI={d['Damage_ci95']}。",f"完整应用Top-1合格率差={d['application_top1_difference']:+.5f}。这与固定原Top-1身份的转换是不同指标，不能混同。",f"预注册开发放行：**{d['pass_stage']}**；逐项判据：{d['criteria']}。",'', '全部32应用配置均将1024修改后候选重新送入原生选择流程，CD在修改之后重算。原score、中心、接近方向、宽度、高度保持不变；未改全局评分。原生等价微测见evaluation/.../MICRO.json。相同几何配置只复用其完全相同的计算结果，不把变姿态后的几何质量当作原质量复用。','', '## 局部结果、边界与下一步','', '逐场景CSV同时报告AP0.4/AP0.8、实际Top-1质量/碰撞/空抓、修改率、被标注样本的质量改善/下降、原成功质量下降、regret与Oracle覆盖。完整逐帧记录另含高/低原分数分层；每帧固定8个高分和8个低分随机候选，另加入原R0 Top-1并全标六动作。它是有界诊断样本，不宣称对全部1024候选无偏。']
if gate['tau'] is None:lines+=['**没有得到符合本卡保守要求的有限门槛选择器。** τ=∞严格保留原动作，主方法等于EG；这不能当作“安全且有效”。S4-zero保留为门槛消融，不能事后改成主方法。按卡完成固定开发后停止，不运行迁移开发或新确认。']
elif not d['pass_stage']:lines+=['主方法未通过冻结开发门槛，停止本卡；次要策略的好结果仅作探索性记录，不替换主方法，不打开新确认挑选赢家。']
else:lines+=['开发通过，迁移开发/确认状态以pipeline_status.json为准，不提前宣称全卡成功。']
lines+=['本卡同时改变监督支持、局部动作目标、选模和门控条件，不能作为NQS失败的单因素归因。若失败，仅说明当前冻结特征、配方与门槛没有获得所需收益，不否定全部姿态质量学习。若校准有限样本/拟合不足或区间宽，也应保留不确定性。','', '## 成本与归档','', '几何CPU先4后8个低优先级单线程worker。时延与校准区间两个短辅助进程曾与8个评价worker短暂重叠，按全部计算进程口径最多10个，这是本轮调度偏差，不能把总进程上限写成严格8；没有增加训练或评价条件。GPU0仅冻结核验和三个小头训练，记录共享负载。新增产物上限10GiB，探索上限40单线程进程小时/2GPU小时。resource_summary.json同时列worker墙钟累计与CPU秒，两者不混算。原大数据、骨干和NQS特征缓存只读复用。', '轻量归档包括卡片、冻结场景与配置、源身份、实现、选模/门槛记录、逐场景结果及双向案例索引；完整权重、逐帧缓存与日志留服务器并记录散列。未自动添加新模块、超参数、种子或第二套配方。']
lines += ['', '## 为什么当前选择器没有过关', '', '|种子|τ=0 Net|τ=0 Damage|τ=1/12 Net|τ=1/12 Damage|', '|---|---:|---:|---:|---:|']
for seed in SEEDS:
 g0=next(x for x in gate['rows'] if x['seed']==seed and x['tau']==0);g1=next(x for x in gate['rows'] if x['seed']==seed and x['tau']==1/12)
 lines.append(f"|{seed}|{g0['Net']:.3%}|{g0['Damage']:.3%}|{g1['Net']:.3%}|{g1['Damage']:.3%}|")
lines += ['', '下述分解来自已用于校准的gate-cal，仅作事后描述，不作为独立泛化证据。组间误差表示能否识别候选整体可抓性；组内误差更直接对应同一候选的姿态差别。MSE变好并不自动保证后者。', '', '|种子|六动作预测范围中位数|组内方差解释比例|组内相关|', '|---|---:|---:|---:|']
for x in json.loads((out/'action_diagnostic.json').read_text())['rows']:
 lines.append(f"|{x['seed']}|{x['predicted_six_action_range_quantiles'][2]:.5f}|{x['within_group_variance_explained']:.3%}|{x['within_group_pearson']:.4f}|")
lines += ['', '三个种子选中的更新步为100/900/800。第一种子更晚的检查点局部regret更低，但其六动作MSE没有超过第100步；本卡按预注册MSE选模，未事后替换成后期权重。这个现象说明选模目标与实际局部选择效用之间存在值得进一步检验的差别，尚不足以断言是唯一失败原因。', '', '### τ=0消融的双向损益', '', '|种子|AP+CD|原Top-1救回R|损坏D|Net|Damage|', '|---|---:|---:|---:|---:|---:|']
for seed in SEEDS:
 v=[x for x in rows if x['configuration']==f'S4-zero_{seed}_CD'];rr=sum(int(x['R']) for x in v);dd=sum(int(x['D']) for x in v);ns=sum(int(x['original_top_success']) for x in v);nn=sum(int(x['frames']) for x in v)
 lines.append(f"|{seed}|{np.mean([float(x['ap']) for x in v]):.3f}|{rr}|{dd}|{(rr-dd)/nn:.3%}|{dd/ns:.3%}|")
lines += ['', '主方法τ=∞没有救回或损坏案例，按NO_CASE记录；为展示双向风险，另外保存预定第一种子的S4-zero最早scene/frame救回、损坏和未修改失败样例，并明确标记为消融。案例不是无偏的机制证据。可视化展示观测RGB/深度、接近轴与宽度线段，不把示意轴线当作完整手指网格。', '', '本次完整开发为8场景×32固定帧的开发指标，不是官方完整256帧场景AP，更不是完整90场景测试。旧transfer未运行、新确认未打开，不能声称完成独立泛化验证。', '', '时延微测约11 ms/帧（CPU单线程，含六动作头与写回，不含EG/CD/NMS）。由于S1停止，没有运行同硬件EG对应路径的S2配对时延，因此不判定通过轻量门槛。', '', '研究决策：停止当前卡。当前固定特征、MSE目标、选模和共同门槛的组合未得到可用保守选择器；不否定所有姿态质量学习，也不把校准保护成立或局部Oracle空间当成已实现AP收益。若另立实验，优先验证组内动作辨别及选模目标是否匹配实际重选效用；本卡没有启动该后续实验。']
base=[x for x in rows if x['configuration']=='R0_CD'];zero=[x for x in rows if x['configuration'].startswith('S4-zero_') and x['configuration'].endswith('_CD')];base_ap=np.mean([float(x['ap']) for x in base]);zero_ap=np.mean([float(x['ap']) for x in zero]);rr=sum(int(x['R']) for x in zero);dd=sum(int(x['D']) for x in zero)
lines[2:2]=[f'**决策：按S1停止。** 未找到三种子共同满足要求的有限门槛；主S4等于原EG。开发EG+CD AP={base_ap:.3f}；S4-zero三种子均值={zero_ap:.3f}（{zero_ap-base_ap:+.3f} pp），固定原Top-1合计救回{rr}、损坏{dd}。未进入旧transfer或新确认。','']
(out/'final_report.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY')
