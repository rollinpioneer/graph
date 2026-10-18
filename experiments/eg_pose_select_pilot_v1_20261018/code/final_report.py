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
lines += ['',f"主效应 S4+CD−R0+CD：**{d['mean_ap_difference_pp']:+.3f} AP pp**，场景95%CI {d['ci95_pp']}；逐种子 {d['seed_ap_difference_pp']}；正向场景 {d['criteria']['positive_scene_count']}/8。",f"同一原R0 Top-1身份：每种子N={d['N_per_seed']}，原成功={d['original_success_per_seed']}；R={d['R_by_seed']}，D={d['D_by_seed']}，Net={d['Net']:.5f}，Damage={d['Damage']}。Net使用同一总数N；Damage按原成功条件分母。Net CI={d['Net_ci95']}，Damage CI={d['Damage_ci95']}。",f"完整应用Top-1合格率差={d['application_top1_difference']:+.5f}。这与固定原Top-1身份的转换是不同指标，不能混同。",f"预注册开发放行：**{d['pass_stage']}**；逐项判据：{d['criteria']}。",'', '全部32应用配置均将1024修改后候选重新送入原生选择流程，CD在修改之后重算。原score、中心、接近方向、宽度、高度保持不变；未改全局评分。原生等价微测见evaluation/.../MICRO.json。相同几何配置只复用其完全相同的计算结果，不把变姿态后的几何质量当作原质量复用。','', '## 局部结果、边界与下一步','', '逐场景CSV同时报告AP0.4/AP0.8、实际Top-1质量/碰撞/空抓、修改率、被标注样本的质量改善/下降、原成功质量下降、regret与Oracle覆盖。完整逐帧记录另含高/低原分数分层；每帧固定8个高分和8个低分随机候选，另加入原R0 Top-1并全标六动作。它是有界诊断样本，不宣称对全部1024候选无偏。']
if gate['tau'] is None:lines+=['**没有得到符合本卡保守要求的有限门槛选择器。** τ=∞严格保留原动作，主方法等于EG；这不能当作“安全且有效”。S4-zero保留为门槛消融，不能事后改成主方法。按卡完成固定开发后停止，不运行迁移开发或新确认。']
elif not d['pass_stage']:lines+=['主方法未通过冻结开发门槛，停止本卡；次要策略的好结果仅作探索性记录，不替换主方法，不打开新确认挑选赢家。']
else:lines+=['开发通过，迁移开发/确认状态以pipeline_status.json为准，不提前宣称全卡成功。']
lines+=['本卡同时改变监督支持、局部动作目标、选模和门控条件，不能作为NQS失败的单因素归因。若失败，仅说明当前冻结特征、配方与门槛没有获得所需收益，不否定全部姿态质量学习。若校准有限样本/拟合不足或区间宽，也应保留不确定性。','', '## 成本与归档','', 'CPU先4后8个低优先级单线程worker；上限8。GPU0仅冻结核验和三个小头训练，记录共享负载。新增产物上限10GiB，探索上限40单线程进程小时/2GPU小时。resource_summary.json同时列worker墙钟累计与CPU秒，两者不混算。原大数据、骨干和NQS特征缓存只读复用。', '轻量归档包括卡片、冻结场景与配置、源身份、实现、选模/门槛记录、逐场景结果及双向案例索引；完整权重、逐帧缓存与日志留服务器并记录散列。未自动添加新模块、超参数、种子或第二套配方。']
(out/'final_report.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY')
