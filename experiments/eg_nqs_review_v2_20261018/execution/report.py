"""Write a faithful report after the registered development gate is evaluated."""
import csv
from common import *
out=R/'results';d=json.loads((out/'subset_development_decision.json').read_text())
fits=list(csv.DictReader(open(out/'subset_fit_summary.csv')))
coverage=list(csv.DictReader(open(out/'subset_label_coverage.csv')))
cost=json.loads((out/'label_cost_summary.json').read_text())
environment=json.loads((out/'environment_receipt.json').read_text())
case_index=json.loads((out/'bad_case_index.json').read_text());cases=json.loads((out/'bad_case_details.json').read_text())
seeds=[20261018,20261019,20261020];summary={v['configuration']:v for v in d['summary']}
def avg(arm,suffix,metric='ap'):
    return summary['R0'+suffix][metric] if arm=='R0' else float(np.mean([summary[f'{arm}_{s}{suffix}'][metric] for s in seeds]))
export=[json.loads(p.read_text()) for p in (R/'receipts').glob('export_subset_*.json')]
labels=[json.loads(p.read_text()) for p in (R/'receipts').glob('label_subset_*_full.json')]
micro=[json.loads(p.read_text()) for p in (R/'receipts').glob('label_subset_*_micro.json')]
fit_receipts=[json.loads((R/f'receipts/fit_subset_{s}.json').read_text()) for s in seeds]
eval_receipts=[json.loads((R/f'evaluation/subset/scene_{s:04d}/COMPLETE.json').read_text()) for s in SPLIT['probe_development']]
validation=json.loads((R/'receipts/eval_cache_validation_subset.json').read_text())
resource=dict(export_worker_seconds=sum(x['wall_seconds'] for x in export),label_worker_seconds=sum(x['wall_seconds'] for x in labels),label_micro_worker_seconds=sum(x['wall_seconds'] for x in micro),fit_worker_seconds=sum(x['wall_seconds'] for x in fit_receipts),evaluation_worker_seconds=sum(x['wall_seconds'] for x in eval_receipts)+validation['wall_seconds'],peak_export_allocated_mib=max(x['cuda_peak'] for x in export)/2**20,peak_fit_allocated_mib=max(x['cuda_peak'] for x in fit_receipts)/2**20,full_label_quality_calls=sum(x['quality_calls'] for x in labels),cached_development_quality_calls=sum(x['quality_calls'] for x in eval_receipts),validation_seconds=validation['wall_seconds'],label_queue_wall_seconds=json.loads((R/'receipts/label_queue_subset.json').read_text())['wall_seconds'])
resource['gpu_process_hours']=(resource['export_worker_seconds']+resource['fit_worker_seconds'])/3600
resource['total_worker_hours']=sum(resource[k] for k in ['export_worker_seconds','label_worker_seconds','label_micro_worker_seconds','fit_worker_seconds','evaluation_worker_seconds'])/3600
resource['accounting_scope']='Successful computation and microchecks; initial failed dtype export and orchestration not included; runtime wall occupancy per worker, not hardware utilization-integrated time.'
resource['disk_bytes']={n:sum(p.stat().st_size for p in (R/n).rglob('*') if p.is_file()) for n in ['inputs','exports','labels','micro_labels','heads','evaluation','logs','bad_cases']}
resource['inference_seconds']={k:float(fn([v['inference_seconds'] for x in export for v in x['rows']])) for k,fn in [('mean',np.mean),('median',np.median),('p95',lambda x:np.quantile(x,.95))]}
write_json(out/'resource_summary.json',resource)
head='EG-NQS-PILOT-V2 执行报告'
lines=[f'# {head}','','分支：`codex/eg-nqs-pilot-v2-20261018`；登记提交 `52496bb04beefac6a67c87f36b800aa352b419a8`；执行前代码提交 `25bc3c1877575dc8dbd5b90b67b7d12939e6375a`。','',f"状态：{'开发门槛通过，尚需条件确认结果' if d['continue_confirmation'] else '探索阶段完成；两项开发门槛均未通过，依卡停止后续确认'}。本轮不是完整 EconomicGrasp 重训，也不是完整官方90场景评价。",'',
'## 实际执行','',
'冻结既有25%训练骨干；24场景×8帧拟合、8场景×8帧校准；八个开发场景×32帧评价。三组 E/B6/A × 三个配对种子，每头固定1000次更新，共九个头，骨干更新次数为0。每帧同一1024候选，所有20种配置重新执行原生选择流程；固定R0候选池结果另报。','',
'在新账号运行，复用数据和校验过的权重。两张空闲GPU并行导出、三张空闲GPU并行三个种子的头拟合，CPU最多八个低优先级单线程场景worker。未改动其他任务。主机身份和绝对数据路径保存在内部交接，公开材料只使用相对路径与环境变量。','',
'## 关键结果','',
'AP单位为百分数，提升单位为百分点。这里的 AP 是抽帧开发集上的原生指标，不是全量官方 AP，也不是机械臂成功率。三头组数值为三个种子的均值。','',
'|评分|无CD AP|带CD AP|无CD固定池AP|带CD固定池AP|','|---|---:|---:|---:|---:|']
for arm in ['R0','E','B6','A']:lines.append(f"|{arm}|{avg(arm,''):.4f}|{avg(arm,'_CD'):.4f}|{avg(arm,'','fixed_pool_ap'):.4f}|{avg(arm,'_CD','fixed_pool_ap'):.4f}|")
lines+=['','|预注册对照|平均差 AP pp|三个种子差|正向场景|开发门槛|','|---|---:|---|---:|---|']
for name,key in [('目标 A−B6，无CD','A-B6'),('实用 A+CD−R0+CD','A-R0_CD')]:
    x=d['effects'][key];lines.append(f"|{name}|{x['mean_pp']:+.4f}|{', '.join(f'{v:+.4f}' for v in x['seed_mean_pp'])}|{x['positive_scenes']}/8|{'通过' if x['pass_gate'] else '未通过'}|")
lines+=['','门槛要求平均≥1 pp、三个种子方向全正、至少6/8场景为正；不能用其他对照代替。场景bootstrap区间在JSON中供探索性描述，开发集区间不作为独立确认。所有种子、场景和配置保存在CSV，未挑选最佳头。','',
'## 可信性与限制','',
'1. 特征hook与同环境同输入的无hook推理逐元素一致；两个骨干权重均核验SHA256。初次导出因标签索引dtype不符失败，按照官方加载器改成int32后重跑，失败日志保留。',
'2. 八个微测帧的397个原始选中抓法，真实几何质量与碰撞复现官方结果；开发四帧×四种排序×CD/no-CD共32组，选择、质量、碰撞与AP逐项完全一致后才启用缓存。',
'3. B6包含原始动作，六种姿态仅改变角度/深度，中心、接近方向、最终宽度和高度固定；全样本B6≥A。推理仅修改分数，标签/评价对象身份不进入评分输入。',
'4. 原卡“top-60”系误读过时注释，执行前勘误为固定源码及现有缓存实际使用的300视角。三组都使用未修改官方匹配函数返回的共同mask；未按结果收放阈值。原始登记卡保留作对照。',
'5. 同一种子三组初始化哈希和采样顺序哈希一致；九个头都能精确重载；只评价第1000步。不同骨干不是随机种子重复，本轮三个种子仅代表评分头。','',
'|用途|预采样候选|匹配候选|覆盖率|B6>A比例（匹配内）|平均B6−A效用|','|---|---:|---:|---:|---:|---:|']
for name,key in [('fit','probe_fit'),('calibration','probe_calibration')]:
    x=cost[key];lines.append(f"|{name}|{x['requested']}|{x['matched']}|{x['matched']/x['requested']:.2%}|{x['B6_greater_A']/x['matched']:.2%}|{x['mean_utility_gap']:.4f}|")
zeros=[x['scene'] for x in coverage if int(x['matched'])==0]
lines+=['',f"低覆盖且跨场景不均：零匹配场景为 {', '.join(zeros)}。没有补抽或更换场景。这与先前原始抓取标签最近邻覆盖率不同，此处使用经济标签保留点与官方视角匹配。结果只支持这个共同支持集训练出的有限预算评分头，不能推出经济监督是或不是总体排序错误的主要原因。",'',
'|组/种子|train MSE|cal MSE|常数cal MSE|','|---|---:|---:|---:|']
for x in fits:lines.append(f"|{x['arm']}/{x['seed']}|{float(x['train_mse']):.5f}|{float(x['cal_mse']):.5f}|{float(x['constant_cal_mse']):.5f}|")
lines+=['','所有头的校准MSE优于各自常数基线，但后期训练MSE继续下降、校准误差上升，存在过拟合迹象。未改训练步数、未挑早停点、未追加种子。因此阴性结果不能证明特征不含信息，也不能排除配方、支持覆盖和域差异的影响。','',
'## 成本','',
f"成功阶段累计worker时间 {resource['total_worker_hours']:.3f} 小时；GPU进程占用合计 {resource['gpu_process_hours']:.4f} 小时。CPU与GPUworker墙钟之和用于保守计量，不等同于作业总墙钟。初次失败导出与人工编排开销另记日志，未纳入此合计。",
f"正式六姿态标签共 {resource['full_label_quality_calls']} 次几何质量查询，worker累计 {resource['label_worker_seconds']:.1f} 秒；八worker标签队列墙钟 {resource['label_queue_wall_seconds']:.1f} 秒。导出GPU峰值allocated {resource['peak_export_allocated_mib']:.1f} MiB，头训练峰值 {resource['peak_fit_allocated_mib']:.1f} MiB，非整卡显存占用。",
f"fit+cal标签文件 {(cost['probe_fit']['label_bytes']+cost['probe_calibration']['label_bytes'])/2**20:.2f} MiB；一次warm-cache读取 {cost['probe_fit']['warm_npz_read_seconds']+cost['probe_calibration']['warm_npz_read_seconds']:.3f} 秒。该成本仅适用于此固定抽样与共同mask，不能外推为完整密集姿态监督的成本。",
f"模型导出单帧forward+decode计时：均值 {resource['inference_seconds']['mean']:.4f} 秒，中位数 {resource['inference_seconds']['median']:.4f} 秒；不含场景加载/质量评价。详细磁盘与阶段计时见resource_summary.json。",'',
'## Bad Case','',
'仅从已完成开发材料选取，固定使用首个登记种子20261018，并按scene/frame最小选例；不是最佳种子或最大提升案例。分别给带/不带CD的三种现象。图中展示中心、接近方向和开口投影，真实碰撞结论来自原生评价。官方确认未执行时不伪造Seen/Similar/Novel案例。','',
'|类别|场景/帧|R0 Top-1 μ|A Top-1 μ|','|---|---|---:|---:|']
for x in cases:
    tops=[v for v in x['rows'] if v['rank']==1];lines.append(f"|{x['category']}|{x['scene']:04d}/{x['frame']:04d}|{tops[0]['mu']:.1f}|{tops[1]['mu']:.1f}|")
lines+=['','μ≤0.4且为正作为案例合格定义，负值表示无有效质量。原RGB/深度、预测姿态、候选分数、μ/碰撞/空抓和场景AP保存在bad_cases目录。上述现象只能说明重排选择发生变化，不能直接证明某个模块的机制原因。','',
'## 研究决定','']
if d['continue_confirmation']:lines+=['开发门槛已通过；本文件为阶段结果，待完成冻结官方骨干复核与12场景确认后另交最终结论。']
else:lines+=['依卡停止：不启动官方骨干九头训练，不开启12场景确认，不增加训练数据、种子或姿态重选。结论是本次有限预算设置未满足继续投入的预注册门槛；不能声称证明经济监督无问题，也不能声称修正评分失配没有潜力。下一步如需继续，应另立实验卡优先检查共同支持覆盖和跨场景过拟合，先确认这些限制是否主导结果。']
lines+=['','## 文件与复验','',
'`subset_per_frame.csv`/`subset_per_scene.csv`/`subset_configuration_summary.csv`：完整20配置；`subset_development_decision.json`：冻结门槛计算；`subset_fit_summary.csv`：九头拟合与checkpoint哈希；`subset_label_coverage.csv`：逐场景标签覆盖；`environment_receipt.json`：依赖、GPU负载与源码哈希；`resource_summary.json`：成本；`bad_case_details.json`：可追溯案例。完整缓存、checkpoint和日志留在实验目录，公开归档仅保留轻量结果与可移植代码。']
(out/'final_experiment_report.md').write_text('\n'.join(lines)+'\n')
manifest={}
for p in sorted(out.iterdir()):
    if p.is_file() and p.name!='results_manifest.json':manifest[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
write_json(out/'results_manifest.json',manifest)
print(json.dumps(dict(continue_confirmation=d['continue_confirmation'],worker_hours=resource['total_worker_hours'],gpu_hours=resource['gpu_process_hours'],report='results/final_experiment_report.md')))
