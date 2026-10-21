"""Full C3 paired scene statistics, fixed criteria, no checkpoint selection."""
from conditioned import *
from collections import defaultdict
cfg=json.loads((R/'frozen_config.json').read_text());groups=cfg['C3'];records=[];manifest=[]
for split,scenes in groups.items():
 for scene in scenes:
  for frame in range(256):
   p=R/f'evaluation/development/{scene:04d}/{frame:04d}.json';zpath=p.with_suffix('.npz');vv=json.loads(p.read_text());assert len(vv)==16
   with np.load(zpath) as z:
    for v in vv:
     ids=z['selected_'+v['configuration']];mu=z['quality'][ids];assert np.isfinite(mu).all();assert v['top_candidate']==(int(ids[0]) if len(ids) else None);assert bool(len(ids) and 0<mu[0]<=.4000001)==v['top1_valid']
   records.extend(dict(split=split,**v) for v in vv);manifest.extend(dict(path=str(x.relative_to(R)),bytes=x.stat().st_size,sha256=sha(x)) for x in [p,zpath])
assert len(records)==49152;metrics=['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty'];buckets=defaultdict(list)
for v in records:buckets[v['scene'],v['configuration']].append(v)
scene_rows=[];table={}
for (scene,name),vv in sorted(buckets.items()):
 assert len(vv)==256;row=dict(scene=scene,split=vv[0]['split'],configuration=name,frames=256,**{m:float(np.mean([v[m] for v in vv])) for m in metrics});table[scene,name]=row;scene_rows.append(row)
csv_write(R/'results/per_scene_results.csv',scene_rows);names=sorted({v['configuration'] for v in records});absolute=[dict(configuration=name,**{m:float(np.mean([v[m] for v in scene_rows if v['configuration']==name])) for m in metrics}) for name in names];write(R/'results/absolute_results.json',absolute)
families=['EG','O','G','cap','concat','FiLM'];family_results=[]
for fam in families:
 for sp in ['overall',*groups]:
  rr=[v for v in scene_rows if v['configuration'].split('_')[0]==fam and (sp=='overall' or v['split']==sp)];family_results.append(dict(family=fam,split=sp,**{m:float(np.mean([v[m] for v in rr])) for m in metrics}))
csv_write(R/'results/family_results.csv',family_results)
rng=np.random.default_rng(20261021);boot={sp:rng.integers(0,len(ss),(10000,len(ss))) for sp,ss in groups.items()};comparisons=[];deltas=[]
for fam,ref in [('cap','G'),('concat','G'),('FiLM','G'),('concat','cap'),('FiLM','concat'),('FiLM','cap')]:
 result=dict(family=fam,reference=ref,use_cd=True,metrics={})
 for m in metrics:
  by={sp:np.array([[table[s,f'{fam}_{seed}_CD'][m]-table[s,f'{ref}_{seed}_CD'][m] for s in scenes] for seed in SEEDS]) for sp,scenes in groups.items()};scale=1 if m.startswith('ap') else 100;dist=np.mean([v.mean(0)[boot[sp]].mean(1) for sp,v in by.items()],axis=0)*scale
  result['metrics'][m]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(dist,[.025,.975]).tolist(),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),split_delta={sp:float(v.mean()*scale) for sp,v in by.items()},scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,scenes in groups.items() for i,s in enumerate(scenes)},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())))
 comparisons.append(result)
 for sp,scenes in groups.items():
  for s in scenes:deltas.append(dict(family=fam,reference=ref,split=sp,scene=s,**{m:result['metrics'][m]['scene_delta'][str(s)] for m in metrics}))
write(R/'results/comparisons.json',comparisons);csv_write(R/'results/per_scene_differences.csv',deltas)
decisions={}
for fam in ['cap','concat','FiLM']:
 v=next(c['metrics'] for c in comparisons if c['family']==fam and c['reference']=='G');ap=v['ap'];top=v['top1_valid'];ap_route=ap['delta']>=.5 and ap['ci95'][0]>0 and top['ci95'][0]>=-.5 and sum(d>0 for d in ap['seed_delta'])>=2;top_route=top['delta']>=1 and top['ci95'][0]>0 and ap['ci95'][0]>=-.2 and sum(d>0 for d in top['seed_delta'])>=2
 decisions[fam]=dict(pass_frozen_gate=bool(ap_route or top_route),AP_route=bool(ap_route),Top1_route=bool(top_route))
passing=[f for f,v in decisions.items() if v['pass_frozen_gate']];decision=dict(status='DEVELOPMENT_PASS' if passing else 'NO_GROUP_PASSES_KEEP_G',groups=decisions,passing_groups=passing,C3_is_development=True,C4_read=False,independent_confirmation=False,additional_training_started=False)
write(R/'results/development_decision.json',decision)
train=json.loads((R/'results/training_complete.json').read_text());ev=json.loads((R/'results/evaluation_complete.json').read_text());assert len(train['heads'])==9 and ev['frames']==3072
for v in train['heads']:assert sha(R/v['checkpoint_relative'])==v['checkpoint_sha256'] and v['step']==1000 and v['frozen_G_unchanged'] and v['reload_scoring_pass']
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'archive_worktree' not in p.parts);assert size<4*1024**3
write(R/'results/evaluation_artifact_manifest.json',manifest);write(R/'results/final_validation.json',dict(status='PASS',frames=3072,configurations=16,records=49152,heads=9,effective_training_updates=9000,executed_training_updates=12000,old_EG_O_G_selection_and_metrics_exact=True,all_selected_quality_finite=True,new_training_quality_labels=0,EG_forward=0,C4_read=False,CPU_worker_hours_conservative=ev['CPU_worker_seconds_conservative']/3600,GPU_successful_training_process_hours=train['GPU_process_seconds_successful']/3600,failed_training_cost_reported_separately=True,actual_CPU_peak=ev['peak_running_processes_observed'],successful_training_overlap_peak=train['successful_training_overlap_peak'],new_runtime_bytes=size))
lines=['# EG-G-CONDITIONED-V1：完整C3开发报告','',f"状态：{decision['status']}。9个适配器各1000步，完整C3 3072帧、主口径带CD。",'', '| 方法 | AP | AP0.4 | AP0.8 | Top-1合格率% | 非空碰撞% | 空抓% |','|---|---:|---:|---:|---:|---:|---:|']
for f in family_results:
 if f['split']=='overall':lines.append(f"| {f['family']} | {f['ap']:.3f} | {f['ap04']:.3f} | {f['ap08']:.3f} | {100*f['top1_valid']:.3f} | {100*f['top1_nonempty_collision']:.3f} | {100*f['top1_empty']:.3f} |")
lines+=['','## 配对比较','', '| 比较 | AP差 [95%CI] | Top-1差pp [95%CI] | AP正向场景 | Top-1正向场景 |','|---|---:|---:|---:|---:|']
for c in comparisons:
 a=c['metrics']['ap'];t=c['metrics']['top1_valid'];lines.append(f"| {c['family']}−{c['reference']} | {a['delta']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}] | {t['delta']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] | {a['positive_scenes']}/12 | {t['positive_scenes']}/12 |")
lines+=['','## 冻结判断','',json.dumps(decision,ensure_ascii=False,indent=2),'','三个种子先在场景内平均，Seen/Similar/Novel各4场景分层配对bootstrap10000次。逐种子、分组及逐场景差值完整保留；不按结果挑种子或调整融合。','', 'C3已经用于方法开发，当前结果不是独立确认；多组开发比较未做家族错误率校正。C4未读取。只有通过冻结门槛的组才执行固定96帧成本评价。属性组超过容量对照才能支持属性增量；FiLM超过concat才能支持调制增量。','', '## 训练与评价核验','', '原G权重全冻结、原六姿态MSE、Top64+随机64候选与混合批次采样流完全保留，三个组固定step1000。14维属性仅从fit计算归一化，六姿态逐动作构造。原1024候选姿态和F05/CD/NMS不变。新入选动作的真实质量只用于官方评价，不回流训练。','', '独立小规模评价在EG、O、G及三个新组各一个种子上对照未修改官方API，选择、质量、碰撞和AP均一致。全部3072帧重复核验旧EG/O/G选择和指标完全一致；全部新入选候选质量有限。','', '## 技术事件与资源','', '跨节点NFS输出目录可见性竞争导致seed20261018三组在1000步后无法保存。保留6个已成功模型，只将3个失败任务按相同初始化重放；最终9模型各1000步，实际共12000次更新，额外3000次为失败保存开销。三个重放任务每100步记录的损失均与首次运行完全一致。没有依据评价结果重训或选模。','', f"成功训练任务累计{train['GPU_process_seconds_successful']:.2f}进程秒；失败保存任务的精确耗时未记录，此成功耗时不代表全部GPU成本。完整评价实际观察峰值{ev['peak_running_processes_observed']}进程，CPU累计保守{ev['CPU_worker_seconds_conservative']/3600:.3f}worker小时。新增运行产物{size/1024**3:.3f}GiB。时间限制已由用户解除，4GiB存储限制保留。",'', '详见results/comparisons.json、family_results.csv、per_scene_results.csv、per_scene_differences.csv、development_decision.json、final_validation.json及evaluation_artifact_manifest.json。完整权重与逐帧缓存留服务器。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision),flush=True)
