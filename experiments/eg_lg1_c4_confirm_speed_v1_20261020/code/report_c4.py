from common import *
from collections import defaultdict,Counter
import datetime
cfg=json.load(open(R/'frozen_config.json'));groups=cfg['C4'];lock=json.load(open(R/'method_lock.json'));ev=json.load(open(R/'results/evaluation_complete.json'));assert ev['status']=='PASS';records=[];manifest=[]
assert sha(R/'frozen_config.json')==lock['frozen_config_sha256']
for v in lock['new_heads']+lock['old_O_heads']:assert sha(v['path'])==v['sha256']
assert sha(lock['backbone_path'])==lock['backbone_sha256'];assert sha(R/'code/geometry_core.py')==lock['geometry_core_sha256']
for split,scenes in groups.items():
 for scene in scenes:
  for frame in range(256):
   p=R/f'evaluation/confirmation/{scene:04d}/{frame:04d}.json';vv=json.load(open(p));assert len(vv)==14
   with np.load(p.with_suffix('.npz')) as z:
    for v in vv:
     ids=z['selected_'+v['configuration']];mu=z['quality'][ids];assert np.isfinite(mu).all();assert v['method_lock_sha256']==sha(R/'method_lock.json')
     assert v['top_candidate']==(int(ids[0]) if len(ids) else None);assert v['top1_valid']==bool(len(ids) and 0<mu[0]<=.4000001)
   records.extend(dict(split=split,**v) for v in vv)
   for q in [p,p.with_suffix('.npz'),R/f'geometry/C4/{scene:04d}/{frame:04d}.npz',R/f'exports/scene_{scene:04d}/{frame:04d}.npz']:manifest.append(dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q)))
assert len(records)==43008;csv_write(R/'results/confirmation_per_frame.csv',records)
metrics=['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty','top1_utility','top5_valid'];buckets=defaultdict(list)
for v in records:buckets[v['scene'],v['configuration']].append(v)
scene_rows=[];table={}
for (scene,name),vv in sorted(buckets.items()):
 assert len(vv)==256;row=dict(scene=scene,split=vv[0]['split'],configuration=name,frames=256,**{m:float(np.mean([v[m] for v in vv])) for m in metrics});den=sum(v['selected_top50_count'] for v in vv);row['top50_nonempty_collision']=sum(v['selected_top50_nonempty_collisions'] for v in vv)/den if den else 0.;table[scene,name]=row;scene_rows.append(row)
metrics+=['top50_nonempty_collision'];csv_write(R/'results/per_scene_results.csv',scene_rows)
names=sorted({v['configuration'] for v in records});absolute=[dict(configuration=name,**{m:float(np.mean([v[m] for v in scene_rows if v['configuration']==name])) for m in metrics}) for name in names]
write(R/'results/absolute_results.json',absolute)
family_absolute=[]
for family in ['EG','O','G']:
 for cd in [True,False]:
  vals=[v for v in absolute if v['configuration'].split('_')[0]==family and v['configuration'].endswith('_CD')==cd];family_absolute.append(dict(family=family,use_cd=cd,**{m:float(np.mean([v[m] for v in vals])) for m in metrics}))
write(R/'results/family_absolute.json',family_absolute)
rng=np.random.default_rng(20261020);boot={sp:rng.integers(0,4,(10000,4)) for sp in groups};comparisons=[];deltas=[]
for fam,ref in [('G','O'),('G','EG'),('O','EG')]:
 for cd in [True,False]:
  suffix='_CD' if cd else '';result=dict(family=fam,reference=ref,use_cd=cd,metrics={})
  for m in metrics:
   by={sp:np.array([[table[s,f'{fam}_{seed}{suffix}'][m]-table[s,('EG' if ref=='EG' else f'{ref}_{seed}')+suffix][m] for s in scenes] for seed in SEEDS]) for sp,scenes in groups.items()};scale=1 if m.startswith('ap') else 100;dist=np.mean([v.mean(0)[boot[sp]].mean(1) for sp,v in by.items()],axis=0)*scale
   result['metrics'][m]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(dist,[.025,.975]).tolist(),one_sided95_low=float(np.quantile(dist,.05)),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),split_delta={sp:float(v.mean()*scale) for sp,v in by.items()},scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,scenes in groups.items() for i,s in enumerate(scenes)},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())))
  comparisons.append(result)
  for sp,scenes in groups.items():
   for s in scenes:deltas.append(dict(family=fam,reference=ref,use_cd=cd,split=sp,scene=s,**{m:result['metrics'][m]['scene_delta'][str(s)] for m in metrics}))
write(R/'results/comparisons.json',comparisons);csv_write(R/'results/per_scene_differences.csv',deltas)
go=next(v['metrics'] for v in comparisons if v['family']=='G' and v['reference']=='O' and v['use_cd']);checks=dict(AP_noninferiority=go['ap']['ci95'][0]>=-.5,Top1_superiority=go['top1_valid']['one_sided95_low']>0)
decision=dict(status='CONFIRMATION_PASS' if all(checks.values()) else 'CONFIRMATION_CRITERIA_NOT_MET',checks=checks,criteria_frozen_before_C4=True,AP_superiority_secondary=go['ap']['ci95'][0]>0,independent_within_known_registered_history=True,no_tuning=True,new_training=0,new_training_quality_labels=0,scenes=12,frames=3072,configurations=14,records=43008,source_lock_sha256=sha(R/'method_lock.json'))
write(R/'results/confirmation_decision.json',decision)
lookup={(v['scene'],v['frame'],v['configuration']):v for v in records};transitions=[]
for fam,ref in [('G','O'),('G','EG'),('O','EG')]:
 for seed in SEEDS:
  counts=Counter((lookup[s,f,('EG' if ref=='EG' else f'{ref}_{seed}')+'_CD']['top1_category'],lookup[s,f,f'{fam}_{seed}_CD']['top1_category']) for s in sum(groups.values(),[]) for f in range(256))
  transitions.extend(dict(family=fam,reference=ref,seed=seed,old_category=a,new_category=b,count=n) for (a,b),n in sorted(counts.items()))
csv_write(R/'results/top1_category_transitions.csv',transitions);write(R/'results/server_artifact_index.json',manifest)
# One additional frozen Novel frame checked through the completely unmodified official evaluator separately.
checks_native=[]
for path in sorted((R/'results').glob('eval_*_profile.json')):
 v=json.load(open(path));assert v['status']=='PASS' and all(all(c[k] for k in ['selection','quality','collision','AP']) for c in v['native_checks']);checks_native.extend(v['native_checks'])
write(R/'results/final_C4_validation.json',dict(status='PASS',frames=3072,records=len(records),native_checks=checks_native,all_selected_quality_finite=True,weights_unchanged=True,geometry_core_unchanged=True,official_definition_unchanged=True,unknown_not_failure=True,resources=ev))
lines=['# EG-LG1 C4独立确认','',f"状态：{decision['status']}。已按事先冻结的方法完成12场景3072帧、EG及O/G各三个头，CD为主口径。",'', '| 方法 | AP | AP04 | AP08 | Top1合格率 | 非空碰撞率 |','|---|---:|---:|---:|---:|---:|']
for v in family_absolute:
 if v['use_cd']:lines.append(f"| {v['family']} | {v['ap']:.4f} | {v['ap04']:.4f} | {v['ap08']:.4f} | {100*v['top1_valid']:.4f}% | {100*v['top1_nonempty_collision']:.4f}% |")
lines+=['','G−O：',f"AP {go['ap']['delta']:+.4f}，95%CI {go['ap']['ci95']}；Top1 {go['top1_valid']['delta']:+.4f}pp，95%CI {go['top1_valid']['ci95']}，单侧95%下界 {go['top1_valid']['one_sided95_low']:+.4f}pp。",f"共同主判据：{json.dumps(checks)}。AP优效为辅助结果，不能替代共同主判据。",'', '## 场景差异','',f"三种子AP差：{go['ap']['seed_delta']}；Top1差pp：{go['top1_valid']['seed_delta']}。",f"Seen/Similar/Novel AP差：{go['ap']['split_delta']}；Top1差pp：{go['top1_valid']['split_delta']}。",f"AP正向场景 {go['ap']['positive_scenes']}/12；Top1正向场景 {go['top1_valid']['positive_scenes']}/12。",'', '## 可信范围与限制','', '沿用预留C4身份清单，并在打开前冻结权重/几何/F05/统计。已知登记历史审计通过，无法排除未登记外部历史。12个场景、单骨干B、单相机Kinect；三头先平均后进行分层场景配对bootstrap，不把3072帧或三个种子当独立场景样本。此为固定12场景确认，不是完整官方90场景AP，也不是真实机械臂成功率。', '所有候选姿态、位置、宽度一致。官方NMS、CD与摩擦/AP定义保持原样；新入选候选真实质量只作测试评价，未回流训练。C4不用于任何模型/融合/几何参数选择。确认结果冻结后再开展数值不变的加速。', '', '## 实际资源','',f"GPU峰值{ev['peak_GPU_processes']}进程，累计{ev['GPU_process_seconds']/3600:.4f}进程小时；CPU峰值{ev['peak_CPU_processes']}评价worker，保守累计{ev['CPU_worker_seconds_conservative']/3600:.4f}worker小时。共享负载见GPU_dispatch.json；首帧回执路径错误修复未重算预测、未改变算法。", '', '逐场景及逐头指标、转移表和轻量配对统计位于results；完整逐帧预测/几何/评价缓存留服务器，通过server_artifact_index.json核验。']
(R/'C4_CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
write(R/'results/C4_result_freeze.json',dict(status='FROZEN_BEFORE_ACCELERATION',time=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha(R/'C4_CONFIRMATION_REPORT.md'),comparisons_sha256=sha(R/'results/comparisons.json'),decision_sha256=sha(R/'results/confirmation_decision.json')))
print(json.dumps(decision),flush=True)
