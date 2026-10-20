from common import *
from collections import defaultdict,Counter
t=time.time();cfg=json.load(open(R/'frozen_config.json'));groups=cfg['C3'];records=[];manifest=[]
for split,scenes in groups.items():
 for scene in scenes:
  for frame in range(256):
   p=R/f'evaluation/development/{scene:04d}/{frame:04d}.json';zz=p.with_suffix('.npz');vv=json.load(open(p));assert len(vv)==20
   with np.load(zz) as z:
    for v in vv:
     ids=z['selected_'+v['configuration']];mu=z['quality'][ids];assert np.isfinite(mu).all();assert v['top_candidate']==(int(ids[0]) if len(ids) else None);assert bool(len(ids) and 0<mu[0]<=.4000001)==v['top1_valid']
   records.extend(dict(split=split,**v) for v in vv);manifest.extend(dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in [p,zz])
assert len(records)==61440;csv_write(R/'results/development_per_frame.csv',records)
metrics=['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty','top1_utility','top5_valid'];buckets=defaultdict(list)
for v in records:buckets[v['scene'],v['configuration']].append(v)
scene_rows=[];table={}
for (scene,name),vv in sorted(buckets.items()):
 assert len(vv)==256;row=dict(scene=scene,split=vv[0]['split'],configuration=name,frames=256,**{m:float(np.mean([v[m] for v in vv])) for m in metrics});den=sum(v['selected_top50_count'] for v in vv);row['top50_nonempty_collision']=sum(v['selected_top50_nonempty_collisions'] for v in vv)/den if den else 0.;table[scene,name]=row;scene_rows.append(row)
metrics+=['top50_nonempty_collision'];csv_write(R/'results/per_scene_results.csv',scene_rows);names=sorted({v['configuration'] for v in records});absolute=[dict(configuration=name,**{m:float(np.mean([v[m] for v in scene_rows if v['configuration']==name])) for m in metrics}) for name in names];write(R/'results/absolute_results.json',absolute)
rng=np.random.default_rng(20261020);boot={sp:rng.integers(0,4,(10000,4)) for sp in groups};comparisons=[];deltas=[]
for fam,ref in [('G','O'),('G','Gshuffle'),('Gshuffle','O'),('G','EG'),('O','EG')]:
 for cd in [True,False]:
  suffix='_CD' if cd else '';result=dict(family=fam,reference=ref,use_cd=cd,metrics={})
  for m in metrics:
   by={sp:np.array([[table[s,f'{fam}_{seed}{suffix}'][m]-table[s,('EG' if ref=='EG' else f'{ref}_{seed}')+suffix][m] for s in scenes] for seed in SEEDS]) for sp,scenes in groups.items()};scale=1 if m.startswith('ap') else 100;dist=np.mean([v.mean(0)[boot[sp]].mean(1) for sp,v in by.items()],axis=0)*scale
   result['metrics'][m]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(dist,[.025,.975]).tolist(),one_sided95_low=float(np.quantile(dist,.05)),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),split_delta={sp:float(v.mean()*scale) for sp,v in by.items()},scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,scenes in groups.items() for i,s in enumerate(scenes)},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())))
  comparisons.append(result)
  for sp,scenes in groups.items():
   for s in scenes:deltas.append(dict(family=fam,reference=ref,use_cd=cd,split=sp,scene=s,**{m:result['metrics'][m]['scene_delta'][str(s)] for m in metrics}))
write(R/'results/comparisons.json',comparisons);csv_write(R/'results/per_scene_differences.csv',deltas)
get=lambda f,r:next(v['metrics'] for v in comparisons if v['family']==f and v['reference']==r and v['use_cd']);go,gs=get('G','O'),get('G','Gshuffle');checks=dict(GO_AP=go['ap']['delta']>=-.25,GO_AP_CI=go['ap']['ci95'][0]>=-.6,GO_Top1=go['top1_valid']['delta']>=1.5,GO_Top1_one_sided95=go['top1_valid']['one_sided95_low']>0,GO_Top1_seed_count=sum(v>0 for v in go['top1_valid']['seed_delta'])>=2,GO_AP04=go['ap04']['delta']>=0,GO_nonempty_collision=go['top1_nonempty_collision']['delta']<=1,GS_Top1=gs['top1_valid']['delta']>=1,GS_AP=gs['ap']['delta']>=0,P0_cost=json.load(open(R/'results/P0_cost.json'))['status']=='PASS')
decision=dict(status='DEVELOPMENT_GO_MAY_REQUEST_FUTURE_CONFIRMATION' if all(checks.values()) else 'DEVELOPMENT_CRITERIA_NOT_MET_STOP_CURRENT_RECIPE',checks=checks,C4_opened=False,C4_requires_separate_user_authorization=True,independent_generalization_claim=False,all_heads_all_frames_evaluated=True,new_training_quality_labels=0)
write(R/'results/development_decision.json',decision)
lookup={(v['scene'],v['frame'],v['configuration']):v for v in records};tr=[]
for fam,ref in [('G','O'),('G','Gshuffle'),('Gshuffle','O')]:
 for seed in SEEDS:
  counts=Counter()
  for scene in sum(groups.values(),[]):
   for frame in range(256):
    old=lookup[scene,frame,f'{ref}_{seed}_CD'];new=lookup[scene,frame,f'{fam}_{seed}_CD'];counts[old['top1_category'],new['top1_category']]+=1
  tr.extend(dict(family=fam,reference=ref,seed=seed,old_category=a,new_category=b,count=n) for (a,b),n in sorted(counts.items()))
csv_write(R/'results/top1_category_transitions.csv',tr)
train=json.load(open(R/'results/training_complete.json'));ev=json.load(open(R/'results/evaluation_complete.json'));assert train['training_updates']==6000 and ev['frames']==3072
for v in train['heads']:assert sha(v['checkpoint'])==v['checkpoint_sha256'] and v['reload_scoring_pass']
for v in json.load(open(P/'C3_method_lock.json'))['heads']['B']:assert sha(v['path'])==v['sha256']
cpu=ev['CPU_worker_seconds_conservative'];gpu=train['GPU_process_seconds'];size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'archive_worktree' not in p.parts);assert cpu<50*3600 and gpu<.15*3600 and size<2*1024**3
write(R/'results/evaluation_artifact_manifest.json',manifest);write(R/'results/final_validation.json',dict(status='PASS',frames=3072,configurations=20,records=61440,six_heads_fixed1000=True,old_EG_O_selection_and_metrics_exact=True,geometry_unchanged=True,CD_unchanged=True,training_stream_exact_old_O=True,paired_initialization=True,all_selected_quality_finite=True,new_training_quality_labels=0,EG_inference=0,C4_opened=False,CPU_worker_hours_conservative=cpu/3600,GPU_process_hours=gpu/3600,new_runtime_bytes=size,actual_CPU_peak=ev['peak_running_processes_observed'],actual_training_process_peak=train['peak_running_processes_observed']))
lines=['# EG-LG-1 完整C3开发实验报告','',f"状态：{decision['status']}。完整3072帧、3个种子、O/G/Gshuffle；主口径CD。C3是已观察开发材料，不能视作独立确认。",'', '## 主要结果','', '| 比较 | AP差 [95%CI] | Top1差pp [95%CI] | 非空碰撞差pp |','|---|---:|---:|---:|']
for fam,ref in [('G','O'),('G','Gshuffle'),('Gshuffle','O')]:
 v=get(fam,ref);lines.append(f"| {fam}−{ref} | {v['ap']['delta']:+.4f} {v['ap']['ci95']} | {v['top1_valid']['delta']:+.4f} {v['top1_valid']['ci95']} | {v['top1_nonempty_collision']['delta']:+.4f} |")
lines+=['','## 冻结判据','',json.dumps(checks,ensure_ascii=False,indent=2),'','全部判据同时通过才能申请未来确认，本轮不打开C4。阴性或未定不追加种子、训练步数或融合搜索。','', '## 方法与必要核验','', '新增6头各1000步，原B骨干及原标签只读。G与Gshuffle逐种子的初始参数、采样流一致；与原O的采样流一致。原324维输入/动作、六姿态逐动作MSE、AdamW、固定step1000、F05与官方CD/NMS/AP保持一致。33维几何及33维缺失指示，归一化仅来自fit。Gshuffle在帧内置换候选组且共享六姿态下标，训练128池、部署1024池，置换算法相同但池大小不同。','', '所有六姿态分别计算实际旋转/深度对应几何，a0与D7精确一致。仅用可见深度和相机内参，不用GT分割/模型/物体姿态作为头输入；测试CD及官方评分仍遵守原评价管线。测试中新入选预测的几何质量只用于评价，未增加或回流训练标签。','', '主结果完整12场景×256帧，三头先平均、Seen/Similar/Novel各4场景分层bootstrap10000次。全部20配置逐帧验证既有EG/O选择与指标一致；未用AUROC挑权重。3种子不是3个独立场景样本。','', '## 成本与限制','',f"1024候选几何单CPU中位数{json.load(open(R/'results/P0_cost.json'))['median']:.3f}秒，只满足预注册研究成本界限，不满足实时部署。CPU累计保守{cpu/3600:.3f}worker小时，GPU {gpu/3600:.4f}进程小时；实际观察CPU峰值{ev['peak_running_processes_observed']}进程、训练峰值{train['peak_running_processes_observed']}进程；GPU只用于6头训练。",'', '几何定义沿用D7的单视角深度近似；法向缺失和不可见接触面限制仍在。新增架构比O多66输入维度，G−O同时含容量改变；G−Gshuffle才更接近几何对应信息的对照。shuffle会改变输入协方差，不能视作完美因果干预。未纳入B预留场景、A迁移或C4，不声称稳定泛化。','', '## 交付索引','', 'results/absolute_results.json、comparisons.json、per_scene_results.csv、per_scene_differences.csv、offline_summary.csv、top1_category_transitions.csv、development_decision.json、final_validation.json。完整逐帧缓存、权重、几何缓存留服务器，通过散列清单索引。', '', '所有下一轮训练、特征变体、融合修改或C4评价均未自动执行。']
if (R/'results/engineering_resume.json').exists():
 er=json.load(open(R/'results/engineering_resume.json'));lines+=['','## 评价工程接续','',f"首段评价出现重复距离矩阵及大批量CD的内存带宽开销。保留{er['completed_frames_preserved']}个完成帧，仅停止本卡已核验PID后，从同一权重和方法锁接续。按原官方精确距离一次计算候选归属并缓存；CD按64候选分块。新旧实现同一帧全部20配置JSON及NPZ数组精确一致，且未修改官方实现的4配置选择/质量/碰撞/AP核验通过。没有更改NMS、AP、摩擦质量或训练标签。首段保守CPU成本{er['prior_CPU_worker_seconds_upper']/3600:.3f}worker小时计入总预算，未因接续清零。原实现见运行前归档提交c9e2e414。"]
off=list(csv.DictReader(open(R/'results/offline_summary.csv')));lines+=['','## 固定checkpoint的离线诊断','', '| 分组 | model-val Top64 AUROC | gate-cal Top64 AUROC |','|---|---:|---:|']
for family in ['O','G','Gshuffle']:
 vals=[float(np.mean([float(v['auc']) for v in off if v['family']==family and v['role']==role and v['zone']=='top64' and v['comparison']=='noncollision_mu'])) for role in ['model_val','gate_cal']];lines.append(f'| {family} | {vals[0]:.4f} | {vals[1]:.4f} |')
lines+=['','该表为三头均值，分区各4场景，不用于checkpoint选择。model-val与gate-cal结果不一致时，不能只引用较好分区；完整应用指标与冻结判据决定本卡结论。所有逐头、逐帧、随机64及含碰撞口径保存在offline_per_frame.csv和offline_summary.csv。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT_COMPLETE',decision['status'],json.dumps(checks),time.time()-t,flush=True)
