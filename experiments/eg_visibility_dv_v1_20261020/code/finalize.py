from common import *
import pandas as pd, shutil, csv
from scipy.stats import spearmanr
start=time.time()
df=pd.read_csv(R/'results/DV1_candidates.csv');top=df[df['rank']==1]
assert len(top)==9216
baseline=json.load(open(H/'results/C3_summary.json'))
ref=next(x for x in baseline['metrics'] if x['family']=='G' and x['use_cd'] and x['split']=='all')
assert np.isclose(top.valid.mean()*100,ref['Top1_pct'])
assert np.isclose((top.collision&~top['empty']).mean()*100,ref['Top1_nonempty_collision_pct'])
assert np.isclose(top['empty'].mean()*100,ref['Top1_empty_pct'])
v1=json.load(open(R/'results/DV1.json'));collision=top[top.collision&~top['empty']]
v1['top1_nonempty_collisions']=len(collision);v1['fraction_collision_occluded80']=float((collision.collision_occluded_fraction>=.8).mean());v1['collision_gate']=False
write(R/'results/DV1.json',v1)
extras=[];mismatches=[];representative=[]
for p in sorted((R/'cache/c3').glob('*/*.npz')):
 z=np.load(p);scene=int(p.parent.name);frame=int(p.stem);ids=z['ids'];di=z['diag_ids']
 # Every selected G Top-1 identity is checked against protected published rows.
 old=json.load(open(H/f'evaluation/development/{scene:04d}/{frame:04d}.json'))
 old={v['configuration']:v for v in old}
 for h,seed_ in enumerate(SEEDS):
  idx=int(z['top'][h,0]);assert idx==old[f'G_{seed_}_CD']['top_candidate']
  loc=np.searchsorted(di,idx);pos=np.searchsorted(ids,idx);cp=z['collision_points'][loc]
  if z['collision'][pos] and not z['empty'][pos]:
   row=dict(scene=scene,frame=frame,head=seed_,candidate=idx,points=int(cp[0]),free=float(cp[2]),surface=float(cp[3]),occluded=float(cp[4]),unknown=float(cp[5]))
   extras.append(row)
   if cp[4]>=.8 and len([x for x in representative if x['type']=='occluded_collision'])<3:representative.append(dict(type='occluded_collision',**row))
   if cp[3]>=.8 and len([x for x in representative if x['type']=='surface_collision'])<3:representative.append(dict(type='surface_collision',**row))
for p in sorted((R/'cache/head').glob('*/*.npz')):
 z=np.load(p);bad=np.flatnonzero(~np.isclose(z['dv0'][:,1],z['dv0_mu'],atol=1e-6))
 for j in bad:mismatches.append(dict(scene=int(p.parent.name),frame=int(p.stem),sample=z['dv0_id'][j].tolist(),cached_mu=float(z['dv0_mu'][j]),recomputed_mu=float(z['dv0'][j,1])))
ex=pd.DataFrame(extras);ex.to_csv(R/'results/Top1_collision_visibility.csv',index=False)
scene_head=ex.groupby(['scene','head'])[['free','surface','occluded','unknown']].mean()
scene_means=scene_head.groupby('scene').mean();scene_means.to_csv(R/'results/collision_visibility_per_scene.csv')
facts=dict(top1_baseline_exact=True,top1_nonempty_collisions=len(ex),
 collision_state_mean_per_grasp={k:float(ex[k].mean()) for k in ['free','surface','occluded','unknown']},
 collision_state_scene_mean={k:float(scene_means[k].mean()) for k in ['free','surface','occluded','unknown']},
 collision_surface80=float((ex.surface>=.8).mean()),collision_front_or_surface80=float(((ex.free+ex.surface)>=.8).mean()),
 controls={name:dict(n=len(g),contact_found=float(g.contact_found.mean()),contact_occluded_or_grazing=float(g.contact_occluded_or_grazing.dropna().mean())) for name,g in [('success',df[df.valid]),('noncolliding_failure',df[(~df.collision)&(~df.valid)])]},
 label_mismatches=mismatches,representative_cases=representative,
 correction='Excluded 12 empty grasps from nonempty-collision denominator; no features, labels or predictions changed.')
write(R/'results/necessary_validation.json',facts)
v0=json.load(open(R/'results/DV0.json'));dv2=json.load(open(R/'results/DV2_decision.json'));contrasts=json.load(open(R/'results/probe_contrasts.json'));absolute=json.load(open(R/'results/probe_absolute.json'))
dispatch=json.load(open(R/'results/dispatch_status.json'));probe=json.load(open(R/'results/probe_status.json'));audit=json.load(open(R/'results/identity_audit.json'))
pred_seconds=sum(json.load(open(p))['seconds'] for p in (R/'results/predict_jobs').glob('*.json'))
projection=json.load(open(R/'results/projection_sanity.json'))
# Charged conservative overhead covers interactive preflight commands, matrix
# preparation and summaries not timed separately. Counts more than observed CPU.
overhead=900
seconds=dispatch['worker_seconds']+probe['worker_seconds']+pred_seconds+audit['worker_seconds']+projection['worker_seconds']+overhead
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
resource=dict(extraction_worker_seconds=dispatch['worker_seconds'],probe_fit_worker_seconds=probe['worker_seconds'],probe_evaluation_worker_seconds=pred_seconds,identity_audit_worker_seconds=audit['worker_seconds'],projection_check_worker_seconds=projection['worker_seconds'],conservative_other_seconds=overhead,total_charged_worker_hours=seconds/3600,upper_bound_not_exact_meter=True,extraction_parallel=24,probe_fit_parallel=8,probe_evaluation_parallel=12,audit_parallel=12,GPU_hours=0,bytes_before_archive=size,new_formal_heads=0,new_training_quality_labels=0,EG_inferences=0,C4_reads=0)
assert seconds<=8*3600 and size<1024**3
write(R/'results/resource_report.json',resource)
def delta(pool,pr,feature,target):
 return next(x for x in contrasts if x['pool']==pool and x['probe']==pr and x['contrast']==feature+'-FG' and x['target']==target)
table=[]
for method,target,label in [('FGV',1,'V：可行性'),('FGK',2,'K-lite：质量')]:
 for pr in ['linear','mlp']:
  vals=[]
  for pool in ['model_val','gate_cal','heldout8','C3']:
   x=delta(pool,pr,method,target);vals.append(f"{x['mean']:+.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]")
  table.append('|'+ '|'.join([label,pr,*vals])+'|')
absolute_table=[]
for key in ['F0','FC','FG','FGV','FGK','FGVK','shuffle','align','conservative']:
 for pr in (['fixed'] if key=='conservative' else ['linear','mlp']):
  vals=[]
  for pool,target in [('heldout8',1),('C3',1),('heldout8',2),('C3',2),('heldout8',3),('C3',3)]:
   x=next((x for x in absolute if x['pool']==pool and x['probe']==pr and x['features']==key and x['target']==target),None)
   vals.append(f"{x['mean']:.5f}" if x and x['mean'] is not None else '—')
  absolute_table.append('|'+ '|'.join([key,pr,*vals])+'|')
report=f"""# D-V 可观测性诊断报告

## 结论

本轮完成D-V0、D-V1和D-V2。**保留G，当前不建议训练VCP，也不建议训练K角度分支。**
V与K-lite在两种简单CPU探针、两个评价池上均未通过冻结增量判据。
这是对本次具体特征、采样和探针预算的阴性诊断，不否定所有可观测性表示或所有轻量评分头。

## 数据、身份与完成范围

- B骨干、G三头和既有质量标签只读。原始头数据256帧（fit192、model-val32、gate-cal32），每帧原Top64+随机64，D-V2只使用原动作a0。
- D-V0不放回复核20,000个既有非碰撞非空六姿态动作；D-V1完整C3 3,072帧、三头9,216个Top-1及92,160个Top-10头候选记录。
- D-V2部署区复用O/G/C/S/S-fill/H已评价候选的并集。所有所用候选质量有限，未把unknown视作失败，也未补标签。
- 3072帧的导出身份、三代评价缓存重叠标签一致性全部通过。G Top-1身份、合格率、非空碰撞和空抓率与受保护结果逐项一致。
- 新正式头0、EG推理0、新训练GT标签0、GPU0；未读取C4。C3仍是已使用的开发集。

## D-V0：物理定义正确，但法向可靠性不足

|检查|结果|冻结条件|
|---|---:|---|
|旧标签与解析摩擦分档一致|99.995%（19,999/20,000）|≥95%，通过|
|找到双侧GT接触|19,399/20,000|601个缺失单列|
|解析分档与官方接触力闭合函数一致|100%（找到接触者）|核对公式|
|至少一侧接触可见，占全部抽样|30.470%|≥30%，勉强通过|
|双侧都可见，占找到接触者|1.438%|覆盖很小|
|可见接触位置误差中位数|1.585 mm|描述性|
|可见且掠射余弦≥0.3的探针法向误差中位数|14.365°|≤10°，未过|
|上述可靠法向对应|3,588对，覆盖80.16%|缺失未混入误差分母|

GT接触像素处法向误差中位数14.936°，说明单靠把闭合线定位更准确，未必能解决法向误差。
按预设将K降为56维状态/位置/可靠性缺失特征；角度、正切、掠射数值和法向方差不进入探针。
4mm阈值下，GT接触点仍有41.66%被深度判作“前方空”，说明模型表面与可见深度的配准、采样和噪声不能忽略；不能据此声称GT接触不存在。
3/4/6mm表面状态比例分别为12.52%/16.43%/23.50%，只记录敏感性，没有调阈值。
唯一标签不一致案例保存在necessary_validation.json，未改动旧标签。

## D-V1：遮挡不是本批非空碰撞的主要解释

- G Top-1非空碰撞为**{len(ex):,}**次。满足“≥80%碰撞点被遮挡”的比例为**{v1['fraction_collision_occluded80']*100:.2f}%**，预设要求60%，明显未过。
- 碰撞点状态按每次抓法先取比例再平均：前方空{facts['collision_state_mean_per_grasp']['free']*100:.2f}%、表面{facts['collision_state_mean_per_grasp']['surface']*100:.2f}%、遮挡{facts['collision_state_mean_per_grasp']['occluded']*100:.2f}%、无深度{facts['collision_state_mean_per_grasp']['unknown']*100:.2f}%。
- {facts['collision_front_or_surface80']*100:.2f}%的非空碰撞，其至少80%碰撞点位于表面带或前方；这是检查CD阈值、几何对齐和传感噪声的线索，不是已确认的机制归因。
- 高可观测层仅528个Top-10头候选、7个可计算同帧AUROC的头帧；低可观测层76,966候选、7,399个可计算头帧。高层0.9286、低层0.5593的原始均值不具备充分可比性，不能当作稳定0.05增量证据。
- 两层合并候选失败率44.13%与48.30%，差4.17pp，未达预设10pp；GT接触缺失14,666条头候选记录单列unknown。
- 弱质量/无闭合候选的接触遮挡或掠射比例90.13%（仅找到接触者）。成功对照相同比例为{facts['controls']['success']['contact_occluded_or_grazing']*100:.2f}%。必须结合成功对照，不能只用失败组的高遮挡率支持因果解释。

完整分场景/分头统计见DV1_scene.csv，碰撞状态场景平均见collision_visibility_per_scene.csv。
工程口径修正：汇总时曾把12个空抓包含在碰撞分母，最终已剔除；候选、标签、特征与预测均未改变。

对每场景首个非空碰撞例（共12例）额外做了投影往返检查，实际深度点的往返误差<1e-9m，排除了这一步的像素坐标/单位实现错误。
其中10例有“GT点位于可见深度前方”的碰撞点，前向距离中位数约4.2–48.4mm；有些远大于4mm阈值。
这项小样本核查不证明GT与实测深度已经精确对齐，也不能直接归因于噪声或CD阈值。深度漏测、GT模型完整性/位姿及传感器配准仍是待区分解释。
因此下一步若继续，应先检查GT—深度观测对应关系，不能直接把“前方空”作为可靠的无碰撞证据。

## D-V2：未见超过G的可靠增量

下表为相对FG的AUROC差，方括号为场景bootstrap 95%区间；frame AUROC先在场景内平均，C3按Seen/Similar/Novel分层重采样。无双类别帧单列，不作为零分。

|特征/目标|探针|model-val 4场景|gate-cal 4场景|合并8场景|C3 12场景|
|---|---|---|---|---|---|
{chr(10).join(table)}

V要求t1增量≥0.03且两池区间下界>0、置乱增量≤0.005；K要求t2增量≥0.02且下界>0。两者均未过。
K-lite非线性探针在C3反而下降约0.01190 AUROC，区间不跨0。不能据此断言所有非线性接触模型都会退化。
两个探针家族、全部特征臂均报告，无超参数搜索、无留出集拟合、无C3选模。

### 全部对照点估计

|输入|探针|留出t1|C3 t1|留出t2|C3 t2|留出t3 Spearman|C3 t3 Spearman|
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(absolute_table)}

FC、Falign、FGVK和固定保守规则均为预设诊断对照，不能根据其中有利点估计事后替换主要比较或宣称AP提高。
完整绝对指标和区间在probe_absolute.json；配对差值在probe_contrasts.json；逐场景结果在probe_per_scene.csv。
MLP为固定50轮64隐藏单层诊断探针，不是新正式评分头；其拟合不足或模型偏差仍是阴性结论的局限。

## 资源与边界

提取24路、探针拟合8路、部署区探针评价12路CPU，均单线程、低优先级；没有使用GPU或终止他人任务。
累计保守计费**{seconds/3600:.3f} worker小时**（含额外15分钟覆盖预备、汇总及交互计算；不是伪装成精确计量），预算8小时。
归档前新增**{size/1024**2:.1f} MiB**，预算1GiB。进程和文件回执见resource_report.json。

## 研究决策与下一步

1. **首选：保留已确认的G，不启动VCP正式训练。** 当前射线状态可以解释一些个案，但没有形成超过G的可靠判别增量。
2. 如果继续研究，最值得单独立卡的是零训练的“官方真实碰撞—CD—深度表面”对齐检查：固定失败/成功配对候选，核对碰撞发生在手指还是底座、离可见表面多远、CD点数阈值和官方GT几何差异。预计1–2 CPU worker小时，GPU0、不补训练标签；如不能定位可修复且普遍的差异，就停止该路径。
3. K角度方案暂缓。只有独立验证接触法向中位误差降至≤10°且覆盖不下降，才值得重新提出最小训练对照。任何后续正式训练须另行批准，仍应先对照G与单一通过模块，不直接上双分支。
4. 本轮没有运行新的AP/Top-1部署评价，诊断分数不是部署收益；不宣称新方法有效或形成新独立确认。

分支：https://github.com/rollinpioneer/graph/tree/codex/eg-visibility-dv-v1-20261020
"""
(R/'D_V_DIAGNOSIS_REPORT.md').write_text(report)
write(R/'results/finalization.json',dict(status='COMPLETE',seconds=time.time()-start,decision='retain_G_do_not_train_VCP',formal_next_stage_started=False))
print('FINALIZED',seconds/3600,size/1024**2,flush=True)
