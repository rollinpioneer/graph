from common_v2 import *
import csv,subprocess
start=time.time();out=R/'results';status=json.loads((out/'pipeline_status.json').read_text());assert status['status']=='COMPLETED_AT_DEVELOPMENT_STOP';decision=json.loads((out/'development_decision.json').read_text());locked=json.loads((out/'locked_method.json').read_text());cal=json.loads((out/'score_calibration.json').read_text());retro=json.loads((out/'H0_2x2.json').read_text());rows=list(csv.DictReader(open(out/'development_per_scene.csv')))
# Apply each predeclared 2x2 cell's chosen gate to both existing diagnostic splits.
retrospective_rows=[]
for group,prediction_key in [('model_val','val'),('gate_cal','gate')]:
 d=load(group)
 for block in retro:
  for seed in SEEDS:
   q=np.load(R/f'predictions/H0/{seed}/{block["selection"]}.npz')[prediction_key];tau=block['chosen']['thresholds'][str(seed)] if block['chosen'] else None;m=metrics(q,d,float('inf') if tau is None else tau);h=json.loads((out/f'H0_{seed}.json').read_text())['selected'][block['selection']]
   retrospective_rows.append(dict(split=group,selection=block['selection'],gate_rule=block['threshold_rule'],seed=seed,step=h['step'],tau=tau,gate_cal_pass=block['chosen'] is not None,Net=m['Net'],Damage=m['Damage'],regret=m['regret'],utility_gain=m['utility_gain'],modified=m['modified'],N=sum(x['N'] for x in m['scenes']),R=sum(x['R'] for x in m['scenes']),D=sum(x['D'] for x in m['scenes'])))
with open(out/'retrospective_2x2_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(retrospective_rows[0]));w.writeheader();w.writerows(retrospective_rows)
# Descriptive scene-cluster uncertainty for all predeclared calibration comparisons.
rng=np.random.default_rng(20261019);uncert=[]
for block in retro:
 for candidate in block['rows']:
  for seed in candidate['seeds']:
   ss=seed['scenes'];ix=rng.integers(0,len(ss),(10000,len(ss)));net=np.array([x['Net'] for x in ss]);damage=np.array([np.nan if x['Damage'] is None else x['Damage'] for x in ss]);uncert.append(dict(selection=block['selection'],gate_rule=block['threshold_rule'],grid_value=candidate['grid_value'],seed=seed['seed'],Net=seed['Net'],Net_ci95=np.quantile(net[ix].mean(1),[.025,.975]).tolist(),Damage=seed['Damage'],Damage_ci95=np.nanquantile(np.nanmean(damage[ix],axis=1),[.025,.975]).tolist(),scene_denominators=[dict(scene=x['scene'],N=x['N'],original_success=x['original_success']) for x in ss]))
write(out/'gate_scene_uncertainty.json',dict(exploratory=True,unit='4 observed scenes; point-estimate gate unchanged',rows=uncert))
# Exact deployed local paired groups; original labels stay in the previous immutable cache.
local=[]
for group,key in [('model_val','val'),('gate_cal','gate')]:
 d=load(group)
 for head in locked['heads']:
  q=np.load(R/f'predictions/H0/{head["seed"]}/regret.npz')[key];choice=select(q,d['ids'],'S4',head['tau']);idx=np.arange(len(q));u0=d['y'][:,0];un=d['y'][idx,choice];og=(d['mu'][:,0]>0)&(d['mu'][:,0]<=.4000001);ng=un>=np.float32(5/6)
  folder=R/'local_diagnostics';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{group}_{head["seed"]}.npz',scene=d['scene'],group_index=idx,actions=d['ids'],selected_action=choice,original_mu=d['mu'][:,0],selected_mu=d['mu'][idx,choice],true_delta=un-u0,predicted_delta=q[idx,choice]-q[:,0],R=(~og&ng),D=(og&~ng))
  local.append(dict(group=group,seed=head['seed'],**metrics(q,d,head['tau'])))
write(out/'deployed_local_summary.json',local)
allframes=[];probe=[];paired=[];correspondence=True
for p in sorted((R/'evaluation/development').glob('*/*.json')):
 if not p.stem.isdecimal():continue
 v=json.loads(p.read_text());assert len(v)==20;allframes+=v;table={x['configuration']:x for x in v};previous={x['configuration']:x for x in json.loads((P/'evaluation/development'/p.parent.name/p.name).read_text())}
 for baseline in ['R0','R0_CD']:
  for k in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty']:assert table[baseline][k]==previous[baseline][k],(p.name,baseline,k)
 for seed in SEEDS:
  names=[f'{m}_{seed}_CD' for m in ['G0','Gabs','Gdelta']];assert all(tuple(table[n][k] for k in ['selected_action','R','D','original_top_mu','selected_top_mu'])==tuple(table[names[0]][k] for k in ['selected_action','R','D','original_top_mu','selected_top_mu']) for n in names)
 for x in v:
  paired.append({k:x[k] for k in ['scene','frame','configuration','top_identity','original_top_success','selected_action','original_top_mu','selected_top_mu','R','D']})
  for st in ['high','low']:probe.append(dict(scene=x['scene'],frame=x['frame'],configuration=x['configuration'],stratum=st,**x[st]))
assert len(allframes)==5120
for name,data in [('paired_transitions.csv',paired),('strata_per_frame.csv',probe)]:
 with open(out/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
strata=[]
for name,st in sorted(set((x['configuration'],x['stratum']) for x in probe)):
 v=[x for x in probe if x['configuration']==name and x['stratum']==st];strata.append(dict(configuration=name,stratum=st,N=sum(x['n'] for x in v),modified=sum(x['modified'] for x in v),improved=sum(x['improved'] for x in v),degraded=sum(x['degraded'] for x in v),mean_utility_gain=float(np.mean([x['utility_gain'] for x in v]))))
write(out/'score_strata_summary.json',strata)
cases=[]
for typ,test in [('rescued',lambda x:x['R']==1),('damaged',lambda x:x['D']==1),('unchanged_failed',lambda x:not x['original_top_success'] and x['selected_action']==0)]:
 v=sorted([x for x in allframes if x['configuration']==f'Gdelta_{SEEDS[0]}_CD' and test(x)],key=lambda x:(x['scene'],x['frame']));cases.append(dict(category=typ,status='SAVED' if v else 'NO_CASE',case=v[0] if v else None,selection='first registered seed; earliest scene/frame; exploratory'))
write(out/'bad_case_index.json',cases)
# Single-thread, same frozen-feature input extra processing timing; no backbone or native evaluator.
from scipy.special import expit
pred,hyp,features,act,action_ids=export(CFG['development'][0],CFG['dev_frames'][0]);latency=[]
for head in locked['heads']:
 ck=torch.load(P/f'heads/{head["seed"]}/step{head["step"]}.pt',map_location='cpu',weights_only=False);m=old_model().eval();m.load_state_dict(ck['model']);times=[]
 for it in range(23):
  t=time.perf_counter();x=np.concatenate([np.broadcast_to(((features-ck['mean'])/ck['std'])[:,None,:],(1024,6,320)),act],-1).copy()
  with torch.no_grad():q=m(torch.from_numpy(x)).squeeze(-1).numpy()
  chosen=select(q,action_ids,'S4',head['tau']);output=hyp[np.arange(1024),chosen].astype(np.float64);delta=q[np.arange(1024),chosen]-q[:,0];delta[chosen==0]=0;output[:,0]=expit(cal['a']*pred[:,0].astype(np.float64)+cal['b'])+delta
  if it>=3:times.append((time.perf_counter()-t)*1000)
 latency.append(dict(seed=head['seed'],median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95))))
write(out/'latency.json',dict(scope='CPU one thread; normalization, six-action head, S4 gating, C+delta and geometry writeback; excludes backbone/IO/CD/NMS',warmups=3,repeats=20,rows=latency))
receipts=[json.loads(p.read_text()) for p in (R/'evaluation').rglob('*.json') if p.name in ['MICRO.json','COMPLETE.json']];resources=dict(gpu_jobs=0,new_trainings=0,new_labels_for_training=0,evaluation_workers=7,controller=1,total_compute_process_cap=8,completed_evaluation_worker_wall_seconds=sum(x['seconds'] for x in receipts),completed_evaluation_cpu_seconds=sum(x['cpu_seconds'] for x in receipts),evaluation_launcher_child_cpu_seconds=status['child_cpu_seconds'],failed_micro_attempts=2,failed_micro_cost_measured=False,failed_micro_budget_reserve_worker_hours=.5,s0_initial=json.loads((out/'S0_pre_gate_check_correction.json').read_text()),s0_corrected_aggregation=json.loads((out/'S0.json').read_text()),calibration=cal['seconds'],export_io_seconds=sum(x['export_io_seconds'] for x in receipts),development_wall_seconds=status['seconds'],budget_worker_hours=40)
files=[p for sub in ['code','results','logs','predictions','evaluation','local_diagnostics'] for p in (R/sub).rglob('*') if p.is_file()];resources['new_artifact_bytes']=sum(p.stat().st_size for p in files);resources['conservative_accounted_worker_hours']=resources['completed_evaluation_worker_wall_seconds']/3600+.5+(cal['seconds']+resources['s0_initial']['seconds']+resources['s0_corrected_aggregation']['seconds'])/3600;assert resources['conservative_accounted_worker_hours']<40 and resources['new_artifact_bytes']<CFG['new_disk_limit_bytes'];write(out/'resource_summary.json',resources)
write(out/'artifact_manifest.json',[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in files if p.suffix in ['.npz','.log'] or p.parent.name=='code'])
write(out/'final_verification.json',dict(status='PASS',frames=256,configurations_per_frame=20,total_records=5120,all_score_variants_same_local_actions=True,monotonic_C_complete_native_equivalence=True,R0_and_R0_CD_exactly_match_previous_V1_all_frames=True,original_micro_passed=True,new_trainings=0,confirmation_opened=False))
lines=['# EG Local Delta / Global Score V2 执行报告','',f"状态：{status['status']}。注册提交c46d0ccd9ff9fb818268b1fc9b9472a988506577。父结果42ac74b8444d5234d79808e98f269fd85420c9c4。分支codex/eg-local-delta-global-score-v2-20261019。",'', '本轮复用30个检查点、196608个原生训练姿态标签。H0通过初步局部筛选，按卡片跳过H1/H2，新增训练0次，EG更新0次，GPU使用0。下面不能作为差值损失或FiLM的效果证据。','', '## 零训练2×2回顾','', '|选模|门控|gate-cal共同可行值|', '|---|---|---|']
for x in retro:lines.append(f"|{x['selection']}|{x['threshold_rule']}|{x['chosen']['grid_value'] if x['chosen'] else '无'}|")
lines += ['', 'S4-regret均选第1000步，未用AP挑权重。主方案锁定κ=0.01，各种子τ分别为'+', '.join(f"{h['tau']:.8f}" for h in locked['heads'])+'。κ根据model-val正预测增量的分位数定义，不保证开发固定修改比例。共同κ来自gate-cal，阈值数值按各模型尺度不同。', '', '### 锁定门槛后的局部开发材料','', '|数据|种子|Net|Damage|修改数|','|---|---|---:|---:|---:|']
for x in local:lines.append(f"|{x['group']}|{x['seed']}|{x['Net']:.4%}|{x['Damage']:.4%}|{x['modified']}|")
lines += ['', 'model-val/gate-cal已多次使用，只是开发材料；其正向点估计不能当安全或泛化保证。场景分母和四场景聚类区间见gate_scene_uncertainty.json。','', '## 同姿态评分对照','', f"C(s)=sigmoid({cal['a']:.9f}×s{cal['b']:+.9f})，仅fit原动作、top64权重1/随机64权重15、场景等权拟合。C(s)与原分数的完整原生候选选择在每帧带/不带CD均相同。Gdelta不截断，不能称成功概率。",'', '|方法（三种子均值）|无CD AP|带CD AP|带CD Top-1合格率|','|---|---:|---:|---:|']
for method in ['R0','G0','Gabs','Gdelta']:
 vals=[]
 for suffix in ['', '_CD']:
  names=[method+suffix] if method=='R0' else [f'{method}_{seed}{suffix}' for seed in SEEDS];v=[x for x in rows if x['configuration'] in names];vals.append(float(np.mean([float(x['ap']) for x in v])))
 top=float(np.mean([float(x['top1_valid']) for x in v]));lines.append(f'|{method}|{vals[0]:.3f}|{vals[1]:.3f}|{top:.3%}|')
lines += ['', f"主效应Gdelta+CD−R0+CD={decision['mean_ap_difference_pp']:+.4f} AP pp；场景95%CI={decision['ci95_pp']}；逐种子={decision['seed_ap_difference_pp']}。Gdelta−G0={decision['global_score_effect_Gdelta_minus_G0']:+.4f} AP pp。",'', '|种子|原Top-1 N|原成功|救回R|损坏D|Net|Damage|','|---|---:|---:|---:|---:|---:|---:|']
for x in decision['local']:lines.append(f"|{x['seed']}|{x['N']}|{x['original_success']}|{x['R']}|{x['D']}|{x['Net']:.3%}|{x['Damage']:.3%}|")
lines += ['', f"开发放行：{decision['pass_stage']}；各项判据：{decision['criteria']}。固定原Top-1身份以R0无CD原生列表为准，不等于修改后整帧Top-1；各评分对照的局部姿态、R/D严格一致。种子重复评价不是独立样本。",'', '本卡按冻结模型序列停止：不根据开发AP再切换H1/H2或修改门槛。旧迁移场景未运行，新确认未打开。开发为8场景×32帧，不是官方完整256帧场景评价或90场景AP。','', '## 证据边界与结论','', 'S4-regret选模与分位门控能在已观察的局部校准材料上形成很小的正信号，但能否成为实用方法以开发AP和同候选损益共同判定。低修改率与小样本下的零损坏不构成安全证明。若开发失败，只否定本次锁定H0配方的放行，不证明所有冻结特征无信息，也不证明差值损失或条件化无效，因为H1/H2本轮未训练。', 'Gdelta相对G0只比较相同姿态下的评分变化；Gabs比较完全替换评分。不能把这些差值直接分解成旧实验失败的因果百分比。','', '## 必要适配与成本','', '初始S0放行代码误用了无门槛model-val统计，已在新增训练及应用AP前修正为锁定门槛后的实际选择器，保留修正前回执；未改权重、门槛网格或标签。', '原生微测发现Gabs一个候选的float32宽度0.10000000149与官方截断0.1存在1.49e-9差异，质量/碰撞/AP均一致。已在评价适配层补齐官方截断，复验四个完整路径的候选、质量、碰撞和AP精确一致。原始预测几何未覆盖。', 'V1应用缓存未保存完整逐帧导出身份，本轮没有冒用其逐姿态质量；正式训练标签哈希全部核验并复用。新应用缓存记录源导出与评价版本。','', f"正式评价worker墙钟累计{resources['completed_evaluation_worker_wall_seconds']/3600:.3f}小时；含两次失败微测的0.5 worker小时保守预留后，计入预算约{resources['conservative_accounted_worker_hours']:.3f}小时。失败微测没有精确CPU计时，不能记为0。新增产物约{resources['new_artifact_bytes']/1024**2:.1f}MiB。S0三个worker加控制、开发七个worker加控制；总计算进程上限8，均CPU单线程，无GPU。", '延迟见latency.json，覆盖特征归一化、小头、门控、标定和写回，不含EG骨干、IO/CD/NMS，不能当完整机械臂时延。完整阶段墙钟、CPU与导出IO分项见resource_summary.json。', '双向案例按第一种子最早scene/frame选取，缺失类别明确NO_CASE。轻量结果/实现进入本卡分支；完整逐帧质量、预测Q与配对组缓存留服务器并保存散列。']
(out/'final_report.md').write_text('\n'.join(lines)+'\n');write(out/'finalization.json',dict(status='PASS',seconds=time.time()-start));print('FINALIZATION_PASS',flush=True)
