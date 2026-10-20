from common import *
import subprocess,datetime
cfg=json.load(open(R/'frozen_config.json'));summary=json.load(open(R/'results/C3_summary.json'))
assert summary['frames']==3072 and summary['records']==73728
lock=json.load(open(R/'method_lock.json'));assert sha(R/'frozen_config.json')==lock['config_sha256']
for name,h in lock['code_sha256'].items():assert sha(R/'code'/name)==h,name
for item in cfg['protected_G_C_heads']+cfg['protected_heads']+[cfg['backbone'],cfg['training']['prepared']]:assert sha(item['path'])==item['sha256']
heads=[]
for fam in ['SF','H']:
 for seed in SEEDS:
  r=json.load(open(R/f'results/train_{fam}_{seed}.json'))
  assert r['step']==1000 and r['reload_exact'] and sha(r['checkpoint'])==r['checkpoint_sha256']
  heads.append({k:r[k] for k in ['family','seed','step','checkpoint','checkpoint_sha256','sample_stream_sha256','identity_stream_sha256','parameters','seconds','peak_gpu_bytes','diagnostics']})
write(R/'results/trained_heads_lock.json',dict(heads=heads))
speed=json.load(open(R/'results/speed_summary.json'))
def comp(name):return next(x['metrics'] for x in summary['comparisons'] if x['comparison']==name and x['use_cd'])
def practical(name):
 v=comp(name);a,t=v['AP'],v['Top1_pct']
 return ((a['delta']>=.5 and a['CI95'][0]>0 and a['positive_seeds']>=2) or (t['delta']>=1 and t['CI95'][0]>0 and t['positive_seeds']>=2)) and a['delta']>=-.25 and t['delta']>=-.5
complementary=practical('H-C') and practical('H-SF')
decisions={}
for fam in ['SF','H']:
 v=summary['preliminary'][fam];s=next((x for x in speed.get('measurements',[]) if x['family']==fam),None)
 a=bool(s and v['A_performance_only'] and s['median_ratio_to_G']<=1.25 and s['p95_ratio_to_G']<=1.5)
 b=bool(s and v['B_performance_only'] and s['median_ratio_to_G']<=.8 and s['p95_ratio_to_G']<=1)
 decisions[fam]=dict(performance=v,A_pass=a,B_pass=b,speed_measured=s is not None)
preferred='H' if complementary and (decisions['H']['A_pass'] or decisions['H']['B_pass']) else 'SF' if decisions['SF']['A_pass'] or decisions['SF']['B_pass'] else 'G'
write(R/'results/development_decision.json',dict(preferred=preferred,H_complementarity=bool(complementary),H_C_pass=bool(practical('H-C')),H_SF_pass=bool(practical('H-SF')),methods=decisions,development_only=True,capacity_control_not_run=True))
# Predeclared secondary comparison with old S, reusing the exact same C3 scenes/seeds.
with (G/'eg_geometry_compact_v1_20261020/results/per_scene_results.csv').open() as f:old=list(csv.DictReader(f))
with (R/'results/per_scene_results.csv').open() as f:new=list(csv.DictReader(f))
scenes=sum(cfg['evaluation']['C3'].values(),[]);rng=np.random.default_rng(20261020)
ix=np.concatenate([rng.integers(4*i,4*i+4,size=(10000,4)) for i in range(3)],1);sf_s={}
for metric in ['AP','Top1_pct','Top1_nonempty_collision_pct','Top1_empty_pct']:
 delta=[]
 for sc in scenes:
  a=next(v for v in new if int(v['scene'])==sc and v['family']=='SF' and v['use_cd']=='True')
  b=next(v for v in old if int(v['scene'])==sc and v['family']=='S' and v['use_cd']=='True')
  delta.append(float(a[metric])-float(b[metric]))
 delta=np.array(delta);sf_s[metric]=dict(delta=float(delta.mean()),CI95=np.quantile(delta[ix].mean(-1),[.025,.975]).tolist(),positive_scenes=int((delta>0).sum()),per_scene=[dict(scene=sc,delta=float(v)) for sc,v in zip(scenes,delta)])
write(R/'results/SF_vs_old_S.json',sf_s)
prep=json.load(open(R/'results/preparation.json'));train=json.load(open(R/'results/training_complete.json'));score=json.load(open(R/'results/scoring_complete.json'));ev=json.load(open(R/'results/evaluation_complete.json'));profile=json.load(open(R/'results/eval_0100_profile.json'));spf=json.load(open(R/'results/score_0100_profile.json'));p0=json.load(open(R/'results/P0_validation.json'))
gpu=p0['GPU_process_seconds_upper']+train['GPU_process_seconds_upper']+score['GPU_process_seconds_upper']+spf['seconds']+speed.get('seconds',0)
cpu=p0['CPU_worker_seconds_upper']+prep['CPU_worker_seconds_upper']+train['CPU_worker_seconds_upper']+score['CPU_worker_seconds_upper']+ev['CPU_worker_seconds_upper']+profile['seconds']+spf['seconds']+speed.get('seconds',0)+60
used=int(subprocess.check_output(['du','-sb',str(R)],text=True).split()[0])
resources=dict(GPU_process_hours_upper=gpu/3600,CPU_worker_hours_upper=cpu/3600,new_logical_bytes=used,limits=cfg['resources'],training_actual_parallel=6,scoring_actual_parallel=score['actual_peak'],evaluation_actual_parallel=ev['actual_peak'],preparation_worker_pool=72,formal_updates=6000,EG_inference=0,EG_training=0,new_training_GT=0,C4=0,engineering_eval_import_failure=True)
assert gpu<=4*3600 and cpu<=70*3600 and used<=15*1024**3
write(R/'results/resource_report.json',resources)
write(R/'results/final_identity_receipt.json',dict(status='PASS',frozen_config_sha256=sha(R/'frozen_config.json'),frozen_sources_unchanged=True,protected_weights_unchanged=True,prepared_identity_unchanged=True,heads_fixed_step1000=True,three_original_sampler_hashes_match=True,official_validation_checks=profile['native_checks'],GPU_training_updates=6000,GT_training_added=0,C4=False,time=time.time()))
def fmt(v):
 return f"{v['delta']:+.3f} [{v['CI95'][0]:+.3f}, {v['CI95'][1]:+.3f}]"
primary=[v for v in summary['metrics'] if v['split']=='all' and v['use_cd']]
lines=['# EG-STAT-SHAPE-H-V1 统计＋形状开发实验结果','',f"完成：SF/H各3头×1000步，完整C3的3072帧及三头评价。按预先冻结标准，当前首选 **{preferred}**。H同时超过C和SF的互补性筛选：**{complementary}**。",'','## 主结果（带CD，三头先在场景内平均）','','|方法|AP|AP0.4|AP0.8|Top1合格%|非空碰撞%|空抓%|','|---|---:|---:|---:|---:|---:|---:|']
for v in primary:lines.append('|'+v['family']+'|'+'|'.join(f"{v[k]:.3f}" for k in ['AP','AP04','AP08','Top1_pct','Top1_nonempty_collision_pct','Top1_empty_pct'])+'|')
lines+=['','|配对比较|AP差值 [95%场景区间]|Top1差值pp [95%场景区间]|AP正向场景|Top1正向场景|','|---|---:|---:|---:|---:|']
for name in ['H-G','H-C','H-SF','SF-G','SF-C']:
 v=comp(name);lines.append(f"|{name}|{fmt(v['AP'])}|{fmt(v['Top1_pct'])}|{v['AP']['positive_scenes']}/12|{v['Top1_pct']['positive_scenes']}/12|")
lines+=['',f"回填相对旧S：AP {fmt(sf_s['AP'])}；Top1 {fmt(sf_s['Top1_pct'])}pp。其余训练设置及1000步不变；这是同一开发数据上的定向采样干预，不能据此宣称新场景泛化。",'','## 判据与解释','',f"H−C筛选={practical('H-C')}；H−SF筛选={practical('H-SF')}。双项同时通过才支持继续研究统计与形状组合；若仅超过SF，不能解释为PointNet为统计增加了可用信息。",'',f"完整替代判据：{json.dumps(decisions,ensure_ascii=False)}",'','H比C参数多、结构不同，本轮没有加宽C容量控制。即使H出现正结果，也只支持下一轮单独预注册容量对照，不能证明两个分支已经分别学会“可行性”和“稳定性”。仅输出统一Q，未引入新监督任务。','',
'## 效率和资源','']
if speed['status']=='PASS':
 lines+=['|方法|完整评分中位秒|p95秒|中位/G|峰值模型计算显存MiB|','|---|---:|---:|---:|---:|']
 for v in speed['measurements']:lines.append(f"|{v['family']}|{v['total']['median']:.4f}|{v['total']['p95']:.4f}|{v['median_ratio_to_G']:.3f}|{v['peak_gpu_bytes']/1024**2:.2f}|")
 lines+=['','同一GPU、同96固定帧轮换顺序；几何/采样与正式评分输入核验通过。完整流程为新增评分部分，不包括EG推理。GPU共享负载见私有速度回执，不能与旧日期绝对耗时直接比较。']
else:lines+=['没有新方法达到预先测速入口，完整条件测速未执行，不声称速度优势。']
lines += ['',f"累计GPU {gpu/3600:.4f}进程小时、CPU {cpu/3600:.3f}worker小时（含保守60秒汇总和兼容修复余量）、新增逻辑存储 {used/1024**3:.3f}GiB。准备72worker池；训练实测6、评分{score['actual_peak']}、评价{ev['actual_peak']}进程并行。并行数是同阶段进程存在数，不是GPU利用率。",'','## 身份与技术事件','','受保护G/C/O权重和EG、原prepared、原混合采样流均核验；6000更新完成，无重复训练。SF和H逐姿态输入分别计算，全部196608个训练/留出姿态的C统计与保护G列精确相同。P0核验回填保留旧点、无重复、确定性、空mask稳定和SF/H编码器初值一致。正式评价先通过四个配置的官方选择、质量、碰撞和AP等价核验。','','首次官方评价导入失败：基础环境Open3D需要GLIBC_2.35。恢复为此前gpu04已用的隔离兼容环境（PyTorch2.5.1+cu121、Open3D0.19），只影响评价入口；训练权重保持原样。原失败日志及续跑控制器均留痕。','','## 可信范围与下一步','','12个C3场景均已用于开发；3个头不算3套独立场景，帧也不算独立场景。区间按场景配对分层bootstrap，属于开发性描述，未校正多比较。C4没有读取或运行。无新增训练GT标签，官方新选中动作质量仅用于本轮评价。','']
if preferred=='G':
 lines+=['继续保留G。本轮未达到替换标准；不自动追加训练、改融合或打开新确认集。若H没有超过C，优先停止双分支路径；若差值方向正向但区间宽，先解释场景差异再决定是否值得容量对照。']
else:
 lines+=[f'{preferred}满足本轮开发筛选，尚不能宣称独立确认成功。若要发展H，最小下一步是接近65953参数的加宽C对照，固定原采样、标签、1000步和F05；新实验须单独批准。']
lines+=['','## 归档状态','','本地独立分支与公开候选文件已准备。自动审批拒绝了本轮实验卡公开提交，认为未明确授权该具体研究载荷、可能泄露内部研究信息。本轮未绕过；需要用户审阅最终脱敏内容后明确批准公开推送。私有路径、完整逐文件清单、权重和数据不在公开候选载荷中。']
audit=json.load(open(R/'results/full_sampling_audit.json'))
lines += ['', '## 回填输入与训练诊断', '', '|集合原动作|旧有效点均值|回填后均值|回填后padding%|选点增加候选%|', '|---|---:|---:|---:|---:|']
for role,v in audit.items():
 a=v['original'];lines.append(f"|{role}|{a['old_mean']:.2f}|{a['filled_mean']:.2f}|{a['filled_padding_fraction']*100:.2f}|{a['increased_fraction']*100:.2f}|")
lines += ['', '点数增加仅证明采样改变，不等于形状可辨性提升。所有原选点保留，没有通过复制点凑满128；不足区域并集容量时继续保留mask。', '', '|方法|末100步训练MSE均值|model-val MSE|gate-cal MSE|', '|---|---:|---:|---:|']
training=json.load(open(R/'results/head_training_comparison.json'))
for fam in ['C','S','SF','H']:
 a=[v for v in training if v['family']==fam];values=[np.mean([v[k] for v in a]) for k in ['last100_training_MSE','model_val_MSE','gate_cal_MSE']];lines.append('|'+fam+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
lines += ['', '训练诊断沿用六姿态MSE，不用于选检查点或放行。实际决策仍依据完整C3官方指标。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
write(R/'results/computation_complete.json',dict(status='PASS',preferred=preferred,report='FINAL_REPORT.md',time=time.time(),remote_push=False))
print('FINAL',json.dumps(dict(primary=primary,preferred=preferred,complementarity=bool(complementary),resources={k:resources[k] for k in ['GPU_process_hours_upper','CPU_worker_hours_upper','new_logical_bytes']})),flush=True)
