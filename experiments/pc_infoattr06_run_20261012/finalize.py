"""Finalize factual lightweight report and archives after all frozen conditions end."""
import os,pathlib,json,csv,hashlib,datetime,subprocess,shutil
import numpy as np
R=pathlib.Path(os.environ['R']);W=pathlib.Path(os.environ['W']);branch='codex/pc-infoattr-06-20261012-card'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,obj):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
jobs=[];peak=0;charged=0;full=[];callcount=0;recd=[]
for out in sorted((R/'runs').iterdir()):
 if not (out/'guard_result.json').exists():continue
 g=json.loads((out/'guard_result.json').read_text());charged+=g['charged_gpu_seconds'];done=out/'data/completed.json';d=json.loads(done.read_text()) if done.exists() else None
 if d:callcount+=d['model_calls']
 j={'run':out.name,'status':'executed_complete' if g['exit_code']==0 and d else 'failed_or_stopped','exit_code':g['exit_code'],'stop_reason':g['stop_reason'],'charged_gpu_seconds':g['charged_gpu_seconds'],'wall_seconds':g['wall_seconds'],'memory_peak_own_mib':g['task_peak_mib'],'n':d['n'] if d else None,'full_episodes':d['full_episodes'] if d else 0,'partial_episodes':d['partial_episodes'] if d else 0,'model_calls':d['model_calls'] if d else 0}
 jobs.append(j);target=R/'receipts/jobs'/out.name;target.mkdir(parents=True,exist_ok=True)
 for name in ['launch.json','guard_result.json','active.json']:
  if (out/name).exists():shutil.copy2(out/name,target/name)
 if d:
  shutil.copy2(done,target/'completed.json');shutil.copy2(out/'data/events.json',target/'events.json');shutil.copy2(out/'data/config.json',target/'config.json');shutil.copy2(out/'data/noise_schedule.json',target/'noise_schedule.json')
  compact=[{k:x[k] for k in ['id','label','seed','batch','slot','elapsed_s','noise_draws','noise_content_sha256','noise_sha256','rng_sha256','rng_path','noise_path','generated_path']} for x in json.loads((out/'data/sampling_index.json').read_text())];write('receipts/jobs/'+out.name+'/sampling_identity.json',compact)
 if not d:write('receipts/jobs/'+out.name+'/failure_excerpt.json',{'last4000chars':(out/'process.log').read_text(errors='replace')[-4000:]})
 for line in (out/'gpu_memory.jsonl').read_text().splitlines():peak=max(peak,json.loads(line).get('aggregate_task_memory_mib',0))
 if d and d['full_episodes']:
  ev=json.loads((out/'data/events.json').read_text());full.append({k:d[k] for k in ['n','scene_seed','policy_seed','policy_steps','goal_fraction','all_success','mean_goal_distance_m','wall_s','trajectory_path','trajectory_sha256']});full[-1].update(run=out.name,**ev)
assert len(full)==4 and all(j['status']=='executed_complete' for j in jobs if j['run'].startswith('normal'))
assert charged<=7200 and peak<12288
ledger={'charged_gpu_seconds':charged,'GPU_hours':charged/3600,'aggregate_peak_mib':peak,'aggregate_peak_GiB':peak/1024,'sampling_interval_s':.5,'counting':'full child wall including setup/errors times max(1,number of task-resident GPU contexts); CPU-only analysis separate','GPU_jobs':len(jobs),'technical_failed_jobs':sum(j['status']!='executed_complete' for j in jobs),'full_episodes':4,'partial_episodes':sum(j['partial_episodes'] for j in jobs),'policy_calls':callcount,'additional_offline_policy_calls':0,'training_steps':0,'limits':{'GPU_seconds':7200,'memory_mib':12288},'CPU_receipts':['analysis_receipt.json','receipts/action_normalizer_audit.json'],'jobs':jobs}
write('results/resource_ledger.json',ledger);write('results/job_status.json',jobs);write('results/normal_full_results.json',full)
for name,rows in [('job_status.csv',jobs),('normal_full_results.csv',full)]:
 with (R/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
mech=json.loads((R/'results/P2_mechanism_review_summary.json').read_text());old=[x for x in mech if x['run']=='validate6_blocked' and x['object']==4][0];new=[x for x in mech if x['run'].startswith('normal')];rep=[x for x in new if x['command_candidate_windows'] or x['execution_candidate_windows']]
gate=json.loads((R/'results/P1_health_gate.json').read_text());identity=json.loads((R/'results/P1_identity_summary.json').read_text());multi=json.loads((R/'results/P1_holdout_multiview_availability.json').read_text())
decision={'P1_B':'not_executed_reader_health_gate_failed; not an encoding-reliability signal','information_replacement':'not_executed_no_valid_readout_or_same_scene_source','bounded_response_validation':'not_executed','reason':'new normal review has no candidate hit; both6C episodes have no eligible20stepprefallwindow due earlyfalls, so absence does not reject mechanism. No discriminating operation; lower priority instead of restoring old blocked case' if not rep else 'normal candidate is observational; original command/response already directly measurable; available intervention cannot distinguish legitimate repositioning from inappropriate request without a valid task-relative reference. No generic oracle push/success test added','new_normal_candidate_objects':rep,'maximum_followup_directions':1,'retained_followup_direction':'none; stop automatic expansion' if not rep else 'task-relative appropriateness of the command in reproduced event; no training or interface changes authorized','old_case_scope':'noise-blocked/permuted discovery only'}
write('receipts/conditional_validation_decision.json',decision)
status={'P1_A':'executed_calibration_and_holdout_selfcheck_failed; diagnostic outputs descriptive','P1_B':'not_executed_failed_prerequisite','P2_A_B':'executed_full_history_and_aligned_step_chain','new_normal_review':'4complete200step_runs, all endpoints retained','conditional_validation':'not_executed_no_discriminating_operation','training':'not_executed_out_of_scope'};write('results/condition_status.json',status)
summary=json.loads((R/'results/P2_object_summaries.json').read_text());blocked=[x for x in summary if x['run']=='validate6_blocked' and x['object']==4][0];maxerr=max(x['controller_transform_max_error_m'] for x in summary)
rows='\n'.join('|{run}|{n}|{scene_seed}/{policy_seed}|{score}|{mean_goal_distance_m:.5f}|{fall}|{candidate}|'.format(**q,score='%d/%d'%(round(q['goal_fraction']*q['n']),q['n']),fall=q['first_fall'],candidate=sum(x['command_candidate_windows'] for x in new if x['run']==q['run'])) for q in full)
report=f'''# PC-INFOATTR-06 执行结果

本轮已按用户“执行”授权完成校准、自检、旧轨迹逐步归因与四个原始 normal 采样完整复核回合。沿用 [{branch}](https://github.com/rollinpioneer/graph/tree/{branch})；起点 fb0a59b6ee6be39e98148ad30d2f99556bc6d926，固定发现证据 6c03f59140bc7d8b416ea4c72c4f71cb35eba823。原始卡片 SHA256 11711ea4242a3dcd412262109c0f575b44b0325d313e83495f3fa14c34750e4d；制卡时“未授权执行”的文字在 REQUESTED_CARD.md 原样保留，其历史状态由本轮用户授权覆盖。

## 得到的定位

P1 的简单前景粒子外观原型读出规则未通过健康自检，因此本轮不能把读出失败归为 DLP 对象信息丢失。P2 旧 6C 分块轨迹中，未完成对象4的退离多次已体现在控制请求，并被实际运动跟随；这削弱该片段的纯执行反向解释，尚不证明请求不合理。新 normal 采样的机制复核是否命中与终点成败分列，不能将规模分数或落桌本身当机制复现。

## P1：校准、留出与停止门槛

校准为旧 supplement3/supplement6 的固定0/10/30/50时点和目标图；不向规则提供终点成败。校准使用投影几何及颜色代理标签，拟合6个颜色的4维外观中位数原型和固定尺度。共有246个校准粒子标签，各颜色60/8/17/29/82/50；这包含数据校准，不称完全无拟合。读出只访问实际 used_current_particles/used_goal_particles 的位置、外观及obj_on；不访问诊断 RGB、背景、真值或回合号。

冻结门槛：obj_on>.1，标准化外观距离<=3，最优和次优距离差>=.15；多粒子按固定权重形成颜色位置。像素容差10px约为这个图像尺度下物体投影宽度的量级，针对粗定位，不能据此宣称毫米级XY可靠性。留出场景为 validate3_permuted 的0/50/100/199和目标，独立于校准场景；validate3_blocked与该留出共享场景，只作描述参照，不增加独立场景数。

健康留出29个可见对象视角帧中，19/29={gate['within_10px_fraction']:.1%}返回正确颜色且位置误差<=10px，9个未返回位置、1个误差31.21px，低于预设80%门槛。当前帧14/23达标，目标5/6达标；逐视角见 P1_holdout_current_goal_breakdown.json。按所有预测的最近几何参考评分，34个颜色位置输出中18个身份正确且<=10px，其余包含多余或错误颜色输出。这进一步限制读出器可靠性，不能把仅检测到某颜色说成稳定对象级识别。

两个视角的正确位置可用性上界为 {multi['at_least_one_view_within_10px']}/{multi['visible_object_frames']}={multi['fraction']:.0%}。这一指标用评分真值判断哪个视角正确；没有建立推理时可靠视角选择、联合不确定性或XY三角测量。因此既不把一视角漏读叫整个输入缺失，也不将该上界冒充已校准联合读出。

冻结的旧诊断0/50/68/88/108/150/199和新复核0/50/100/150/199均保留描述性读出。原始数据来自直接保存的策略前景输入，不补造历史中间输入。P1-B局部位置编码探针及信息替换未启动：自检失败没有提供可信对象级疑点；未重新调参、增加读出器训练或无限搜索。RGB参考为投影区域内可见颜色质心，部分遮挡会偏移；同时报告投影中心误差，均用像素，未混称物理位置误差。旧局部重建证据仅在 joined_evidence_chain.json 中引用 PC-INFOCHAIN-05 已保存输出，没有重新包装为新解码成果。

## P2：动作前输入、命令与动作后状态

实际源码核查：RGB[t]/粒子[t]进入策略，H5首动作[t]执行后得到状态/关节/末端[t+1]。controller_goal[t]虽然在env.step返回后读出，却是该步开始设置且整个控制周期保持的目标；数值核验最大目标变换误差 {maxerr:.3g}m。控制目标是clip(eef[t]+clip(action,-1,1)*.125)并受桌面与高度限制。该核验定位时义和接口变换，不声称关节即时到达目标。

P2_step_evidence.csv 覆盖旧6条及新4条轨迹每个对象、每个动作，包括H5反归一化动作、控制目标、当前/下一关节与末端、对象目标距离与运动进展、落桌状态和请求/响应径向投影。原始反归一化动作已保存；采样器clip前归一化proposal没有被旧记录器保存，不能反推它。action_normalizer_audit.json从冻结训练回放的有效路径精确重算上下界，链表仅提供clip后动作的逆归一化值。没有追加模型调用补造缺失proposal。

旧blocked对象4从状态30开始成为唯一未达标对象。完整轨迹终点目标误差0.215052m；局部2mm改善的最后动作137也不能代表取得新的最佳进展，P2_event_sequences.json另列new-best时间及全程事件。该对象待完成的200步中，{blocked['requested_departure_followed']}步为请求退离且实际退离，{blocked['requested_approach_actual_departure']}步出现请求接近而实际退离，微小/其他变化单列。请求/实际方向一致并不表示操作有效，中心距离近也不是接触证据。

冻结的20步机制候选窗口要求：待完成且起始EEF到对象<75mm、期间无落桌、无>=2mm目标改善，末端距离增加>=40mm，请求和实际径向退离正路径各>=40mm，显著运动中>=80%的请求/实际方向cos>.8。竞争执行线索为同样窗口中>=3步请求接近>2mm却实际退离>1mm。规则在开启新复核结果前冻结，但前两个新回合已经开始生成数据，此时间边界在回执明确，不伪称全部采集前预注册。

旧blocked对象4命中{old['command_candidate_windows']}个重叠窗口，执行竞争窗口{old['execution_candidate_windows']}个；旧permuted成功对象4也命中1个请求退离候选。因此它不是充分失败条件，也不能据此证明“退离不合理”。成功参考按同对象操作事件描述对照；不同闭环状态不称同状态因果配对。

## 原始 normal 采样复核

成本8步试跑后冻结4回合档：3C与6C各2个独立场景，每场景一次。200步、H5/T100、batch1/slot0、场景/策略随机流独立；无每步重置、无噪声块化。两个独立worker并行，每个随后执行其第二场景，所有终点包含落桌。

|回合|物体数|场景/策略种子|目标数|终点平均XY误差m|首次确认落桌状态|请求退离候选窗口|
|---|---:|---|---:|---:|---:|---:|
{rows}

候选窗口重叠，表中计数不是独立样本或群体发生率，落桌后的片段不用于无落桌机制规则，但仍在完整终点和轨迹中。两个新6C回合首次低于落桌阈值分别在状态17和20，五个连续状态确认在21和24；两条轨迹均没有满足“待完成、接近且完整20步无落桌”的候选窗口机会。因此0次命中受早期落桌限制，不能用0除以0当阴性发生率，也不能排除normal采样中的命令机制。只降低旧长时无进展线索本轮复核优先级，不追旧分块案例恢复。精确机会数在 normal_review_opportunity_and_censoring.json。4个场景只作机制复核，不能重新估计总体成功率、训练种子稳定性或数量因果贡献。

条件验证决定：{decision['reason']}。P1-B、信息替换及额外完整验证回合均未执行；剩余预算不作为必须消耗的任务。后续最多保留的机制方向：{decision['retained_followup_direction']}。不自动训练、改架构、扫噪声块长或扩展P3底层接口。

## 资源、技术失败与可复现性

合计 {charged:.3f} GPU秒={charged/3600:.5f} GPUh；0.5秒采样的任务相关进程合计峰值 {peak}MiB={peak/1024:.5f}GiB。计入全部模型/渲染加载、失败、8步试跑和上下文；同任务在GPU0产生附属上下文时也乘入GPU计费，独立worker相加。CPU校准和分析单列时间；没有训练、额外离线策略调用或GPU解码。资源硬限7200GPU秒、12GiB；normal每回合预留1200GPU秒，共4800，超出0.75h建议分配的可能成本在启动前记录，以四回合档和未动用离线GPU预算保留条件验证余量，硬限未改。

7个GPU工作中2个加载失败（libpython共享库搜索路径、Ninja PATH），修正环境后同一成本场景8步成功，失败目录与费用均保留。正式4回合无技术失败、无预算停止，完整模型调用800次，试跑8次，共{callcount}次。初始SSH沙箱内别名解析失败，获准网络权限后本轮仅建立并复用一个远端交互shell。两个正式worker结束后清理自己的SSH，不结束其他用户进程。

recording_audit.json 核验冻结源码、每回合200个不同随机内容、每调用101次draw、原采样方式、初始保存输入一致、只执行H5首动作、物理配置与旧官方正常回合完全一致、状态t/t+1长度、有限数值及接口变换。三项模型/数据资产本轮重新SHA核验不变，工程commit6b5e482与既有三项兼容补丁保留。

完整权重、原始张量、RGB、完整随机流和日志留在服务器 {R}。results/server_artifact_manifest.json 为路径、大小、SHA256索引；results/source_inputs.json核验旧数据身份。Git仅本轮源码、冻结配置、回执、轻量表、图、报告和复现说明；失败、未执行条件明确区分。
'''
(R/'RESULTS.md').write_text(report)
write('receipts/archival_whitespace_policy.json',{'requested_card_preserved_bytes':True,'excluded_from_blank_at_eol':['REQUESTED_CARD.md','official_compatibility.patch'],'reason':'source card CRLF/intentional Markdown hardbreaks and raw patch context spaces; source hashes checked; generated Python/docs checked normally'})
artifact=[]
for folder in [R/'runs',R/'logs']:
 for f in sorted(folder.rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts:artifact.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f),'storage':'server only; selected light receipts copied to receipts/jobs'})
write('results/server_artifact_manifest.json',artifact)
card=W/'experiments/pc_infoattr06_card_20261012'
write('receipts/final_scientific_status.json',{'complete':True,'condition_status':status,'GPU_worker_groups_all_exited':True,'SSH_status':'pending_close_after_push','optional_branch':decision})
(card/'EXECUTION_ANNEX.md').write_text('# PC-INFOATTR-06 execution annex\n\nUser authorized execution on2026-10-12. Original card is a frozen historical request; its card-only status is superseded by this annex. Execution and archive use '+branch+'.\n\nResults: ../pc_infoattr06_run_20261012/RESULTS.md. Four complete normal-sampling episodes; reader selfcheck failed, so no strong DLP attribution. Optional validation not executed; see conditional_validation_decision.json.\n\nGPU hours %.5f, aggregate memory peak %.5fGiB.\n'%(charged/3600,peak/1024))
(card/'BRANCH_INFO.json').write_text(json.dumps({'branch':branch,'branch_url':'https://github.com/rollinpioneer/graph/tree/'+branch,'status':'executed_complete_with_prerequisite_stop_and_optional_not_executed','base_commit':'fb0a59b6ee6be39e98148ad30d2f99556bc6d926','results_path':'experiments/pc_infoattr06_run_20261012/RESULTS.md','source_card_sha256':'11711ea4242a3dcd412262109c0f575b44b0325d313e83495f3fa14c34750e4d'},indent=2))
for p in (R/'results').glob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
light=[p for p in sorted(R.rglob('*')) if p.is_file() and not any(x in p.relative_to(R).parts for x in ['runs','logs','__pycache__']) and p.name not in ['light_files.txt','light_manifest.json']]
assert all(p.stat().st_size<30*1024*1024 for p in light)
(R/'light_files.txt').write_text(''.join(str(p.relative_to(W))+'\n' for p in light)+str((R/'light_files.txt').relative_to(W))+'\n'+str((R/'light_manifest.json').relative_to(W))+'\n')
write('light_manifest.json',[{'relative_path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in light])
print('FINALIZED',len(light),charged,peak,'new_candidates',len(rep),flush=True)
