"""Final pure export/audit; reads existing results, never runs models or physics."""
import os,sys,json,csv,hashlib,time,subprocess,shutil,resource,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent
sys.path.insert(0,'/home/xushijie3/work/graph_saw_goalprobe05_20261013/experiments/saw_goalprobe05_20261013/vendor/analysis_deps')
import numpy as np,pandas as pd
assert os.sched_getaffinity(0)=={75}
started=time.time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def text(p,x):
    assert not Path(p).exists();Path(p).write_text(x,encoding='utf-8')
def export(p,x):assert not Path(p).exists();x.to_csv(p,index=False)
def gone(pid):return not Path('/proc',str(pid)).exists()

auth=read(ROOT/'receipts/AUTHORIZATION.json');assert auth['time_cost_limits_removed'] and not auth['X']
gate=read(ROOT/'results/I_GATE.json');both=bool(gate['H_R_pass']);arms=[0,1,2,3] if both else [0,1];mode='FINAL' if both else 'I';stats=ROOT/'results'/f'{mode}_STATISTICS'
finalgate=read(stats/'HYPOTHESIS_GATE.json')
phases=['I','II'] if both else ['I'];workflows=[read(ROOT/'receipts'/f'{s}_WORKFLOW_MONITOR.json') for s in phases]
assert all(x['status']=='COMPLETE' and x['all_owned_children_exited'] for x in workflows)
assert all(gone(j['pid']) for x in workflows for j in x['jobs'])
assert all(gone(x['cpu_evaluation']['pid']) for x in workflows)
ident=read(ROOT/'REFERENCE_AND_INIT_IDENTITY.json');cal=read(ROOT/'results/P_continuation/GATE_CALIBRATION.json')
assert all(sha(x['reference']['path'])==x['reference']['sha256'] and sha(x['init_path'])==x['init_sha256'] for x in ident)
assert all(sha(x['path'])==x['sha256'] for x in cal)
for x in cal:
    c=np.load(x['path']);assert len(c['cost'])==16384 and np.isfinite(c['cost']).all()
    assert float(np.sort(c['cost'])[int(np.ceil(.95*16384))-1])==x['tau']
old_initial=ROOT/'runs/CALIBRATION_b0.npz'
if old_initial.exists():
    old=np.load(old_initial);new=np.load(cal[0]['path'])
    assert np.array_equal(old['indices'],new['indices']) and np.array_equal(old['goal_indices'],new['goal_indices'])
pilots=read(ROOT/'results/P_continuation/P_PILOTS.json');assert sum(x['updates'] for x in pilots)==4000 and all(x['weights_discarded'] for x in pilots)
audit=read(ROOT/'results/P_continuation/LOSS_AND_GRADIENT_AUDIT.json');assert all(x['frozen_native_gradients_zero'] for x in audit)
replay=read(ROOT/'results/LEGACY_LOG_REPLAY_AUDIT.json');assert replay['status']=='COMPLETE'
checks=[];pairing=[];allcps=[]
for r in range(3):
    hashes={}
    for a in arms:
        st=read(ROOT/'results'/f'b{r}_a{a}_status.json');assert st['status']=='COMPLETE_500K' and st['student_updates']==500000
        cps=read(ROOT/'results'/f'b{r}_a{a}_checkpoints.json');assert [x['step'] for x in cps]==[0,100000,200000,300000,400000,500000]
        assert cps[0]['student_parameter_sha256']==ident[r]['four_arm_parameter_sha256'][a]
        assert all(x['adam_count']==x['step'] and x['audit_tuple_count']==256 and x['frozen_reference_exactly_unchanged'] and x['reference_full_state_sha256']==ident[r]['reference_full_state_sha256'] for x in cps)
        for x in cps:assert sha(x['path'])==x['sha256']
        hashes[a]={x['step']:x['base_F_original_R_rng_state_sha256'] for x in cps};allcps+=cps
        checks.append(dict(block=r,arm=a,checkpoint_count=6,all_reference_full_state_exact=True,source_files_readonly=True,student_new_adam_count_matches_updates=True,student_initial_identity_matches=True,student_updates=500000))
    assert all(hashes[a]==hashes[0] for a in arms)
    pairing.append(dict(block=r,arms=arms,base_F_original_R_state_equal_at_all_six_checkpoints=True,checkpoint_rng_hashes=hashes[0],actual_a0_stream_replay_verified=True))
write(ROOT/'FROZEN_PARAMETER_AUDIT.json',dict(status='PASS',formal=checks,P_pilots=pilots,reference_native_gradients_zero=True,original_full_reference_and_optimizer_exactly_unchanged=True))
write(ROOT/'SAMPLER_PAIRING_AUDIT.json',dict(status='PASS',P_actual_tuple_tests=read(ROOT/'results/P_continuation/SAMPLER_PAIRING_AUDIT.json'),formal=pairing,legacy_input_replay=replay,scope='Immutable sampler and exact shared RNG checkpoint states; P direct tuple comparison; all actual 25k tuple exports retained.'))

episodes=pd.concat([pd.read_csv(ROOT/'results'/f'{tag}_episodes.csv') for tag in [f'source_A_b{r}' for r in range(3)]+[f'b{r}_a{a}_u{u}' for r in range(3) for a in arms for u in (100000,200000,300000,400000,500000)]],ignore_index=True)
assert len(episodes)==750+len(arms)*3*950
assert not episodes.duplicated(['block','arm','student_step','scene_id']).any()
export(ROOT/'EPISODES.csv',episodes)
for f in ['CHECKPOINT_SCORES.csv','PRIMARY_WINDOW_SCORES.csv','PAIRED_BLOCK_CONTRASTS.csv','TRAINING_BLOCK_DESCRIPTIVES.csv','DISTANCE_SUMMARY.csv','DISTANCE_CONTRASTS.csv','DISTANCE_START_CLUSTERS.csv']:
    assert not (ROOT/f).exists();shutil.copyfile(stats/f,ROOT/f)
distance=pd.concat([pd.read_csv(ROOT/'results'/f'b{r}_a{a}_u500000_distance.csv') for r in range(3) for a in arms],ignore_index=True)
assert len(distance)==len(arms)*3*93;export(ROOT/'DISTANCE_PROBE.csv',distance)
local_errors=[]
for row in distance.to_dict('records'):
    assert sha(row['trace_path'])==row['trace_sha256']
    with open(row['trace_path'],'rb') as f:trace=pickle.load(f)
    xy=np.asarray(trace['states']);action=np.asarray(trace['actions']);raw=np.asarray(trace['raw_actions'])
    assert 1<=len(action)<=25 and len(action)==row['steps'] and len(xy)==len(action)+1 and np.array_equal(np.clip(raw,-1,1),action)
    assert len(action)==25 or row['terminated'] or row['truncated']
    cells=np.trunc((xy+6)/4).astype(int)[:,::-1];start=tuple(json.loads(row['start_cell']));firsts=[tuple(x) for x in json.loads(row['firsts'])]
    exited=[(i,tuple(c)) for i,c in enumerate(cells[1:],1) if tuple(c)!=start]
    first_exit=exited[0][1] if exited else None;correct=int(first_exit in firsts) if first_exit else 0
    assert correct==row['correct'] and first_exit==(tuple(json.loads(row['first_exit'])) if json.loads(row['first_exit']) is not None else None)
    path=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());maximum=float(np.linalg.norm(xy-xy[0],axis=1).max())
    assert abs(path-row['path_length'])<=1e-9 and abs(maximum-row['maximum_displacement'])<=1e-9
    norm=np.linalg.norm(action,axis=1);assert abs(norm.mean()-row['mean_action_norm'])<=1e-9
    def reversals(v):
        return sum(np.linalg.norm(a)>.01 and np.linalg.norm(b)>.01 and np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))<-.5 for a,b in zip(v[:-1],v[1:]))
    dr=reversals(np.diff(xy,axis=0));ar=reversals(action)
    repeat=sum(any(np.linalg.norm(xy[i]-xy[j])<=.1 for j in range(i-2)) for i in range(3,len(xy)))
    assert dr==row['displacement_reversals'] and ar==row['action_reversals'] and repeat==row['repeated_visit_steps']
    inferred=np.linalg.norm(xy[1:]-trace['goal_xy'],axis=1)<=1.0
    assert np.array_equal(inferred,np.asarray(trace['termination_success'])[:,2].astype(bool))
    local_errors.append(dict(block=row['block'],arm=row['arm'],condition_id=row['condition_id'],CBR_exact=True,flags_exact=True,clip_exact=True,morphology_exact=True,path_length_error=abs(path-row['path_length']),maximum_displacement_error=abs(maximum-row['maximum_displacement'])))
export(ROOT/'LOCAL_INDEPENDENT_RECOMPUTATION.csv',pd.DataFrame(local_errors))
legacy=pd.read_csv(ROOT/'results/LEGACY_A0_CORRECTED_SUPERVISION.csv')
new=[]
for r in range(3):
    for a in arms[1:]:
        df=pd.read_csv(ROOT/'results'/f'b{r}_a{a}_training.csv');assert list(df.step)==list(range(25000,500001,25000))
        df['logging_origin']='DURING_ACTUAL_TRAINING_BATCH';df['cost_numeric_device']='GPU highest';new.append(df)
sup=pd.concat([legacy]+new,ignore_index=True);assert len(sup)==len(arms)*3*20
export(ROOT/'SUPERVISION_EXPOSURE.csv',sup)
actual_pairing=[]
for r in range(3):
    for u in range(25000,500001,25000):
        a0=sup[(sup.block==r)&(sup.arm==0)&(sup.step==u)].iloc[0]
        assert sha(a0.supervision_raw_path)==a0.supervision_raw_sha256
        with open(a0.supervision_raw_path,'rb') as f:base=pickle.load(f)
        for a in arms[1:]:
            row=sup[(sup.block==r)&(sup.arm==a)&(sup.step==u)].iloc[0]
            assert sha(row.supervision_raw_path)==row.supervision_raw_sha256
            with open(row.supervision_raw_path,'rb') as f:sample=pickle.load(f)
            for k in ['i','future','original_random','random_waypoint']:assert np.array_equal(base['indices'][k],sample['indices'][k])
            for k in ['goal','waypoint']:assert np.array_equal(base['indices'][k][:512],sample['indices'][k][:512])
            assert sample['sampler_state']['generators'][:3]==base['sampler_state']['generators'][:3]
            assert np.array_equal(sample['indices']['waypoint'][512:],sample['indices']['random_waypoint'][512:])
            assert sample['tau']==cal[r]['tau']
            if a==1:assert np.array_equal(sample['indices']['goal'][512:],sample['indices']['original_random'][512:])
            if a==2:assert np.all(sample['cost'][512:]<=sample['tau'])
            if a==3:assert np.all(sample['cost'][512:]>sample['tau'])
            actual_pairing.append(dict(block=r,arm=a,step=u,full_base_F_original_R_exact=True,actual_first_F512_exact=True,R512_present=True,random_waypoint_unchanged=True,accepted_goal_rule_verified=True))
formal_pair_audit=read(ROOT/'SAMPLER_PAIRING_AUDIT.json');formal_pair_audit['actual_logged_25k_tuple_comparisons']=actual_pairing;write(ROOT/'SAMPLER_PAIRING_AUDIT.json',formal_pair_audit)
export(ROOT/'GRADIENT_AUDIT.csv',pd.DataFrame([{k:v for k,v in x.items() if k not in ('path',)} for x in allcps]))
man=read(ROOT/'results/P_continuation/GATE_MANIPULATION.json');export(ROOT/'GATE_MANIPULATION.csv',pd.DataFrame(man))
oldcal=ROOT/'GATE_CALIBRATION.json'
if oldcal.exists():
    initial=ROOT/'results/P_initial_unadmitted';initial.mkdir(exist_ok=False);shutil.move(str(oldcal),str(initial/oldcal.name))
write(oldcal,cal)
write(ROOT/'P_RESULT_SELECTION.json',dict(authoritative_P_directory='results/P_continuation',authoritative_calibration='GATE_CALIBRATION.json',original_unadmitted_calibration='results/P_initial_unadmitted/GATE_CALIBRATION.json',initial_P_failure='P_ADMISSION.json',initial_budget_prediction='results/P_continuation/P_ADMISSION.json',time_cost_override='receipts/TIME_COST_AMENDMENT.json',original_budget_prediction_preserved=True,highest_precision_fixed_before_all_training_updates=True))

counter=read(ROOT/'receipts/EVALUATION_COUNTER.json');assert counter['complete_episodes']==len(episodes) and counter['local_trajectories']==len(distance) and counter['local_control_steps']==int(distance.steps.sum()) and counter['technical_episodes']==0
assert counter['complete_episodes']<=12200 and counter['local_control_steps']<=32000
assert all(read(ROOT/'results'/f'b{r}_a{a}_u{u}_evaluation.json')['independent_success_termination_recomputation'] for r in range(3) for a in arms for u in (100000,200000,300000,400000,500000))
if not both:
    assert not any((ROOT/'results'/f'b{r}_a{a}_status.json').exists() for r in range(3) for a in (2,3))
execution=dict(status='COMPLETE_ALL_AUTHORIZED_SCIENTIFIC_STAGES' if both else 'COMPLETE_I_STOP_II_REGISTERED_H_R_GATE',P='COMPLETE_WITH_RETAINED_ZERO_UPDATE_FAILURE_AND_REPAIR',I='COMPLETE_SIX_500K_STUDENTS',II='COMPLETE_SIX_500K_STUDENTS' if both else 'NOT_EXECUTED_H_R_GATE_FAILED',X='NOT_AUTHORIZED_NOT_EXECUTED',formal_updates=len(arms)*3*500000,pilot_updates=4000,pilot_weights_discarded=True,formal_restarts=0,complete_episodes=counter['complete_episodes'],source_A_episodes=750,local_trajectories=counter['local_trajectories'],local_control_steps=counter['local_control_steps'],student_checkpoint_selection='All planned points; primary fixed average300k/400k/500k',time_cost_limits_removed=True,scientific_sample_and_memory_caps_retained=True)
write(ROOT/'EXECUTION_STATUS.json',execution)
Pmon=[read(q) for q in (ROOT/'receipts').glob('P_*_MONITOR.json')];assert len(Pmon)==4 and all(x['child_exited'] for x in Pmon)
Pseconds=sum(x['conservative_gpu_seconds'] for x in Pmon);Iseconds=workflows[0]['conservative_gpu_seconds'];IIseconds=workflows[1]['conservative_gpu_seconds'] if both else 0
resource_record=dict(status='ALL_EXPERIMENT_WORKERS_EXITED',snapshot_epoch=time.time(),P_GPU_seconds=Pseconds,I_GPU_seconds=Iseconds,II_GPU_seconds=IIseconds,total_conservative_GPU_seconds=Pseconds+Iseconds+IIseconds,total_conservative_GPU_hours=(Pseconds+Iseconds+IIseconds)/3600,GPU_accounting='Each owned GPU job entire child lifetime, includes loading/JIT/gates/checkpoints; sum across concurrent workers. Prior reference1M costs remain prior costs.',owned_GPU_peak_bytes=max([x['peak_owned_gpu_bytes'] for x in Pmon+workflows]),owned_RSS_peak_bytes=max([x.get('peak_rss_bytes',x.get('peak_owned_rss_bytes',0)) for x in Pmon+workflows]),CPU_affinity=[72,73,74,75],CPU_core_union_limit=4,CPU_evaluation_process_wall_seconds_including_watch_idle=sum(read(ROOT/'receipts'/f'{s}_EVALUATION_EXIT.json')['elapsed_seconds'] for s in phases),CPU_preparation_job_wall_seconds=read(ROOT/'RUN_APPENDIX.json')['seconds']+sum(x['elapsed_seconds'] for x in Pmon),CPU_log_replay_seconds=replay['elapsed_seconds'],counter=counter,time_cost_limits_removed=True,memory_and_scientific_count_caps_retained=True,SSH_closed=False,SSH_note='Session remains open for publishing. Actual client closure is recorded afterward in local publication receipt.',round_wall_seconds_so_far=time.time()-read(ROOT/'receipts/REGISTRATION.json')['start_epoch'])
write(ROOT/'RESOURCE_AND_EXIT.json',resource_record)
admin=read(ROOT/'receipts/ADMINISTRATIVE_CPU_AFFINITY.json');advance=read(ROOT/'receipts/ADVANCE_PROCESS.json')
resource_record.update(administrative_CPU_monitoring_limitation=admin['limitation'],subsequent_administrative_CPU_affinity=[75],advance_driver_RSS_peak_bytes=advance['process_peak_RSS_bytes'],conservative_combined_RSS_peak_bound_bytes=resource_record['owned_RSS_peak_bytes']+advance['process_peak_RSS_bytes'],CPU_statistics_job_wall_seconds=sum(x.get('elapsed_seconds',0) for x in advance['jobs'] if x['script']=='analyze07.py'))
assert resource_record['conservative_combined_RSS_peak_bound_bytes']<16*1024**3

scores=pd.read_csv(stats/'PRIMARY_WINDOW_SCORES.csv');contrasts=pd.read_csv(stats/'PAIRED_BLOCK_CONTRASTS.csv');local=pd.read_csv(stats/'DISTANCE_SUMMARY.csv');summary=pd.read_csv(stats/'TRAINING_BLOCK_DESCRIPTIVES.csv')
ys=finalgate['raw_window_scores'];rows=[]
for r in range(3):
    d=contrasts[(contrasts.block==r)&(contrasts.contrast=='a0-a1')].iloc[0]
    rows.append(f"|{r}|{ys[str(r)]['a0']:.2f}|{ys[str(r)]['a1']:.2f}|{d.effect_pp:.2f}|[{d.scene_ci_low:.2f}, {d.scene_ci_high:.2f}]|")
source=[read(ROOT/'results'/f'source_A_b{r}_evaluation.json')['overall']*100 for r in range(3)]
hdr='通过' if both else '未通过';ii='按门槛完成阶段 II' if both else '按预登记规则未启动阶段 II'
scientific=finalgate['selection_verdict'] if both else 'REEXTRACTION_GAP_NOT_CONFIRMED_IN_THIS_PLATFORM'
detail=''
if both:
    detail='\nII：'+json.dumps({k:finalgate[k] for k in ['recovery_ratio','a2_minus_a3_pp','a3_minus_a1_pp','a2_minus_a1_far_CBR_pp','a2_minus_a0_pp','selection_verdict','method_verdict']},ensure_ascii=False,indent=2)+'\n'
report=f'''# SAW-REEXTRACT-07 执行报告

**I 的 H-R 门{hdr}；{ii}。** 本轮完成 {len(arms)*3} 条各500k学生训练，共{execution['formal_updates']:,}正式更新，{len(episodes):,}完整评价回合（含750个来源A回合），{len(distance)}条封存局部轨迹、{counter['local_control_steps']:,}局部控制步。X未授权、未执行；未补跑或挑检查点。

分支：[codex/saw-reextract-07-20261013](https://github.com/rollinpioneer/graph/tree/codex/saw-reextract-07-20261013)。来源基点6d16c1271b3fad563c58d9b2e34b48418b910860。用户后续明确“解除时间成本限制”；原GPU/CPU时限与成本预测门已撤销，500k、三个配对块、科学准入门、4核与12GiB显存/16GiB RSS/50GiB产物上限保留。

## 正式结果

Y为300k/400k/500k三个点、五任务等权均分，单位百分数；差值单位百分点。区间为5000次任务×评价流分层配对bootstrap，只覆盖固定模型的场景不确定性，三个检查点随同一场景携带，不是独立重复训练。

|配对块|a0 Y (%)|a1 Y (%)|a0−a1 (pp)|95% 场景区间 (pp)|
|---|---:|---:|---:|---|
{chr(10).join(rows)}

正式H-R要求三个原始差值各≥20pp，结果{hdr}。来源A-1M在同银行的一次能力参照分别为{', '.join(f'{x:.2f}%' for x in source)}；它们使用既有1M完整模型，不能当作500k学生的公平成本基线。逐任务、全部开发点与正式点见[CHECKPOINT_SCORES.csv](CHECKPOINT_SCORES.csv)，Task2/4/5和弱a0表现全部保留。n=3为参照来源与学生种子一起变化的配对训练单位，不是9个种子，也不支持精确总体泛化保证。
{detail}
## 技术准入与记录

首轮P在零更新时出现默认GPU矩阵精度下损失/梯度等价性失败，原失败回执保留。零更新诊断定位后固定`jax_default_matmul_precision=highest`，按原容差通过三个块的官方损失/学生梯度核验；不放宽容差。校准seed、索引与nearest-rank Q95规则保留，最高精度下正式tau为{', '.join(f'{x["tau"]:.8f}' for x in cal)}，均在训练前冻结。两个学生模块重新初始化、同块四臂参数相同，新Adam计数0/矩全零，原完整模型及原优化器只读。

四次各1000更新P功能试跑共4000步，权重丢弃。三个V操纵检查均通过；合法BFS映射100%，原F全量轨迹后缀支持最大BFS为4。这些仅核验采样工具，不是策略结果。P预测原时间预算超限的记录保留，后续人类撤销时间/成本门没有改变数学设计。

正式每个检查点核对原完整参照状态散列、模型文件、学生Adam计数、256个固定审计tuple的两项梯度范数与内积；全部通过。共享base/F/原R流在每个检查点状态完全相同；拒绝补抽流独立，P直接tuple审计通过。500k、同银行、温度0、原clip、原成功阈值≤1.0及1000步限制全部保持，保存状态独立重算成功/终止逐回合一致。

训练日志曾有x/y到行列的区域映射错误，仅影响诊断导出。运行中的a0未重启，旧源码/原CSV保留；其真实25k训练输入由冻结随机流重放，100k至500k全部Generator状态与真实检查点完全一致，修正区域并补齐实际tuple/轨迹身份/c/tau。学生/教师幅度和原贡献保留，没有用其他检查点猜测缺失学生状态。后续臂仅补充日志输出，不改损失、初始化、优化器或训练随机流。重放成本在CPU以最高精度描述性计算，不重新决定接受/拒绝。

## 局部探针与解释边界

原93条件全部复用，没有增加起点、目标、重复或删除6格选择偏差。68个主共同支持条件含12个中档6格，近/远主CBR只使用其余56条件，按目标→距离→通道→起点等权、8起点整簇bootstrap。所有93条逐条件数据在[DISTANCE_PROBE.csv](DISTANCE_PROBE.csv)，该银行已消费，属于辅助证据。6格和15格单列，不改写SAW06 seed1近缺口12.5pp造成旧联合门失败的永久裁定。

当前裁定：`{scientific}`。固定成熟V/teacher、随机新学生和500k有限重抽取与原1M逐步联合训练有实质区别。未重现门槛不能推出原差距不存在，也不能证明teacher/value演化是必要原因；a0较弱时应视为当前平台辨别力的限制。即使筛选有效，也只能支持该输入训练分布规则的受控效果，不能分离几何距离与区域/多样性、AWR与Waypoint、目标编码器与动作平均，不能写`ACTION_AVERAGING_PROVEN`。

## 资源与归档

GPU驻留逐作业整段生命周期相加：P {Pseconds/3600:.4f}小时，I {Iseconds/3600:.4f}小时，II {IIseconds/3600:.4f}小时，总计{resource_record['total_conservative_GPU_hours']:.4f}小时。包含加载、JIT、门控、训练及保存；既有参照训练成本没有归零。CPU评价进程墙钟（含等待检查点）{resource_record['CPU_evaluation_process_wall_seconds_including_watch_idle']/3600:.3f}小时。自有显存峰值{resource_record['owned_GPU_peak_bytes']/1024**3:.3f}GiB，自有RSS峰值{resource_record['owned_RSS_peak_bytes']/1024**3:.3f}GiB，CPU核并集72–75。全部实验子进程已退出；最终Git发布后再记录SSH客户端真实退出。

上述CPU并集来自正式训练/评价/分析工作进程的固定亲和性。前期管理shell/Git未纳入此工作进程监控；一次小型推送报告默认112个压缩线程，其实际同时使用核数未测，不能把这一阶段的管理开销宣称也已核验为4核以内。发现后shell及后续归档子进程绑定75，Git压缩显式1线程；原回执见receipts/ADMINISTRATIVE_CPU_AFFINITY.json。

轻量代码、配置、逐回合CSV、区间、审计、图与回执上传本分支；权重、银行、完整轨迹和日志留服务器，由[RAW_ARTIFACT_INDEX.json](RAW_ARTIFACT_INDEX.json)给路径、大小和SHA。最终提交及传输核验见本地交接材料。
'''
text(ROOT/'REPRODUCTION_REPORT.md',report)
text(ROOT/'HYPOTHESIS_VERDICT.md',f'''# 预登记裁定

I：`{gate['verdict']}`，三个a0−a1原始差值{gate['raw_gap_pp']}pp，门槛各≥20pp，不依据区间或单点成绩替换门。

II：{execution['II']}。H-SEL/H-FAR/H-M：{'见最终门控JSON' if both else 'NOT_TESTED，不是阴性成绩或0%'}。最终规则裁定`{scientific}`。

动作平均根因未证明，原1M联合训练差距未被完全解释。旧SAW06联合门失败保留。P失败与数值修复、日志导出修正完整保留。n=3配对训练块、区间只含固定模型场景或固定起点不确定性。
''')
text(ROOT/'NEXT_DECISION.md',f'''# 后续决定

本卡结束：{execution['status']}。{'分析既定筛选模式与相对a0方法门；结论见报告。' if both else 'H-R未过，按原门停止II。当前固定成熟参照＋随机学生＋500k平台不足以复制注册差距；不自动换参照、调tau、延训或补seed。'}

X的GCIVL/QRL范围问题仍可独立讨论，但必须另卡冻结与授权；本轮不启动。若a0能力有限，应区分平台无辨别力与a1不差，不能宣布联合学习动态必不可少。保留原SAW06反例，不追认旧结果。
''')
text(ROOT/'README.md','''# SAW-REEXTRACT-07

先读[执行报告](REPRODUCTION_REPORT.md)、[真实状态](EXECUTION_STATUS.json)、[裁定](HYPOTHESIS_VERDICT.md)。用户原卡SAW07.md保留，执行登记见EXPERIMENT_CARD.md，时间/成本撤销见receipts/TIME_COST_AMENDMENT.json。P初始失败与预算预测是历史回执，最终选择见P_RESULT_SELECTION.json。无训练大产物进Git。
''')
write(ROOT/'HYPOTHESIS_GATE.json',finalgate)
append=read(ROOT/'RUN_APPENDIX.json');append.update(status=execution['status'],final_execution=execution,original_time_cost_fields_are_historical_and_superseded=True,active_time_cost_limits=None,final_source_commit_before_export=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip());write(ROOT/'RUN_APPENDIX.json',append)
card=ROOT/'EXPERIMENT_CARD.md';card.write_text(card.read_text()+'\n最终执行：'+execution['status']+'；详见REPRODUCTION_REPORT.md。全部实验子进程结束后归档，SSH真实退出另由本地发布回执记录。\n')

# Pure figures consume only existing CSV; no new rollouts or model probes.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plots=ROOT/'figures';plots.mkdir(exist_ok=False);colors=['#2878b5','#d45b38','#278455','#8759a6']
checkpoint=pd.read_csv(stats/'CHECKPOINT_SCORES.csv');allscore=checkpoint[checkpoint.task=='ALL_EQUAL_WEIGHT']
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for r,ax in enumerate(axs):
    for a in arms:
        g=allscore[(allscore.block==r)&(allscore.arm==a)].sort_values('student_step');ax.plot(g.student_step/1000,g.success_rate_percent,'o-',label=f'a{a}',color=colors[a])
    ax.axhline(source[r],ls=':',color='#555555',label='source A-1M');ax.set(title=f'Paired block {r}',xlabel='Student updates (k)',ylim=(0,100));ax.grid(alpha=.2)
axs[0].set_ylabel('Five-task success (%)');axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(plots/'LEARNING_CURVES.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for a in arms:ax.plot(range(3),[ys[str(r)][f'a{a}'] for r in range(3)],'o-',label=f'a{a}',color=colors[a])
ax.set(xlabel='Paired training block',ylabel='Fixed 300/400/500k window success (%)',xticks=[0,1,2],ylim=(0,100));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(plots/'PAIRED_WINDOW_SCORES.png',dpi=180);plt.close(fig)
oldpath=Path('/home/xushijie3/work/graph_saw_distance06_20261013/experiments/saw_distance06_20261013/DISTANCE_CONTRASTS.csv');old=pd.read_csv(oldpath)
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for r,ax in enumerate(axs):
    for a in arms:
        g=local[(local.block==r)&(local.arm==a)].set_index('band');ax.plot([0,1],[100*g.loc[b,'CBR'] for b in ['near','far']],'o-',label=f'new a{a}',color=colors[a])
    prev=old[old.seed==r].iloc[0]
    for name,col in [('A','#555555'),('B','#999999')]:ax.plot([0,1],[100*prev['CBR_'+name+'_'+b] for b in ['near','far']],'s--',color=col,label='SAW06 '+name)
    ax.set(title=f'Block {r}; consumed bank',xticks=[0,1],xticklabels=['near','far'],ylim=(0,105));ax.grid(alpha=.2)
axs[0].set_ylabel('Correct branch rate (%)');axs[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(plots/'DISTANCE_EXISTING_BANK_COMPARISON.png',dpi=180);plt.close(fig)
write(ROOT/'figures/PROVENANCE.json',dict(only_existing_results_used=True,new_model_queries=0,new_physics_steps=0,old_distance_csv_path=str(oldpath),old_distance_csv_sha256=sha(oldpath),new_scientific_CSV_unchanged=True,II_arms_in_figures=both))
aux=distance[distance.distance.isin([6,15])].groupby(['block','arm','bank','role','distance'],dropna=False).agg(conditions=('condition_id','count'),CBR=('correct','mean')).reset_index();export(ROOT/'DISTANCE_AUXILIARY_6_15.csv',aux)

# Protect original worktrees, and index all newly stored raw artifacts.
registration=read(ROOT/'receipts/REGISTRATION.json');protected={}
for path,prior in registration['original_worktrees'].items():
    head=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip();state=subprocess.check_output(['git','-C',path,'status','--short'],text=True)
    actual=dict(head=head,status_count=len(state.splitlines()),status_sha256=hashlib.sha256(state.encode()).hexdigest());assert actual==prior;protected[path]=actual
write(ROOT/'receipts/ORIGINAL_WORKTREE_PROTECTION.json',dict(status='PASS',original_worktrees=protected))
raw=[]
for d in [ROOT/'runs',ROOT/'source_ref',ROOT/'results/P_continuation/runs']:
    for q in sorted(d.rglob('*')):
        if q.is_file():raw.append(dict(path=str(q),sha256=sha(q),bytes=q.stat().st_size))
assert sum(x['bytes'] for x in raw)<50*1024**3
write(ROOT/'RAW_ARTIFACT_INDEX.json',dict(raw_files=raw,total_new_raw_bytes=sum(x['bytes'] for x in raw),reference_files=[x['reference'] for x in ident],evaluation_bank=read(ROOT/'EVALUATION_BANK.json'),raw_outputs_stay_on_server=True))
resource_record.update(final_export_seconds=time.time()-started,final_export_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_round_bytes_snapshot=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file()))
assert resource_record['new_round_bytes_snapshot']<50*1024**3;write(ROOT/'RESOURCE_AND_EXIT.json',resource_record)
print('FINAL_EXPORT_DONE',execution['status'],scientific,flush=True)
