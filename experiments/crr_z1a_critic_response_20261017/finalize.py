from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent
def load(n):return json.loads((R/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
crit=[load(f'runs/critic_s{s}/complete.json') for s in range(4)];health=[load(f'runs/healthy_s{s}/complete.json') for s in [1,2]]
assert all(load(f'runs/critic_s{s}/exit.json')['exit_code']==0 for s in range(4)) and all(load(f'runs/healthy_s{s}/exit.json')['exit_code']==0 for s in [1,2])
assert sum(x['critic_only_updates'] for x in crit)==60000 and sum(x['updates'] for x in health)==80000
for a in load('asset_manifest.json'):
 if a['sha256']:assert sha(a['path'])==a['sha256']
for s in [1,2]:
 for a in load(f'runs/healthy_s{s}/complete.json')['new_checkpoints']:assert sha(a['path'])==a['sha256']
for s in range(4):
 for a in load(f'runs/critic_s{s}/complete.json')['checkpoints']:assert sha(a['path'])==a['sha256']
rows=list(csv.DictReader(open(R/'metrics.csv')));end=[x for x in rows if x['updates']=='5000' and x['region']=='start'];decision=load('decision.json');label=decision['status']
labels={'LIMITED_GO_MORE_CANDIDATES_AND_LATER_FPU_ONLY':'有限GO：只支持扩大候选时间点并另行验证行为后果','INCONCLUSIVE_ONLY_ONE_BAD_CANDIDATE':'INCONCLUSIVE：只有一个坏轨迹候选达到预注册条件','NO_GO_CURRENT_SHORT_RESPONSE_DISCRIMINATION':'NO-GO：本轮短期响应指标未形成预注册的坏/健康区分'}
cost=dict(healthy_GFP_updates=80000,critic_only_updates=60000,total_optimizer_steps_across_scopes=140000,new_environment_construct_reset_step=0,new_behavior_episodes=0,successful_job_core_wall_seconds={**{f'critic_s{s}':crit[s]['seconds'] for s in range(4)},**{f'healthy_s{s}':health[j]['seconds'] for j,s in enumerate([1,2])}},successful_jobs_cpu_user_seconds=sum(x['cpu_user_seconds'] for x in crit+health),successful_jobs_cpu_system_seconds=sum(x['cpu_system_seconds'] for x in crit+health),successful_jobs_peak_rss_kib={**{f'critic_s{s}':crit[s]['peak_rss_kib'] for s in range(4)},**{f'healthy_s{s}':health[j]['peak_rss_kib'] for j,s in enumerate([1,2])}},zero_update_startup_failures='initial missing runtime field plus native-admission compilation-input mismatch; all0 updates, retained logs and lifecycle receipts',extra_readonly_audits=2,readonly_audit_GPU_seconds='not independently metered; logs retained; do not claim exact total GPU hours',cost_caution='Shared GPU lifecycle/core wall time is not exclusive device compute time. CPU totals here cover successful workers only; failed startup and readonly-audit CPU time not independently measured.')
save('cost_summary.json',cost)
generated=[]
for s in [1,2]:
 for x in load(f'runs/healthy_s{s}/complete.json')['new_checkpoints']:generated.append(dict(seed=s,role='new_ordinary_GFP_continuation_NOT_original_unsaved_checkpoint',**x))
save('generated_healthy_assets.json',generated)
text=f'''# CRR-Z1a：受控 TD Actor 候选替换的 Critic-only 筛查

**裁决：{labels[label]}。** 这不是CRR门控性能验证，也没有后续行为结果。所有后续候选扩展、F/P/U完整续训或新种子实验均未启动。

## 实际完成

四个候选：坏轨迹s0 650k→660k、s3 310k→320k；预定健康种子s1 650k→660k、s2 310k→320k。每个候选F/P/F′各5000次Critic-only更新，共60000；真实退出码全部0。

健康权重最初缺失，经用户另外明确批准，从s1 600k续训60000步、s2 300k续训20000步，共80000次原版GFP更新，生成四份权重，另计预算、退出0。这些是本轮新续训产物，不冒充原始未保存权重；没有补做行为评价来确认新生成候选的健康程度。

没有新增环境构造、reset、step或评价回合。Critic-only阶段Actor、VaBC与两种TD生成器均冻结；仅在线Critic及按原时序更新的Target Critic变化。

## 主指标与数值结果

主指标 R = 起点状态上 mean(||∇a Q_P−∇a Q_F||) / 分叉时 mean(||∇a Q_fork||)。N用相同公式将P/F换成F/F′。它是原始在线双头均值Q的**动作输入梯度**，不是完整Actor参数梯度，也没有乘Actor损失的全局Q归一化系数。

| 候选种子 | latent | R（95%数据轨迹区间） | F/F′噪声N | 分叉梯度范数 | F末梯度范数 | P末梯度范数 |
|---|---:|---:|---:|---:|---:|---:|
'''
for x in end:
 text+=f"| s{x['seed']} | {x['latent']} | {float(x['R_PminusF_vector']):.4f} [{float(x['R_lo']):.4f}, {float(x['R_hi']):.4f}] | {float(x['N_FminusFprime_vector']):.4f} | {float(x['G_fork']):.4f} | {float(x['G_F']):.4f} | {float(x['G_P']):.4f} |\n"
text+='''
所有0/1000/2000/3000/4000/5000点、起点/非起点、两套latent，以及每臂绝对范数变化、相对分叉变化、向量漂移、Target Q绝对漂移和局部对比度均在`metrics.csv`。不能只根据P/F比值判断P有害；须同时读F、P绝对曲线。

预注册的保守充分条件：两个坏候选各自R的下界，均超过两个健康候选R的上界，并超过自身N的上界，两套latent分别满足。区间按原始数据轨迹聚类重采样400次；它们不估计优化随机流或独立训练种子的方差。只有一个F′，无法宣称充分刻画所有优化噪声。

| 坏候选 | 两套latent均通过上述条件 |
|---|---|
'''
for s in [0,3]:text+=f"| s{s} | {decision['per_bad_candidate_passed'][str(s)]} |\n"
text+='''
## 本轮证据与推测分开

主指标上，两个坏候选的点估计都高于健康对照，但s0约0.45的候选响应与自身约0.45的F/F′噪声重叠；s3约0.80明显高于自身约0.50的噪声。因此只有s3满足预注册的充分条件，不能升级为两个坏候选均可分。

绝对范数提供另一层事实：s3的P从约0.599升至0.841，而F降至约0.554，P自身确实增加约41%，不是单纯由F下降制造比值。s0的P也升约13%，F大致不变。这个辅助范数变化值得保留，但不能在看到结果后把它替换为新的主指标来宣布通过；向量噪声与范数噪声回答的不是同一问题。

可以推测s3这个更新点对Critic局部梯度更敏感，但目前没有F/P/U完整行为后果，不能据此把该Actor更新定性为有害，不能将局部梯度放大等同价值估计错误或行为崩塌。

## 如何解释，哪些还不能判断

本轮只能判断这四个已指定候选在固定离线数据探针上的短期响应是否可分。坏候选来自已消费的坏训练轨迹，不等于该10k Actor变化本身已经被证明有害。健康对照来自原先表现较好的种子，但新生成路径没有新的行为验收；不能把种子标签当作每个新检查点行为健康的证明。

即使排序全部正确，也只是开发筛查，不能宣布风险预测、泛化或长期Actor–Critic反馈机制成立。若区分失败，应降低当前短期响应量的优先级，不能看到某一子区域或中间时间点表现较好就事后改变主裁决。若仅一个坏候选通过，保留INCONCLUSIVE，不挑选训练阶段包装正结果。

若下一阶段另获批准，必须检验完整F/P/U续训的行为后果；本轮不新增此类更新或回合。

## 计算接口与可信性

- F/P起点使用后一个检查点的同一在线Critic、Target Critic、Adam一阶/二阶矩及计数。F/P数据批次一致；F′只改批次流，TD噪声键保持一致。新数据随机流已预注册，没有冒充原轨迹后续批次恢复。
- TD Actor分别为前/后检查点副本；VaBC固定为后检查点副本。VaBC在当前状态生成动作、在下一状态评价；target_use_next=False，q_agg与target_agg均mean。
- 更新损失直接调用固定官方GFPAgent.critic_loss；Target EMA保持原源码使用更新前在线Critic的时序。只提取并恢复Critic的Adam状态，其他参数组保持源文件身份，不声称自定义Critic-only状态可直接作为普通完整GFP续训状态。
- 四组首4次真实更新均计入5000预算；在线Critic确实变化，指定冻结参数不变，保存/加载自定义Critic-only状态精确一致，Adam计数正确。分叉点各臂探针一致，原始源权重和已加载源状态前后不变。
- 所有数值必须有限；实际各臂计数及退出码、哈希见回执。原版健康生成调用官方fused4，不改超参数或在线训练范围。

## 启动问题与额外成本：不隐瞒

最初四个作业因恢复配置漏补运行时log_metrics开关，在零更新阶段退出；随后Critic零更新准入发现重写损失/原生对照不一致。改为直接使用原生损失后，对照还需让两侧VaBC权重以一致的动态参数传入，消除不同编译输入方式的偏差。统一后，独立对照的损失一致、梯度最大误差约6e-8；没有放宽3e-5容差。

这些失败均在真实优化器更新之前，未生成已更新Critic检查点；目录、日志和真实退出码全部保留。健康生成在运行时配置修复后成功，未因后续Critic准入修正而重跑。四个Critic候选在准入修正后各执行一次完整5000×3；没有重跑已执行的训练更新。

零更新失败和两次只读审计仍消耗编译、推理、CPU/GPU时间，不能当作免费。完整生命周期、资源采样及失败日志留服务器。只读审计和失败CPU时间未独立精确计量，因此本报告不声称精确的总GPU·小时。

'''
text+=f"成功作业核心墙钟秒数：`{cost['successful_job_core_wall_seconds']}`。成功作业CPU用户/系统秒数：{cost['successful_jobs_cpu_user_seconds']:.2f}/{cost['successful_jobs_cpu_system_seconds']:.2f}。这些是共享GPU上的运行时间，不是独占设备基准。数据准备另见`data_manifest.json`。\n\n"
text+='''## 交付与边界

`gradient_response.png`：主响应与F/F′噪声；`absolute_gradients.png`：F/P/F′绝对梯度；`q_response.png`：局部Q对比度。完整逐点数值和预注册裁决分别见`metrics.csv`、`decision.json`。

服务器：`/home/xushijie3/work/crr_z1a_critic_response_20261017`。完整权重、数据、向量探针和日志仅留服务器，`server_artifact_index.json`给出路径/大小/哈希。轻量文件按显式白名单归档。

分支：https://github.com/rollinpioneer/graph/tree/codex/crr-z1a-critic-response-20261017 。普通提交推送后验证远端SHA，并同步本地reports；无PR、合并、旧资产覆盖或追加实验。
'''
(R/'RESULTS.md').write_text(text)
(R/'HANDOFF.md').write_text(f'''# CRR-Z1a handoff
Status: COMPLETE; decision={label}.
Root: /home/xushijie3/work/crr_z1a_critic_response_20261017
Branch: codex/crr-z1a-critic-response-20261017
URL: https://github.com/rollinpioneer/graph/tree/codex/crr-z1a-critic-response-20261017
Exactly60000 Critic-only updates plus separately approved80000 ordinary GFP healthy generation. All successful exits0. No environment operations/evaluation episodes. Startup/admission failures consumed0 updates and are preserved separately; costs not hidden. Full interpretation RESULTS.md. No further candidates, F/P/U, gate training or new seeds authorized. No ongoing jobs need monitoring or restart.
''')
art=[]
for folder in ['arrays','runs','startup_failure_zero_updates','startup_native_loss_admission_failure','receipts']:
 for p in sorted((R/folder).rglob('*')):
  if p.is_file():art.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
save('server_artifact_index.json',art)
print('FINALIZED',label,'ARTIFACTS',len(art))
