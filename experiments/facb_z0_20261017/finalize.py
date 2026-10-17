"""Integrity, factual report and explicit lightweight archive list."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent;t=time.monotonic()
def load(n):return json.loads((R/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
r=load('separability.json');forward=load('receipts/inference.json');ex=load('receipts/inference_exit.json');data=load('data_identity.json');records=load('checkpoint_manifest.json');src=load('source_manifest.json');audit=load('receipts/compiler_boundary_audit.json')
assert ex['exit_code']==0 and forward['forbidden_attempts']==0
assert ex['total_gpu_hours_conservative_upper_bound']<=2
for row in records:
 assert sha(row['path'])==row['sha256'] and sha(row['config_path'])==row['config_sha256']
for row in src['sources']:assert sha(row['path'])==row['sha256']
assert sha(data['path'])==data['sha256'] and sha(R/'arrays/samples.npz')==data['sample_sha256']
assert sha(R/'sample_indices.csv')==data['index_csv_sha256']
assert sha(R/'calibration.json')==load('receipts/calibration_lock.json')['sha256']
validations=load('receipts/forward_validation.json');maxq=max(x.get('native_q_max_abs_error',0) for x in validations);maxv=max(x.get('value_direct_max_abs_error',0) for x in validations);maxmapping=max(x.get('target_parameter_mapping_error',0) for x in validations)
final=dict(status='PASS',source_checkpoints_unchanged=len(records),configs_unchanged=True,data_and_indices_unchanged=True,source_snapshots_unchanged=True,calibration_lock_unchanged=True,parameters_optimizer_rng_unchanged=True,training_updates=0,environment_construct_reset_step=0,new_episodes=0,new_value_weights_initialized=0,model_q_max_abs_error=maxq,model_v_max_abs_error=maxv,target_parameter_mapping_max_abs_error=maxmapping,process_exit_code=ex['exit_code'],gpu_budget_pass=True,seconds=time.monotonic()-t)
save('receipts/final_integrity.json',final)
cal=r['calibration'];h=r['healthy_overall'];p=lambda x:f'{100*x:.3f}%'
signed=r['calibration_signed_audit']
regions=[('all_unique','预定7168状态并集（主判据）'),('global_uniform','均匀全局4096'),('start_dedicated','起点专门2048'),('near_goal_dedicated','近目标专门1024'),('all_next_outside_start','并集中全部非起点')]
region_rows='\n'.join(f"| {name} | {p(h[key]['pooled']['upper'])} | {p(h[key]['pooled']['lower'])} |" for key,name in regions)
seed_rows='\n'.join(f"| {s} | {p(x['all_unique']['upper'])} | {p(x['all_unique']['lower'])} | {p(x['start_dedicated']['upper'])} | {p(x['near_goal_dedicated']['lower'])} |" for s,x in r['healthy_by_seed'].items())
island_rows='\n'.join(f"| {name} | {p(x['pooled']['upper'])} | {p(x['0']['upper'])} | {p(x['1']['upper'])} | {x['pooled']['quantiles'][4]:.3f} | {x['pooled']['quantiles'][6]:.3f} |" for name,x in r['island_start'].items())
noise_rows='\n'.join(f"| {n} | {p(h['all_unique'][n]['upper'])} | {p(h['all_unique'][n]['lower'])} |" for n in ['0','1'])
sensitivity=load('anchor_sensitivity.json');sens_rows=[]
for name in r['island_start']:
 a=[x for x in sensitivity if x['id']==name and not x['own_seed']]
 sens_rows.append(f"| {name} | {p(min(x['upper'] for x in a))}–{p(max(x['upper'] for x in a))} | {sum(x['upper']>.5 for x in a)}/7 |")
aux=[]
for x in r['details']:
 if x['group'].startswith('aux_') and x['region']=='start_dedicated' and x['noise']=='pooled':aux.append(f"| {x['id']} | {x['group']} | {p(x['upper'])} | {x['quantiles'][4]:.3f} |")
gate_rows='\n'.join(f"| {x['noise']} | {x['healthy_upper_pass']} | "+' / '.join(f"{k}: {v}" for k,v in x['islands_upper_pass'].items())+' |' for x in r['threshold_checks'])
if r['decision']=='GO_FOR_CONTROLLED_SCREEN':
 conclusion='当前代理锚满足预先冻结的基本可分性门槛，可作为下一张受控训练筛查卡的依据。它不证明FACB有效，也不验证正式独立预训练锚。'
 choice='健康整体下界触发超过5%，按卡降低双边优先级，仅保留单边上界候选。' if r['healthy_lower_above_5percent'] else '健康整体下界未超过5%；按卡，未来可比较单边与双边，但仍须查看局部区域风险。'
else:
 conclusion='当前代理锚未满足冻结的可分性条件，停止使用这一代理条件进入FACB训练；不修改c、百分位或挑选种子挽救。'
 choice='主可分性未通过，单边/双边的后续训练选择不放行；下界风险仅作描述。'
report=f'''# FACB-Z0 冻结价值锚可分性检查

日期：2026-10-17。裁决：**{r['decision']}**。

{conclusion}

## 冻结标定与身份

仅使用GFP健康s1/s2的600k、800k、1M与各自I臂500k V。标定记录数{cal['calibration_samples']}，来自六个检查点、两套固定噪声与7168个不同状态；不是相互独立的同等数量数据轨迹。
奖励范围[-1,0]、γ=0.995、S=200；99%绝对偏离分位数给出 **κ={cal['kappa']:.9f}，c={cal['c']:.6f}**。带宽在任何留出/价值岛G检查点评分前写入并锁定哈希，此后未变。

在有效标定组中，d<0占{p(signed['negative_fraction'])}，d中位数{signed['median']:.3f}；绝对偏离最大的约1%记录中，负侧占{p(signed['top_absolute_1percent_negative_fraction'])}。这说明当前绝对偏离标定的尾部来自哪一侧，仅描述两个模型的差异，不证明任何一方正确。即使分布图显示某些差别，也不能改用另一阈值替换本卡预定c。

I臂代理来自正式确认实验的同种子500k，8个种子全部使用，未选择有利锚。结构4×512、LayerNorm、单头、69维状态输入，无观测归一化。expectile=0.7，回归数据动作处Target Critic两头均值；V与其他网络同期训练、每批先更新V。官方IQL源码使用Q头最小值，FACB计划是独立预训练后冻结，因此本代理不能冒充正式独立锚。
实际资产路径为gfp_eta0_ablation_20261015/analysis_p0p1/confirm_scratch，卡片的gfp_confirm_scratch_20261016为Git归档前缀。I实现与固定归档提交8c7caf61b8b91b85c7340df04a2db5c10eeb0a6f一致；官方GFP关键文件与1abbd5f0e287caf9bda70e7bc655d0a499ac8401身份记录一致。全部34个文件（26个G、8个I）及配置SHA见checkpoint_manifest.json。

## 健康检查：s4–s7，各600k/800k/1M

上界是d>c，下界是d<−c，d=qG−V500k。整体主判据使用所有预定样本并集等权合并；全局均匀样本另列，不能把区域增强抽样当作原数据总体频率。

| 区域 | 上界触发 | 下界触发 |
|---|---:|---:|
{region_rows}

| 种子（三检查点合并） | 整体上界 | 整体下界 | 起点上界 | 近目标下界 |
|---|---:|---:|---:|---:|
{seed_rows}

两套噪声分别报告；噪声编号0/1对应预注册基种子2026101701/2026101702。

| 噪声 | 健康整体上界 | 健康整体下界 |
|---|---:|---:|
{noise_rows}

## 已知价值岛：预定2048条起点状态

| 检查点 | 合并上界触发 | 噪声0 | 噪声1 | d中位数 | d的95%分位 |
|---|---:|---:|---:|---:|---:|
{island_rows}

“价值岛组”表示此前被诊断为有岛的检查点，不表示所采2048条起点状态逐条都具有真实估值错误。这里测到的是相对代理锚越界比例，不是真实价值误差检出率。

| 判定所用噪声 | 健康≤1% | 两个价值岛分别>50% |
|---|---|---|
{gate_rows}

{choice} 两者始终使用同一c，未作单边专属调参。完整逐种子、逐检查点、区域、分位数见separability.json及trigger_rates.csv。

## 预先声明的辅助与敏感性

| 辅助检查点 | 角色 | 起点上界触发 | d中位数 |
|---|---|---:|---:|
{chr(10).join(aux)}

上述异常前与阶段匹配检查没有参与带宽或主裁决。稀疏检查点和已有异常标签不构成前瞻预警验证。

把同一G评分分别与其余7个种子的V比较，仍沿用原c：

| 价值岛检查点 | 换其他种子V后的上界比例范围 | 超过50%的其他V数 |
|---|---|---:|
{chr(10).join(sens_rows)}

全部交叉结果见anchor_sensitivity.json。跨种子结果不用于挑选锚或重新标定。分布图distributions.png及200箱密度重叠系数仅用于描述，不添加事后通过标准。

## 计算语义及零更新核验

- qG保持原TD：Actor(s′)、VaBC(s)，在s′用Target Critic各自双头均值评分后取两个来源的均值。两分支使用同一个官方sample_rng。没有修改target_use_next=False。
- OGBench1.1.0纯加载器读取训练实际数据。当前状态qpos在目标0.5米内时r=0、m=0，否则r=−1、m=1；不改为下一状态奖励。区域使用next_observations。
- 训练数据有效转移{data['valid_transitions']}条，任务mask=0共{data['goal_terminal_mask0_count']}条，另有{data['dataset_boundary_mask1_count']}条数据轨迹边界但原生mask仍为1。无效终端占位行剔除；mask=0不进入主要偏离统计。构造的mask=0前向夹具确认目标退化为r；未执行环境。
- 原生TD评分对照最大绝对误差{maxq:.9g}，V同编译方式恢复对照误差{maxv:.9g}，每个G的Target分支与保存Target参数直接应用的对照误差最大{maxmapping:.9g}，均通过原3e−5容差；原生critic_loss前向也通过预注册rtol/atol。没有执行优化器更新、生成新网络参数或改写旧检查点。
- 最终重新核验34个检查点、配置、源码、数据、样本与带宽哈希不变；保存模型/优化器/RNG树前后不变。完整回执见receipts/final_integrity.json、forward_validation.json、integrity.json。

## 失败记录与真实成本

首次CPU准备因无.git源码快照的版本读取失败，保留prepare_attempt1.log；改为校验既有源码身份记录，未计算模型评分。
两次GPU核验失败均保留真实退出码1和日志：首次是V的JIT/非JIT边界对照差异{audit['jitted_module_vs_eager_direct']:.9f}，同JIT或同非JIT调用差异均0；第二次是重建损失残留的非JIT Critic前向（误差0.0001086），统一完整JIT边界后该夹具差异0。未放宽容差。
随后独立参数映射审计发现本轮只读模型恢复时复用了模块对象，Target分支错误读取在线参数，对照误差4.580662。按官方create的deepcopy方式分开模块后，直接Target参数对照误差0。此前所有G评分、代理相关核验及带宽全部作废，保留在receipts/invalid_restoration_attempt2，不参与任何结论；这不是根据结果挑选阈值。修正提交后，使用完全相同的样本/噪声/种子/百分位重算，得到本报告唯一有效带宽，并再次在正式留出/价值岛评分前锁定。原有训练源文件和权重全程未改。
正式有效推理真实退出码0。完成后的每个G均另对照保存的Target参数直接前向，避免两个错误路径互相验证。

完成26组G×两套噪声×7168状态={forward['G_state_noise_scores']}次评分（主20组、辅助6组），另8组V共{forward['anchor_state_scores']}次前向。未以这些相关样本数量作独立样本统计显著性推断。
修正后GPU进程墙钟{ex['wall_seconds']:.2f}秒，推理CPU {forward['cpu_user_seconds']+forward['cpu_system_seconds']:.2f}秒，最大进程RSS {forward['peak_rss_kib']/1024**2:.3f}GiB，采样到本进程GPU显存峰值{ex['own_gpu_peak_mib']}MiB。共享GPU0，不能把设备总利用率当本作业独占利用率。
前两次失败GPU检查合计{ex['prior_failed_attempt_seconds']:.2f}秒；专门编译审计计算墙钟{audit['wall_seconds']:.2f}秒、CPU {audit['cpu_seconds']:.2f}秒，另外的损失与Target映射审计成本见loss_boundary_audit.json、target_identity_audit.json。三项审计GPU驻留合计保守预留60秒。包含失败和审计后，总GPU驻留成本保守上界 **{ex['total_gpu_hours_conservative_upper_bound']:.6f} GPU·h**，低于2GPU·h。失败过程还包含重复V/部分G推理，不隐藏这部分计算；不能只把最后一次运行计为总成本。前两次失败进程CPU未完整采集，明确留缺，不计为0。CPU数据准备、汇总和最终哈希检查另有成本，分别留回执。

**参数更新0；新回合0；环境构造/reset/step均0；新价值网络权重初始化0。** 本轮不自动启动FACB训练，也不因未通过而更换带宽或代理锚。相对V的偏离不等于相对真实回报的误差；结论限于当前代理条件。

分支：[codex/facb-z0-20261017](https://github.com/rollinpioneer/graph/tree/codex/facb-z0-20261017)。权重、原始数组与完整成功日志留服务器，轻量清单归档并同步本地。
'''
(R/'RESULTS.md').write_text(report,encoding='utf-8')
large=[p for p in (R/'arrays').iterdir() if p.is_file()]+list((R/'receipts/invalid_restoration_attempt2').glob('*.npy'))
save('server_artifact_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in large])
names=['EXPERIMENT_CARD_ORIGINAL.md','EXECUTION_FREEZE.md','prepare.py','infer.py','run_inference.py','audit_forward.py','audit_loss.py','audit_target_identity.py','analyze.py','finalize.py','protocol.json','checkpoint_manifest.json','configs.json','source_manifest.json','anchor_identity.json','data_identity.json','sample_indices.csv','calibration.json','separability.json','anchor_sensitivity.json','trigger_rates.csv','distributions.png','RESULTS.md','server_artifact_index.json']
names += [str(p.relative_to(R)) for p in (R/'sources').iterdir() if p.is_file()]
names += [str(p.relative_to(R)) for p in (R/'receipts').glob('*.json')]
names += ['receipts/prepare_attempt1.log','receipts/inference_attempt1.log','receipts/inference_attempt2.log']
names += [str(p.relative_to(R)) for p in (R/'receipts/invalid_restoration_attempt2').glob('*.json')]
names=sorted(set(names));assert all((R/n).is_file() and (R/n).stat().st_size<2000000 for n in names)
save('archive_manifest.json',{n:dict(bytes=(R/n).stat().st_size,sha256=sha(R/n)) for n in names})
print('FINAL',r['decision'],'FILES',len(names),flush=True)
