from pathlib import Path
import json,csv,hashlib,resource
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
rows=list(csv.DictReader(open(R/'metrics.csv')));sens=list(csv.DictReader(open(R/'sensitivity.csv')))
def at(s,d):return [x for x in rows if int(x['seed'])==s and int(x['delta'])==d]
def rng(s,d,k):return min(float(x[k]) for x in at(s,d)),max(float(x[k]) for x in at(s,d))
def fmt(s,d,k,n=3):
 lo,hi=rng(s,d,k);return f'{lo:.{n}f}–{hi:.{n}f}'
v=json.loads((R/'receipts/validation.json').read_text());run=json.loads((R/'receipts/inference.json').read_text());ex=json.loads((R/'receipts/execution.json').read_text());alarm=json.loads((R/'alarm_timing.json').read_text());ev=json.loads((R/'historical_evaluations.json').read_text())
assert v['status']=='PASS' and ex['exit_code']==0
for r in json.loads((R/'checkpoint_manifest.json').read_text()):assert sha(r['path'])==r['sha256']
for r in json.loads((R/'source_manifest.json').read_text()):assert sha(r['path'])==r['sha256']
for seed,d in [(0,60000),(3,20000)]:
 h=[x for x in sens if int(x['seed'])==seed and int(x['delta'])==d];assert len(h)==4 and min(float(x['gradient_ratio']) for x in h)>1
decision=dict(necessary_condition_screen='LIMITED_GO_FOR_RELATIVE_GRADIENT_CANDIDATE_ONLY',prospective_behavior_prediction='INCONCLUSIVE',deployable_CRR_gate_or_method_training='NOT_AUTHORIZED_AND_NOT_ESTABLISHED',local_Q_large_divergence='only 5k before finer historical alarm in both development trajectories; do not claim >=10k clear warning from this component',early_gradient_examples=[dict(seed=seed,delta=d,gradient_ratio=rng(seed,d,'gradient_ratio'),lead_to_fine_alarm=alarm[str(seed)]['first_existing_5k_original_rule_delta']-d,all_fixed_latents_and_probe_halves_same_direction=True) for seed,d in [(0,60000),(3,20000)]],no_numeric_alarm_threshold_fitted=True,interpretation='There is a repeatable early relative-gradient difference above numerical noise. This is enough only to consider the next separately approved Critic-only candidate-response screen. It is not a calibrated alarm, causation, independent validation, or demonstrated prospective performance warning.',single_next_question='Does a small, controlled candidate TD-Actor replacement induce an adverse fixed-probe Critic response distinguishable from a benign replacement, rather than merely reproducing the difference to a separately frozen training trajectory?',new_training_or_env_interaction=0)
save('decision.json',decision)
report=f'''# CRR 历史可分性筛查：结果与裁决

完成范围：task1 s0 600k→700k、s3 300k→400k，B0/B1a 共44个逻辑模型点（42份独立权重文件），每10k；固定起点2048、非起点2048，两个固定latent。参数更新、环境构造/reset/step、新评价回合均为0。真实推理退出码0。

## 结论先行

**必要条件筛查有限通过：存在提前、可重复的相对动作梯度分离。只允许推荐下一张小规模 Critic-only 候选响应筛查，不足以进入 CRR 门控开发或训练。**

**固定动作上的大幅局部 Q 抬升不是已经证实的≥10k提前信号。行为退化的前瞻性仍是 INCONCLUSIVE。** 这里没有看到曲线后设置报警阈值；以下时间点用于描述完整曲线，不定义“首次预警”。两个latent和两个状态子集不是独立训练种子。

## 1. 较早出现的是什么

| 轨迹与检查点 | 在线梯度比 B0/B1a（两个latent） | ΔC（两个latent，Q单位） | 距更细历史报警 |
|---|---:|---:|---:|
| s0 660k（+60k） | {fmt(0,60000,'gradient_ratio')} | {fmt(0,60000,'delta_C')} | 15k |
| s3 320k（+20k） | {fmt(3,20000,'gradient_ratio')} | {fmt(3,20000,'delta_C')} | 15k |

梯度比在四个latent×固定子集组合中均大于1。s0上述时间点的子集比值约1.280–1.295，s3约1.311–1.333。轨迹聚类bootstrap的95%描述区间：s0约1.205–1.377，s3约1.261–1.373；这些区间描述959条原始数据轨迹的抽样敏感性，不是独立训练复现、风险阈值或多重检验后的显著性。

差异远大于本轮原实现对照的浮点误差（实测0）。因此“完全没有提前数值分离”不符合本轮结果。但也不能据此把1.3或任何事后选定比值设置为风险门槛。

**重要反例：s3的梯度比上升并不是B0自身已经梯度爆炸。** 分叉时两者约0.779；320k时B0约0.590、B1a约0.447，两者均比起点下降，B1a下降更多。s0则从约0.365变为660k的B0约0.447、B1a约0.345。两轨迹相同的是相对比值，不是相同的绝对梯度演化。

更早的局部对比度也没有统一正向风险模式：s0 660k ΔC约+0.68，s3 320k约−0.81。不能把“有差异”直接改写成“局部价值岛已经提前抬升”。

## 2. 大幅局部价值分离出现较晚

| 点 | ΔC | 起点平均绝对Q差 | 非起点平均绝对Q差 | 梯度比 |
|---|---:|---:|---:|---:|
| s0 670k | {fmt(0,70000,'delta_C',2)} | {fmt(0,70000,'Q_abs_start',2)} | {fmt(0,70000,'Q_abs_nonstart',2)} | {fmt(0,70000,'gradient_ratio',2)} |
| s3 330k | {fmt(3,30000,'delta_C',2)} | {fmt(3,30000,'Q_abs_start',2)} | {fmt(3,30000,'Q_abs_nonstart',2)} | {fmt(3,30000,'gradient_ratio',2)} |

这两点比原10k报警（680k/340k）早10k，**但比已存在的5k首次越界（675k/335k）只早5k**。图上同时保留两种标记，不能舍弃较细记录来宣布达到10k。s0历史规则为起点在线V抬升20，s3为起点减全局对比度抬升15；两者定义不同，原样保留。s3另有335k开始的历史联合异常记录，时间一致。

固定Q差异不仅来自B0向上漂移。以latent0为例：s0 670k起点B0相对分叉约+7.82，B1a约−1.04；s3 330k起点B0约+12.39，B1a约−16.47。s3同时包含冻结分支显著下移，不能将29点分支差全部解释成B0抬升29。非起点的两分支漂移也在第二组图完整报告。

## 3. 行为时间关系无法补齐

| 种子 | 轨迹 | 已有检查点 | 成功率 | 终点仍在起点区 |
|---|---|---:|---:|---:|
'''
for x in sorted(ev,key=lambda x:(x['seed'],x['step'],x['arm'])):report+=f"| {x['seed']} | {x['arm']} | {x['step']//1000}k | {x['success_percent']:g}% | {x['final_start_percent']:g}% |\n"
report+='''
s3分叉点300k本身只有9%成功率，不能描述成健康性能之后才开始下降。s0在680k/700k的B0成功率5%/0%，但没有本轮时间网格上的连续匹配行为评价，无法知道成功率是否早于数值分离下降。历史评价还缺少严格完整初始状态哈希核对；本轮只沿用其聚合观测，不冒充精确配对重放。没有补做任何环境回合来填空。

## 4. 实际计算语义和验证

- B0是当前在线Actor进入TD，不是EMA Actor。B1a仅在TD中固定分叉Actor，在线Actor、VaBC及Critic历史上仍更新。
- 固定Actor动作由下一状态生成；固定VaBC动作由当前状态生成、在下一状态评分，保持target_use_next=False。动作各自裁剪后固定，先各自做Target Critic ensemble mean，再按target_agg=mean平均两个来源；不平均动作向量。
- ΔC是新固定动作Target评分的起点减非起点对比度，不能混同旧探针的当前Actor在线评分、起点减全局对比度。
- 动作梯度用在线Critic ensemble mean，在固定已裁剪Actor动作处对动作输入求导；不穿过Actor或clip，不使用guidance_agg=min，也不将全局Q归一化系数混入原始Q梯度范数。
- 分叉点两臂输出逐位一致；原始模块映射、Target参数身份、Actor/VaBC、TD聚合和在线动作梯度对照最大绝对误差均0（容差3e-5）。全部结果有限，TrainState.step=Adam count+1。模型/优化器/RNG载入对象和42份原始权重前后哈希不变。
- B1a文件保存普通GFP状态，没有冻结生成器sidecar；足够只读恢复Critic，不宣称具备完整B1a续训恢复能力。

## 5. 决策表

| 问题 | 裁决 |
|---|---|
| 两条开发轨迹有≥10k提前、跨latent/子集一致且超出数值噪声的梯度相对差异吗？ | 有；给出的+60k/+20k点均比细记录早15k。不是拟定报警阈值，也不声称首次出现时间。 |
| 大幅局部Q抬升有同等明确提前量吗？ | 未建立；明显放大的点距细报警仅5k，较早ΔC方向不统一。 |
| 可预测成功率退化吗？ | INCONCLUSIVE；行为时间稀疏，s3分叉时已弱。 |
| 现在开发/训练CRR门控吗？ | 不可以；仅完成必要条件筛查，没有风险特异性或独立验证。 |
| 下一项值得单独检验什么？ | 小幅、受控的TD Actor候选替换，是否使Critic在固定探针上的响应与良性替换可分。不能只拿历史B0/B1a距离作为可部署门控。 |

后一问题对应原卡允许推荐的下一轮小规模Critic-only候选响应筛查，**本轮没有启动**。B0/B1a差异依赖两条已经走完的训练轨迹；单模型实际运行中没有现成的B1a反事实，因此还不是可直接使用的门控量。正结果也只属于两条已消费开发轨迹，不能推广为跨种子可靠预测器、独立确认或因果证据。

## 6. 成本、文件和归档

'''
report+=f"推理核心耗时{run['seconds']:.2f}秒；含启动与资源采样等待的子任务占用窗口{ex['wall_seconds']:.2f}秒；CPU用户/系统时间{run['cpu_user_seconds']:.2f}/{run['cpu_system_seconds']:.2f}秒，峰值RSS约{run['peak_rss_kib']/1024**2:.2f}GiB。共享物理GPU0，采样到本推理进程显存峰值690MiB；这不是独占GPU基准或精确设备忙时。分析约1.93秒。360448次模型×状态×latent评估，额外原实现验证另列回执；没有任何训练或物理控制步。\n\n"
report+='''三组图：`contrast.png`、`q_differences.png`、`gradient_ratio.png`。完整逐点表`metrics.csv`，子集敏感性`sensitivity.csv`。`checkpoint_manifest.json`给出原权重位置及哈希，`server_artifact_index.json`索引完整数值数组和运行资源回执；不将权重、数据、完整日志纳入Git。

服务器根：`/home/xushijie3/work/crr_historical_separation_z0_20261017`。

分支：[codex/crr-historical-separation-z0-20261017](https://github.com/rollinpioneer/graph/tree/codex/crr-historical-separation-z0-20261017)。本轮普通提交推送后核对远端SHA；本地reports同步交付。无PR、无合并、无旧实验变更。
'''
(R/'RESULTS.md').write_text(report)
(R/'HANDOFF.md').write_text('''# Handoff — CRR historical separation Z0

Status: completed, zero training / environment operations. Full interpretation: RESULTS.md; decision.json has scoped decisions.
Relative-gradient necessary condition: limited GO only for recommending a future separately approved Critic-only candidate-response screen. Large fixed-Q separation has only5k lead to fine alarms. Prospective performance warning remains INCONCLUSIVE. No method development or additional experiment launched.
Root: /home/xushijie3/work/crr_historical_separation_z0_20261017
Branch: codex/crr-historical-separation-z0-20261017
URL: https://github.com/rollinpioneer/graph/tree/codex/crr-historical-separation-z0-20261017
Current inference finished with true exit0. No monitoring or restart needed. Two latents/probe halves are not training replications. s3 fork already success9%; sparse behavior timing prevents foresight claims. Source B1a checkpoint lacks frozen actor sidecar; no complete training resume assertion.
''')
art=[]
for p in sorted(list((R/'arrays').glob('*'))+[R/'receipts/resources.json',R/'receipts/infer.log',R/'receipts/analyze.log']):
 art.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
save('server_artifact_index.json',art)
print('FINAL_REPORT_READY',len(art),decision['necessary_condition_screen'])
