from pathlib import Path
import json,numpy as np,csv
R=Path(__file__).resolve().parents[1];P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015');C=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
load=lambda n:json.loads((R/'reports'/n).read_text())
data=load('data_support.json');top=load('value_topology.json');fit=load('proposal_fit.json');rob=load('first_edge_robustness.json')
cost=json.loads((R/'receipts/probe_exit.json').read_text());probe=json.loads((R/'receipts/probe.json').read_text())
prior1=json.loads((C/'reports/final_checkpoint_results.json').read_text());prior4=json.loads((P/'reports/matched_results.json').read_text())['checkpoints']
performance=[]
for task in [1,4]:
 for seed in [0,1]:
  rows=[x for x in prior1 if x['model']=='G0' and x['seed']==seed and x['protocol']=='matched_verified' and x['component']=='Actor'] if task==1 else [x for x in prior4 if x['algorithm']=='GFP' and x['seed']==seed]
  rows=sorted(rows,key=lambda x:x['step'] if task==1 else x['checkpoint']);scores=[x['success'] if task==1 else x['success_rate'] for x in rows];returns=[x['return_mean'] if task==1 else x['return_'] for x in rows]
  performance.append(dict(task=task,seed=seed,checkpoints=[800000,900000,1000000],success=scores,last3_mean=float(np.mean(scores)),return_mean=returns,reused=True))
decisions=[
 dict(hypothesis='H-data',status='SIMPLE_SCARCITY_EXPLANATION_WEAKENED',evidence='Both tasks have few complete observed paths (11 vs 9 train, 1 vs 1 validation); task4 all-shortest-route edges and turns have substantial episode support.',unknown='Goal-consistent quality, joint posture/action support at deployed states and composability are not established; start-radius1 routes are 11 vs5.',decision='Do not attribute the gap to nine routes alone; do not claim data sufficiency proven.'),
 dict(hypothesis='H-value',status='REPRODUCIBLE_SPATIAL_ASSOCIATION_NOT_CAUSAL',evidence='Task4 first required east edge has higher observed-data Q in the start cell across both seeds, three checkpoints, train/validation and same-episode stable-posture sensitivity.',unknown='Different states and data actions; Q follows learned continuation, not optimal BFS. No long-term counterfactual action returns. Cannot separate inaccurate critic from accurately poor downstream policy.',decision='Highest-priority retained clue; DIAGNOSE, not permission to change Critic/Guidance.'),
 dict(hypothesis='H-policy',status='TASK4_SPECIFIC_FIT_DEFECT_NOT_ESTABLISHED',evidence='Both networks produce data-near proposals, but closeness does not demonstrate executable detours. Task4 controlled actor fit penalty is not consistent; VaBC flow-fit detour penalty also exists, more clearly, in task1.',unknown='Finite candidates do not exhaust distributions; action RMSE/flow error do not establish correct multi-step behavior or long-horizon utility.',decision='Do not claim missing action expressivity or failed distillation; no module or training GO.')]
(R/'reports/decision_table.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2));(R/'reports/reused_performance.json').write_text(json.dumps(performance,indent=2))
lines=['# Task4 全局数据—价值—策略联合诊断','', '**裁决：DIAGNOSE；本轮执行完成。保留“起点出口附近价值结构异常”这一可重复线索，但三个假设尚未形成解释任务差距的因果证据。停止本轮实验，不自动追加训练、环境交互或模块。**','',
'| 假设 | 证据与反例 | 本轮判断 |', '|---|---|---|',
'| H-data：有效支持不足 | task1/task4 完整观察路径为 11/9 条，验证集各 1 条；task4 关键边、转弯有上百条训练轨迹支持。更严格起点半径下仍有 11/5 的差异。 | 简单的“只有 9 条，所以学不会”解释被削弱；任务相关的联合支持是否充分仍未知。 |',
'| H-value：拓扑价值失配 | task4 必须先向东，但起点数据状态—动作的 Q 稳定高于东侧下一格；两个种子、三个检查点与验证集均出现。多数其他路线段和隔墙关系方向合理。 | 最集中的机制线索；尚不能判定 Critic 错误，更不能证明其导致失败。 |',
'| H-policy：行为学习/优化失配 | task4 绕行组没有跨种子一致的 Actor 拟合恶化；VaBC 流匹配误差的绕行差异在成功的 task1 也出现，且更清楚。 | 未建立 task4 特异的表达或蒸馏缺陷；生成相近动作不等于能执行有效绕行。 |','',
'## 1. 对照结果与数据支持','', '所有成绩均复用已有 matched 评价，每检查点 100 回合；本轮未重新评价。task1/task4 的任务与初始条件不同，数值仅说明已有表现差距，不能当成单因素因果对照。','', '| 任务 | 种子 | 800k / 900k / 1M 成功率 | 最后三点均值 |','|---|---|---|---|']
for x in performance:lines.append(f"| task{x['task']} | {x['seed']} | {' / '.join(f'{v*100:.1f}%' for v in x['success'])} | {x['last3_mean']*100:.2f}% |")
lines+=['','两任务使用相同 navigate 原始数据：训练 1,000 条轨迹、2,000,000 个有效转移；验证 100 条轨迹、200,000 个有效转移。OGBench 固定 1.1.0，discount 均为 0.995。只按各自任务目标离线计算几何量。','', '| 数据支持口径 | task1 训练 | task4 训练 | task1 验证 | task4 验证 |','|---|---:|---:|---:|---:|']
order=[next(x for x in data if x['task']==t and x['split']==s) for s in ['train','validation'] for t in [1,4]]
for label,key in [('曾访问起点格的轨迹','start_cell_episodes'),('起点格之后到达目标半径 0.5 m 的轨迹','complete_routes'),('起点半径 1 m 后到达目标的轨迹','complete_routes_radius1')]:lines.append('| '+label+' | '+' | '.join(str(x[key]) for x in order)+' |')
for label,key in [('最短路线每条有向边的独立轨迹数','shortest_edges'),('最短路线每个直角转弯的独立轨迹数','shortest_turns')]:lines.append('| '+label+' | '+' | '.join(f"{min(z['episodes'] for z in x[key])}–{max(z['episodes'] for z in x[key])}" for x in order)+' |')
lines+=['','统计覆盖所有最短路线，task1 有 12 条边、task4 有 10 条边；不是只选一条方便的路径。完整路径仍是“同一采集轨迹中先到起点、后到目标”，采集目标会变化且原文件没有目标字段，不能称为 task4 专家示范。局部覆盖计数也不能证明姿态—速度—动作的联合支持已充分。',
'','按预先定义的 100 步代理（BFS 至少进展一格、同时欧氏目标距离增加超过 0.5 m），task4 训练数据有 39,441 个状态、333 条轨迹；验证集有 5,356 个状态、47 条轨迹。task1 对应 5,156/270 与 326/27。该代理标记实际记录的绕行片段，不代表当前动作的真实优势，也不代表整段都在追求本任务目标。','',
'## 2. 价值：异常集中在 task4 起点出口','',
'在已有均匀随机批次上读取 Target Critic 对真实数据动作的评分；未制造观测，也未重新按距离或行为类别归一化 Guidance。比较起点格 (6,5) 与必须先到达的东侧格 (6,6)，控制 head height、upright 和三个 COM 速度分量，按原始轨迹聚类估计区间。下表为 Q(起点) − Q(东侧)，正值表示起点分数更高：','',
'| 种子 | 200k | 600k | 1M（95% 聚类区间） |','|---|---:|---:|---|']
for seed in [0,1]:
 d=sorted([x for x in top if x['task']==4 and x['split']=='validation' and x['seed']==seed and x['kind']=='shortest_edge' and x['far']==[6,5] and x['near']==[6,6]],key=lambda x:x['checkpoint']);x=d[-1];lines.append(f"| {seed} | {d[0]['far_minus_near']:.2f} | {d[1]['far_minus_near']:.2f} | {x['far_minus_near']:.2f} [{x['ci95_cluster_normal'][0]:.2f}, {x['ci95_cluster_normal'][1]:.2f}] |")
lines+=['','验证样本为两格 613/1,112 个观察记录、合计 32 条原始轨迹；训练集也同向。一次事后稳健性检查仅保留同一条轨迹在两格都有站立样本（head≥1.2、upright≥0.7）的 23 条验证轨迹，1M 差值仍为 seed0 **5.27 [3.43, 6.96]**、seed1 **2.89 [1.72, 4.01]**；两个 ensemble 成员均同向。此检查未完全匹配所有身体状态，不能视为同模拟器状态干预。',
'','它不是全局“看不见墙”：1M 验证集的隔墙配对中，task1 两种子均 8/8 与 BFS 远近方向一致；task4 为 10/11、11/11。最短路线边中 task4 为 8/10、9/10 同向，最稳定的反向关系正好在必经的第一条东向边。',
'','**限制至关重要：**这是不同状态、不同数据动作的空间评分关系，不是同状态动作排序。BFS 是几何距离，不是带身体控制、折扣和当前后续策略的真实 Q。这个形状可能来自价值学习失配，也可能正确反映“进入该区域后 GFP 继续控制很差”。因此不能据此宣布 Critic 高估、规划错误或 Guidance 有害。','',
'## 3. 策略与 Guidance：未出现 task4 特异的拟合证据','',
'预先按数据行为分层抽样，每组每任务/划分最多 256 个状态、每原始轨迹最多 8 个。对每个状态生成 8 组固定噪声的 Actor/VaBC 候选，未执行任何动作。下面为 1M 验证集绕行代理组；RMSE 在 [-1,1] 动作尺度上计算，先对每状态噪声平均，再按轨迹等权汇总。','',
'| 任务/种子 | Actor RMSE | VaBC RMSE | Actor best-of-8 | VaBC best-of-8 | 无权重 VaBC Flow MSE |','|---|---:|---:|---:|---:|---:|']
for x in fit:
 if x['checkpoint']==1000000 and x['split']=='validation' and x['group']=='detour_progress':
  keys=['actor_mean_rmse','vabc_mean_rmse','actor_best8_rmse','vabc_best8_rmse','vabc_flow_mse'];lines.append(f"| task{x['task']}/s{x['seed']} | "+' | '.join(f"{x['metrics'][k]['trajectory_equal_mean']:.3f}" for k in keys)+' |')
lines+=['','不同任务的绕行代理组不是相同状态，不能用上表直接给模型能力排名。限制到共同格子、控制姿态和速度后，task4 绕行组相对退行组的 Actor RMSE 差为 +0.011 与 +0.004，两个区间均跨 0；VaBC Flow MSE 差为 +0.049 [0.012,0.087] 与 +0.036 [-0.005,0.077]。task1 反而两种子都有更清楚的 Flow 拟合差（约 +0.071、+0.068，区间均大于 0）。所以这项误差不能独立解释 task4 特有失败。',
'','task4 绕行组的数据动作只在约 28.9% / 34.5% 的有限 Actor 候选比较中获得更高 Q；但 task1 也只有 32.8% / 37.5%。Actor 本来经过 Q 优化，原数据动作也不是专家真值，这不是排序错误证据。',
'','复用原始均匀批次的 Guidance：1M task4 绕行组均值为 0.452 / 0.486，退行组为 0.424 / 0.426；中性区比例为 30.1% / 37.0%。尚未得到“有用绕行被一致压制得比退行更低”的证据。task1 绕行样本稀少（验证原批次仅少量轨迹），区间较宽，不应挑均值比较。',
'','## 4. 本轮停止点与可追溯性','',
'已有足够证据削弱“单纯缺少绕行动作”和“task4 独有的动作拟合变差”这两种简单解释；价值空间形状提供了更具体的关联线索。**仍没有哪条假设获得足以解释性能差距、支持新模块的独立因果证据。按本轮边界暂停机制实验，不自动扩大采样、训练新种子或调整温度。** 后续若另行立项，须区分错误的价值学习与真实较差的后续策略，不能仅重复观察同一张 Q 地图。','',
f"本轮处理 3,990 个任务—状态条目、12 份冻结模型，23,940 次模型—状态评估、{probe['state_noise_trials']:,} 次状态—噪声试验；固定批次另有 {probe['padded_state_noise_trials']:,} 个填充计算，全部丢弃、不计作证据。纯推理进程墙钟 {cost['wall_seconds']:.1f} 秒，GPU7 共享驻留 {cost['gpu_residency_hours']:.4f} 小时，2秒采样显存峰值 {cost['peak_gpu_mib_sampled']} MiB（采样值可能漏掉瞬时峰值）。数据统计成功运行45.1秒、进程峰值约12.2GiB内存；首次数据JSON导出失败后重算，额外CPU耗时未单独计时，未隐去失败日志。一次分析脚本语法错误在计算前修正。没有重复模型推理。",
'','全部 12 份模型在推理前后的参数/Adam/Agent RNG 哈希及 checkpoint 文件哈希相同；环境构造、reset/step、优化器更新入口被禁止，计数均为 **0**。OGBench1.1.0、JAX/JAXlib0.6.2、Flax0.10.7、Optax0.2.5、NumPy1.26.4；保持 float32 和原默认矩阵精度。完整配置、权重哈希、输入索引和来源数组哈希见同目录 JSON。',
'','区间按原始轨迹处理（400次轨迹重采样；控制回归使用轨迹聚类 sandwich 正态区间）。未对全部探索性比较进行多重检验校正，不能用个别区间跨不跨0宣称新机制。两个训练种子不能提供稳定的总体效应估计；观测代理和动作RMSE均不是长期控制真值。','',
'![联合证据](joint_evidence.png)','',
'复核入口：依次 `code/data.py`、`code/probe.py`（由单次 `run_probe.py` 调度）、`code/analyze.py`、`code/topology.py`、`code/finalize.py`、`code/report.py`。旧结果仅读取；新原始数组位于本卡 `arrays/`，不入 Git。精确配置与哈希见 `reused_asset_manifest.json`、`array_index.json`、`receipts/probe.json`。',
'','分支：https://github.com/rollinpioneer/graph/tree/codex/gfp-task4-global-diagnosis-v1-20261016','服务器结果：`/home/xushijie3/work/gfp_task4_global_diagnosis_v1_20261016/reports/JOINT_DIAGNOSIS.md`。']
(R/'reports/JOINT_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
with (R/'reports/decision_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
print('REPORT_DONE',[(x['task'],x['seed'],x['success']) for x in performance])
