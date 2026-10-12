"""Four requested deliverables, raw identity index, resources and scientific plots."""
import datetime
import importlib.metadata
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval_core import *

def document(name,text):(ROOT/name).write_text(text.rstrip()+'\n')
def rows(name):return list(csv.DictReader(open(ROOT/'results'/name)))
def pp(x):return f'{100*x:.1f}'
def main():
    start=time.monotonic();cpu=time.process_time()
    a=json.load(open(ROOT/'results/A_AUDIT.json'));b=json.load(open(ROOT/'results/B_SUMMARY.json'));tech=json.load(open(ROOT/'results/TECHNICAL_RNG.json'))
    assert a['status']=='PASS' and b['total_formal_episodes']==2000
    measured={name:json.load(open(ROOT/path)) for name,path in [('timing','receipts/TIMING_PROBE.json'),('A','results/A_AUDIT.json'),('B_navigate','receipts/B_navigate_DONE.json'),('B_stitch','receipts/B_stitch_DONE.json'),('technical','results/TECHNICAL_RNG.json'),('analysis','results/B_SUMMARY.json')]}
    assert all(x['gpu_seconds']==0 for x in measured.values())
    identity={k:dict(path=str(OLD/'runs'/f'formal_{k}'/'params_1000000.pkl'),sha256=EXPECTED[k]) for k in EXPECTED}
    for x in identity.values():assert sha(x['path'])==x['sha256']
    document('EVALUATOR_AUDIT.md',f'''# 评价器逐步一致性审计

结论：A {a['status']}，50 组真实完整初态匹配场景，共 100 条独立仿真。不是仅共享种子或在同一轨迹上比较动作。

官方固定源码 utils.evaluation.evaluate 调用 sample_actions(temperature=0)，项目实际 RD01 common.evaluate 调用 actor.mode()，都按原代码裁剪动作。通过回放 reset 包装器将相同完整 MuJoCo INTEGRATION、qpos/qvel、目标、环境/动作/全局 NumPy/Python RNG 与原包装器计数恢复到两个独立环境。初始化最大差 {a['max_initial_difference']:.12g}，RNG 全部相等。

逐步动作最大绝对差 {a['max_action_difference']:.12g}（门槛 1e-6）；位置差 {a['max_qpos_difference']:.12g}、速度差 {a['max_qvel_difference']:.12g}（各门槛 1e-5）。终止、截断、奖励、成功和步数完全相同。按原 PointMaze 距离阈值 1、终止选项与 TimeLimit 1000 步独立重算全部通过。

轻量逐步表：results/A_steps_<model>_task<id>_scene<id>.csv（50 文件）；汇总 A_CASES.csv、A_AUDIT.json；完整匹配场景与双轨迹在 runs/A_*.pkl（服务器）。原函数源码散列与权重身份在 EVIDENCE_INDEX.json。

适配边界：RD01 函数在这里只显示所选任务的单任务视图，真实环境仍保留全部官方任务，回放恢复实际 task ID。该函数固定生成五项任务汇总，未显示任务的空均值会有日志警告；这些未使用的汇总不作为审计结果、不写入成绩。审计采用完整逐步记录和独立判定；两个函数源码均未修改。H1 支持仅限这些冻结权重、温度和所测 50 场景，不扩展到随机温度、视觉任务或所有环境。
''')
    score=rows('B_TASK_SCORES.csv');diffs=rows('B_TASK_DIFFERENCES.csv')
    lines=['# 两重置协议成绩及不确定性','', '2000 正式回合已完成；每个模型/条件/随机流 5×50 回合。O=初始化一次后连续 reset；U=原项目逐回合 reset(seed) 和动作空间 seeding。统一 clipped actor mode，温度 0，无视频，固定 1000 步、原成功阈值。旧 62.8%/49.6% 保留。','', '| 模型 | O 总成功 | U 总成功 | O-U（百分点） | 95% 区间（百分点） | 流 55001 / 65001 差 | 判定 |','|---|---:|---:|---:|---|---|---|']
    for model,x in b['overall'].items():
        lines.append(f"| {model} | {pp(x['O'])}% | {pp(x['U'])}% | {pp(x['O_minus_U'])} | [{pp(x['bootstrap95'][0])}, {pp(x['bootstrap95'][1])}] | "+' / '.join(pp(v) for v in x['stream_differences'].values())+f" | {x['interpretation']} |")
    lines+=['','| 模型 | 任务 | O 成功率 [Wilson95%] | U 成功率 [Wilson95%] | O-U [bootstrap95%]（百分点） |','|---|---:|---|---|---|']
    for d in diffs:
        cells={c:next(r for r in score if r['model']==d['model'] and r['condition']==c and r['task']==d['task'] and r['stream']=='pooled') for c in ('O','U')}
        fmt=lambda r:f"{pp(float(r['success_rate']))}% [{pp(float(r['wilson95_low']))}, {pp(float(r['wilson95_high']))}]"
        lines.append(f"| {d['model']} | {d['task']} | {fmt(cells['O'])} | {fmt(cells['U'])} | {pp(float(d['O_minus_U']))} [{pp(float(d['bootstrap95_low']))}, {pp(float(d['bootstrap95_high']))}] |")
    lines+=['','逐流/逐任务样本数、成功数和区间完整保存在 B_TASK_SCORES.csv。独立二项重采样 20000 次保持 task×stream×condition 单元等权，种子 57001/57002。区间条件于固定权重、固定任务与这两条随机流，不含训练种子或全体重置随机流的不确定性。即使数值流编号相同，主差分仍按非配对统计；偶然发现相同位置不事后换更窄的配对区间。任务区间未作多重比较校正，不作家族显著性声明。','', '预登记大效应：绝对差至少 10pp、95% 区间排除 0、两流同向；未发现大效应：区间包含在 (-10,+10)pp 内，仅适用于当前条件。否则未知。两流不足以精确估计小影响或代替多训练种子。']
    document('PROTOCOL_SCORES.md','\n'.join(lines))
    matching=rows('B_SCENE_MATCHING.csv');features=rows('B_FAILURE_FEATURES.csv')
    lines=['# 随机场景与失败轨迹分布比较','',f"事后实测 {len(matching)} 组同 task/episode/stream 索引：位置+目标完全相同 {b['exact_position_goal_pairs']}；完整 qpos+qvel+目标相同 {b['exact_full_state_pairs']}；后续位置、动作、奖励和判定逐元素相同 {b['exact_states_actions_flags_pairs']}。这些是对保存数据的核验，不是以同种子假定配对。完整初始速度不同的场景不能称完整初态匹配对照。",'', '各条件/任务/流的坐标、初始速度、目标、起点距离、任务中心偏移的均值、SD、5/50/95 分位、KS 与标准化均值差在 B_SCENE_DISTRIBUTIONS.csv。KS 仅描述、不作机制证明；相同分布不等于相同有限场景。初始化 RNG 哈希在每个 B_<model>_<condition>_<stream>.csv，完整随机状态保存在对应原始 pickle。','', '源码解释：Maze.add_noise 从全局 NumPy 取 XY 和目标噪声；主比较两条件持有相同全局流，reset(seed) 只重置内部生成器。PointEnv.step 每步先将 qvel 置零，再推进 MuJoCo；模型观测为 qpos。若本次不同初始速度没有改变后续轨迹，这符合该具体控制实现，而不是所有 OGBench 系统的普遍等价。','', '| 模型 | 条件 | 成功/失败 | 回合 | 持续无效移动 | 未触发 | 平均终点距离 |','|---|---|---:|---:|---:|---:|']
    for model in ('navigate','stitch'):
        for c in ('O','U'):
            rs=[r for s in (55001,65001) for r in rows(f'B_{model}_{c}_{s}.csv')]
            for success in (1,0):
                f=[r for r in features if r['model']==model and r['condition']==c and float(r['success'])==success]
                distances=[float(r['final_distance']) for r in rs if float(r['success'])==success];n=len(f);events=sum(r['persistent_no_effect']=='True' for r in f)
                lines.append(f"| {model} | {c} | {'成功' if success else '失败'} | {n} | {events} | {n-events} | {np.mean(distances):.4f} |")
    lines+=['','每回合原始成绩表含终点距离、路径长度、首次达标、终止/截断；B_FAILURE_FEATURES.csv 含反复访格、持续无效移动、贴墙代理和小动作比例。所有成功、失败、未触发均保留。阈值沿用独立校准的 RD01 ratio .02、request>.05、8/10 window 与 wall .71，未重新拟合；不把 floor contact 当碰墙，不把小动作自动判为异常。','',f"独立技术检查（不混入主成绩）：官方无视频 {tech['official_no_video_reset_count']} 回合；额外视频指定回合 {tech['official_extra_video_reset_count']} 回合；统一模式 4 回合。额外回合后下一任务初始位置改变={tech['task2_position_changed_after_video']}，目标改变={tech['task2_goal_changed_after_video']}；官方 per-task 动作 RNG 的 NumPy draw 改变首场景位置={tech['task1_position_changed_by_official_action_key_draw']}。视频渲染被零图像占位替代，只测实际 reset/actions 的随机流消耗，没有真实 GPU 图形或视频质量核验。完整技术轨迹独立在 TECHNICAL_RNG.pkl。"]
    document('SCENE_DISTRIBUTIONS.md','\n'.join(lines))
    p=ROOT/'SCENE_DISTRIBUTIONS.md';p.write_text(p.read_text()+f"\n实际核验第一控制步之后 qvel 全数组相同 {b['exact_post_step_qvel_pairs']}/{len(matching)} 对，保留初始 qvel 差，不把完整初态声称相同。\n")
    decisions='\n'.join(f"- {k}: {v['interpretation']}；O/U={pp(v['O'])}%/{pp(v['U'])}%。" for k,v in b['overall'].items())
    document('PROTOCOL_VERDICT.md',f'''# 最终协议裁定与复现边界

H1：匹配完整初态下，两实际评价函数在本次 50 场景逐步一致，未发现温度 0 的 sample_actions 与 mode 引入成绩差异。

H2：按预登记的独立流与效应/区间标准：

{decisions}

仅测试连续与每回合环境/动作随机重置流程。统一主评价器不做官方 evaluate 在每个 task 开头额外的 np.randint 动作-key draw；该操作在温度 0 不改变相同场景动作，却可改变之后的有限 XY/目标场景。视频额外回合也消耗随机流。它们是另行检查的协议边界，不能把本轮 O 条件写成作者完整历史评价的恢复。

裁定：原 RD01 的 62.8%/49.6% 保留，未据此选择更好权重、重训练或改成功标准。未来每张评价卡必须明确 O/U、全局 NumPy/环境/动作/JAX 随机流初始化与任务调用的消费顺序、视频回合数和种子。可将 O 连续重置作为明确标记的未来惯例；本轮 O 只称官方风格确定性运行，不是作者未知的 2025 随机流。额外 RNG 消费检查不能单独估计正式成绩的大小变化。

两条训练权重仍为 seed 0 单种子；Stitch 专用作者训练配置仍未完全确认，保留 PROTOCOL_APPROXIMATION。即使分数更接近用户卡中的作者参考 85%/8%，也不能升级为论文级复现；分数仍不同也不自动触发重训练。当前结果只核验这个评价因素，不证明其它复现差距的来源。停止本轮，无新的方法或种子试探。
''')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model in zip(axes,('navigate','stitch')):
        for j,c in enumerate(('O','U')):
            r=[x for x in score if x['model']==model and x['condition']==c and x['stream']=='pooled'];p=np.array([float(x['success_rate']) for x in r]);lo=np.array([float(x['wilson95_low']) for x in r]);hi=np.array([float(x['wilson95_high']) for x in r])
            ax.bar(np.arange(1,6)+(j-.5)*.34,p,.34,yerr=np.maximum(np.vstack([p-lo,hi-p]),0),capsize=3,label=c)
        ax.set_title(model+' fixed 1M checkpoint');ax.set_xticks(range(1,6));ax.set_xlabel('Official task');ax.set_ylim(0,1.12);ax.set_ylabel('Success, n=100/task/condition');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'results/PROTOCOL_SUCCESS.png',dpi=160);plt.close(fig)
    cost={k:dict(wall_seconds=x.get('wall_seconds'),cpu_process_seconds=x.get('cpu_process_seconds'),gpu_seconds=x['gpu_seconds']) for k,x in measured.items()}
    elapsed=time.time()-datetime.datetime.fromisoformat('2026-10-12T17:48:05+08:00').timestamp();assert elapsed<14400
    write(ROOT/'results/RESOURCE_TOTAL.json',dict(round_wall_elapsed_to_report_seconds=elapsed,wall_cap_seconds=14400,stages=cost,
         measured_stage_cpu_seconds=sum(x['cpu_process_seconds'] for x in cost.values()),gpu_seconds=0,training_updates=0,formal_episodes=2000,A_simulations=100,
         timing_probe_rollouts=2,technical_rollouts=13,failed_probe_attempts=1,note='Initial preparation and failed tuple-copy probe CPU core time not fully instrumented; round wall includes them. Stage sums include parallel children and are not round wall. No GPU or actual video rendering.'))
    document('RESOURCE_REPORT.md',f'''# 资源报告

GPU 0 秒，训练更新 0；环境隐藏 CUDA、JAX 仅 CPU，所有模型恢复身份含 CPU devices。没有实际图形渲染。

截至生成报告的整轮墙钟 {elapsed:.3f} 秒，4 小时上限 14400 秒；测得阶段 CPU core seconds 合计 {sum(x['cpu_process_seconds'] for x in cost.values()):.3f}。B 两子进程并行，不能把子墙钟和当整轮墙钟。各阶段详见 RESOURCE_TOTAL.json，归档/SSH 关闭后的最终墙钟另在本地关闭回执。

试运行最大每步 .000695922 秒；2000 回合按满 1000 步估计 1597.875 秒（含加载与预留），两倍安全后通过，样本数先于正式结果提交。两个有效耗时回合之外，第一次探针因目标 tuple 没有 copy 接口失败；适配为 np.asarray 后修复，原错误日志保留，无模型/算法改动。该失败阶段 CPU core time 未完整测量，不编造精确值。

正式 B 2000 回合、A 100 条匹配独立仿真、技术检查 13 回合、有效耗时试跑 2 回合；所有辅助成本列明，没有当作正式成绩。初始准备/失败探针的精确 core time 未全量计量，整轮墙钟含其开销。
''')
    raw=[]
    for directory in ('runs','logs'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file():raw.append(dict(path=str(p),relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    sources=[OLD/'common.py',OLD/'vendor/saw/impls/utils/evaluation.py',OLD/'vendor/saw/impls/agents/saw.py',OLD/'vendor/saw/impls/utils/networks.py',OLD/'vendor/saw/ogbench/locomaze/maze.py',OLD/'vendor/saw/ogbench/locomaze/point.py']
    write(ROOT/'EVIDENCE_INDEX.json',dict(server_root=str(ROOT),parent_project_commit='6462e06c540594acca6bc8630c37a803fe36dd4a',checkpoints=identity,
          source=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources],raw_files=raw))
    write(ROOT/'DEPENDENCIES.json',{name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','numpy','gymnasium','mujoco','ogbench','scipy','matplotlib')})
    write(ROOT/'BRANCH_INFO.json',dict(branch='codex/saw-eval-02-20261012',branch_url='https://github.com/rollinpioneer/graph/tree/codex/saw-eval-02-20261012',
         parent='6462e06c540594acca6bc8630c37a803fe36dd4a',status='COMPLETED_ZERO_TRAINING',A=a['status'],B_episodes=2000,gpu_seconds=0,training_updates=0,original_scores_unchanged=True))
    write(ROOT/'receipts/REPORT_COST.json',dict(wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0))
    resource=json.load(open(ROOT/'results/RESOURCE_TOTAL.json'))
    resource['stages']['report']=json.load(open(ROOT/'receipts/REPORT_COST.json'))
    resource['measured_stage_cpu_seconds']=sum(x['cpu_process_seconds'] for x in resource['stages'].values())
    write(ROOT/'results/RESOURCE_TOTAL.json',resource)
    p=ROOT/'RESOURCE_REPORT.md';p.write_text(p.read_text()+'\n报告生成、图与索引散列成本另在 REPORT_COST.json，并计入 RESOURCE_TOTAL.json 的 report 阶段和累计 measured_stage_cpu_seconds；上文阶段合计为报告脚本启动前阶段。归档核验 CPU 成本不伪称已全量计量。\n')
    print('FOUR_DELIVERABLES_DONE',flush=True)

if __name__=='__main__':main()
