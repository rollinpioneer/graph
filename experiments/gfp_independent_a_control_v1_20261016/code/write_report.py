import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
s=read('reports/summary.json');c=s['costs'];pp=read('runs/A/probes.json');events=read('receipts/reference_events.json')
decision={
 'LOCAL_ORDINARY_CONTROL_SUFFICIENT':'本窗口支持普通动作变化控制可能已经足够：独立 A 同时通过持续更新、无连续联合异常、无行为崩塌及较强收益标准。没有必要仅依据这个窗口优先引入复杂密度模型。',
 'INCOMPLETE_AMPLITUDE_DIAGNOSTIC':'持续更新幅度未达标，本轮诊断未完成；即使成功率较好，也不能据此宣称方法有效。',
 'COLLAPSE_WITH_INDEPENDENT_UPDATES':'独立更新幅度达标，但重新出现行为崩塌。此前共同减速可能重要，需重新研究参考 Actor 的更新方式；单次历史比较尚不能证明唯一原因。',
 'MIXED_RESULT':'持续更新达标，但没有同时通过较强收益标准。按混合结果保留，不能把未触发崩塌自动称为修复成功。'
}[s['decision']]
table='\n'.join(f"| +{x['updates']//1000}k | {x['success_percent']:.0f}% | {x['return_mean']:.2f} | {x['start_stay_percent']:.0f}% |" for x in s['evaluation'])
probe_table='\n'.join(f"| {p['updates']//1000}k | {p['historical_gap_increase']:.3f} | {p['historical_gradient_ratio']:.3f} | {p['dqda_cos_start']:.3f} | {'是' if p['joint_anomaly'] else '否'} |" for p in pp)
move='\n'.join(f"| {x['split']} / {x['region']} | {x['independent_coarse_path']:.5f} | {x['historical_A_coarse_path']:.5f} | {x['independent_final_net_rms']:.5f} | {x['historical_A_final_net_rms']:.5f} |" for x in s['fixed_sample_movement'])
pair='\n'.join(f"| +{x['updates']//1000}k / 历史{x['historical_arm']} | {x['independent_minus_historical_pp']:+.1f} | [{x['paired_episode_bootstrap_95'][0]:.1f}, {x['paired_episode_bootstrap_95'][1]:.1f}] |" for x in read('reports/historical_comparison.json'))
first=read('receipts/first_event_source_comparison.json');ad=read('receipts/admission_result.json');temporal=read('reports/temporal_comparison.json')
text=f'''# 独立 A：参考 Actor 动作变化控制诊断

**裁决：{s['decision']}。** {decision}

实验卡 GFP-INDEPENDENT-A-CONTROL-V1 于2026-10-16冻结，执行跨至2026-10-17。只有 Seed 3 全局350k这个分叉状态、一个50k续训分支。没有启动 B，没有加载密度模型，也没有增加任务、种子或搜索参数。

## 核心验收

| 项目 | 实测 | 预定标准 |
|---|---|---|
| 足额参考更新 | {s['continuous_events']}/500（{100*s['continuous_fraction']:.1f}%） | 至少450/500 |
| 实际全局RMS / 标定半径，均值 | {s['radius_fraction_mean']:.4f} | 单事件至少0.5 |
| 幅度比最小 / 中位 / 最大 | {s['radius_fraction_quantiles'][0]:.4f} / {s['radius_fraction_quantiles'][2]:.4f} / {s['radius_fraction_quantiles'][4]:.4f} | 起点与非起点均非零 |
| 零幅度事件 / 不同参考参数哈希 | {s['zero_events']} / {s['distinct_reference_hashes']} | 不把哈希变化代替幅度标准 |
| 最长连续联合异常 | {s['longest_joint_anomaly_run']}个探针点 | 连续2点才判异常 |
| +50k行为崩塌 | {s['behavior_collapse']} | 成功≤5%且起点滞留≥90% |
| +50k较强收益 | {s['strong_benefit']} | 成功≥25%、起点滞留≤30%、无连续联合异常 |

旧配对A平均实际半径比例为{s['historical_paired_A_mean_radius_fraction']:.4f}，本轮为{s['radius_fraction_mean']:.4f}。这排除了本轮依赖B额外压低共同步长的情况；A仍保留自己的动作变化上界，不能说所有减速都不重要。

参考更新每100次GFP更新执行一次。校准rho0=0.0005490051116794348，与前轮相同；保持独立起点/非起点幅度上界和跟随损失检验。A采用自己的可行候选，不再取A/B共同半径；初始候选/最多8次减半/6次区间细化均保留。删除的额外匹配阶段只用于前轮A/B配对。有限搜索不保证理论最大可行步长。

## 在线 Actor 评价

每点100回合，只部署原在线Actor，参考Actor不部署。

| 检查点 | 成功率 | 平均Return | 终点在起点区域 |
|---|---:|---:|---:|
{table}

旧配对A的25k/50k成功率为31%/41%，50k起点滞留14%；旧B为39%/41%、18%。历史L0原EMA为3%/0%、50k滞留100%；历史L1a固定TD Actor最终36%、滞留17%。这些是只读历史对照，没有重复训练；不能宣称逐位重放或当作多个独立训练重复。

本轮与历史A/B的初始观测100/100逐回合匹配，策略随机序列协议一致。下面区间只是固定模型间的配对回合bootstrap，不包含训练随机性，不能据此证明跨种子优势或算法等价。

| 比较 | 本轮减历史成功率（百分点） | 配对回合95%区间 |
|---|---:|---|
{pair}

25k到50k的成功率下降13个百分点；同批初始条件中，29个回合由成功变失败，16个由失败变成功，8个均成功。配对回合bootstrap区间为[-26, 0]个百分点，仅描述这两个冻结模型，不能代替独立训练重复。50k相对历史配对A低17个百分点，配对回合区间[-29, -5]；仍不是跨训练种子的差异证明。

24%与25%的操作门槛只差一个回合，未达预设标准必须如实报告，但不等于已经证明方法无效。起点滞留34%也未达≤30%要求。综合判为混合结果：更新幅度恢复后没有重现原定义崩塌，但能力保留不足。此前较小步幅可能有帮助，目前不能把该历史差异归因于密度信息。

## 完整探针与动作变化

历史诊断锚点保持原300k，未改为350k。联合条件是起点相对全局Q差增量≥15且固定历史动作处梯度范数比≥2。方向相似度只作辅助，不是异常判据。

| 续训 | Q差增量 | 梯度范数比 | 梯度方向cos | 单点联合条件 |
|---|---:|---:|---:|---|
{probe_table}

![完整曲线](full_curves.png)

固定4032状态及两份固定噪声的纯前向审计提供另一种动作变化测量；下表路径是相邻5k检查点的RMS之和，不能等同连续参数路径。每100更新的独立拟合/检验批次路径另见regional_paths.csv。没有使用密度评分决定更新或验收。

| split / 区域 | 独立A粗路径 | 旧配对A粗路径 | 独立A最终净RMS | 旧配对A最终净RMS |
|---|---:|---:|---:|---:|
{move}

## 正确性与成本

- 同一350k源检查点SHA256：ffba11e943f7e192134b86093f4b2dc49efbe239cf2d32e3e884406358de1cca。源文件结束后未变，完整检查点索引见checkpoint_index.json。
- 零更新TD完全一致，总损失及梯度对照最大差0；首四次真实更新计入50k，在线网络确实变化而参考无隐藏EMA；模型、Adam计数、Agent/NumPy/Python RNG保存恢复通过。首500更新5/5事件足额，准入通过。
- 首事件与历史A输入身份一致；实际RMS差{first['difference']:.3g}，不声称浮点计算逐位重现。GPU共享及编译差异属于历史比较限制。
- GFP更新总计{c['gfp_updates']}，参考事件{c['reference_events']}；训练无环境交互，密度模型训练与加载均为0。评价{c['evaluation_episodes']}回合，参数更新0。
- 评价外部步{c['external_steps']}，包含构造/reset的控制调用{c['control_calls']}，物理子步{c['physics_substeps']}，均在400000 / 416666 / 2083330预算内。
- 训练含准入墙钟{c['training_wall_seconds']:.2f}秒、CPU {c['training_cpu_seconds']:.2f}秒，最大单训练进程RSS {c['max_training_rss_kib']/1048576:.2f} GiB。评价进程及worker成本逐项见costs.json。GPU为共享使用，设备总利用率不能当作本作业独占利用率；未作独占算力效率比较。
- 训练、两点评价真实退出码均0；无重试、无额外训练、无参数调整；两点评价worker全部退出。轻量文件归档，权重、输入及完整日志留服务器。

## 证据边界与下一步

{decision}

这只回答一个已知异常中途窗口，不能证明长期训练、从头训练、其他种子或Task4有效；也不能据此否定数据支持信息在其他条件下的作用。收益不能被解释为已证明原始崩塌的唯一根因。下一步最值得区分的是：为何25k到50k的成功率下降、起点滞留增加，却没有触发预定义价值异常？可先只读定位受损回合与参考动作变化的关系，再决定是否需要新的干预；当前不据此升级密度模型，也不追加训练。

分支：[codex/gfp-independent-a-control-v1-20261016](https://github.com/rollinpioneer/graph/tree/codex/gfp-independent-a-control-v1-20261016)。提交与远端核验见receipts/git_verified.json。
'''
(R/'reports/RESULTS.md').write_text(text)
(R/'HANDOFF.md').write_text('# 已完成：独立A诊断\n\n'+decision+'\n\n50000新增GFP更新、500参考事件、200评价回合；不追加训练。\n分支 codex/gfp-independent-a-control-v1-20261016\n报告 reports/RESULTS.md；源和完整权重留服务器，提交核验 receipts/git_verified.json。\n')
print(decision)
