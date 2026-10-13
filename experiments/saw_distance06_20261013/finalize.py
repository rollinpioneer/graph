"""Report fixed results and limits; no policy forward or simulation."""
from core import *
import pandas as pd
def main():
    guard(); b=json.loads((ROOT/'receipts/BEHAVIOR_ANALYSIS.json').read_text()); s=json.loads((ROOT/'results/SIGNAL_INTERPRETATION.json').read_text())
    audit=json.loads((ROOT/'AUDIT06.json').read_text()); fa=json.loads((ROOT/'results/SIGNAL_FORWARD_AUDIT.json').read_text())
    assert audit['status']==fa['status']=='PASS'
    phases=[json.loads(p.read_text()) for p in sorted((ROOT/'receipts').glob('*_PROCESS_RESOURCE.json'))]
    assert all(x['exit_code']==0 and x['GPU_MiB']==0 and not x['own_GPU_process_matches'] for x in phases)
    resource=json.loads((ROOT/'receipts/RESOURCE_START.json').read_text()); elapsed=time.time()-resource['start_epoch']; controls=json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text())
    assert elapsed<14400 and controls['used']<=25000
    rows_stat=b['per_seed']; table=['|seed|A近CBR|B近CBR|A远CBR|B远CBR|近缺口pp|远缺口pp|交互pp [95%起点区间]|单seed门|','|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows_stat:
        table.append(f"|{r['seed']}|{100*r['CBR_A_near']:.2f}%|{100*r['CBR_B_near']:.2f}%|{100*r['CBR_A_far']:.2f}%|{100*r['CBR_B_far']:.2f}%|{r['gap_near_pp']:.2f}|{r['gap_far_pp']:.2f}|{r['distance_interaction_pp']:.2f} [{r['distance_interaction_pp_ci_low']:.2f},{r['distance_interaction_pp_ci_high']:.2f}]|{'通过' if r['operational_gate_pass'] else '未过'}|")
    signal=pd.read_csv(ROOT/'SIGNAL_BY_DISTANCE.csv'); conditional=pd.read_csv(ROOT/'CONDITIONAL_ACTION_AUDIT.csv'); df=pd.read_csv(ROOT/'BEHAVIOR_ROLLOUTS.csv'); morph=pd.read_csv(ROOT/'TRAJECTORY_MORPHOLOGY.csv')
    sigtable=['|seed|R近WP log-AUROC|R远WP log-AUROC|近负图进展权重占比|远负图进展权重占比|','|---|---:|---:|---:|---:|']
    for seed in (0,1,2):
        z=signal[(signal.seed==seed)&(signal.variant=='B')&(signal.branch=='R')].set_index('distance_bin');n=z.loc['S_NEAR_2_5'];f=z.loc['S_FAR_8_15']
        sigtable.append(f"|{seed}|{n.WP_log_AUROC:.3f}|{f.WP_log_AUROC:.3f}|{100*n.negative_WP_weight_share:.2f}%|{100*f.negative_WP_weight_share:.2f}%|")
    frozen=json.loads((ROOT/'results/SIGNAL_SAMPLE_FREEZE.json').read_text()); max_loss=max(x.get('max_absolute_loss_error',0) for x in fa['checks'])
    main_verdict=b['verdict']
    report=f'''# SAW-DISTANCE-06 复现与诊断报告

三份固定训练seed均观察到远档A/B缺口比近档更大；正式联合H-DIST门未过，因为seed1近档缺口12.5pp不满足预登记<10pp。不能将此写成没有距离趋势，也不把阈值失败改成阳性结论。监督审计保留远条件图区分下降与固定目标邻域动作抵消两个机制线索，因果未建立。

## 固定设计和完成范围

分支：[codex/saw-distance-06-20261013](https://github.com/rollinpioneer/graph/tree/codex/saw-distance-06-20261013)。基点19d1b4b411a00b92866b4512150339472a086f0b；正式D/S科学代码5d4ef4c40fedf93abe3a1e36227268b06ad720bc。官方SAW版本3535a16c901e5de1233ac5a586401b3ddfa5e30d。六份既有A/B×seed0/1/2的1M权重只读，原训练配置和原生计数保持不变。

8个起点，82主条件、11辅助条件，93个均通过独立几何参考准入；全部六模型完成558条各25步局部轨迹，没有新1000步任务评价。主估计使用8起点、9同首通道共同支持块、68主条件；近={{2,4}}、远={{8,10,12,14}}，目标→距离→首通道→起点等权，起点整簇bootstrap5000次，seed26061303。6格和官方15格另列。

## D：行为距离与区域

{chr(10).join(table)}

三个固定模型的距离交互区间下限均>0；seed0/2过单seed门，seed1仅近缺口门失败，因此总体裁定为 `{main_verdict}`。8个起点很少，区间只描述这些固定模型的位置不确定性，不能当训练总体置信保证。seed1近档失败集中(3,1)的2/4格两个目标，都有预登记往返证据；它们不是两个独立起点。

NW/SW区域内部的近远趋势方向在三seed重复，SE有限支持下交互为0；NE无区域内近远共同支持。NW仅3起点、远8/10格，SW仅3起点、远8/12/14格，SE仅2起点、远8格。NW seed1的退化区间[50,50]来自固定起点相同估计，不代表训练精度。结果不能称全区域一致，也不能因总门失败自动宣称区域主导。

同起点(1,8)、同首通道(1,7)、同10格，NW目标(1,2)的B通道结果seed0/1/2为1/0/1，SW目标(7,4)为0/0/0，A两目标都通过，保留目标区域/身份交互候选。但(3,1)的8/10格NE和SE目标B跨区都失败，而(7,4)的8格NW/SE B跨区都通过，反驳单一坏区域解释。全部39个匹配跨区比较与反例保留，不用某一目标对认定区域根因。

## S：真实采样器的冻结训练信号

S1固定8192个合法底层transition，真实HGCDataset从同一NumPy随机流F/R配对，16384 tuples；保留原索引、轨迹、原奖励/masks、value/low/high goals与waypoint。R分支和实际跨轨迹分别记录。S2冻结32组、1642 tuples；25组满足至少16样本及4原轨迹，7组不足。科学tuple银行共18026，六模型前向；另有384个native损失检查tuple评价和150个有效S2参考actor查询，未扩样、不求梯度、不更新。

两头价值均值和两项alpha=3、cap100权重按冻结源码复算。12项原actor/waypoint_loss一致性检查最大误差{max_loss:.9g}，全部网络及优化器树前后散列相同。固定sigma=1，等效标签数据系数为rho_AWR/(2 sigma²)、教师系数rho_WP。

{chr(10).join(sigtable)}

这里S近为2–5、远8–15两个预登记分层集合，与D精确档不同。图进展正负类和原轨迹支持足够时才报告AUROC；全部距离档、neutral、log/截断权重、饱和、尾部、ESS与原轨迹区间保留。F远8–15无样本，近档无负类，相应F/R同档比较及AUC不可估，不能填0.5。R远档图区分下降与负进展权重占比提高是训练指导线索，不等于真实价值错误，也不是真实SGD批次回放证明。

固定同一数据目标的B组合二次系数合力比中位数近档约0.756/0.771/0.788，远档0.184/0.232/0.203；实际s_ref输出也较小。但A的R可能条件探针也有远档低合力而较常正确，且模型输出不等于所算等效均值。因此保留条件抵消候选，不称神经网络就是标签平均器或唯一根因。完整逐组对应、目标偏差、原轨迹与缺支持均保留。

七个不足S2组全部来自两个官方起点，S2不能直接解释原P3两个官方初态。S2固定数据目标g_j距D格中心g_ref最大约0.002627，仍不是严格同一连续目标；近邻位置也不是完整物理初态相同。A/B的value/teacher探针输出同异见SIGNAL_INTERPRETATION，不由相似曲线冒称相关参数逐位一致。

## 逐步形态和独立审计

558新D轨迹和192旧P3轨迹均从带SHA的逐步文件独立重算通过，核心数值最大差{audit['maximum_numeric_error']:.9g}；完整初态integration/qpos/qvel、RNG、wrapper、目标与冻结银行一致。旧P3原目标B的往返证据分别3/16、2/16、6/16，小动作候选7/16、2/16、5/16；没有近静止。不能将全部失败统称不动或摇摆，不能用路径比替代反向和重复访问证明。

## 实施偏差、资源和复现边界

目标平局规则在模型查询前登记了额外首通道覆盖排序，导致(5,6)的6格第二目标选(3,10)而非卡片字典序(1,8)。已封存银行未按结果重选；该唯一中间档偏差不影响主近远估计，仍明确保留，不能宣称逐字完全复现原卡选择规则。

正式运行前修复signal.py遮蔽Python标准库signal的命名问题，保留原失败日志；故障发生在D开始前，新增模型查询与物理步为0。初始源码参考包缺一个预期文件与Markdown两空格换行检查也在准备期处理，卡片原始散列保持。没有重跑正式轨迹、改阈值或补样本。

全轮控制步{controls['used']}：正式D13950、参考准入1026、恢复重复186；GPU0小时/0MiB、参数更新0，限定4核72–75。CPU总墙钟包括准备、协调、实现、分析和故障修复，最终精确值见RESOURCE_AND_EXIT_AUDIT；各阶段峰值RSS均低于16GiB。大NPZ/PKL和完整日志留服务器独立目录，由RAW_ARTIFACT_INDEX路径/散列定位；旧权重、数据、配置、索引与历史结果保持不变。正式SSH关闭与最终发布哈希另以发布回执核验，不能在尚未关闭时提前宣称完成。

本轮没有因果干预和新训练；完成后收口。距离、目标身份、区域和有限地图支持尚未完全拆开，支持监督候选不等于方法贡献。GCIVL/HIQL/QRL等后续只可另卡审批，不属本轮执行。
'''
    (ROOT/'REPRODUCTION_REPORT.md').write_text(report,encoding='utf8')
    (ROOT/'HYPOTHESIS_VERDICT.md').write_text(f'''# 预登记裁定

H-DIST：{main_verdict}。三seed距离交互为48.96/30.21/41.67pp，区间均>0；seed1近缺口12.5pp不满足<10pp，不能更改门槛。正相关趋势保留，联合支持未过。

H-REGION：REGION_OR_INTERACTION_CANDIDATE。部分同距离同通道目标对有重复差异；亦有跨区共同失败/成功。区域内支持稀少且距离档不同，不能裁定区域主导或根因。

H-TIE：DESCRIPTIVE_ONLY。全部最短首通道接受，tie/官方锚点与unique分别报告；唯一通道近档亦有往返，歧义不是这些案例的必要条件。

H-SIGNAL：CONDITIONAL_SIGNAL_CANDIDATES_RETAINED。R远档图权重区分下降、固定目标邻域合力变弱，与部分B小动作/往返对应；F远支持缺失，官方起点S2不足，A的低合力反例及实际输出偏离等效均值均保留，训练因果未建立。

唯一6格目标选择偏差已备案，不改变主近远集合。研究结论限于这六个既有模型、8起点和25步局部窗口。
''',encoding='utf8')
    (ROOT/'NEXT_DECISION.md').write_text('''# 下一步裁定与停止边界

SAW06测量完成并收口。保留“异质的远目标行为缺口及监督候选”，撤下“B普遍忽略目标”及“已找到标签平均根因”的充分解释。联合预登记门未过不等于没有距离趋势；现有地图的区域支持亦不足决定单一原因。

不增加距离档、起点、邻域，不延长旧T，不自动重训练或新增损失。后续如有研究价值，可另提GCIVL A/B各2seed完整训练范围卡，先按该方法实测冻结资源；本卡不启动它，也不沿用历史GPU余量。更广方法与数据集亦另卡。保持全部负例、旧分数和原模型。
''',encoding='utf8')
    write(ROOT/'RESOURCE_AND_EXIT_AUDIT.json',dict(status='ANALYSIS_COMPLETE_PENDING_PUBLICATION_AND_SSH_EXIT',start_epoch=resource['start_epoch'],analysis_end_epoch=time.time(),total_CPU_wall_seconds=elapsed,CPU_affinity=[72,73,74,75],CPU_wall_limit_seconds=14400,peak_RSS_bytes_observed=max(x['peak_RSS_bytes_observed'] for x in phases),RSS_limit_bytes=17179869184,phase_resources=phases,physical_budget=controls,GPU_hours=0,GPU_MiB=0,new_parameter_updates=0,D_rollouts=len(df),S1_pairs=8192,S2_tuples=1642,scientific_tuple_bank_count=18026,native_loss_check_tuple_evaluations=384,S2_reference_actor_queries=150,weights_datasets_configs_indices_unchanged=True,independent_audit_status=audit['status'],SSH_closed=False))
    card=(ROOT/'EXPERIMENT_CARD.md').read_text();card=card.replace('**状态：待用户批准执行。当前仅完成材料核查、静态地图可行性枚举和本卡编制。**','**状态：执行完成。原编制卡及身份见SAW06.md与MANIFEST.json。**')
    card+='\n\n## 完成登记\n\n完成93条件、558局部轨迹、S1 8192对、S2 32组，0训练/GPU；全部逐步审计通过。正式H-DIST联合门未过，正距离交互与监督候选保留。实际6格选择偏差及技术故障如实见报告。CPU/退出精确回执见RESOURCE_AND_EXIT_AUDIT及发布SSH回执。\n'
    (ROOT/'EXPERIMENT_CARD.md').write_text(card,encoding='utf8')
    print('FINAL_REPORT_COMPLETE',main_verdict,flush=True)
if __name__=='__main__':main()
