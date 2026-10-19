from common import *
def load(name):
 p=R/name;return json.loads(p.read_text()) if p.exists() else None
diag=load('results/diagnosis_summary.json');exp=load('results/exploration_decision.json');conf=load('results/confirmation_decision.json');pipe=load('results/pipeline_status.json');reserve=load('C2_reservation.json')
lines=['# EG-LQR-CD-DIAGNOSIS-V4 研究报告','',f"分支：{CFG['branch']}；父结果：{CFG['parent_commit']}。",'',f"执行状态：{pipe['status'] if pipe else 'DIAGNOSIS_RUNNING'}。0 次新训练；原 EG 与三个已有头、全部候选几何固定。",'','C1 已揭晓，本轮 C1 复盘及 F05 对照均为探索性，不是独立确认。','候选质量覆盖率表示有界缓存实际评价了多少候选，不是最近标签匹配成功率；这些已评价候选使用实际姿态的官方完整几何/碰撞/摩擦质量。','']
if diag:
 d=diag['full_C1_AP_decomposition'];c=diag['scene0136_CD_counts'];lines+=['## CD 为什么缩小收益','',f"完整 C1：S 相对 EG 在无 CD 时提升 {d['delta_noCD']:+.3f} AP，带 CD 时 {d['delta_CD']:+.3f} AP，减少 {-d['CD_change_in_rescoring_gain']:.3f} AP。CD 本身让 EG 增加 {d['EG_benefit_CD']:.3f} AP，而让 S 增加 {d['S_benefit_CD']:.3f} AP。此处是同一候选、同一场景的精确指标分解，不将其称为独立机制因果证据。",'',f"scene0136 的候选诊断覆盖原 1024 候选中的 {c['known_quality_coverage']*100:.1f}%（累计逐帧候选）。仅在已有官方质量的候选中，CD 拒绝合格抓法 {int(c['known_good_removed'])} 次、保留合格抓法 {int(c['known_good_kept'])} 次；拒绝真实几何碰撞 {int(c['known_true_collisions_removed'])} 次、仍保留 {int(c['known_true_collisions_kept'])} 次。CD 合格拒绝率 {c['CD_good_rejection_rate_among_known_good']*100:.2f}%，漏过碰撞率 {c['CD_missed_collision_rate_among_known_collisions']*100:.2f}%。分母均为已有质量的候选，不能外推到全部 1024 候选，也不是物理机器人真值。",'','## scene0136 排序表现','','|头|失败→成功|成功→失败|被损伤 EG 合格 Top1 的 Q 排名中位数|其中不在 S 的前50|','|---|---:|---:|---:|---:|']
 for row in diag['scene0136_top1']:lines.append(f"|{row['seed']}|{row['failure_to_success']}|{row['success_to_failure']}|{row['EG_good_demoted_Q_rank_median']}|{row['EG_good_absent_S_selected50']}|")
 lines+=['','以上检查的是实际官方 NMS/CD 路径输出。抓法几何没有变；错误来自选中不同候选。Q 排名是 CD 后尚存的原候选中的分数名次；候选还可能在 NMS/物体候选配额中消失。关联不能单独证明冻结特征无姿态信息、MSE 是根因，或某一几何参数导致退化。','', '宽度、深度、相机接近方向及可见像素面积分层见 diagnosis_geometry_strata.csv。分割可见面积不是完整物体可见比例，也不能直接当作遮挡率。代表性 Bad Case 是按已揭晓 AP 选择的探索性例子。','']
highlight=load('results/geometry_highlights.json')
if highlight:
 h=highlight['models'];eg=h['R0'];q=h['S'];lines+=['## 几何分层的描述性线索','',f"scene0136、CD后官方前50：宽度>=8cm候选的官方几何碰撞率，EG为{eg['wide_collision_rate']*100:.2f}%，S为{q['wide_collision_rate']*100:.2f}%；1cm浅抓取占比分别为{eg['shallow_fraction']*100:.2f}%和{q['shallow_fraction']*100:.2f}%。这些是实际选中候选的条件分布，跨帧/头相关，不能据此声称宽度或深度是退化原因。",'']
lines+=['CD使用相同原始候选几何，因而所有评分方案的逐帧CD mask及全局拒绝率相同。不同AP收益来自过滤后排名及官方NMS/物体配额选择的交互；没有改变CD阈值。','']
if exp:
 x=exp['primary'];s=exp['S_primary'];delta=exp['F05_minus_S'];lines+=['## 固定 F05 探索','','F05 始终是 0.5*z(EG)+0.5*z(Q)，全1024候选、float64、ddof0；没有系数、头或检查点搜索。','',f"F05+CD 相对 EG 平均 {x['delta']:+.3f} AP，描述性95%CI=[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]；三头差值 {x['seed_delta']}，正向场景 {x['positive_scenes']}/12。S 相对 EG {s['delta']:+.3f} AP。",'',f"scene0136 中 F05 相对 S {delta['scene_delta']['136']:+.3f} AP；最差场景差值从 S 的 {min(s['scene_delta'].values()):+.3f} 变为 F05 的 {min(x['scene_delta'].values()):+.3f} AP。",'',f"冻结放行条件：{exp['checks']}；判决：{exp['status']}。即便通过，也只能支持选定 F05 进入新数据确认，不能把 C1 当独立确认。",'']
if reserve:lines += ['## 新确认使用边界','',f"已知项目使用历史审计：{reserve['status']}；预留场景：{reserve['selected']}。审计不能保证未知外部历史没有暴露。只有探索闸门通过才写入 C2_method_lock 并打开新性能。",'']
if conf:
 x=conf['primary'];lines+=['## C2 固定方法确认','',f"结论：**{conf['status']}**。F05 相对 EG 平均 {x['delta']:+.3f} AP，场景分层95%CI=[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]；三头差值 {x['seed_delta']}；正向场景 {x['positive_scenes']}/12。",'',f"Seen/Similar/Novel 差值：{x['split_delta']}。",'','仅代表冻结的12场景×完整256帧，不是官方90场景全评测。三头先场景内平均，划分/场景等权，分层场景bootstrap10000次，种子20261019；不把帧、候选或头当独立样本。AP不等于机器人抓取成功率。','']
lines+=['## 可信度、资源与停止','','所有旧结果只读，新目录补算固定F05选中新候选的官方完整几何/摩擦质量。复用质量绑定输入、几何、骨干、评价源码身份。R0逐帧复验须与V3一致；两个固定帧额外对照完整原生官方路径。未评价质量仍是未知，不以失败填充。','', '诊断曾因提前把候选转float64造成CD边界重放不一致，已恢复原始dtype；原失败日志和旧诊断保留，最终608帧统一重放原始dtype。此修改没有改变V3结果、CD阈值或模型。','', '诊断最初上限8；首个F05性能前按资源补充将F05/C2上限改为13：12个CPU场景子任务+控制；新确认导出期间GPU1+CPU11+控制，导出后CPU12+控制。每子任务单线程、低优先级；无监控自动化，不终止其他任务。资源和实际条件状态以pipeline_status及各COMPLETE回执为准。','', '本卡结束后不自动更换融合系数、扩增训练或追加新场景。大输入、逐候选/逐帧文件、权重和完整日志留服务器；代码与轻量结果及哈希归档。']
(R/'results/research_report.md').write_text('\n'.join(lines)+'\n')
print('REPORT_BUILT',pipe['status'] if pipe else 'DIAGNOSIS_RUNNING',flush=True)
