from common import *
def load(name):
 p=R/name;return json.loads(p.read_text()) if p.exists() else None
diag=load('results/diagnosis_summary.json');exp=load('results/exploration_decision.json');conf=load('results/confirmation_decision.json');pipe=load('results/pipeline_status.json');reserve=load('C2_reservation.json')
lines=['# EG-LQR-CD-DIAGNOSIS-V4 研究报告','',f"分支：{CFG['branch']}；父结果：{CFG['parent_commit']}。",'',f"执行状态：{pipe['status'] if pipe else 'DIAGNOSIS_RUNNING'}。0 次新训练；原 EG 与三个已有头、全部候选几何固定。",'','C1 已揭晓，本轮 C1 复盘及 F05 对照均为探索性，不是独立确认。','候选质量覆盖率表示有界缓存实际评价了多少候选，不是最近标签匹配成功率；这些已评价候选使用实际姿态的官方完整几何/碰撞/摩擦质量。','']
if diag:
 d=diag['full_C1_AP_decomposition'];c=diag['scene0136_CD_counts'];lines+=['## CD 为什么缩小收益','',f"完整 C1：S 相对 EG 在无 CD 时提升 {d['delta_noCD']:+.3f} AP，带 CD 时 {d['delta_CD']:+.3f} AP，减少 {-d['CD_change_in_rescoring_gain']:.3f} AP。CD 本身让 EG 增加 {d['EG_benefit_CD']:.3f} AP，而让 S 增加 {d['S_benefit_CD']:.3f} AP。此处是同一候选、同一场景的精确指标分解，不将其称为独立机制因果证据。",'',f"scene0136 的候选诊断覆盖原 1024 候选中的 {c['known_quality_coverage']*100:.1f}%（累计逐帧候选）。仅在已有官方质量的候选中，CD 拒绝合格抓法 {int(c['known_good_removed'])} 次、保留合格抓法 {int(c['known_good_kept'])} 次；拒绝官方碰撞/空抓联合拒绝候选 {int(c['known_true_collisions_removed'])} 次、仍保留 {int(c['known_true_collisions_kept'])} 次。CD 合格拒绝率 {c['CD_good_rejection_rate_among_known_good']*100:.2f}%，联合拒绝候选保留率 {c['CD_missed_collision_rate_among_known_collisions']*100:.2f}%。分母均为已有质量的候选，不能外推到全部 1024 候选，也不是物理机器人真值。",'','## scene0136 排序表现','','|头|失败→成功|成功→失败|被损伤 EG 合格 Top1 的 Q 排名中位数|其中不在 S 的前50|','|---|---:|---:|---:|---:|']
 for row in diag['scene0136_top1']:lines.append(f"|{row['seed']}|{row['failure_to_success']}|{row['success_to_failure']}|{row['EG_good_demoted_Q_rank_median']}|{row['EG_good_absent_S_selected50']}|")
 lines+=['','以上检查的是实际官方 NMS/CD 路径输出。抓法几何没有变；错误来自选中不同候选。Q 排名是 CD 后尚存的原候选中的分数名次；候选还可能在 NMS/物体候选配额中消失。关联不能单独证明冻结特征无姿态信息、MSE 是根因，或某一几何参数导致退化。','', '宽度、深度、相机接近方向及可见像素面积分层见 diagnosis_geometry_strata.csv。分割可见面积不是完整物体可见比例，也不能直接当作遮挡率。代表性 Bad Case 是按已揭晓 AP 选择的探索性例子。','']
highlight=load('results/geometry_highlights.json')
if highlight:
 h=highlight['models'];eg=h['R0'];q=h['S'];lines+=['## 几何分层的描述性线索','',f"scene0136、CD后官方前50：宽度>=8cm候选的官方联合拒绝率，EG为{eg['wide_collision_rate']*100:.2f}%，S为{q['wide_collision_rate']*100:.2f}%；1cm浅抓取占比分别为{eg['shallow_fraction']*100:.2f}%和{q['shallow_fraction']*100:.2f}%。这些是实际选中候选的条件分布，跨帧/头相关，不能据此声称宽度或深度是退化原因。",'']
lines+=['CD使用相同原始候选几何，因而所有评分方案的逐帧CD mask及全局拒绝率相同。不同AP收益来自过滤后排名及官方NMS/物体配额选择的交互；没有改变CD阈值。','']
if exp:
 x=exp['primary'];s=exp['S_primary'];delta=exp['F05_minus_S'];lines+=['## 固定 F05 探索','','F05 始终是 0.5*z(EG)+0.5*z(Q)，全1024候选、float64、ddof0；没有系数、头或检查点搜索。','',f"F05+CD 相对 EG 平均 {x['delta']:+.3f} AP，描述性95%CI=[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]；三头差值 {x['seed_delta']}，正向场景 {x['positive_scenes']}/12。S 相对 EG {s['delta']:+.3f} AP。",'',f"scene0136 中 F05 相对 S {delta['scene_delta']['136']:+.3f} AP；最差场景差值从 S 的 {min(s['scene_delta'].values()):+.3f} 变为 F05 的 {min(x['scene_delta'].values()):+.3f} AP。",'',f"冻结放行条件：{exp['checks']}；判决：{exp['status']}。即便通过，也只能支持选定 F05 进入新数据确认，不能把 C1 当独立确认。",'']
if reserve:lines += ['## 新确认使用边界','',f"已知项目使用历史审计：{reserve['status']}；预留场景：{reserve['selected']}。审计不能保证未知外部历史没有暴露。只有探索闸门通过才写入 C2_method_lock 并打开新性能。",'']
semantics=load('results/collision_semantics.json')
if semantics:
 lines+=['## 官方拒绝标记的语义','', 'GraspNet collision标记=几何相交 OR 空抓；不能将联合标记直接称为纯几何碰撞。非空碰撞定义为collision AND NOT empty，空抓也可能同时相交，现有缓存不进一步拆分其相交状态。初版诊断CSV中known_true_collisions字段沿用旧名，数值实际是官方联合标记；AP始终使用官方定义，没有改动。','', 'ModelFree CD检查观测点云密度及5cm接近路径，官方质量使用完整物体模型、静态抓手和空抓判定；二者不是同一判据。CD拒绝官方合格候选不自动证明CD错误，也不能把所有保留的官方联合拒绝候选叫作纯碰撞漏检。','']
 ds=load('results/collision_disaggregated_comparisons.json') or []
 for v in ds:
  if v['use_cd']:
   d=v['delta_percentage_points'];lines.append(f"{v['phase']} {v['family']}相对EG：Top1官方联合拒绝{d['top1_collision']:+.3f}个百分点，空抓{d['top1_empty']:+.3f}，非空碰撞{d['top1_nonempty_collision']:+.3f}；描述性拆分，非新增确认终点。")
 lines+=['']
if conf:
 x=conf['primary'];lines+=['## C2 固定方法确认','',f"结论：**{conf['status']}**。F05 相对 EG 平均 {x['delta']:+.3f} AP，场景分层95%CI=[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]；三头差值 {x['seed_delta']}；正向场景 {x['positive_scenes']}/12。",'',f"Seen/Similar/Novel 差值：{x['split_delta']}。",'','仅代表冻结的12场景×完整256帧，不是官方90场景全评测。三头先场景内平均，划分/场景等权，分层场景bootstrap10000次，种子20261019；不把帧、候选或头当独立样本。AP不等于机器人抓取成功率。','']
raw_audit=load('results/C2_raw_observation_audit.json')
if raw_audit:
 lines+=['## 原始观测核验','',f"C2 {raw_audit['unique_current']}/{raw_audit['current_count']}帧联合RGB/深度/K指纹互异，与原C1精确重复数{raw_audit['overlap_with_prior_C1']}。审计只覆盖C2内部和原C1原始指纹；其他历史按场景ID审计，不保证未知外部使用。完整256帧协议保留；无重复时去重敏感性与主结果相同。",'']
for phase,title in [('exploration','C1探索'),('confirmation','C2确认')]:
 path=R/f'results/{phase}_per_scene.csv'
 if not path.exists():continue
 with path.open(newline='') as f:scene_rows=list(csv.DictReader(f))
 lines+=[f'## {title}绝对AP与辅助指标','', '下表仅为每划分固定4场景的AP，不是每划分30场景全量AP；F05三头分别评价后平均。','', '|组别|Seen AP|Similar AP|Novel AP|等权平均AP|Top1合格率%|','|---|---:|---:|---:|---:|---:|']
 families=['R0','S','F05'] if phase=='exploration' else ['R0','F05']
 for cd in [False,True]:
  suffix='_CD' if cd else ''
  for family in families:
   names=['R0'+suffix] if family=='R0' else [f'{family}_{seed}{suffix}' for seed in SEEDS];vals=[];valids=[]
   for split in SPLITS:
    selected=[v for v in scene_rows if v['split']==split and v['configuration'] in names];vals.append(float(np.mean([float(v['ap']) for v in selected])));valids.append(float(np.mean([float(v['top1_valid']) for v in selected]))*100)
   lines.append(f"|{family}{'+CD' if cd else ''}|{vals[0]:.3f}|{vals[1]:.3f}|{vals[2]:.3f}|{np.mean(vals):.3f}|{np.mean(valids):.3f}|")
 comps=load(f'results/{phase}_comparisons.json');primary=next(v for v in comps if v['family']=='F05' and v.get('reference','R0')=='R0' and v['use_cd'])
 lines+=['', '|F05+CD相对EG+CD辅助指标|差值|描述性95%CI|','|---|---:|---|']
 for metric in ['ap04','ap08','top1_valid','top1_collision','top1_empty']:
  v=primary['metrics'][metric];lines.append(f"|{metric}|{v['delta']:+.3f}|[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]|")
 lines+=['', 'AP差值为AP百分点，其余为率的百分点；collision为含empty的官方联合标记。辅助区间为描述性，不作新增成功标准。','']
visual=load('results/visualized_bad_cases.json')
if visual:
 lines+=['## 探索性示例','', '三个例子固定来自此前最差AP列表；官方抓手网格投影到RGB/深度，仅作位置/姿态示意，质量依据仍是实际姿态的完整官方评价。图注reject含empty。','']
 for f in visual['files']:lines.append(f"![探索性Bad Case]({Path(f).name})")
 lines+=['', '例如0136/0093头20261018：S选中Q=0.820的非空碰撞候选，F05换到mu=0.4的合格候选；但该帧AP仍从EG53.81降到F0539.62。0102/0144中S候选无碰撞却质量=-1，F05换到mu=0.2后AP仍低于EG。0136/0077三个方法的Top1都未合格。它们显示修复Top1不等于恢复整张候选列表的AP，也不能声称F05修好了每个失败例子。','']
lines+=['## 可信度、资源与停止','','所有旧结果只读，新目录补算固定F05选中新候选的官方完整几何/摩擦质量。复用质量绑定输入、几何、骨干、评价源码身份。R0逐帧复验须与V3一致；两个固定帧额外对照完整原生官方路径。未评价质量仍是未知，不以失败填充。','', '诊断曾因提前把候选转float64造成CD边界重放不一致，已恢复原始dtype；原失败日志和旧诊断保留，最终608帧统一重放原始dtype。此修改没有改变V3结果、CD阈值或模型。','', '诊断最初上限8；首个F05性能前按资源补充将F05/C2上限改为13：12个CPU场景子任务+控制；新确认导出期间GPU1+CPU11+控制，导出后CPU12+控制。每子任务单线程、低优先级；无监控自动化，不终止其他任务。资源和实际条件状态以pipeline_status及各COMPLETE回执为准。','', '本卡结束后不自动更换融合系数、扩增训练或追加新场景。大输入、逐候选/逐帧文件、权重和完整日志留服务器；代码与轻量结果及哈希归档。']
(R/'results/research_report.md').write_text('\n'.join(lines)+'\n')
print('REPORT_BUILT',pipe['status'] if pipe else 'DIAGNOSIS_RUNNING',flush=True)
