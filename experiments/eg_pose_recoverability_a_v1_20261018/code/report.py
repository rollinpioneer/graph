from core import *
out=R/'results';d=json.loads((out/'summary.json').read_text());replay=json.loads((out/'replay_receipt.json').read_text());quality=[json.loads((out/f'quality_{s:04d}.json').read_text()) for s in SCENES]
lines=['# EG 姿态候选可恢复性诊断 A','', '分支：`codex/eg-pose-recoverability-a-v1-20261018`；预注册 `546c6d70b19e3c8b87ec4437d3ea32f779fb750e`。','', '零训练；原开发10场景640个Top-1：187失败、453成功。固定中心、接近方向、预测宽度，12角度×4深度；真实几何/碰撞/摩擦质量，正μ≤0.4合格。失败中170例可恢复。是已观察开发案例的离线诊断，不是AP、机器人成功率或独立泛化评价。','', '## 失败候选覆盖','', '|规则|找回数/187|覆盖187比例|覆盖170比例|场景95%区间（187分母）|','|---|---:|---:|---:|---|']
for x in d['strategies']:
 lo,hi=x['failure_coverage_ci95'];lines.append(f"|{x['strategy']}|{x['recovered_or_expected']:.3f}|{100*x['failure_coverage']:.2f}%|{100*x['recoverable_coverage']:.2f}%|[{100*lo:.2f}%,{100*hi:.2f}%]|")
lines+=['','随机为无放回抽样的精确超几何期望，不挑随机种子。keep1保留原动作。联合排名是softmax角度×softmax深度，等价于对应logit和排序；Top12为联合前12。','', '## 预注册配对比较','', '|策略/同预算保留原动作随机|差值pp|95%CI|四项校正单侧下界pp|可靠优势|','|---|---:|---|---:|---|']
for x in d['paired_tests']:
 lo,hi=x['ci95_pp'];lines.append(f"|{x['strategy']}/{x['control']}|{x['difference_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{x['family_corrected_lower_pp']:+.2f}|{x['reliable_advantage']}|")
lines+=['',f"冻结判据：{d['decision']}；通过策略：{d['reliable_strategies']}。10场景聚类bootstrap20,000次、案例加权，4项Bonferroni校正。只有10个簇，不能外推为普遍保证。",'', '## 单参数可行性（互斥类别）','', '|类别|数量|','|---|---:|']
for k,v in d['categories'].items():lines.append(f'|{k}|{v}|')
lines+=['','这是有限格点的反事实可行性，不是训练机制归因。单独改变角度或深度均可恢复的案例，不能称作两分支均有因果错误。','', '## 最佳合格姿态的分支排名','', '最佳指最小合格μ；全部并列排名和任意合格姿态排名保存在per_case.json。下表canonical在最佳μ并列中按联合分数、再类索引选一个，对模型排名有利，不应单独作为有效性证据。','', '|排名|角度数量|深度数量|','|---|---:|---:|']
h=np.array(d['canonical_best_rank_histogram'])
for i in range(12):lines.append(f"|{i+1}|{h.sum(1)[i]}|{h.sum(0)[i] if i<4 else '—'}|")
lines+=['','## 原成功抓法的风险','', '|规则|oracle保留率|集合内均匀选择失败率|强制选最高分非原动作失败率|','|---|---:|---:|---:|']
for x in d['strategies']:
 v=x['success_forced_top_alternative_failure'];t='—' if v is None else f'{100*v:.2f}%';lines.append(f"|{x['strategy']}|{100*x['success_oracle_retention']:.2f}%|{100*x['success_uniform_choice_failure']:.2f}%|{t}|")
lines+=['','保留原动作的集合oracle覆盖必为100%，不是选择器安全证据。联合argmax仍是原动作：不会救回失败，也不会损坏成功。强制改选和均匀选择只是压力测试，不代表未来质量头的实际损坏率。','', '## 核验与成本','',f"640个选定帧、原batch4共2560帧推理，预测最大绝对差 {replay['max_abs_raw_prediction_difference']}，旋转映射误差 {replay['max_rotation_symmetry_error']:.3g}。原动作质量逐例核对通过。",f"失败复用8976次质量；成功新增{sum(x['new_native_calls'] for x in quality)}次原生评价。新增计算用于用户要求的成功组检查。10个单线程低优先级CPU进程，GPU0仅冻结推理{replay['seconds']:.1f}s；质量进程时间合计{sum(x['seconds'] for x in quality)/3600:.3f}h，最慢场景{max(x['seconds'] for x in quality):.1f}s；不是独占整机成本。",'模型/输入/历史缓存只读，哈希及逐案例、场景结果可核验。原生48格点通过平行夹爪π对称映射到模型类别；不使用标签坐标旋转或标签宽度。没有新训练或新AP。']
(out/'final_report.md').write_text('\n'.join(lines)+'\n')
write(out/'resource_summary.json',dict(replay=replay,quality=quality,new_native_calls=sum(x['new_native_calls'] for x in quality),reused_native_calls=8976,cpu_workers=10,threads=1,training_updates=0))
print('REPORT_COMPLETE')
interpretation='''
## 结论与后续边界

Top3×2找回141/187（覆盖可恢复案例82.94%）；Top2×3找回149/187（87.65%）；联合Top6为133/187（78.24%）；联合Top12为159/187（93.53%）；全48为170/187。随机6期望94.68例，保留原动作随机5期望86.86例。预注册校正后，Top3×2、Top2×3、联合Top12通过可靠优势判据，联合Top6没有通过。Top2×3点估计较好不等于已证明优于Top3×2，不能事后改成主胜出策略。

分支排名保留有用信息，停止条件“所有Top-K均无可靠随机优势”未触发。但联合分数Top1仍是原失败动作，因此候选覆盖不等于无需新质量证据就能选择正确。联合Top12较联合Top6多找回26例，全48再多11例；增加候选有收益，也扩大误选空间。

124例只改角度即可恢复，103例只改深度即可恢复，其中69例两种单独改变都可恢复。158/170（92.94%）至少有一种单参数解，只有12/170（7.06%）在本格点需要同时改变。角度单独可修复55例、深度单独可修复34例，不能将其转写成分类头的因果错误比例。当前证据更支持先比较角度重选、深度重选与联合重选，不足以主张联合机制不可替代。

453个原成功例中，强制选择最高联合分数的非原动作损坏112例（24.72%）。各确定候选集合都包含这一第二高联合组合，所以此压力测试结果相同；并不代表所有候选规则等价。Top3×2集合内均匀选择失败率34.36%，Top2×3为37.01%，联合Top6为30.46%，联合Top12为48.42%。保留原动作时oracle100%覆盖是构造保证，不能当作选择器安全性。

建议保留局部姿态重选作为可研究方向，但先建立保守选择/拒绝修改机制，并用单参数对照验证复杂联合头是否必要。未来任何方法需同时报告救回数和成功损坏数；这些压力测试不能冒充其实际损坏率。本卡未训练任何新网络，未自动启动下一轮。这里只是已观察开发场景上的动机证据，仍需另设未用于选择的确认材料。
'''
with (out/'final_report.md').open('a') as f:f.write(interpretation)
write(out/'conclusion.json',dict(status='COMPLETE',decision=d['decision'],reliable_strategies=d['reliable_strategies'],failure_recovery_top3x2=141,failure_recovery_top2x3=149,failure_recovery_joint6=133,failure_recovery_joint12=159,failure_recovery_all48=170,single_parameter_suffices=158,joint_required=12,success_cases=453,forced_change_damaged=112,no_training=True,interpretation='Ranking information is useful; oracle coverage is not learned selection efficacy; joint modeling necessity is not established.'))
