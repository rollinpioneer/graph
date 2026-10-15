import json,hashlib,ast
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def rd(f):return json.loads((R/f).read_text())
def save(f,d):(R/f).write_text(json.dumps(d,indent=2,ensure_ascii=False))
s=rd('reports/summary.json');rows=s['states'];raw=rd('reports/branch_results.json');res=rd('receipts/probe_result.json');ex=rd('receipts/exit.json')
index=[]
for x in raw:
 p=Path(x['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==x['sha256'];index.append(dict(state=x['state'],candidate=x['candidate'],sequence_seed=x['sequence_seed'],steps=x['steps'],path=str(p),bytes=p.stat().st_size,sha256=h))
save('reports/branch_index.json',index)
checks=[]
for i in [0,1,100,126,180,251]:
 x=raw[i]
 with np.load(x['path']) as a:
  assert len(a['actions'])==x['steps'];assert float(a['success'].max())==x['success'];assert float(a['rewards'].sum())==x['return_'];assert int(np.count_nonzero(a['wall_contacts']))==x['wall_contact_steps'];assert all(np.isfinite(v).all() for v in a.values());checks.append(dict(row=i,state=x['state'],candidate=x['candidate'],verified=True))
forbidden={'update','batch_update','multi_sample_and_update','apply_loss_fn','apply_gradients'}
tree=ast.parse((R/'code/probe.py').read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden];assert set(calls)<={'os.environ.update','arr.update'},calls
save('receipts/verification.json',dict(branch_file_hashes_verified=len(index),coarse_numeric_checks=checks,zero_update_calls_in_entry=True,full_agent_and_checkpoint_hashes_unchanged=res['assets_unchanged'],maximum_first_transition_error=s['integrity']['first_transition_max_error'],optimizer_updates=0,exit_code=ex['exit_code']))
text=['# GFP 首动作替代的稳健性验证','', '结论：存在局部正向线索，但本轮没有建立“稳定生成有效替代动作却未执行”的普遍机制。此前四个获救配对均未达到预先冻结的重复获益门槛（0/4）；按卡停止扩展，不增加模块、不调参、不训练。此结论不是证明替代动作永远无效。','', '完成12状态 × 3个旧首动作 × 7条新共同后续随机序列，共252条新分支、62,041个环境步，参数更新0。原来的36条分支只读取，不重跑。以下主结果完全排除旧序列。','', '## 四个预先指定的复验对象','', '配对胜：替代动作成功而原动作失败；负为反向；其余为平。全部是在相同后续随机序列下比较。','', '|状态|替代动作|原动作成功/7|替代成功/7|胜/负/平|', '|---|---|---:|---:|---|']
names={'original_actor':'原 Actor','resampled_actor':'Actor 重采样','vabc':'VaBC'}
for t in s['predeclared_replication']:
 x=next(v for v in rows if v['state']==t['state']);text.append(f"|{t['state']}|{names[t['candidate']]}|{x['new_successes']['original_actor']}|{x['new_successes'][t['candidate']]}|{t['wins']}/{t['losses']}/{t['ties']}|")
text+=['','运行前冻结的操作性门槛：同一配对至少4胜、至多1负；至少2/4个旧获救配对达到门槛才继续该窄假设。本轮0/4通过。门槛是小样本研究决策规则，不是显著性检验，也不是对所有可能收益的否定。','','## 所有状态：新7条与旧1条分开','', '每格顺序为 原 Actor / Actor 重采样 / VaBC。旧序列已经参与候选线索发现，合并8条只能作为辅助记录。','', '|状态|新7条成功数|旧1条成功数|合并8条成功数|','|---|---|---|---|']
for x in rows:
 vals=[' / '.join(str(v) for v in x[k].values()) for k in ['new_successes','old_successes','pooled_successes']];text.append('|'+x['state']+'|'+'|'.join(vals)+'|')
text+=['','![新序列成功次数](new_suffix_successes.png)','','## 正反证据与研究解释','','- s0 e007 仍有正向线索：Actor 重采样比原动作2胜0负；VaBC 为3胜0负，成功数5/7。但这些不足以建立跨状态稳定缺陷。','- s0 e021 的原动作在新序列下成功5/7，旧获救 VaBC 为4/7，配对1胜2负。旧的单次获救不能预测此次相对优势。','- s1 e033 的旧获救 Actor 重采样与原动作均1/7，配对1胜1负。另一个固定 VaBC 动作成功3/7，是应保留的探索性线索，不能事后替代预先指定的复验对象。','- s0 e027 停滞前状态的旧获救 VaBC 成功0/7，原动作1/7。首动作效果依赖后续随机序列。','- s1 e050 的 Actor 重采样成功5/7，原动作3/7，但配对4胜2负，存在收益也存在损失。VaBC 2/7，不能据此宣称 VaBC 普遍优于普通 Actor 重采样。','','本轮只检验固定候选动作在不同后续随机序列下的效果，没有测试在线多采样选择器。能由事后观察挑出某个成功分支，不等于策略知道如何选择。不同状态的后续随机种子共用，两个成功对照还来自同一原始回合，不能把所有分支当作独立同分布样本做总体推断。','','## 停滞前状态：不能只看300步内成功','', '以下为7条新序列平均值，距离进展为初始减最终距离，正值表示末端更接近目标；碰墙为有墙接触的步骤占比。停滞为所有完整100步窗口中，离窗口起点最远距离仍小于0.5m的比例。','', '|状态|首动作|末端距离进展(m)|墙接触比例|停滞窗口比例|','|---|---|---:|---:|---:|']
for x in rows:
 if x['category']=='pre_stall':
  for a,v in x['secondary_means'].items():text.append(f"|{x['state']}|{names[a]}|{v['progress_final']:.3f}|{v['wall_contact_fraction']:.3f}|{v['stagnation_fraction']:.3f}|")
text+=['','停滞前4个状态中，原动作总共1次300步内成功，两个替代来源均0次。这不是说没有短期位置变化：例如 s1 e038 三者平均末端距离进展约7.64/8.15/8.49m，均未成功，说明短窗口成功指标受起点距离限制；s0 e023 重采样末端进展改善，但墙接触与停滞比例同时升高。位置进展不等于路线正确，低头也不等于跌倒原因。完整次级指标和每序列配对差保存在 summary.json 与 new_branch_metrics.csv。','','## 可信度、身份和成本','', '原始 GFP commit：`1abbd5f0e287caf9bda70e7bc655d0a499ac8401`；仅用 seeds 0/1 的1M冻结权重。任务 HumanoidMaze-medium task1；OGBench1.1.0、MuJoCo3.3.5、JAX0.6.2。完整模型/优化器/Agent RNG 和 checkpoint 哈希前后不变；全部252次初始恢复及首步状态、奖励、终止检查通过，最大首步误差0。六条分支的数值指标另作粗粒度复算，252个文件哈希核验通过。没有调用参数更新入口。','', f"实际进程退出码{ex['exit_code']}；墙钟{ex['wall_seconds']:.2f}秒，CPU {res['cpu_seconds']:.2f}秒，进程峰值RSS {res['max_rss_kib']/1024:.1f} MiB。GPU4共享运行，5秒采样的本进程显存峰值{ex['peak_sampled_gpu_mib']} MiB；进程墙钟折合{ex['wall_seconds']/3600:.5f} GPU驻留小时上界，非独占整卡算力消耗。正显存采样跨度205.45秒；采样不能给出精确GPU忙碌积分。代码准备、归档时间未算训练或推理吞吐。",'', '## 停止点与交付','', '此次窄问题已完成，停止新增环境交互。只保留“特定状态下候选动作有局部收益且依赖后续随机性”的观察，不升级为 Critic 排序错误、动作时间组织缺陷或新模块依据。样本量小，未排除小幅真实收益；但当前证据不足以支持继续包装为稳定复杂控制机制问题。','', '分支：[codex/gfp-proposal-robustness-v1-20261015](https://github.com/rollinpioneer/graph/tree/codex/gfp-proposal-robustness-v1-20261015)。冻结协议提交594fc91eacd16280f08efae0cf6dc7eb1e886bcd，执行代码提交94428fa117e12d54e3cdebed2fde0e1e6a11ec73，均在新增交互之前推送验证。','', '服务器根目录：`/home/xushijie3/work/gfp_proposal_robustness_v1_20261015`。完整252条数值轨迹在 branches/，身份索引 reports/branch_index.json；代码、卡、配置、原始轻量指标与回执入新分支，旧文件与旧结果不变。最终归档提交见 receipts/final_archive.json。']
(R/'reports/REPORT.md').write_text('\n'.join(text)+'\n')
print('VERIFIED_AND_REPORTED',len(index))
